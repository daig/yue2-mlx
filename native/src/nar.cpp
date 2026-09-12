#include "lyra/nar.hpp"
#include "lyra/conversion.hpp"
#include "lyra/nar_attention.hpp"
#include "lyra/protocol.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <mlx/io.h>
#include <tuple>

namespace lyra {
namespace {
using Shapes = std::map<std::string, mx::Shape>;
int integer(const Json &config, const char *name) {
  if (!config.contains(name) || !config.at(name).is_number_integer() ||
      config.at(name).is_boolean() || config.at(name).get<double>() < 1 ||
      config.at(name).get<double>() > std::numeric_limits<int>::max())
    throw Error("ValueError", std::string("config.json field '") + name +
                                  "' must be an integer >= 1");
  return config.at(name).get<int>();
}
void check_cancelled(const Cancelled &cancelled, const char *message) {
  if (cancelled && cancelled())
    throw Error("InterruptedError", message);
}
void layer_shapes(Shapes &shapes, const Json &config, int index,
                  bool acoustic) {
  const int hidden = config.at("hidden_size"),
            intermediate = config.at("intermediate_size");
  const int dim = config.at("head_dim"),
            q = int(config.at("num_attention_heads")) * dim;
  const int kv = int(config.at("num_key_value_heads")) * dim;
  const std::string base = "model.layers." + std::to_string(index) + ".";
  const std::string attention =
      base + (acoustic ? "nar_self_attn." : "self_attn.");
  shapes[base + (acoustic ? "nar_input_layernorm.weight"
                          : "input_layernorm.weight")] = {hidden};
  shapes[base + (acoustic ? "nar_pre_mlp_layernorm.weight"
                          : "post_attention_layernorm.weight")] = {hidden};
  shapes[attention + "q_proj.weight"] = {q, hidden};
  shapes[attention + "k_proj.weight"] = {kv, hidden};
  shapes[attention + "v_proj.weight"] = {kv, hidden};
  shapes[attention + "o_proj.weight"] = {hidden, q};
  shapes[attention + "q_norm.weight"] = {dim};
  shapes[attention + "k_norm.weight"] = {dim};
  const std::string mlp = base + (acoustic ? "nar_mlp." : "mlp.");
  shapes[mlp + "gate_proj.weight"] = {intermediate, hidden};
  shapes[mlp + "up_proj.weight"] = {intermediate, hidden};
  shapes[mlp + "down_proj.weight"] = {hidden, intermediate};
}
void validate_weights(const Weights &weights, const Shapes &expected,
                      bool acoustic) {
  for (const auto &[name, shape] : expected) {
    auto found = weights.find(name);
    if (found == weights.end())
      throw Error("ValueError",
                  "Missing " +
                      std::string(acoustic ? "acoustic" : "shared AR") +
                      " tensor '" + name + "'");
    if (found->second.shape() != shape)
      throw Error("ValueError",
                  "Tensor '" + name + "' has an incompatible shape");
    if (found->second.dtype() != mx::bfloat16)
      throw Error("ValueError",
                  "Tensor '" + name +
                      "' must be BF16; quantized AR cannot condition NAR");
  }
  if (acoustic)
    for (const auto &[name, value] : weights)
      if (!expected.contains(name))
        throw Error("ValueError", "Unexpected acoustic tensor: " + name);
}
mx::array bf16(const mx::array &x) { return mx::astype(x, mx::bfloat16); }
mx::array fp32(const mx::array &x) { return mx::astype(x, mx::float32); }
mx::array sequence_slice(const mx::array &x, int end) {
  return mx::slice(x, {0, 0, 0, 0}, {x.shape(0), x.shape(1), end, x.shape(3)});
}
std::pair<mx::array, mx::array> rope_factors(const mx::array &inverse,
                                             int start, int length) {
  auto angles =
      mx::reshape(mx::arange(start, start + length, mx::float32), {length, 1}) *
      mx::reshape(inverse, {1, -1});
  return {mx::reshape(bf16(mx::cos(angles)), {1, 1, length, -1}),
          mx::reshape(bf16(mx::sin(angles)), {1, 1, length, -1})};
}
mx::array apply_rope(const mx::array &x,
                     const std::pair<mx::array, mx::array> &factors) {
  const int half = x.shape(3) / 2;
  auto first =
      mx::slice(x, {0, 0, 0, 0}, {x.shape(0), x.shape(1), x.shape(2), half});
  auto second = mx::slice(x, {0, 0, 0, half}, x.shape());
  auto first_cos = bf16(first * factors.first),
       second_sin = bf16(second * factors.second);
  auto second_cos = bf16(second * factors.first),
       first_sin = bf16(first * factors.second);
  return mx::concatenate(
      {bf16(first_cos - second_sin), bf16(second_cos + first_sin)}, -1);
}
std::tuple<mx::array, mx::array, mx::array>
project(const AcousticModel &model, const Weights &weights,
        const std::string &name, const mx::array &x,
        const std::pair<mx::array, mx::array> &factors) {
  const int heads = model.config.at("num_attention_heads"),
            kv = model.config.at("num_key_value_heads");
  const int dim = model.config.at("head_dim");
  const float eps = model.config.at("rms_norm_eps");
  auto q = mx::reshape(model_ops::linear(x, weights, name + "q_proj"),
                       {1, x.shape(1), heads, dim});
  auto k = mx::reshape(model_ops::linear(x, weights, name + "k_proj"),
                       {1, x.shape(1), kv, dim});
  auto v = mx::reshape(model_ops::linear(x, weights, name + "v_proj"),
                       {1, x.shape(1), kv, dim});
  q = mx::transpose(
      model_ops::rms_norm(q, weights.at(name + "q_norm.weight"), eps),
      {0, 2, 1, 3});
  k = mx::transpose(
      model_ops::rms_norm(k, weights.at(name + "k_norm.weight"), eps),
      {0, 2, 1, 3});
  return {apply_rope(q, factors), apply_rope(k, factors),
          mx::transpose(v, {0, 2, 1, 3})};
}
class CachedNAR {
  AcousticModel &model_;
  const Cancelled &cancelled_;
  int length_, block_;
  std::pair<mx::array, mx::array> factors_;
  mx::array position_, initial_, frequencies_, shift_, shift_delta_, one_;
  std::vector<std::pair<mx::array, mx::array>> cache_;

public:
  CachedNAR(AcousticModel &model, const std::vector<int> &tokens,
            const float *noise, int frames, int block,
            const Cancelled &cancelled, int visible_end = 0)
      : model_(model), cancelled_(cancelled), length_(frames + 2),
        block_(block), factors_{mx::array(0), mx::array(0)}, position_(0),
        initial_(bf16(mx::array(noise, {frames, 64}, mx::float32))),
        frequencies_(model.time_frequencies), shift_(model.timestep_shift),
        shift_delta_(model.timestep_shift_delta), one_(model.one) {
    check_cancelled(cancelled_, "Cancelled before acoustic prefill");
    const int ar_length = static_cast<int>(tokens.size());
    if (tokens.empty() || *std::min_element(tokens.begin(), tokens.end()) < 0 ||
        *std::max_element(tokens.begin(), tokens.end()) >=
            model.config.at("vocab_size").get<int>())
      throw Error("ValueError",
                  "AR conditioning tokens are outside the model vocabulary");
    if (visible_end < 0)
      throw Error("ValueError", "nar_cond_end must be a nonnegative integer");
    if (int64_t(ar_length) + length_ >
        model.config.at("max_position_embeddings").get<int>())
      throw Error("ValueError",
                  "Original acoustic chunk exceeds the model context");
    const int visible =
        visible_end ? std::min(visible_end, ar_length) : ar_length;
    const auto &inverse = model.rope_inverse_frequency;
    factors_ = rope_factors(inverse, ar_length, length_);
    auto positions = mx::minimum(
        mx::arange(length_, mx::int32),
        mx::array(model.config.at("max_latent_frames").get<int>() - 1));
    position_ = mx::expand_dims(
        mx::take(model.weights.at("latent_pos_embed.pe"), positions, 0), 0);
    mx::eval({initial_, factors_.first, factors_.second, position_,
              frequencies_, shift_, shift_delta_, one_});
    auto ar_factors = rope_factors(inverse, 0, ar_length);
    auto ids = mx::array(tokens.data(), {1, ar_length}, mx::int32);
    auto x = mx::take(model.ar.weights.at("model.embed_tokens.weight"), ids, 0);
    const float eps = model.config.at("rms_norm_eps");
    const int layers = model.config.at("num_hidden_layers");
    cache_.reserve(layers);
    for (int i = 0; i < layers; ++i) {
      check_cancelled(cancelled_, "Cancelled during acoustic prefill");
      const std::string base = "model.layers." + std::to_string(i) + ".";
      auto normalized = model_ops::rms_norm(
          x, model.ar.weights.at(base + "input_layernorm.weight"), eps);
      auto [q, k, v] = project(model, model.ar.weights, base + "self_attn.",
                               normalized, ar_factors);
      auto cached_key = mx::contiguous(sequence_slice(k, visible));
      auto cached_value = mx::contiguous(sequence_slice(v, visible));
      auto attended = nar_attention(q, k, v, true, block_);
      attended = mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                             {1, ar_length, -1});
      x = x + model_ops::linear(attended, model.ar.weights,
                                base + "self_attn.o_proj");
      x = x + model_ops::mlp(model_ops::rms_norm(
                                 x,
                                 model.ar.weights.at(
                                     base + "post_attention_layernorm.weight"),
                                 eps),
                             model.ar.weights, base + "mlp");
      mx::eval({cached_key, cached_value, x});
      cache_.emplace_back(std::move(cached_key), std::move(cached_value));
    }
    mx::eval({ar_factors.first, ar_factors.second});
  }
  mx::array velocity(const mx::array &state, double raw_t) const {
    if (!std::isfinite(raw_t))
      throw Error("ValueError", "raw_t must be a finite real number");
    if (state.shape() != mx::Shape{length_ - 2, 64})
      throw Error("ValueError", "ODE state shape changed");
    auto boundary =
        mx::pad(state, std::vector<std::pair<int, int>>{{1, 1}, {0, 0}});
    auto raw = fp32(mx::array(static_cast<float>(raw_t), mx::bfloat16));
    auto sigmoid = fp32(bf16(mx::sigmoid(raw)));
    auto numerator = bf16(shift_ * sigmoid),
         delta = bf16(shift_delta_ * sigmoid);
    auto shifted = numerator / (one_ + delta);
    auto x = model_ops::linear(mx::expand_dims(boundary, 0), model_.weights,
                               "vae2llm");
    auto timesteps = mx::broadcast_to(shifted, {length_});
    auto arguments = mx::reshape(fp32(timesteps), {length_, 1}) *
                     mx::reshape(frequencies_, {1, 128});
    auto embedding =
        bf16(mx::concatenate({mx::cos(arguments), mx::sin(arguments)}, -1));
    auto time = model_ops::linear(
        model_ops::silu(model_ops::linear(embedding, model_.weights,
                                          "time_embedder.mlp.0")),
        model_.weights, "time_embedder.mlp.2");
    x = x + mx::expand_dims(time, 0);
    x = x + position_;
    const float eps = model_.config.at("rms_norm_eps");
    for (size_t i = 0; i < cache_.size(); ++i) {
      check_cancelled(cancelled_, "Cancelled during acoustic velocity");
      const std::string base = "model.layers." + std::to_string(i) + ".";
      auto normalized = model_ops::rms_norm(
          x, model_.weights.at(base + "nar_input_layernorm.weight"), eps);
      auto [q, k, v] = project(model_, model_.weights, base + "nar_self_attn.",
                               normalized, factors_);
      k = mx::concatenate({cache_[i].first, k}, 2);
      v = mx::concatenate({cache_[i].second, v}, 2);
      auto attended = nar_attention(q, k, v, false, block_);
      attended =
          mx::reshape(mx::transpose(attended, {0, 2, 1, 3}), {1, length_, -1});
      x = x + model_ops::linear(attended, model_.weights,
                                base + "nar_self_attn.o_proj");
      x = x +
          model_ops::mlp(
              model_ops::rms_norm(
                  x, model_.weights.at(base + "nar_pre_mlp_layernorm.weight"),
                  eps),
              model_.weights, base + "nar_mlp");
    }
    x = model_ops::rms_norm(x, model_.ar.weights.at("model.norm.weight"), eps);
    x = model_ops::linear(x, model_.weights, "llm2vae");
    return mx::reshape(mx::slice(x, {0, 1, 0}, {1, length_ - 1, 64}),
                       {length_ - 2, 64});
  }
  void solve(float *output, int steps, const StepCallback &progress) const {
    auto state = initial_;
    const double dt = 1.0 / steps;
    auto half_dt = mx::array(static_cast<float>(dt / 2)),
         full_dt = mx::array(static_cast<float>(dt));
    mx::eval({half_dt, full_dt});
    auto logit = [](double t) {
      return std::clamp(std::log(t / (1.0 - t)), -20.0, 20.0);
    };
    for (int step = 0; step < steps; ++step) {
      check_cancelled(cancelled_, "Cancelled during acoustic flow matching");
      const double t = 1.0 - step * dt;
      auto first = velocity(state, logit(t));
      mx::eval(first);
      auto midpoint = state - bf16(fp32(first) * half_dt);
      check_cancelled(cancelled_, "Cancelled during acoustic flow matching");
      auto second = velocity(midpoint, logit(t - dt / 2));
      state = state - bf16(fp32(second) * full_dt);
      mx::eval(state);
      if (progress)
        progress(step + 1, steps);
    }
    auto result = mx::contiguous(fp32(state));
    mx::eval(result);
    const float *data = result.data<float>();
    const size_t count = static_cast<size_t>(length_ - 2) * 64;
    for (size_t i = 0; i < count; ++i) {
      if (!std::isfinite(data[i]))
        throw Error("FloatingPointError",
                    "Acoustic flow matching produced non-finite latents");
      output[i] = data[i];
    }
  }
};
} // namespace

AcousticModel::AcousticModel(const fs::path &input, ARModel &shared,
                             bool verify)
    : ar(shared) {
  const fs::path directory = expand_user(input);
  if (!fs::is_directory(directory))
    throw Error("FileNotFoundError",
                "Converted model directory does not exist: " +
                    directory.string());
  const Json manifest = verify ? verify_conversion(directory)
                               : read_json(directory / "conversion.json");
  const Json expected = {{"ar", "ar-bf16.safetensors"},
                         {"nar", "nar-bf16.safetensors"},
                         {"dtype", "bfloat16"}};
  if (!manifest.is_object() || manifest.value("schema", Json()) != 1)
    throw Error("ValueError", "conversion.json must be a schema-1 manifest");
  if (!manifest.contains("precisions") ||
      !manifest.at("precisions").is_object() ||
      manifest.at("precisions").value("bf16", Json()) != expected)
    throw Error("ValueError", "BF16 conversion manifest entry is incompatible");
  config = read_json(directory / "config.json");
  if (!config.is_object())
    throw Error("ValueError", "config.json must contain a JSON object");
  for (const char *name :
       {"hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size",
        "max_position_embeddings", "latent_dim", "max_latent_frames"})
    integer(config, name);
  for (const char *name : {"rms_norm_eps", "rope_theta", "timestep_shift"})
    if (!config.contains(name) || !config.at(name).is_number() ||
        !std::isfinite(config.at(name).get<double>()) ||
        config.at(name).get<double>() <= 0)
      throw Error("ValueError", std::string("config.json field '") + name +
                                    "' must be a finite positive number");
  if (config.value("latent_type", Json()) != "vae")
    throw Error("ValueError",
                "YuE2 acoustic loading requires config latent_type='vae'");
  if (config.at("latent_dim") != 64)
    throw Error("ValueError", "YuE2 acoustic loading requires latent_dim=64");
  if (config.at("head_dim").get<int>() % 2)
    throw Error("ValueError", "YuE2 RoPE requires an even head_dim");
  if (config.at("num_attention_heads").get<int>() %
      config.at("num_key_value_heads").get<int>())
    throw Error("ValueError",
                "num_attention_heads must be divisible by num_key_value_heads");
  if (!config.contains("tie_word_embeddings") ||
      config.at("tie_word_embeddings") != Json(false))
    throw Error("ValueError", "YuE2 requires untied token embeddings");
  for (const char *name :
       {"hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size",
        "rms_norm_eps", "rope_theta", "max_position_embeddings",
        "tie_word_embeddings"})
    if (!ar.config.contains(name) || ar.config.at(name) != config.at(name))
      throw Error(
          "ValueError",
          std::string(
              "Shared AR configuration does not match generator field '") +
              name + "'");
  const Json scaling = ar.config.value("rope_scaling", Json());
  if (!scaling.is_null() && scaling != Json::object())
    throw Error("ValueError",
                "YuE2 acoustic conditioning does not support scaled RoPE");
  if (ar.precision != "bf16")
    throw Error("ValueError", "Quantized AR cannot condition NAR");
  const int hidden = config.at("hidden_size"),
            layers = config.at("num_hidden_layers");
  Shapes shared_shapes = {{"model.embed_tokens.weight",
                           {config.at("vocab_size").get<int>(), hidden}},
                          {"model.norm.weight", {hidden}}};
  Shapes acoustic_shapes = {
      {"vae2llm.weight", {hidden, 64}},
      {"vae2llm.bias", {hidden}},
      {"llm2vae.weight", {64, hidden}},
      {"llm2vae.bias", {64}},
      {"time_embedder.mlp.0.weight", {hidden, 256}},
      {"time_embedder.mlp.0.bias", {hidden}},
      {"time_embedder.mlp.2.weight", {hidden, hidden}},
      {"time_embedder.mlp.2.bias", {hidden}},
      {"latent_pos_embed.pe",
       {config.at("max_latent_frames").get<int>(), hidden}}};
  for (int i = 0; i < layers; ++i) {
    layer_shapes(shared_shapes, config, i, false);
    layer_shapes(acoustic_shapes, config, i, true);
  }
  validate_weights(ar.weights, shared_shapes, false);
  const fs::path checkpoint = directory / "nar-bf16.safetensors";
  if (!fs::is_regular_file(checkpoint))
    throw Error("FileNotFoundError",
                "Missing acoustic checkpoint: " + checkpoint.string());
  try {
    weights = mx::load_safetensors(checkpoint.string()).first;
  } catch (const std::exception &) {
    throw Error("ValueError",
                "Unable to load acoustic checkpoint: " + checkpoint.string());
  }
  validate_weights(weights, acoustic_shapes, true);
  const int dim = config.at("head_dim");
  const auto exponent =
      mx::arange(0, dim, 2, mx::float32) / mx::array(float(dim));
  rope_inverse_frequency =
      mx::array(1.0f) /
      mx::power(mx::array(config.at("rope_theta").get<float>()), exponent);
  time_frequencies = mx::exp(mx::array(static_cast<float>(-std::log(10000.0))) *
                             mx::arange(128, mx::float32) / mx::array(128.0f));
  timestep_shift = mx::array(config.at("timestep_shift").get<float>());
  timestep_shift_delta = mx::array(
      static_cast<float>(config.at("timestep_shift").get<double>() - 1.0));
  mx::eval({rope_inverse_frequency, time_frequencies, timestep_shift,
            timestep_shift_delta, one});
  materialize();
}
void AcousticModel::materialize() const {
  std::vector<mx::array> parameters;
  parameters.reserve(weights.size());
  for (const auto &[name, value] : weights)
    parameters.push_back(value);
  mx::eval(parameters);
}
FloatMatrix synthesize(AcousticModel &model, const std::vector<int> &prefix,
                       const std::vector<int> &codec, const FloatMatrix &noise,
                       int steps, int context, int query_chunk_size,
                       const Cancelled &cancelled,
                       const StepCallback &on_progress) {
  if (steps < 1)
    throw Error("ValueError", "steps must be a positive integer");
  if (query_chunk_size < 1)
    throw Error("ValueError", "query_chunk_size must be a positive integer");
  if (prefix.empty() || codec.empty())
    throw Error(
        "ValueError",
        "prefix and codec must be nonempty sequences of integer token IDs");
  if (*std::min_element(prefix.begin(), prefix.end()) < 0 ||
      *std::min_element(codec.begin(), codec.end()) < 0 ||
      *std::max_element(codec.begin(), codec.end()) >= CODEC_SIZE)
    throw Error("ValueError", "Token IDs are outside their allowed vocabulary");
  if (context < 1 || context > CONTEXT)
    throw Error("ValueError", "context must be an integer in 1..24576");
  if (codec.size() > std::numeric_limits<int>::max() ||
      prefix.size() > std::numeric_limits<int>::max())
    throw Error("ValueError", "Token sequence exceeds supported context");
  const auto ranges = chunk_ranges(static_cast<int>(codec.size()),
                                   static_cast<int>(prefix.size()), context);
  noise.validate("noise", 64);
  if (noise.rows != static_cast<int64_t>(codec.size()))
    throw Error("ValueError",
                "Supplied noise must have one frame per semantic token");
  if (ranges.size() >
      static_cast<size_t>(std::numeric_limits<int>::max() / steps))
    throw Error("ValueError",
                "Acoustic progress step count exceeds supported range");
  const int total_steps = steps * static_cast<int>(ranges.size());
  FloatMatrix output{noise.rows, 64, std::vector<float>(noise.values.size())};
  for (size_t index = 0; index < ranges.size(); ++index) {
    check_cancelled(cancelled, "Cancelled before acoustic prefill");
    const auto [start, end] = ranges[index];
    std::vector<int> tokens;
    tokens.reserve(prefix.size() + end - start + 1);
    tokens.insert(tokens.end(), prefix.begin(), prefix.end());
    for (int i = start; i < end; ++i)
      tokens.push_back(codec[i] + CODEC_OFFSET);
    tokens.push_back(MUSIC_END);
    CachedNAR engine(model, tokens,
                     noise.values.data() + static_cast<size_t>(start) * 64,
                     end - start, query_chunk_size, cancelled);
    StepCallback progress;
    if (on_progress)
      progress = [&, index](int completed, int) {
        on_progress(static_cast<int>(index) * steps + completed, total_steps);
      };
    engine.solve(output.values.data() + static_cast<size_t>(start) * 64, steps,
                 progress);
  }
  return output;
}
} // namespace lyra
