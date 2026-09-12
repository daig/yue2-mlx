#include "lyra/pipeline.hpp"
#include "lyra/conversion.hpp"
#include "lyra/storage.hpp"
#include <cmath>
#include <iostream>

namespace lyra {
namespace {
void require_precision(std::string_view value) {
  if (value != "bf16" && value != "8bit" && value != "4bit")
    throw Error("ValueError", "precision must be bf16, 8bit or 4bit");
}
void release_mlx_cache() {
  mx::synchronize();
  mx::clear_cache();
}
} // namespace

Pipeline::Pipeline(PipelineOptions options) : options_(std::move(options)) {
  require_precision(options_.precision);
  options_.generation.validate();
  if (options_.vae_core_frames < 1 || options_.query_chunk_size < 1)
    throw Error(
        "ValueError",
        "Decoder core and attention query sizes must be positive integers");
  initialize_runtime();
  execution_ = std::make_unique<GPUExecution>(options_.require_ac);
  const double started = monotonic_seconds();
  {
    Progress progress(options_.progress, "Resolving model files");
    auto model = expand_user(options_.model);
    if (fs::is_regular_file(model / "pipeline.json")) {
      auto metadata = read_json(model / "pipeline.json");
      if (!metadata.is_object() || !metadata.contains("model") ||
          !metadata["model"].is_string() || !metadata.contains("vae") ||
          !metadata["vae"].is_string())
        throw Error("ValueError", "Invalid saved pipeline metadata");
      if (options_.vae == VAE_REPO)
        options_.vae = (model / metadata["vae"].get<std::string>()).string();
      model /= metadata["model"].get<std::string>();
    }
    if (fs::is_regular_file(model / "conversion.json")) {
      model_dir_ = model;
    } else {
      auto source =
          resolve_model(model.string(), MODEL_REVISION, options_.offline);
      execution_->check();
      model_dir_ = prepare(source, options_.converted_dir, options_.precision);
    }
    vae_dir_ = resolve_model(options_.vae, VAE_REVISION, options_.offline);
    execution_->check();
  }
  {
    Progress progress(options_.progress, "Verifying model files");
    auto conversion = verify_conversion(model_dir_);
    if (!fs::is_regular_file(model_dir_ /
                             ("ar-" + options_.precision + ".safetensors")))
      throw Error("FileNotFoundError",
                  "Convert the requested AR precision first: " +
                      options_.precision);
    auto vae_identity = model_identity(vae_dir_);
    auto pinned = pinned_vae_files();
    if (vae_identity["files"] ==
            Json{{"model.safetensors", pinned.at("model.safetensors")}} &&
        vae_identity["config_sha256"] ==
            pinned.at("config.json").at("sha256")) {
      vae_identity["source"] = {{"repository", VAE_REPO},
                                {"revision", VAE_REVISION},
                                {"revision_proof", "verified-pinned-content"}};
    }
    weights = {{"mot", std::move(conversion)},
               {"vae", std::move(vae_identity)}};
    tokenizer_ = std::make_unique<Tokenizer>(model_dir_ / "qwen.tiktoken");
    runtime_ = runtime_info();
    runtime_sha256_ = runtime_.at("runtime_sha256").get<std::string>();
    decoder_release_ = read_json(vae_dir_ / "config.json")
                           .value("release_variant", Json(nullptr));
  }
  load_timing["resolve_conversion_integrity_seconds"] =
      monotonic_seconds() - started;
  execution_->check();
}

Pipeline::~Pipeline() {
  try {
    close();
  } catch (const std::exception &error) {
    std::cerr << "Pipeline cleanup failed: " << error.what() << '\n';
  }
}

void Pipeline::close() {
  if (closed_)
    return;
  closed_ = true;
  std::exception_ptr error;
  // Model storage must die while the execution guard still owns the GPU.
  try {
    mx::synchronize();
  } catch (...) {
    error = std::current_exception();
  }
  vae_.reset();
  nar_.reset();
  ar_.reset();
  bf16_ar_.reset();
  try {
    mx::clear_cache();
  } catch (...) {
    if (!error)
      error = std::current_exception();
  }
  try {
    execution_->close();
  } catch (...) {
    if (!error)
      error = std::current_exception();
  }
  if (error)
    std::rethrow_exception(error);
}

void Pipeline::record_load(const std::string &stage, double seconds) {
  auto key = stage + "_load_events_seconds";
  if (!load_timing.contains(key))
    load_timing[key] = Json::array();
  load_timing[key].push_back(seconds);
  load_timing[stage + "_load_seconds"] =
      load_timing.value(stage + "_load_seconds", 0.0) + seconds;
}

ARModel &Pipeline::load_ar() {
  execution_->check();
  if (options_.precision != "bf16" && !ar_ && (bf16_ar_ || nar_)) {
    nar_.reset();
    bf16_ar_.reset();
    release_mlx_cache();
    execution_->check();
  }
  if (!ar_) {
    if (options_.precision == "bf16" && bf16_ar_) {
      ar_ = bf16_ar_;
    } else {
      const double started = monotonic_seconds();
      Progress progress(options_.progress,
                        "Loading " + options_.precision + " AR model");
      ar_ = std::make_shared<ARModel>(model_dir_, options_.precision, false);
      ar_->materialize();
      record_load("ar", monotonic_seconds() - started);
    }
  }
  execution_->check();
  return *ar_;
}

AcousticModel &Pipeline::load_nar() {
  execution_->check();
  if (options_.precision != "bf16" && ar_) {
    ar_.reset();
    release_mlx_cache();
    execution_->check();
  }
  if (!bf16_ar_) {
    if (options_.precision == "bf16" && ar_) {
      bf16_ar_ = ar_;
    } else {
      const double started = monotonic_seconds();
      Progress progress(options_.progress,
                        "Loading BF16 acoustic conditioning");
      bf16_ar_ = std::make_shared<ARModel>(model_dir_, "bf16", false);
      bf16_ar_->materialize();
      record_load("conditioning", monotonic_seconds() - started);
      execution_->check();
    }
  }
  if (!nar_) {
    const double started = monotonic_seconds();
    Progress progress(options_.progress, "Loading acoustic model");
    nar_ = std::make_unique<AcousticModel>(model_dir_, *bf16_ar_, false);
    nar_->materialize();
    record_load("nar", monotonic_seconds() - started);
  }
  execution_->check();
  return *nar_;
}

TokenGeneration Pipeline::generate_phase(const std::vector<int> &prefix,
                                         const Sampling &sampling,
                                         uint64_t seed, std::string_view phase,
                                         const std::vector<int> &negative,
                                         double guidance, bool legacy_off) {
  execution_->check();
  sampling.validate();
  if (prefix.size() + static_cast<size_t>(sampling.max_tokens) > CONTEXT)
    throw Error("ValueError", "Prefix + requested generation budget exceeds "
                              "24576; no implicit truncation");
  if (!negative.empty() &&
      negative.size() + static_cast<size_t>(sampling.max_tokens) > CONTEXT)
    throw Error("ValueError",
                "Negative prefix + generation budget exceeds context");
  check_cancelled();
  auto &model = load_ar();
  Progress progress(options_.progress,
                    phase == "abc" ? "Planning score" : "Generating song",
                    "tokens");
  auto guarded = [this] {
    execution_->check();
    return cancellation_requested();
  };
  auto result = generate_tokens(
      model, prefix, sampling, seed, phase, negative, guidance, legacy_off,
      guarded, [&](std::string_view, int) { progress.advance(); });
  progress.finish(result.truncated);
  execution_->check();
  return result;
}

SymbolicPlan Pipeline::plan(const SongRequest &request,
                            const Json &abc_sampling) {
  execution_->check();
  request.validate();
  SymbolicPlan result;
  result.request = request;
  if (request.cot == "off") {
    result.prefix = token_prefixes(request, *tokenizer_);
  } else if (request.abc) {
    Progress progress(options_.progress, "Using provided score");
    result.abc = request.abc;
    result.abc_ids = tokenizer_->encode(*request.abc);
    result.prefix = token_prefixes(request, *tokenizer_, result.abc_ids);
    result.timing = {{"seconds", 0.0},
                     {"output_tokens", 0},
                     {"external_prefix_tokens", result.abc_ids.size()}};
  } else {
    auto sampling = Sampling::from_json(abc_sampling, options_.generation.abc);
    auto generated = generate_phase(token_prefixes(request, *tokenizer_),
                                    sampling, request.seed, "abc");
    result.abc_ids = std::move(generated.tokens);
    result.abc = tokenizer_->decode(result.abc_ids);
    result.prefix = token_prefixes(request, *tokenizer_, result.abc_ids);
    result.timing = std::move(generated.timing);
    result.truncated = generated.truncated;
  }
  execution_->check();
  return result;
}

SemanticResult Pipeline::generate_semantic(const SymbolicPlan &plan,
                                           const Json &overrides) {
  execution_->check();
  if (token_prefixes(plan.request, *tokenizer_, plan.abc_ids) != plan.prefix)
    throw Error("ValueError",
                "Plan prefix disagrees with request/exact ABC IDs");
  auto sampling = Sampling::from_json(overrides, options_.generation.semantic);
  std::vector<int> negative;
  if (plan.request.guidance() != 1)
    negative = negative_prefix(plan.request, *tokenizer_, plan.abc_ids);
  auto result = generate_phase(plan.prefix, sampling, plan.request.seed,
                               "semantic", negative, plan.request.guidance(),
                               plan.request.cot == "off");
  if (result.tokens.empty())
    throw Error("RuntimeError", "Semantic generation produced no codec frames");
  for (auto &token : result.tokens)
    token -= CODEC_OFFSET;
  execution_->check();
  return {plan, std::move(result.tokens), std::move(result.timing),
          result.truncated};
}

FloatMatrix Pipeline::synthesize(const SemanticResult &semantic,
                                 const FloatMatrix &noise) {
  execution_->check();
  if (token_prefixes(semantic.plan.request, *tokenizer_,
                     semantic.plan.abc_ids) != semantic.plan.prefix)
    throw Error("ValueError",
                "Semantic result does not retain the request's exact prefix");
  if (semantic.tokens.empty())
    throw Error("ValueError",
                "Semantic result must contain at least one codec frame");
  check_cancelled();
  auto &model = load_nar();
  Progress progress(options_.progress, "Synthesizing audio", "steps");
  auto result = lyra::synthesize(
      model, semantic.plan.prefix, semantic.tokens, noise,
      options_.generation.ode_steps, options_.generation.context,
      options_.query_chunk_size,
      [this] {
        execution_->check();
        return cancellation_requested();
      },
      [&](int complete, int total) {
        execution_->check();
        progress.update(complete, total);
      });
  execution_->check();
  return result;
}

FloatMatrix Pipeline::decode(const FloatMatrix &latents) {
  execution_->check();
  latents.validate("VAE latents", 64);
  if (!vae_) {
    const double started = monotonic_seconds();
    Progress progress(options_.progress, "Loading audio decoder");
    vae_ = std::make_unique<VAE>(vae_dir_, [this] {
      execution_->check();
      return cancellation_requested();
    });
    record_load("vae", monotonic_seconds() - started);
  }
  execution_->check();
  const int tiles = static_cast<int>(
      (latents.rows + options_.vae_core_frames - 1) / options_.vae_core_frames);
  Progress progress(options_.progress, "Decoding audio", "chunks", tiles);
  auto audio = vae_->decode(
      latents, options_.vae_core_frames, 16,
      [this] {
        execution_->check();
        return cancellation_requested();
      },
      [&](int complete, int total) {
        execution_->check();
        progress.update(complete, total);
      });
  if (audio.rows != 1920 * latents.rows - 64 || audio.cols != 2)
    throw Error("RuntimeError",
                "VAE output violates its natural stereo length contract");
  audio.validate("VAE audio", 2);
  execution_->check();
  return audio;
}

Json Pipeline::effective_config(const SongRequest &request,
                                const Json &abc_sampling,
                                const Json &semantic_sampling) const {
  auto generation = options_.generation;
  generation.abc = Sampling::from_json(abc_sampling, generation.abc);
  generation.semantic =
      Sampling::from_json(semantic_sampling, generation.semantic);
  auto config = generation.to_json();
  auto defaults = GenerationConfig{}.to_json();
  auto overrides = Json::object();
  for (const auto &[key, value] : config.items())
    if (defaults.at(key) != value)
      overrides[key] = value;
  const Json guidance = request.cfg_scale ? request.to_json().at("cfg_scale")
                                          : Json(request.guidance());
  if (request.guidance() != (request.cot == "off" ? 1.01 : 1.0))
    overrides["cfg_scale"] = guidance;
  return {
      {"generation", config},
      {"overrides", overrides},
      {"cot", request.cot},
      {"cfg_scale", guidance},
      {"cfg_negative", request.cot == "off" ? "instruction_only"
                                            : "same_instruction_and_exact_abc"},
      {"backend", "mlx-native"},
      {"quantization", options_.precision},
      {"ar_precision", options_.precision},
      {"model_dtype", "bfloat16"},
      {"kv_dtype", "bfloat16"},
      {"nar_dtype", "bfloat16"},
      {"conditioning_precision", "bf16"},
      {"attention_opmath", "float32"},
      {"mlx_tf32_enabled", false},
      {"vae_dtype", "float32"},
      {"vae_decode", "halo_crop"},
      {"vae_core_frames", options_.vae_core_frames},
      {"vae_halo_frames", 16},
      {"query_chunk_size", options_.query_chunk_size},
      {"device", "mps"},
      {"memory_policy", "observe_only"},
      {"offload_ar", false},
      {"runtime_sha256", runtime_sha256_},
      {"runtime", runtime_},
      {"upstream_commit", UPSTREAM_COMMIT},
      {"execution_context", "GPUExecution"},
      {"decoder_release", decoder_release_},
      {"rng",
       {{"ar", "request_local_mlx"}, {"acoustic", "torch_cpu_fp32_full_song"}}},
      {"validation_status", "unvalidated"}};
}

SongResult Pipeline::render(const SymbolicPlan &plan,
                            const Json &semantic_sampling) {
  const double started = monotonic_seconds();
  auto config = effective_config(plan.request, nullptr, semantic_sampling);
  auto stamp = identity({{"request", plan.request.to_json()},
                         {"config", config},
                         {"weights", weights}});
  return render_with_config(plan, semantic_sampling, std::move(config),
                            std::move(stamp), started);
}

SongResult Pipeline::render_with_config(const SymbolicPlan &plan,
                                        const Json &semantic_sampling,
                                        Json config, std::string stamp,
                                        double started) {
  auto semantic = generate_semantic(plan, semantic_sampling);
  auto noise = initial_noise(static_cast<int>(semantic.tokens.size()),
                             plan.request.seed);
  const double nar_started = monotonic_seconds();
  auto latents = synthesize(semantic, noise);
  const double nar_seconds = monotonic_seconds() - nar_started;
  execution_->check();
  check_cancelled();
  const double vae_started = monotonic_seconds();
  auto audio = decode(latents);
  const double vae_seconds = monotonic_seconds() - vae_started;
  Json timing = {{"abc", plan.timing},
                 {"semantic", semantic.timing},
                 {"nar_seconds", nar_seconds},
                 {"vae_seconds", vae_seconds},
                 {"load", load_timing},
                 {"e2e_seconds", monotonic_seconds() - started}};
  execution_->check();
  return {std::move(audio),  48000,   std::move(semantic), std::move(latents),
          std::move(config), weights, std::move(timing),   std::move(stamp),
          std::move(noise)};
}

SongResult Pipeline::generate(const SongRequest &request,
                              const Json &abc_sampling,
                              const Json &semantic_sampling) {
  execution_->check();
  request.validate();
  auto config = effective_config(request, abc_sampling, semantic_sampling);
  auto stamp = identity({{"request", request.to_json()},
                         {"config", config},
                         {"weights", weights}});
  const double started = monotonic_seconds();
  auto symbolic = plan(request, abc_sampling);
  auto result =
      render_with_config(symbolic, semantic_sampling, std::move(config),
                         std::move(stamp), started);
  if (options_.progress) {
    std::cerr << "Generated " << static_cast<double>(result.audio.rows) / 48000
              << " seconds of audio in "
              << result.timing["e2e_seconds"].get<double>() << " seconds";
    if (result.semantic.truncated || result.semantic.plan.truncated)
      std::cerr << " (truncated)";
    std::cerr << '\n';
  }
  execution_->check();
  return result;
}
} // namespace lyra
