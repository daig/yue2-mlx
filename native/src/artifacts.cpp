#include "lyra/artifacts.hpp"
#include "lyra/runtime.hpp"
#include "lyra/storage.hpp"
#include "lyra/vae.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <regex>
#include <set>

namespace lyra {
namespace {
[[noreturn]] void invalid(const std::string &message) {
  throw Error("ValueError", message);
}
struct Npy {
  std::string bytes, dtype;
  std::vector<int64_t> shape;
  size_t offset = 0, count = 0, itemsize = 0;
  bool swap = false, fortran = false;
};
Npy read_npy(const fs::path &path) {
  Npy n;
  n.bytes = read_text(path);
  const auto *p = reinterpret_cast<const unsigned char *>(n.bytes.data());
  if (n.bytes.size() < 10 || std::memcmp(p, "\x93NUMPY", 6))
    invalid("Invalid NPY magic");
  size_t prefix = 0, length = 0;
  if (p[6] == 1 && p[7] == 0) {
    prefix = 10;
    length = p[8] | (size_t(p[9]) << 8);
  } else if ((p[6] == 2 || p[6] == 3) && p[7] == 0 && n.bytes.size() >= 12) {
    prefix = 12;
    for (int i = 0; i < 4; ++i)
      length |= size_t(p[8 + i]) << (8 * i);
  } else
    invalid("Unsupported NPY version");
  if (length > n.bytes.size() - prefix)
    invalid("Truncated NPY header");
  std::string header = n.bytes.substr(prefix, length);
  if (header.empty() || header.back() != '\n')
    invalid("Invalid NPY header");
  // Parse only the literal scalar dtype / shape dictionary emitted by NumPy;
  // never evaluate pickle or Python expressions from an artifact.
  const std::regex field(
      R"(['"](descr|fortran_order|shape)['"]\s*:\s*(['"][^'"]*['"]|True|False|\([^)]*\))\s*,?)");
  auto begin = std::sregex_iterator(header.begin(), header.end(), field),
       end = std::sregex_iterator();
  std::set<std::string> seen;
  size_t cursor = 0;
  auto whitespace = [](std::string_view s) {
    return std::all_of(s.begin(), s.end(), [](unsigned char c) {
      return c == ' ' || c == '\t' || c == '\n' || c == '\r';
    });
  };
  size_t opening = header.find_first_not_of(" \t\r\n");
  if (opening == std::string::npos || header[opening] != '{')
    invalid("Invalid NPY dictionary");
  cursor = opening + 1;
  for (auto it = begin; it != end; ++it) {
    const auto &match = *it;
    if (!whitespace(
            std::string_view(header).substr(cursor, match.position() - cursor)))
      invalid("Invalid NPY header field");
    std::string key = match[1], value = match[2];
    if (!seen.insert(key).second)
      invalid("Duplicate NPY field");
    if (key == "descr") {
      if (value.size() < 3 || (value[0] != '\'' && value[0] != '"') ||
          value.back() != value[0])
        invalid("Invalid NPY dtype");
      n.dtype = value.substr(1, value.size() - 2);
    } else if (key == "fortran_order") {
      if (value != "True" && value != "False")
        invalid("Invalid NPY order");
      n.fortran = value == "True";
    } else {
      if (value.front() != '(' || value.back() != ')')
        invalid("Invalid NPY shape");
      std::string_view dimensions(value.data() + 1, value.size() - 2);
      while (!dimensions.empty()) {
        auto first = dimensions.find_first_not_of(" \t");
        if (first == std::string_view::npos)
          break;
        dimensions.remove_prefix(first);
        int64_t dimension = 0;
        auto parsed =
            std::from_chars(dimensions.data(),
                            dimensions.data() + dimensions.size(), dimension);
        if (parsed.ec != std::errc{} || parsed.ptr == dimensions.data() ||
            dimension < 0)
          invalid("Invalid NPY dimension");
        n.shape.push_back(dimension);
        dimensions.remove_prefix(parsed.ptr - dimensions.data());
        auto separator = dimensions.find_first_not_of(" \t");
        if (separator == std::string_view::npos)
          break;
        if (dimensions[separator] != ',')
          invalid("Invalid NPY shape separator");
        dimensions.remove_prefix(separator + 1);
      }
    }
    cursor = match.position() + match.length();
  }
  auto rest = std::string_view(header).substr(cursor);
  auto closing = rest.find_first_not_of(" \t\n\r");
  if (seen.size() != 3 || closing == std::string_view::npos ||
      rest[closing] != '}' || !whitespace(rest.substr(closing + 1)))
    invalid("Invalid NPY dictionary");
  if (n.dtype.size() != 3 ||
      (n.dtype[0] != '<' && n.dtype[0] != '>' && n.dtype[0] != '=' &&
       n.dtype[0] != '|') ||
      (!((n.dtype[1] == 'i' || n.dtype[1] == 'u') &&
         (n.dtype[2] == '1' || n.dtype[2] == '2' || n.dtype[2] == '4' ||
          n.dtype[2] == '8')) &&
       n.dtype.substr(1) != "f4"))
    invalid("Unsupported NPY dtype; expected integer tokens or float32");
  if (n.dtype[0] == '|' && n.dtype[2] != '1')
    invalid("Invalid byte order for multibyte NPY dtype");
  n.itemsize = n.dtype[2] - '0';
  n.count = 1;
  for (auto dimension : n.shape) {
    if (static_cast<uint64_t>(dimension) >
            std::numeric_limits<size_t>::max() / n.itemsize ||
        (dimension && n.count > std::numeric_limits<size_t>::max() /
                                    static_cast<size_t>(dimension)))
      invalid("NPY shape is too large");
    n.count *= dimension;
  }
  n.offset = prefix + length;
  if (n.count > std::numeric_limits<size_t>::max() / n.itemsize ||
      n.bytes.size() - n.offset != n.count * n.itemsize)
    invalid("NPY data length disagrees with shape");
  n.swap = (n.dtype[0] == '>' && std::endian::native == std::endian::little) ||
           (n.dtype[0] == '<' && std::endian::native == std::endian::big);
  return n;
}
template <class T> T element(const Npy &n, size_t index) {
  std::array<unsigned char, sizeof(T)> bytes{};
  std::memcpy(bytes.data(), n.bytes.data() + n.offset + index * sizeof(T),
              sizeof(T));
  if (n.swap)
    std::reverse(bytes.begin(), bytes.end());
  T value;
  std::memcpy(&value, bytes.data(), sizeof(T));
  return value;
}
std::string npy_header(std::string_view dtype,
                       const std::vector<int64_t> &shape) {
  std::string header = "{'descr': '" + std::string(dtype) +
                       "', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i)
      header += ", ";
    header += std::to_string(shape[i]);
  }
  if (shape.size() == 1)
    header += ',';
  header += "), }";
  header.append((64 - (10 + header.size() + 1) % 64) % 64, ' ');
  header += '\n';
  std::string result("\x93NUMPY", 6);
  result += '\1';
  result += '\0';
  result += static_cast<char>(header.size() & 255);
  result += static_cast<char>(header.size() >> 8);
  result += header;
  return result;
}
template <class T> void append_scalar(std::string &bytes, T value) {
  std::array<char, sizeof(T)> data;
  std::memcpy(data.data(), &value, sizeof(T));
  if constexpr (std::endian::native == std::endian::big)
    std::reverse(data.begin(), data.end());
  bytes.append(data.data(), data.size());
}
void validate_config(const Json &config, std::string_view message) {
  if (!config.is_object() || !config.contains("generation") ||
      !config["generation"].is_object())
    invalid(std::string(message));
  try {
    GenerationConfig::from_json(config["generation"]);
  } catch (const std::exception &) {
    invalid(std::string(message));
  }
}
void validate_weights(const Json &weights) {
  if (!weights.is_object() || !weights.contains("mot") ||
      !weights.contains("vae"))
    invalid("Saved song lacks model weight identities");
}
void validate_tokens(const std::vector<int> &tokens) {
  if (tokens.empty())
    invalid("Expected nonempty integer semantic token vector");
  for (int token : tokens)
    if (token < 0 || token >= CODEC_SIZE)
      invalid("Saved semantic tokens are outside the codec vocabulary");
}
void validate_latents(const FloatMatrix &matrix, size_t frames,
                      std::string_view name) {
  matrix.validate(name, 64);
  if (matrix.rows != static_cast<int64_t>(frames))
    invalid(std::string(name) + " must have shape [T,64]");
}
std::vector<int> json_tokens(const Json &array) {
  if (!array.is_array())
    invalid("Saved plan tokens must be an integer array");
  std::vector<int> result;
  result.reserve(array.size());
  for (const auto &value : array) {
    if (!value.is_number_integer())
      invalid("Saved plan token must be an integer");
    if (value.is_number_unsigned()) {
      if (value.get<uint64_t>() > uint64_t(INT32_MAX))
        invalid("Saved token is outside int32 range");
    } else if (value.get<int64_t>() < INT32_MIN ||
               value.get<int64_t>() > INT32_MAX)
      invalid("Saved token is outside int32 range");
    result.push_back(value.get<int>());
  }
  return result;
}
} // namespace
void save_ints(const fs::path &path, const std::vector<int> &values,
               bool int64) {
  auto bytes =
      npy_header(int64 ? "<i8" : "<i4", {static_cast<int64_t>(values.size())});
  static_assert(sizeof(int) == sizeof(int32_t));
  if constexpr (std::endian::native == std::endian::little) {
    if (!int64) {
      const std::array<std::string_view, 2> buffers = {
          bytes, std::string_view(reinterpret_cast<const char *>(values.data()),
                                  values.size() * sizeof(int))};
      write_buffers(path, buffers);
      return;
    }
  }
  bytes.reserve(bytes.size() + values.size() * (int64 ? 8 : 4));
  for (int value : values) {
    if (int64)
      append_scalar<int64_t>(bytes, value);
    else
      append_scalar<int32_t>(bytes, value);
  }
  write_text(path, bytes);
}
std::vector<int> load_ints(const fs::path &path) {
  auto n = read_npy(path);
  if (n.shape.size() != 1 || (n.dtype[1] != 'i' && n.dtype[1] != 'u'))
    invalid("Expected integer NPY token vector");
  std::vector<int> result;
  result.reserve(n.count);
  for (size_t i = 0; i < n.count; ++i) {
    int64_t value;
    if (n.dtype[1] == 'u') {
      uint64_t unsigned_value = n.itemsize == 1   ? element<uint8_t>(n, i)
                                : n.itemsize == 2 ? element<uint16_t>(n, i)
                                : n.itemsize == 4 ? element<uint32_t>(n, i)
                                                  : element<uint64_t>(n, i);
      if (unsigned_value > uint64_t(INT32_MAX))
        invalid("Saved token is outside int32 range");
      value = static_cast<int64_t>(unsigned_value);
    } else
      value = n.itemsize == 1   ? element<int8_t>(n, i)
              : n.itemsize == 2 ? element<int16_t>(n, i)
              : n.itemsize == 4 ? element<int32_t>(n, i)
                                : element<int64_t>(n, i);
    if (value < INT32_MIN || value > INT32_MAX)
      invalid("Saved token is outside int32 range");
    result.push_back(static_cast<int>(value));
  }
  return result;
}
void save_matrix(const fs::path &path, const FloatMatrix &matrix) {
  matrix.validate("matrix");
  auto bytes = npy_header("<f4", {matrix.rows, matrix.cols});
  if constexpr (std::endian::native == std::endian::little) {
    const std::array<std::string_view, 2> buffers = {
        bytes,
        std::string_view(reinterpret_cast<const char *>(matrix.values.data()),
                         matrix.values.size() * sizeof(float))};
    write_buffers(path, buffers);
  } else {
    bytes.reserve(bytes.size() + matrix.values.size() * sizeof(float));
    for (float value : matrix.values)
      append_scalar<float>(bytes, value);
    write_text(path, bytes);
  }
}
FloatMatrix load_matrix(const fs::path &path) {
  auto n = read_npy(path);
  if (n.shape.size() != 2 || n.dtype.substr(1) != "f4")
    invalid("Expected FP32 NPY matrix");
  FloatMatrix result{n.shape[0], n.shape[1], std::vector<float>(n.count)};
  if (!n.swap && !n.fortran)
    std::memcpy(result.values.data(), n.bytes.data() + n.offset,
                n.count * sizeof(float));
  else
    for (int64_t row = 0; row < result.rows; ++row)
      for (int64_t col = 0; col < result.cols; ++col)
        result.values[row * result.cols + col] = element<float>(
            n, n.fortran ? col * result.rows + row : row * result.cols + col);
  result.validate("Saved matrix");
  return result;
}
void save_plan(const SymbolicPlan &plan, const fs::path &directory) {
  fs::create_directories(directory);
  if (plan.abc)
    write_text(directory / "score.abc", *plan.abc);
  save_ints(directory / "abc_tokens.npy", plan.abc_ids);
  save_ints(directory / "prefix.npy", plan.prefix);
  write_json(directory / "plan.json",
             {{"request", plan.request.to_json()},
              {"timing", plan.timing},
              {"truncated", plan.truncated},
              {"prefix", plan.prefix},
              {"abc_ids", plan.abc_ids},
              {"abc", plan.abc ? Json(*plan.abc) : Json(nullptr)}});
  Json manifest = Json::object();
  for (auto name : {"plan.json", "abc_tokens.npy", "prefix.npy"})
    manifest[name] = sha256_file(directory / name);
  if (plan.abc)
    manifest["score.abc"] = sha256_file(directory / "score.abc");
  write_json(directory / "plan_manifest.json", manifest);
}
SymbolicPlan load_plan(const fs::path &directory) {
  if (fs::is_symlink(directory / "plan_manifest.json"))
    invalid("Invalid plan artifact");
  auto hashes = read_json(directory / "plan_manifest.json");
  if (!hashes.is_object())
    invalid("Incomplete saved plan");
  for (auto name : {"plan.json", "abc_tokens.npy", "prefix.npy"})
    if (!hashes.contains(name))
      invalid("Incomplete saved plan");
  const std::set<std::string> allowed = {"plan.json", "abc_tokens.npy",
                                         "prefix.npy", "score.abc"};
  for (auto it = hashes.begin(); it != hashes.end(); ++it) {
    if (!allowed.contains(it.key()) || fs::is_symlink(directory / it.key()))
      invalid("Invalid plan artifact");
    if (!it.value().is_string() ||
        sha256_file(directory / it.key()) !=
            it.value().get_ref<const std::string &>())
      invalid("Saved plan changed; supply modified ABC as an external planner "
              "input");
  }
  auto data = read_json(directory / "plan.json");
  if (!data.is_object() || !data.contains("request") ||
      !data.contains("timing") || !data["timing"].is_object() ||
      !data.contains("truncated") || !data["truncated"].is_boolean() ||
      !data.contains("abc") ||
      !(data["abc"].is_null() || data["abc"].is_string()) ||
      !data.contains("abc_ids") || !data.contains("prefix"))
    invalid("Invalid saved plan metadata");
  SymbolicPlan plan;
  plan.request = SongRequest::from_json(data["request"]);
  plan.abc_ids = json_tokens(data["abc_ids"]);
  plan.prefix = json_tokens(data["prefix"]);
  if (load_ints(directory / "abc_tokens.npy") != plan.abc_ids ||
      load_ints(directory / "prefix.npy") != plan.prefix)
    invalid("Saved plan token array mismatch");
  if (!data["abc"].is_null()) {
    plan.abc = data["abc"].get<std::string>();
    if (!hashes.contains("score.abc") ||
        read_text(directory / "score.abc") != *plan.abc)
      invalid("Saved ABC text mismatch");
  }
  plan.timing = data["timing"];
  plan.truncated = data["truncated"].get<bool>();
  return plan;
}
void save_plan_artifacts(const SymbolicPlan &plan, const fs::path &directory,
                         const GenerationConfig &config) {
  Progress progress(true, "Saving score");
  save_plan(plan, directory);
  auto data = read_json(directory / "plan.json");
  data["generation_config"] = config.to_json();
  write_json(directory / "plan.json", data);
  auto manifest = read_json(directory / "plan_manifest.json");
  manifest["plan.json"] = sha256_file(directory / "plan.json");
  write_json(directory / "plan_manifest.json", manifest);
}
std::pair<SymbolicPlan, std::optional<GenerationConfig>>
load_plan_artifacts(const fs::path &directory) {
  auto plan = load_plan(directory);
  auto data = read_json(directory / "plan.json");
  auto saved = data.value("generation_config", Json());
  if (saved.is_null()) {
    if (!fs::is_regular_file(directory / "config.json"))
      return {std::move(plan), std::nullopt};
    auto config = read_json(directory / "config.json");
    saved = config.is_object() ? config.value("generation", Json()) : Json();
  }
  if (!saved.is_object())
    invalid("Saved plan generation configuration is invalid");
  try {
    return {std::move(plan), GenerationConfig::from_json(saved)};
  } catch (const std::exception &) {
    invalid("Saved plan generation configuration is invalid");
  }
}
Json save_artifacts(const SongResult &song, const fs::path &directory) {
  Progress progress(true, "Saving recording");
  if (fs::exists(directory) && !fs::is_empty(directory))
    throw Error("FileExistsError",
                "Use an empty artifact directory to avoid mixing recordings");
  validate_tokens(song.semantic.tokens);
  auto frames = song.semantic.tokens.size();
  validate_config(song.config, "Song generation configuration is invalid");
  validate_weights(song.weights);
  if (!song.timing.is_object() ||
      song.timing.value("abc", Json()) != song.semantic.plan.timing ||
      song.timing.value("semantic", Json()) != song.semantic.timing)
    invalid("Song timing must retain its planning and semantic stages");
  validate_latents(song.latents, frames, "Song latents");
  if (song.noise)
    validate_latents(*song.noise, frames, "Song noise");
  song.audio.validate("Song audio", 2);
  if (frames > size_t(INT64_MAX / 1920) ||
      song.audio.rows != 1920 * static_cast<int64_t>(frames) - 64 ||
      song.sample_rate != 48000)
    invalid(
        "Song audio must be finite FP32 stereo at its natural 48 kHz length");
  if (song.request_identity !=
      identity({{"request", song.semantic.plan.request.to_json()},
                {"config", song.config},
                {"weights", song.weights}}))
    invalid(
        "Song identity does not match its request, configuration and weights");
  fs::create_directories(directory);
  if (song.noise)
    save_matrix(directory / "noise.npy", *song.noise);
  save_plan(song.semantic.plan, directory);
  write_audio(directory / "audio.flac", song.audio, song.sample_rate);
  save_ints(directory / "semantic.npy", song.semantic.tokens);
  save_matrix(directory / "latent.npy", song.latents);
  write_json(directory / "request.json", song.semantic.plan.request.to_json());
  write_json(directory / "config.json", song.config);
  Json result = {{"status", "complete"},
                 {"identity", song.request_identity},
                 {"truncated", song.truncated()},
                 {"sample_rate", song.sample_rate},
                 {"audio_seconds", double(song.audio.rows) / song.sample_rate},
                 {"weights", song.weights},
                 {"timing", song.timing},
                 {"artifacts", collect_hashes(directory)}};
  write_json(directory / "result.json", result);
  return result;
}
SavedSong load_artifacts(const fs::path &directory) {
  auto result = verify_result(directory);
  auto plan = load_plan(directory);
  auto request = read_json(directory / "request.json");
  if (request != plan.request.to_json())
    invalid("Saved plan and song request disagree");
  auto config = read_json(directory / "config.json");
  validate_config(config, "Saved generation configuration is invalid");
  auto weights = result.value("weights", Json());
  validate_weights(weights);
  if (result.value("identity", Json()) !=
      identity(
          {{"request", request}, {"config", config}, {"weights", weights}}))
    invalid("Saved result identity disagrees with its request, configuration "
            "or weights");
  auto tokens = load_ints(directory / "semantic.npy");
  validate_tokens(tokens);
  auto latents = load_matrix(directory / "latent.npy");
  validate_latents(latents, tokens.size(), "Saved latents");
  bool has_noise = result["artifacts"].contains("noise.npy");
  if (fs::is_regular_file(directory / "noise.npy") != has_noise)
    invalid("Saved solver noise and its artifact manifest disagree");
  std::optional<FloatMatrix> noise;
  if (has_noise) {
    noise = load_matrix(directory / "noise.npy");
    validate_latents(*noise, tokens.size(), "Saved noise");
  }
  auto timing = result.value("timing", Json()),
       truncated = result.value("truncated", Json());
  if (!timing.is_object() || !timing.contains("semantic") ||
      !timing["semantic"].is_object() ||
      timing.value("abc", Json()) != plan.timing)
    invalid("Saved result lacks consistent stage timing");
  if (!truncated.is_object() || !truncated.contains("abc") ||
      !truncated["abc"].is_boolean() || !truncated.contains("semantic") ||
      !truncated["semantic"].is_boolean() || truncated["abc"] != plan.truncated)
    invalid("Saved result lacks consistent truncation metadata");
  AudioInfo audio;
  try {
    audio = audio_info(directory / "audio.flac");
  } catch (const std::exception &) {
    invalid("Saved audio is unreadable");
  }
  if (tokens.size() > size_t(INT64_MAX / 1920) || audio.sample_rate != 48000 ||
      audio.channels != 2 ||
      audio.frames != 1920 * static_cast<int64_t>(tokens.size()) - 64)
    invalid("Saved audio violates the natural 48 kHz stereo length contract");
  if (result.value("sample_rate", Json()) != audio.sample_rate ||
      result.value("audio_seconds", Json()) !=
          double(audio.frames) / audio.sample_rate)
    invalid("Saved audio metadata disagrees with the audio artifact");
  SemanticResult semantic{std::move(plan), std::move(tokens),
                          timing["semantic"],
                          truncated["semantic"].get<bool>()};
  return {std::move(semantic), std::move(latents), std::move(noise),
          std::move(config), std::move(result)};
}
} // namespace lyra
