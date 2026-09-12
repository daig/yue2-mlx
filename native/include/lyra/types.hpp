#pragma once
#include <nlohmann/json.hpp>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
namespace lyra {
using Json = nlohmann::json;
namespace fs = std::filesystem;
inline constexpr int EOD=151643, ABC_START=151847, ABC_END=151848;
inline constexpr int MUSIC_START=151851, MUSIC_END=151852, CODEC_OFFSET=151853, CODEC_SIZE=32768;
inline constexpr int LATENT_START=184621, LATENT_END=184622, LATENT_PAD=184623;
inline constexpr int VOCAB_SIZE=184704, CONTEXT=24576;
inline constexpr std::string_view PROTOCOL_VERSION="yue2-native-v1";
struct Error : std::runtime_error {
  std::string type;
  Error(std::string kind, const std::string& message) : std::runtime_error(message), type(std::move(kind)) {}
};
using Cancelled = std::function<bool()>;
using TokenCallback = std::function<void(std::string_view,int)>;
using StepCallback = std::function<void(int,int)>;
struct Sampling {
  double temperature=1, top_p=.95;
  int top_k=100;
  double repetition_penalty=1.2;
  int penalty_window=50, min_tokens=200, max_tokens=9000;
  Json numeric_values=Json::object();
  void validate() const;
  Json to_json() const;
  static Sampling from_json(const Json&, const Sampling& defaults);
};
struct GenerationConfig {
  Sampling abc{.7,.9,30,1.005,100,32,4096}, semantic{};
  int ode_steps=32;
  std::string ode_method="midpoint";
  int context=CONTEXT;
  Json version=std::string(PROTOCOL_VERSION);
  Json context_value=CONTEXT;
  void validate() const;
  Json to_json() const;
  static GenerationConfig from_json(const Json&);
};
struct SongRequest {
  std::string style, lyrics, cot="full";
  uint64_t seed=831001;
  std::optional<std::string> abc;
  std::optional<double> cfg_scale;
  std::string id="song";
  Json numeric_values=Json::object();
  void validate() const;
  double guidance() const;
  std::string text() const;
  Json to_json() const;
  static SongRequest from_json(const Json&);
};
struct FloatMatrix {
  int64_t rows=0, cols=0;
  std::vector<float> values;
  void validate(std::string_view name, int64_t columns=0) const;
};
struct SymbolicPlan {
  SongRequest request;
  std::optional<std::string> abc;
  std::vector<int> abc_ids, prefix;
  Json timing=Json::object();
  bool truncated=false;
};
struct SemanticResult {
  SymbolicPlan plan;
  std::vector<int> tokens;
  Json timing=Json::object();
  bool truncated=false;
};
struct SongResult {
  FloatMatrix audio;
  int sample_rate=48000;
  SemanticResult semantic;
  FloatMatrix latents;
  Json config, weights, timing;
  std::string request_identity;
  std::optional<FloatMatrix> noise;
  Json truncated() const { return {{"abc",semantic.plan.truncated},{"semantic",semantic.truncated}}; }
};
struct SavedSong {
  SemanticResult semantic;
  FloatMatrix latents;
  std::optional<FloatMatrix> noise;
  Json config, result;
};
}
