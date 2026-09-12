#include "lyra/cli.hpp"
#include "lyra/artifacts.hpp"
#include "lyra/conversion.hpp"
#include "lyra/pipeline.hpp"
#include "lyra/storage.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>

namespace lyra {
namespace {
struct UsageError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct Args {
  std::string command, positional;
  std::map<std::string, std::string> values;
  std::set<std::string> flags;
  bool has(const std::string &key) const {
    return values.contains(key) || flags.contains(key);
  }
  std::string get(const std::string &key,
                  const std::string &fallback = "") const {
    auto i = values.find(key);
    return i == values.end() ? fallback : i->second;
  }
  bool quiet() const { return flags.contains("quiet"); }
};
using Options = std::map<std::string, std::pair<std::string, bool>>;
Options options(const std::string &command) {
  Options result;
  auto value = [&](std::string key) { result["--" + key] = {key, false}; };
  auto flag = [&](std::string key) { result["--" + key] = {key, true}; };
  value("output");
  value("precision");
  flag("offline");
  if (command == "prepare") {
    value("source");
    value("cache-dir");
    return result;
  }
  for (auto key : {"model", "vae", "converted-dir", "vae-core-frames"})
    value(key);
  flag("quiet");
  result["--no-progress"] = {"quiet", true};
  flag("require-ac");
  if (command == "generate" || command == "plan" || command == "batch") {
    value("mode");
    result["--cot"] = {"mode", false};
  }
  if (command == "generate" || command == "plan") {
    for (auto key : {"request", "abc-file", "id", "style", "lyrics",
                     "lyrics-file", "seed", "cfg-scale"})
      value(key);
    result["--abc"] = {"abc-file", false};
    result["--tags"] = {"style", false};
  }
  if (command == "generate" || command == "batch")
    flag("resume");
  if (command == "batch") {
    value("input");
    value("concurrency");
  }
  if (command == "doctor")
    flag("verify-hashes");
  if (command == "replay")
    value("stage");
  return result;
}
void help(const std::string &command) {
  if (command.empty()) {
    std::cout
        << "usage: lyra [-h] "
           "{prepare,generate,plan,render-plan,replay,batch,doctor} "
           "...\n\nOffline generation, serial batches, replay and local "
           "runtime diagnostics.\n\ncommands:\n  prepare      Fetch pinned "
           "checkpoints and convert the generator\n  generate     Generate a "
           "song\n  plan         Save a symbolic plan\n  render-plan  Render a "
           "saved plan\n  replay       Replay synthesis or decoding\n  batch   "
           "     Generate requests serially from JSONL\n  doctor       "
           "Diagnose the native runtime without downloading\n";
    return;
  }
  std::cout << "usage: lyra " << command << " [options]";
  if (command == "generate" || command == "plan")
    std::cout << " [request]";
  if (command == "render-plan" || command == "replay")
    std::cout << " request";
  std::cout << "\n\noptions:\n  -h, --help\n";
  for (const auto &[name, spec] : options(command))
    std::cout << "  " << name << (spec.second ? "" : " VALUE") << "\n";
}
void integer(const std::string &value, const std::string &name) {
  if (!std::regex_match(value, std::regex("[+-]?[0-9]+")))
    throw UsageError("argument --" + name + ": invalid int value: '" + value +
                     "'");
}
double number(const std::string &value, const std::string &name) {
  try {
    size_t end = 0;
    double n = std::stod(value, &end);
    if (end != value.size())
      throw std::invalid_argument("trailing");
    return n;
  } catch (const std::exception &) {
    throw UsageError("argument --" + name + ": invalid float value: '" + value +
                     "'");
  }
}
Args parse(int argc, char **argv) {
  Args a;
  if (argc < 2)
    throw UsageError("the following arguments are required: command");
  std::string first = argv[1];
  if (first == "--help" || first == "-h") {
    help("");
    return a;
  }
  const std::set<std::string> commands = {"prepare",     "generate", "plan",
                                          "render-plan", "replay",   "batch",
                                          "doctor"};
  if (!commands.contains(first))
    throw UsageError("invalid command: '" + first + "'");
  a.command = first;
  auto available = options(first);
  bool positional_only = false, position_seen = false;
  for (int i = 2; i < argc; ++i) {
    std::string token = argv[i];
    if (!positional_only && (token == "-h" || token == "--help")) {
      help(first);
      a.command.clear();
      return a;
    }
    if (!positional_only && token == "--") {
      positional_only = true;
      continue;
    }
    if (!positional_only && token.starts_with("-")) {
      auto eq = token.find('=');
      std::string key = token.substr(0, eq);
      auto found = available.find(key);
      if (found == available.end()) {
        std::vector<Options::iterator> matches;
        for (auto it = available.begin(); it != available.end(); ++it)
          if (it->first.starts_with(key))
            matches.push_back(it);
        if (matches.size() != 1)
          throw UsageError((matches.empty() ? "unrecognized argument: "
                                            : "ambiguous option: ") +
                           key);
        found = matches.front();
      }
      const auto &[name, is_flag] = found->second;
      if (is_flag) {
        if (eq != std::string::npos)
          throw UsageError("argument " + key + ": ignored explicit argument");
        a.flags.insert(name);
      } else {
        std::string v;
        if (eq != std::string::npos)
          v = token.substr(eq + 1);
        else {
          if (i + 1 >= argc || std::string(argv[i + 1]).starts_with("--"))
            throw UsageError("argument " + key + ": expected one argument");
          v = argv[++i];
        }
        a.values[name] = v;
      }
    } else {
      if (position_seen || !(first == "generate" || first == "plan" ||
                             first == "render-plan" || first == "replay"))
        throw UsageError("unrecognized argument: " + token);
      a.positional = token;
      position_seen = true;
    }
  }
  if ((first == "render-plan" || first == "replay") &&
      (!position_seen || !a.has("output")))
    throw UsageError("the following arguments are required: request, --output");
  if (first == "batch" && !a.has("input"))
    throw UsageError("the following arguments are required: --input");
  if (position_seen && a.has("request"))
    throw UsageError(
        "Pass request JSON either positionally or with --request, not both");
  for (auto key : {"seed", "vae-core-frames", "concurrency"})
    if (a.has(key))
      integer(a.get(key), key);
  if (a.has("cfg-scale"))
    number(a.get("cfg-scale"), "cfg-scale");
  auto choice = [&](std::string key, const std::set<std::string> &allowed) {
    if (a.has(key) && !allowed.contains(a.get(key)))
      throw UsageError("argument --" + key + ": invalid choice: '" +
                       a.get(key) + "'");
  };
  choice("precision", {"bf16", "8bit", "4bit"});
  choice("mode", {"full", "melody", "off"});
  choice("stage", {"synthesize", "decode"});
  if (a.has("concurrency")) {
    try {
      if (std::stoll(a.get("concurrency")) != 1)
        throw std::out_of_range("choice");
    } catch (const std::exception &) {
      throw UsageError(
          "argument --concurrency: invalid choice (choose from 1)");
    }
  }
  return a;
}
Json request(const Args &a, std::optional<Json> supplied = std::nullopt,
             fs::path base = fs::current_path()) {
  Json data;
  if (supplied)
    data = std::move(*supplied);
  else {
    std::string path = a.get("request", a.positional);
    if (path.empty())
      data = Json::object();
    else {
      data = read_json(path);
      base = fs::path(path).parent_path();
    }
  }
  if (!data.is_object())
    throw Error("ValueError", "Request must be a JSON object");
  for (auto key : {"id", "style", "lyrics", "seed", "cfg-scale"})
    if (a.has(key)) {
      std::string field = key;
      if (field == "cfg-scale") {
        data["cfg_scale"] = number(a.get(key), key);
        continue;
      }
      if (field == "seed") {
        try {
          const auto value = a.get(key);
          if (value.starts_with('-'))
            data[field] = std::stoll(value);
          else
            data[field] = std::stoull(value);
        } catch (const std::exception &) {
          throw Error("ValueError", "seed must be in [0, 2**64)");
        }
        continue;
      }
      data[field] = a.get(key);
      if (field == "style")
        data.erase("tags");
      if (field == "lyrics")
        data.erase("lyrics_path");
    }
  if (a.has("mode"))
    data["cot"] = a.get("mode");
  for (std::string field : {"lyrics", "abc"}) {
    if (a.has(field + "-file")) {
      data[field] = read_text(a.get(field + "-file"));
      data.erase(field + "_path");
    } else if (data.contains(field + "_path")) {
      if (data.contains(field) && !data[field].is_null())
        throw Error("ValueError",
                    "Pass " + field + " or " + field + "_path, not both");
      if (!data[field + "_path"].is_string())
        throw Error("TypeError", field + "_path must be a path string");
      data[field] = read_text(base / data[field + "_path"].get<std::string>());
      data.erase(field + "_path");
    }
  }
  Json generation = data.value("generation_config", Json());
  data.erase("generation_config");
  Json result = request_kwargs(std::move(data), base);
  if (!generation.is_null())
    result["generation_config"] = generation;
  return result;
}
PipelineOptions pipeline_options(const Args &a,
                                 const Json &generation = nullptr) {
  PipelineOptions o;
  o.model = a.get("model", std::string(MODEL_REPO));
  o.vae = a.get("vae", std::string(VAE_REPO));
  o.converted_dir = a.get("converted-dir", "models/converted");
  o.precision = a.get("precision", "bf16");
  o.offline = a.has("offline");
  o.progress = !a.quiet();
  o.require_ac = a.has("require-ac");
  if (a.has("vae-core-frames")) {
    try {
      o.vae_core_frames = std::stoi(a.get("vae-core-frames"));
    } catch (const std::exception &) {
      throw Error("ValueError", "vae_core_frames is out of range");
    }
  }
  if (!generation.is_null())
    o.generation = GenerationConfig::from_json(generation);
  return o;
}
ResourceMonitor monitor(const Args &a, const fs::path &output) {
  return ResourceMonitor(
      a.has("require-ac"), fs::path(output.string() + ".resources.jsonl"),
      fs::path(output.string() + ".resources.json"),
      {{"command", a.command},
       {"memory_policy", "observe_only"},
       {"vae_core_frames", pipeline_options(a).vae_core_frames},
       {"precision", a.get("precision", "bf16")}});
}
Json failure(const fs::path &path, const std::exception &exc,
             Json metadata = Json::object()) {
  Json receipt = {{"status", "failed"},
                  {"type", exception_type(exc)},
                  {"reason", exc.what()}};
  receipt.update(metadata);
  try {
    write_json(path, receipt);
  } catch (const std::exception &e) {
    receipt["receipt_error"] = exception_type(e) + ": " + e.what();
    std::cerr << "Could not write failure receipt " << path.string() << ": "
              << e.what() << '\n';
  }
  return receipt;
}
bool nonempty(const fs::path &p) {
  return fs::exists(p) && fs::is_directory(p) &&
         fs::directory_iterator(p) != fs::directory_iterator();
}
void directory_guard(const fs::path &p, const std::string &label = "Output") {
  if (fs::is_symlink(p) || (fs::exists(p) && !fs::is_directory(p)))
    throw Error("FileExistsError",
                label + " is not a regular directory: " + p.string());
}
bool managed(std::string name) {
  static const std::set<std::string> allowed = {
      "request.json",  "config.json", "result.json",        "failure.json",
      "noise.npy",     "audio.flac",  "prefix.npy",         "semantic.npy",
      "latent.npy",    "plan.json",   "plan_manifest.json", "score.abc",
      "abc_tokens.npy"};
  if (allowed.contains(name))
    return true;
  if (!name.ends_with(".tmp"))
    return false;
  name.resize(name.size() - 4);
  auto dot = name.rfind('.');
  if (dot == std::string::npos || dot + 1 == name.size())
    return false;
  return allowed.contains(name.substr(0, dot)) &&
         std::all_of(name.begin() + dot + 1, name.end(),
                     [](unsigned char c) { return c >= '0' && c <= '9'; });
}
SongRequest normalized(const Json &req) {
  Json d = req;
  for (auto key : {"abc_sampling", "semantic_sampling", "generation_config"})
    d.erase(key);
  return SongRequest::from_json(d);
}
Json generate(const Args &a, const Json &req) {
  Pipeline pipe(pipeline_options(a, req.value("generation_config", Json())));
  auto song = normalized(req);
  auto abc = req.value("abc_sampling", Json()),
       semantic = req.value("semantic_sampling", Json());
  auto effective = pipe.effective_config(song, abc, semantic);
  std::string expected = identity({{"request", song.to_json()},
                                   {"config", effective},
                                   {"weights", pipe.weights}});
  fs::path directory =
               a.get("output", (fs::path("runs/default") / song.id).string()),
           attempt = directory.string() + ".attempt.json";
  directory_guard(directory);
  if (a.has("resume") && fs::exists(directory / "result.json")) {
    auto result = read_json(directory / "result.json");
    if (result.value("status", "") == "complete" && !fs::exists(attempt)) {
      auto saved = load_artifacts(directory);
      if (saved.result.at("identity") != expected)
        throw Error("ValueError", "Request/config/weight identity changed; use "
                                  "a new output directory");
      Json receipt = {{"output", directory.string()}, {"resumed", true}};
      receipt.update(saved.result);
      pipe.close();
      return receipt;
    }
  }
  if (nonempty(directory)) {
    if (!a.has("resume") || !fs::is_regular_file(attempt) ||
        fs::is_symlink(attempt))
      throw Error("FileExistsError", "Nonempty output " + directory.string() +
                                         "; use a new output directory");
    auto prior = read_json(attempt);
    if (prior.value("identity", "") != expected ||
        (prior.value("status", "") != "running" &&
         prior.value("status", "") != "failed"))
      throw Error("ValueError",
                  "Prior attempt identity changed; use a new output directory");
    for (const auto &entry : fs::directory_iterator(directory))
      if (entry.is_symlink() || !entry.is_regular_file() ||
          !managed(entry.path().filename().string()))
        throw Error("FileExistsError", "Interrupted output contains unrelated "
                                       "artifacts; use a new output directory");
    fs::rename(directory,
               directory.string() + ".interrupted-" + unique_suffix());
  }
  if (fs::exists(attempt) || fs::is_symlink(attempt)) {
    if (!a.has("resume") || fs::is_symlink(attempt))
      throw Error("FileExistsError",
                  "Prior attempt receipt exists: " + attempt.string() +
                      "; use --resume or a new output");
    if (read_json(attempt).value("identity", "") != expected)
      throw Error("ValueError",
                  "Prior attempt identity changed; use a new output directory");
  }
  write_json(attempt, {{"status", "running"}, {"identity", expected}});
  Json receipt;
  try {
    auto resources = monitor(
        a, a.has("resume")
               ? fs::path(directory.string() + ".retry-" + unique_suffix())
               : directory);
    try {
      auto result = pipe.generate(song, abc, semantic);
      pipe.close();
      receipt = save_artifacts(result, directory);
      resources.close();
    } catch (...) {
      resources.record_exception(std::current_exception());
      throw;
    }
  } catch (const std::exception &exc) {
    failure(attempt, exc, {{"identity", expected}});
    failure(directory / "failure.json", exc, {{"identity", expected}});
    throw;
  }
  std::error_code ec;
  fs::remove(attempt, ec);
  if (ec)
    std::cerr << "Could not remove completed attempt receipt "
              << attempt.string() << ": " << ec.message() << '\n';
  Json out = {{"output", directory.string()}, {"resumed", false}};
  out.update(receipt);
  return out;
}
int batch(const Args &a) {
  struct Row {
    int line;
    Json data;
    std::string error, type;
  };
  std::vector<Row> rows;
  std::set<std::string> ids;
  fs::path input = a.get("input"), output = a.get("output", "runs/batch");
  std::istringstream stream(read_text(input));
  std::string line;
  int n = 0;
  while (std::getline(stream, line)) {
    ++n;
    if (line.find_first_not_of(" \t\r\n") == std::string::npos)
      continue;
    try {
      auto row = Json::parse(line);
      bool valid =
          row.is_object() && row.contains("id") && row["id"].is_string();
      std::string id = valid ? row["id"].get<std::string>() : "";
      if (!valid || id.empty() || id == "." || id == ".." ||
          id.find_first_of("/\\") != std::string::npos ||
          id.find('\0') != std::string::npos || id.ends_with(".json") ||
          id.ends_with(".jsonl") || ids.contains(id))
        throw Error(
            "ValueError",
            "Every batch row needs a unique, single-component string id");
      ids.insert(id);
      rows.push_back({n, std::move(row), "", ""});
    } catch (const std::exception &exc) {
      rows.push_back({n, nullptr, exc.what(), exception_type(exc)});
    }
  }
  directory_guard(output, "Batch output");
  if (nonempty(output) && !a.has("resume"))
    throw Error(
        "FileExistsError",
        "Nonempty batch output; use --resume or a new output directory");
  if (nonempty(output)) {
    auto p = output / "batch.json";
    if (fs::is_symlink(p) || !fs::is_regular_file(p))
      throw Error(
          "FileExistsError",
          "Nonempty batch output lacks a batch receipt; use a new directory");
    auto prior = read_json(p);
    if (!prior.is_object() || !prior.contains("results") ||
        !prior["results"].is_array())
      throw Error("ValueError",
                  "Invalid prior batch receipt; use a new directory");
  }
  fs::create_directories(output);
  Json report = {{"complete", false},
                 {"expected", rows.size()},
                 {"failed", 0},
                 {"results", Json::array()}};
  write_json(output / "batch.json", report);
  for (const auto &row : rows) {
    Json id = row.data.is_null() ? Json() : row.data["id"];
    if (!a.quiet())
      std::cerr << "Batch line " << row.line << ": "
                << (id.is_null() ? "invalid request" : id.get<std::string>())
                << std::endl;
    try {
      if (!row.error.empty())
        throw Error(row.type, row.error);
      Args child = a;
      child.values["output"] = (output / id.get<std::string>()).string();
      auto receipt =
          generate(child, request(child, row.data, input.parent_path()));
      report["results"].push_back({{"line", row.line},
                                   {"id", id},
                                   {"status", "complete"},
                                   {"identity", receipt.at("identity")},
                                   {"resumed", receipt.at("resumed")}});
    } catch (const std::exception &exc) {
      if (exception_type(exc) == "InterruptedError")
        throw;
      report["failed"] = report["failed"].get<int>() + 1;
      report["results"].push_back(
          failure(output / ("line-" + std::to_string(row.line) + "." +
                            unique_suffix() + ".failure.json"),
                  exc, {{"line", row.line}, {"id", id}}));
    }
    report["complete"] =
        report["results"].size() == rows.size() && report["failed"] == 0;
    try {
      write_json(output / "batch.json", report);
    } catch (const std::exception &exc) {
      std::cerr << "Could not update batch receipt: " << exc.what() << '\n';
      return 1;
    }
  }
  if (rows.empty()) {
    report["complete"] = true;
    write_json(output / "batch.json", report);
  }
  return report["failed"] == 0 ? 0 : 1;
}
Json save(const SongResult &result, const fs::path &output) {
  save_artifacts(result, output);
  return {{"output", output.string()},
          {"sample_rate", result.sample_rate},
          {"audio_seconds", double(result.audio.rows) / result.sample_rate},
          {"truncated", result.truncated()},
          {"timing", result.timing}};
}
SongResult replay(Pipeline &pipe, SavedSong saved, const std::string &stage) {
  double start = monotonic_seconds();
  std::string noise_source = saved.noise ? "source_artifact" : "unavailable";
  double nar_seconds = 0;
  if (stage == "synthesize") {
    if (!saved.noise) {
      saved.noise =
          initial_noise(static_cast<int>(saved.semantic.tokens.size()),
                        saved.semantic.plan.request.seed);
      noise_source = "regenerated_from_request_seed";
    }
    double t = monotonic_seconds();
    saved.latents = pipe.synthesize(saved.semantic, *saved.noise);
    nar_seconds = monotonic_seconds() - t;
  }
  double vae_start = monotonic_seconds();
  auto audio = pipe.decode(saved.latents);
  Json timing = {
      {"abc", saved.semantic.plan.timing},
      {"semantic", saved.semantic.timing},
      {"nar_seconds", nar_seconds},
      {"vae_seconds", monotonic_seconds() - vae_start},
      {"load", pipe.load_timing},
      {"e2e_seconds", monotonic_seconds() - start},
      {"replay", {{"stage", stage}, {"reused_latents", stage == "decode"}}}};
  auto config = pipe.effective_config(saved.semantic.plan.request);
  config["replay"] = {{"stage", stage},
                      {"source_identity", saved.result.at("identity")},
                      {"source_config", saved.config},
                      {"source_weights", saved.result.at("weights")},
                      {"source_artifacts", saved.result.at("artifacts")},
                      {"source_timing", saved.result.at("timing")},
                      {"source_truncated", saved.result.at("truncated")},
                      {"noise_source", noise_source},
                      {"noise_used_for_synthesis", stage == "synthesize"}};
  auto stamp = identity({{"request", saved.semantic.plan.request.to_json()},
                         {"config", config},
                         {"weights", pipe.weights}});
  return {std::move(audio),          48000,
          std::move(saved.semantic), std::move(saved.latents),
          std::move(config),         pipe.weights,
          std::move(timing),         std::move(stamp),
          std::move(saved.noise)};
}
int doctor(const Args &a) {
  auto runtime = runtime_info();
  Json errors = runtime.at("errors"), backends = runtime.at("backends");
  Json report = {
      {"dependencies_ready", runtime.at("dependencies_ready")},
      {"versions", runtime.at("versions")},
      {"errors", errors},
      {"runtime", runtime},
      {"backends", backends},
      {"model",
       expand_user(a.get("model", a.get("converted-dir", "models/converted")))
           .string()},
      {"vae", a.has("vae") ? Json(expand_user(a.get("vae")).string()) : Json()},
      {"hashes_verified", false},
      {"validated", false},
      {"note", "Readiness is not quality, performance, or real-memory "
               "acceptance. Diagnostics never download models."}};
  if (a.has("verify-hashes")) {
    try {
      if (!a.has("vae"))
        throw Error(
            "ValueError",
            "--verify-hashes requires --vae pointing to a local checkpoint");
      fs::path model = report["model"].get<std::string>(),
               vae = report["vae"].get<std::string>();
      if (!fs::is_directory(model) || !fs::is_directory(vae))
        throw Error(
            "FileNotFoundError",
            "Hash verification requires local model and VAE directories");
      Json weights = {{"mot", verify_conversion(model)},
                      {"vae", model_identity(vae)}};
      if (!fs::is_regular_file(
              model / ("ar-" + a.get("precision", "bf16") + ".safetensors")))
        throw Error("FileNotFoundError",
                    "Converted AR precision is unavailable: " +
                        a.get("precision", "bf16"));
      report["weights"] = std::move(weights);
      report["hashes_verified"] = true;
    } catch (const std::exception &exc) {
      errors["weights"] = exception_type(exc) + ": " + exc.what();
    }
  }
  report["errors"] = errors;
  bool backend_ready = !backends.empty();
  for (const auto &value : backends)
    backend_ready = backend_ready && value.get<bool>();
  report["ready"] =
      runtime.at("dependencies_ready").get<bool>() && backend_ready &&
      runtime.at("supported_os").get<bool>() &&
      runtime.at("supported_arch").get<bool>() &&
      runtime.at("unsafe_environment").empty() &&
      (!a.has("verify-hashes") || report["hashes_verified"].get<bool>());
  if (a.has("output"))
    write_json(a.get("output"), report);
  std::cout << report.dump(2) << '\n';
  return report["ready"].get<bool>() ? 0 : 1;
}
int run(Args a) {
  if (a.command.empty())
    return 0;
  if (a.command == "doctor")
    return doctor(a);
  if (a.command == "batch")
    return batch(a);
  if (a.command == "prepare") {
    initialize_runtime();
    std::optional<fs::path> cache;
    if (a.has("cache-dir"))
      cache = a.get("cache-dir");
    auto [model, vae] = fetch_models(cache, a.has("offline"));
    auto output = prepare(a.has("source") ? fs::path(a.get("source")) : model,
                          a.get("output", "models/converted"),
                          a.get("precision", "bf16"));
    std::cout
        << Json({{"model", output.string()}, {"vae", vae.string()}}).dump()
        << '\n';
    return 0;
  }
  Json req;
  if (a.command == "generate" || a.command == "plan") {
    try {
      req = request(a);
      if (req.value("style", req.value("tags", Json())).is_null() ||
          req.value("lyrics", Json()).is_null())
        throw Error("ValueError",
                    "Provide style and lyrics in request JSON or inline flags");
      if (req.contains("generation_config"))
        GenerationConfig::from_json(req["generation_config"]);
      if (!a.has("output")) {
        auto id = req.value("id", Json("song"));
        if (!id.is_string() || id == "." || id == ".." ||
            !std::regex_match(id.get<std::string>(),
                              std::regex("[A-Za-z0-9][A-Za-z0-9_.-]{0,179}")))
          throw Error("ValueError", "id must be a filename-safe identifier");
        a.values["output"] =
            (fs::path("runs/default") / id.get<std::string>()).string();
      }
    } catch (const std::exception &exc) {
      throw UsageError(exc.what());
    }
  }
  fs::path output = a.get("output");
  if (a.command == "generate") {
    if (!a.has("resume") && fs::exists(output) &&
        (!fs::is_directory(output) || nonempty(output)))
      throw UsageError("Output directory must be empty; recordings are never "
                       "silently overwritten");
    std::cout << generate(a, req).dump(2) << '\n';
    return 0;
  }
  directory_guard(output);
  if (nonempty(output))
    throw UsageError("Output directory must be empty; recordings are never "
                     "silently overwritten");
  auto resources = monitor(a, output);
  try {
    if (a.command == "plan") {
      Pipeline pipe(
          pipeline_options(a, req.value("generation_config", Json())));
      auto abc = req.value("abc_sampling", Json()),
           semantic = req.value("semantic_sampling", Json());
      auto plan = pipe.plan(normalized(req), abc);
      auto config = pipe.effective_config(plan.request, abc, semantic);
      pipe.close();
      save_plan_artifacts(plan, output,
                          GenerationConfig::from_json(config.at("generation")));
      resources.close();
      std::cout << Json({{"output", output.string()},
                         {"truncated", plan.truncated},
                         {"timing", plan.timing}})
                       .dump(2)
                << '\n';
    } else if (a.command == "render-plan") {
      auto [plan, config] = load_plan_artifacts(a.positional);
      Pipeline pipe(
          pipeline_options(a, config.value_or(GenerationConfig{}).to_json()));
      auto result = pipe.render(plan);
      pipe.close();
      auto receipt = save(result, output);
      resources.close();
      std::cout << receipt.dump(2) << '\n';
    } else {
      auto saved = load_artifacts(a.positional);
      Pipeline pipe(pipeline_options(a, saved.config.at("generation")));
      auto result = replay(pipe, std::move(saved), a.get("stage", "decode"));
      pipe.close();
      auto receipt = save(result, output);
      resources.close();
      std::cout << receipt.dump(2) << '\n';
    }
  } catch (...) {
    resources.record_exception(std::current_exception());
    throw;
  }
  return 0;
}
} // namespace
int cli_main(int argc, char **argv) {
  try {
    return run(parse(argc, argv));
  } catch (const UsageError &exc) {
    std::cerr << "usage: lyra [-h] "
                 "{prepare,generate,plan,render-plan,replay,batch,doctor} "
                 "...\nlyra: error: "
              << exc.what() << '\n';
    return 2;
  } catch (const std::exception &exc) {
    std::cerr << Json({{"status", "failed"},
                       {"type", exception_type(exc)},
                       {"reason", exc.what()}})
                     .dump()
              << '\n';
    return exception_type(exc) == "InterruptedError" ? 130 : 1;
  }
}
} // namespace lyra
