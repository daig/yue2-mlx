#include "lyra/workflow.hpp"
#include "lyra/artifacts.hpp"
#include "lyra/conversion.hpp"
#include "lyra/pipeline.hpp"
#include "lyra/storage.hpp"
#include <algorithm>
#include <limits>
#include <regex>
#include <set>
#include <sstream>

namespace lyra {
namespace {
struct WorkflowOptions {
  std::string command;
  Json values;
  bool has(const std::string &key) const {
    auto it = values.find(key);
    return it != values.end() && (!it->is_boolean() || it->get<bool>());
  }
  std::string get(const std::string &key,
                  const std::string &fallback = "") const {
    return values.value(key, fallback);
  }
};
std::string operation_name(Operation operation) {
  switch (operation) {
  case Operation::Prepare:
    return "prepare";
  case Operation::Generate:
    return "generate";
  case Operation::Plan:
    return "plan";
  case Operation::RenderPlan:
    return "render-plan";
  case Operation::Replay:
    return "replay";
  case Operation::Batch:
    return "batch";
  case Operation::Doctor:
    return "doctor";
  }
  throw WorkflowInputError("Unsupported operation");
}
void validate_options(Operation operation, const Json &options) {
  if (!options.is_object())
    throw WorkflowInputError("Workflow options must be a JSON object");
  const std::set<std::string> strings = {
      "model",        "vae",         "converted_dir", "precision",
      "request_file", "lyrics_file", "abc_file",      "input",
      "output",       "source",      "cache_dir",     "stage"};
  const std::set<std::string> booleans = {"offline", "require_ac", "resume",
                                          "verify_hashes"};
  for (auto it = options.begin(); it != options.end(); ++it) {
    if (strings.contains(it.key())) {
      if (!it->is_string())
        throw WorkflowInputError(it.key() + " must be a string");
    } else if (booleans.contains(it.key())) {
      if (!it->is_boolean())
        throw WorkflowInputError(it.key() + " must be a boolean");
    } else if (it.key() == "vae_core_frames") {
      if (!it->is_number_integer() || *it < 1 ||
          *it > std::numeric_limits<int>::max())
        throw WorkflowInputError(
            "vae_core_frames must be a positive integer in range");
    } else if (it.key() == "request" || it.key() == "overrides") {
      if (!it->is_object())
        throw WorkflowInputError(it.key() + " must be a JSON object");
    } else {
      throw WorkflowInputError("Unknown workflow option: " + it.key());
    }
  }
  if (options.contains("request") && options.contains("request_file"))
    throw WorkflowInputError("Pass request or request_file, not both");
  auto choice = [&](const char *key, const std::set<std::string> &allowed) {
    if (options.contains(key) &&
        !allowed.contains(options.at(key).get<std::string>()))
      throw WorkflowInputError(std::string("Invalid ") + key);
  };
  choice("precision", {"bf16", "8bit", "4bit"});
  choice("stage", {"synthesize", "decode"});
  if (options.contains("overrides")) {
    const auto &overrides = options.at("overrides");
    for (auto it = overrides.begin(); it != overrides.end(); ++it) {
      if (it.key() == "id" || it.key() == "style" || it.key() == "lyrics" ||
          it.key() == "cot" || it.key() == "abc") {
        if (!it->is_string())
          throw WorkflowInputError(it.key() + " override must be a string");
      } else if (it.key() == "seed") {
        if (!it->is_number_integer())
          throw WorkflowInputError("seed override must be an integer");
      } else if (it.key() == "cfg_scale") {
        if (!it->is_number())
          throw WorkflowInputError("cfg_scale override must be a number");
      } else {
        throw WorkflowInputError("Unknown request override: " + it.key());
      }
    }
    if (overrides.contains("cot") && overrides.at("cot") != "off" &&
        overrides.at("cot") != "melody" && overrides.at("cot") != "full")
      throw WorkflowInputError("cot must be off, melody or full");
  }
  auto require_path = [&](const char *key) {
    if (!options.contains(key) || options.at(key).get<std::string>().empty())
      throw WorkflowInputError(std::string("Required workflow option: ") + key);
  };
  if (operation == Operation::Batch || operation == Operation::RenderPlan ||
      operation == Operation::Replay)
    require_path("input");
  if (operation == Operation::RenderPlan || operation == Operation::Replay)
    require_path("output");
}
Json request(const WorkflowOptions &a,
             std::optional<Json> supplied = std::nullopt,
             fs::path base = fs::current_path()) {
  Json data;
  if (supplied)
    data = std::move(*supplied);
  else if (a.values.contains("request"))
    data = a.values.at("request");
  else {
    std::string path = a.get("request_file");
    if (path.empty())
      data = Json::object();
    else {
      data = read_json(path);
      base = fs::path(path).parent_path();
    }
  }
  if (!data.is_object())
    throw Error("ValueError", "Request must be a JSON object");
  const auto overrides = a.values.value("overrides", Json::object());
  for (auto it = overrides.begin(); it != overrides.end(); ++it) {
    data[it.key()] = it.value();
    if (it.key() == "style")
      data.erase("tags");
    if (it.key() == "lyrics" || it.key() == "abc")
      data.erase(it.key() + "_path");
  }
  for (std::string field : {"lyrics", "abc"}) {
    if (a.has(field + "_file")) {
      data[field] = read_text(a.get(field + "_file"));
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
PipelineOptions pipeline_options(const WorkflowOptions &a,
                                 const Json &generation = nullptr) {
  PipelineOptions o;
  o.model = a.get("model", std::string(MODEL_REPO));
  o.vae = a.get("vae", std::string(VAE_REPO));
  o.converted_dir = a.get("converted_dir", "models/converted");
  o.precision = a.get("precision", "bf16");
  o.offline = a.has("offline");
  o.progress = true;
  o.require_ac = a.has("require_ac");
  if (a.has("vae_core_frames"))
    o.vae_core_frames = a.values.at("vae_core_frames").get<int>();
  if (!generation.is_null())
    o.generation = GenerationConfig::from_json(generation);
  return o;
}
ResourceMonitor monitor(const WorkflowOptions &a, const fs::path &output) {
  return ResourceMonitor(
      a.has("require_ac"), fs::path(output.string() + ".resources.jsonl"),
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
    report_diagnostic("Could not write failure receipt " + path.string() +
                      ": " + e.what());
  }
  emit_event(
      {{"type", "failure"}, {"receipt", receipt}, {"path", path.string()}});
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
Json generate(const WorkflowOptions &a, const Json &req) {
  auto song = normalized(req);
  const auto options =
      pipeline_options(a, req.value("generation_config", Json()));
  (void)Sampling::from_json(req.value("abc_sampling", Json()),
                            options.generation.abc);
  (void)Sampling::from_json(req.value("semantic_sampling", Json()),
                            options.generation.semantic);
  Pipeline pipe(options);
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
    report_diagnostic("Could not remove completed attempt receipt " +
                      attempt.string() + ": " + ec.message());
  Json out = {{"output", directory.string()}, {"resumed", false}};
  out.update(receipt);
  return out;
}
WorkflowResult batch(const WorkflowOptions &a) {
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
    check_cancelled();
    emit_event({{"type", "batch_item"}, {"line", row.line}, {"id", id}});
    try {
      if (!row.error.empty())
        throw Error(row.type, row.error);
      WorkflowOptions child = a;
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
      report_diagnostic(std::string("Could not update batch receipt: ") +
                        exc.what());
      report["receipt_error"] = exception_type(exc) + ": " + exc.what();
      return {std::move(report), false};
    }
  }
  if (rows.empty()) {
    report["complete"] = true;
    write_json(output / "batch.json", report);
  }
  const bool succeeded = report["failed"] == 0;
  return {std::move(report), succeeded};
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
WorkflowResult doctor(const WorkflowOptions &a) {
  auto runtime = runtime_info();
  Json errors = runtime.at("errors"), backends = runtime.at("backends");
  Json report = {
      {"dependencies_ready", runtime.at("dependencies_ready")},
      {"versions", runtime.at("versions")},
      {"errors", errors},
      {"runtime", runtime},
      {"backends", backends},
      {"model",
       expand_user(a.get("model", a.get("converted_dir", "models/converted")))
           .string()},
      {"vae", a.has("vae") ? Json(expand_user(a.get("vae")).string()) : Json()},
      {"hashes_verified", false},
      {"validated", false},
      {"note", "Readiness is not quality, performance, or real-memory "
               "acceptance. Diagnostics never download models."}};
  if (a.has("verify_hashes")) {
    try {
      if (!a.has("vae"))
        throw Error(
            "ValueError",
            "verify_hashes requires vae pointing to a local checkpoint");
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
      (!a.has("verify_hashes") || report["hashes_verified"].get<bool>());
  if (a.has("output"))
    write_json(a.get("output"), report);
  const bool ready = report["ready"].get<bool>();
  return {std::move(report), ready};
}
WorkflowResult run(WorkflowOptions a) {
  if (a.command == "doctor")
    return doctor(a);
  if (a.command == "batch")
    return batch(a);
  if (a.command == "prepare") {
    initialize_runtime();
    std::optional<fs::path> cache;
    if (a.has("cache_dir"))
      cache = a.get("cache_dir");
    auto [model, vae] = fetch_models(cache, a.has("offline"));
    auto output = prepare(a.has("source") ? fs::path(a.get("source")) : model,
                          a.get("output", "models/converted"),
                          a.get("precision", "bf16"));
    return {Json{{"model", output.string()}, {"vae", vae.string()}}, true};
  }
  Json req;
  if (a.command == "generate" || a.command == "plan") {
    try {
      req = request(a);
      if (req.value("style", req.value("tags", Json())).is_null() ||
          req.value("lyrics", Json()).is_null())
        throw Error("ValueError",
                    "Provide style and lyrics in the request or overrides");
      (void)normalized(req);
      const auto config =
          pipeline_options(a, req.value("generation_config", Json()))
              .generation;
      (void)Sampling::from_json(req.value("abc_sampling", Json()), config.abc);
      (void)Sampling::from_json(req.value("semantic_sampling", Json()),
                                config.semantic);
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
      throw WorkflowInputError(exc.what());
    }
  }
  fs::path output = a.get("output");
  if (a.command == "generate") {
    if (!a.has("resume") && fs::exists(output) &&
        (!fs::is_directory(output) || nonempty(output)))
      throw WorkflowInputError(
          "Output directory must be empty; recordings are never "
          "silently overwritten");
    return {generate(a, req), true};
  }
  directory_guard(output);
  if (nonempty(output))
    throw WorkflowInputError(
        "Output directory must be empty; recordings are never "
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
      return {Json{{"output", output.string()},
                   {"truncated", plan.truncated},
                   {"timing", plan.timing}},
              true};
    } else if (a.command == "render-plan") {
      auto [plan, config] = load_plan_artifacts(a.get("input"));
      Pipeline pipe(
          pipeline_options(a, config.value_or(GenerationConfig{}).to_json()));
      auto result = pipe.render(plan);
      pipe.close();
      auto receipt = save(result, output);
      resources.close();
      return {std::move(receipt), true};
    } else {
      auto saved = load_artifacts(a.get("input"));
      Pipeline pipe(pipeline_options(a, saved.config.at("generation")));
      auto result = replay(pipe, std::move(saved), a.get("stage", "decode"));
      pipe.close();
      auto receipt = save(result, output);
      resources.close();
      return {std::move(receipt), true};
    }
  } catch (...) {
    resources.record_exception(std::current_exception());
    throw;
  }
  throw WorkflowInputError("Unsupported operation");
}
} // namespace
WorkflowResult run_workflow(Operation operation, const Json &options,
                            const ExecutionContext &context) {
  ExecutionScope scope(context);
  auto name = operation_name(operation);
  if (context.event)
    emit_event({{"type", "workflow_started"}, {"workflow", name}});
  check_cancelled();
  validate_options(operation, options);
  auto result = run({std::move(name), options});
  check_cancelled();
  return result;
}
} // namespace lyra
