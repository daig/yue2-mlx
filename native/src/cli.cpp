#include "lyra/cli.hpp"
#include "lyra/storage.hpp"
#include "lyra/terminal.hpp"
#include "lyra/workflow.hpp"
#include <algorithm>
#include <cerrno>
#include <csignal>
#include <iostream>
#include <mach-o/dyld.h>
#include <map>
#include <regex>
#include <set>
#include <system_error>
#include <vector>

namespace lyra {
namespace {
struct UsageError : WorkflowInputError {
  using WorkflowInputError::WorkflowInputError;
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

Operation operation(const std::string &command) {
  if (command == "prepare")
    return Operation::Prepare;
  if (command == "generate")
    return Operation::Generate;
  if (command == "plan")
    return Operation::Plan;
  if (command == "render-plan")
    return Operation::RenderPlan;
  if (command == "replay")
    return Operation::Replay;
  if (command == "batch")
    return Operation::Batch;
  return Operation::Doctor;
}
Json native_options(const Args &a) {
  Json result = Json::object(), overrides = Json::object();
  for (const auto &[key, value] : a.values) {
    if (key == "concurrency")
      continue; // The CLI accepts only serial execution.
    if (key == "id" || key == "style" || key == "lyrics") {
      overrides[key] = value;
    } else if (key == "mode") {
      overrides["cot"] = value;
    } else if (key == "cfg-scale") {
      overrides["cfg_scale"] = number(value, key);
    } else if (key == "seed" || key == "vae-core-frames") {
      try {
        if (key == "seed") {
          if (value.starts_with('-'))
            overrides["seed"] = std::stoll(value);
          else
            overrides["seed"] = std::stoull(value);
        } else {
          result["vae_core_frames"] = std::stoll(value);
        }
      } catch (const std::exception &) {
        throw UsageError(key + " is out of range");
      }
    } else {
      std::string name = key == "request" ? "request_file" : key;
      std::replace(name.begin(), name.end(), '-', '_');
      result[name] = value;
    }
  }
  for (auto key : a.flags) {
    if (key == "quiet")
      continue;
    std::replace(key.begin(), key.end(), '-', '_');
    result[key] = true;
  }
  if (!a.positional.empty())
    result[a.command == "generate" || a.command == "plan" ? "request_file"
                                                          : "input"] =
        a.positional;
  if (!overrides.empty())
    result["overrides"] = std::move(overrides);
  return result;
}
volatile std::sig_atomic_t interrupted = 0;
void on_signal(int signal) { interrupted = signal; }
class SignalScope {
public:
  SignalScope() {
    interrupted = 0;
    struct sigaction action{};
    action.sa_handler = on_signal;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, &previous_int_) != 0)
      throw std::system_error(errno, std::generic_category(),
                              "Installing SIGINT handler");
    if (sigaction(SIGTERM, &action, &previous_term_) != 0) {
      const int error = errno;
      sigaction(SIGINT, &previous_int_, nullptr);
      throw std::system_error(error, std::generic_category(),
                              "Installing SIGTERM handler");
    }
  }
  ~SignalScope() {
    sigaction(SIGTERM, &previous_term_, nullptr);
    sigaction(SIGINT, &previous_int_, nullptr);
    interrupted = 0;
  }
  SignalScope(const SignalScope &) = delete;
  SignalScope &operator=(const SignalScope &) = delete;

private:
  struct sigaction previous_int_{}, previous_term_{};
};
fs::path cli_metallib() {
  uint32_t size = 1024;
  std::vector<char> path(size);
  if (_NSGetExecutablePath(path.data(), &size) != 0) {
    path.resize(size);
    if (_NSGetExecutablePath(path.data(), &size) != 0)
      throw Error("RuntimeError", "Could not resolve CLI executable path");
  }
  return (fs::canonical(fs::path(path.data())).parent_path() /
          "../lib/mlx.metallib")
      .lexically_normal();
}
int run(const Args &a) {
  if (a.command.empty())
    return 0;
  auto options = native_options(a);
  configure_runtime(cli_metallib());
  TerminalProgress terminal(!a.quiet());
  ExecutionContext context;
  context.progress = !a.quiet();
  context.event = [&terminal](const Json &event) { terminal(event); };
  context.cancelled = [] { return interrupted != 0; };
  WorkflowResult result;
  {
    SignalScope signals;
    result = run_workflow(operation(a.command), options, context);
  }
  if (a.command != "batch")
    std::cout << result.value.dump(a.command == "prepare" ? -1 : 2) << '\n';
  return result.succeeded ? 0 : 1;
}
} // namespace
int cli_main(int argc, char **argv) {
  try {
    return run(parse(argc, argv));
  } catch (const WorkflowInputError &exc) {
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
