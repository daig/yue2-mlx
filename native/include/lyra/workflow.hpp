#pragma once
#include "runtime.hpp"

namespace lyra {
// Filesystem-backed operations shared by the terminal and application
// frontends. Options use native JSON values and underscore-separated names,
// never argv.
enum class Operation {
  Prepare,
  Generate,
  Plan,
  RenderPlan,
  Replay,
  Batch,
  Doctor
};

struct WorkflowResult {
  Json value;
  bool succeeded = true;
};

// Invalid requests/output selections are distinguished from execution failures
// so each frontend can present an input error without interpreting error text.
struct WorkflowInputError : Error {
  explicit WorkflowInputError(const std::string &message)
      : Error("ValueError", message) {}
};

// Synchronous. The caller owns scheduling; callbacks run during this call and
// must not reenter the engine. No terminal I/O or signal handlers are
// installed.
WorkflowResult run_workflow(Operation operation, const Json &options,
                            const ExecutionContext &context = {});
} // namespace lyra
