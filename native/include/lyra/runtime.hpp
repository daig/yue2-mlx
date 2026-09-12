#pragma once
#include "types.hpp"
#include <exception>
#include <memory>
namespace lyra {
// Callbacks run synchronously on the operation thread and are serialized.
// They must not throw or reenter Lyra. A throwing callback cancels this scope.
// Captured callback state must remain alive until run_workflow returns.
struct ExecutionContext {
  std::function<void(const Json &)> event;
  Cancelled cancelled;
  bool progress = true;
};
class ExecutionScope {
public:
  explicit ExecutionScope(const ExecutionContext &context);
  ~ExecutionScope();
  ExecutionScope(const ExecutionScope &) = delete;
  ExecutionScope &operator=(const ExecutionScope &) = delete;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
void emit_event(const Json &event);
void report_diagnostic(std::string message) noexcept;
// Frontends supply the packaged MLX asset explicitly. Once MLX is configured,
// only the same resolved path may be supplied again.
void configure_runtime(const fs::path &metallib);
void initialize_runtime();
// Digest of the compiled core and pinned native dependency archives.
const char *core_build_sha256();
Json runtime_info();
Json memory_snapshot();
Json power_source();
void check_cancelled();
bool cancellation_requested();
double monotonic_seconds();
class ResourceMonitor {
public:
  ResourceMonitor(bool require_ac = false,
                  const std::optional<fs::path> &log_path = std::nullopt,
                  const std::optional<fs::path> &report_path = std::nullopt,
                  Json metadata = Json::object());
  ~ResourceMonitor();
  void check() const;
  Json report() const;
  void close();
  void record_exception(std::exception_ptr error);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
class GPUExecution {
public:
  explicit GPUExecution(bool require_ac = false);
  ~GPUExecution();
  void check() const;
  void close();
  Json report() const;
  void record_exception(std::exception_ptr error);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
class Progress {
public:
  Progress(bool enabled, std::string label, std::string unit = "",
           int total = 0);
  ~Progress();
  void update(int complete, int total = 0);
  void advance();
  void finish(bool truncated = false);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace lyra
