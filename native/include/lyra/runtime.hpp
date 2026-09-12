#pragma once
#include "types.hpp"
#include <exception>
#include <memory>
namespace lyra {
void initialize_runtime();
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
