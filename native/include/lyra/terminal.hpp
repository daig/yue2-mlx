#pragma once
#include "types.hpp"
#include <memory>
namespace lyra {
// CLI-only presentation. Destruction joins the presentation heartbeat thread.
class TerminalProgress {
public:
  explicit TerminalProgress(bool enabled = true);
  ~TerminalProgress();
  void operator()(const Json &event);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace lyra
