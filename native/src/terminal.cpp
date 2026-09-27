#include "lyra/terminal.hpp"
#include "lyra/runtime.hpp"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iterator>
#include <mutex>
#include <sstream>
#include <sys/ioctl.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace lyra {
namespace {
std::string ascii(std::string value) {
  for (auto &c : value)
    if (c < ' ' || c > '~')
      c = ' ';
  return value;
}
} // namespace
struct TerminalProgress::Impl {
  bool enabled = true, progress_enabled, tty = false, stop = false;
  int columns = 80, width = 0;
  double last = 0, interval = 5;
  struct Stage {
    Json event;
    double received;
  };
  std::vector<Stage> stages;
  std::mutex mutex;
  std::condition_variable wake;
  std::thread thread;
  explicit Impl(bool e) : progress_enabled(e) {
    if (!progress_enabled)
      return;
    tty = isatty(STDERR_FILENO);
    if (tty) {
      struct winsize ws{};
      const char *cols = std::getenv("COLUMNS");
      char *end = nullptr;
      long value = cols ? std::strtol(cols, &end, 10) : 0;
      if (cols && end != cols && !*end && value > 0 && value <= 100000)
        columns = int(value);
      else if (!ioctl(STDERR_FILENO, TIOCGWINSZ, &ws) && ws.ws_col)
        columns = ws.ws_col;
      tty = columns >= 60;
    }
    interval = tty ? .25 : 5.;
    thread = std::thread([this] {
      std::unique_lock lock(mutex);
      while (!wake.wait_for(lock, std::chrono::duration<double>(interval),
                            [this] { return stop; })) {
        try {
          if (!stages.empty())
            render(stages.back());
        } catch (...) {
          enabled = false;
        }
      }
    });
  }
  ~Impl() {
    {
      std::lock_guard lock(mutex);
      stop = true;
    }
    wake.notify_all();
    if (thread.joinable())
      thread.join();
    if (width)
      std::fprintf(stderr, "\n");
  }
  void output(std::string text, bool final) {
    if (!enabled)
      return;
    if (tty) {
      text.resize(std::min(text.size(), size_t(columns - 1)));
      std::string padding(std::max(0, width - int(text.size())), ' ');
      if (std::fprintf(stderr, "\r%s%s%s", text.c_str(), padding.c_str(),
                       final ? "\n" : "") < 0)
        enabled = false;
      width = final ? 0 : int(text.size());
    } else if (std::fprintf(stderr, "%s\n", text.c_str()) < 0)
      enabled = false;
    if (std::fflush(stderr))
      enabled = false;
  }
  void render(const Stage &stage, const char *status = nullptr,
              bool force = false) {
    double now = monotonic_seconds();
    if (!enabled || (!force && now - last < interval))
      return;
    last = now;
    const auto &event = stage.event;
    double elapsed = event.value("elapsed_seconds", 0.) +
                     (status ? 0. : std::max(0., now - stage.received));
    int complete = event.value("completed", 0);
    int total = event.at("total").is_null() ? 0 : event.at("total").get<int>();
    auto unit = ascii(event.value("unit", ""));
    if (unit == "prompt_tokens")
      unit = "prompt tokens";
    else if (unit == "conditioning_layers")
      unit = "conditioning layers";
    std::ostringstream suffix;
    suffix << std::fixed << std::setprecision(1);
    if (unit == "codec_frames") {
      suffix << event.value("content_seconds", 0.) << "s audio represented";
      if (auto limit = event.find("limit_seconds");
          limit != event.end() && limit->is_number())
        suffix << " (limit " << limit->get<double>() << "s)";
      suffix << " | ";
    } else if (total > 0) {
      if (tty) {
        int fill = std::clamp(int(8. * complete / total), 0, 8);
        suffix << "[" << std::string(fill, '#') << std::string(8 - fill, '-')
               << "] ";
      }
      suffix << complete << "/" << total << " "
             << (unit.empty() ? "items" : unit) << " (" << std::setprecision(0)
             << 100. * complete / total << "% of stage) | "
             << std::setprecision(1);
    } else if (!unit.empty() || complete) {
      suffix << complete << " " << (unit.empty() ? "items" : unit);
      if (auto limit = event.find("limit");
          limit != event.end() && limit->is_number())
        suffix << " (limit " << limit->get<int>() << ")";
      suffix << " | ";
    }
    if (unit == "tokens" && total == 0)
      suffix << (elapsed > 0 ? complete / elapsed : 0) << " tokens/s | ";
    suffix << (tty ? "" : "elapsed ") << elapsed << "s";
    std::string prefix = status ? status
                         : tty ? std::string(1, "|/-\\"[int(elapsed / .25) % 4])
                         : force && complete == 0 ? "Starting"
                                                  : "Running";
    std::string label = ascii(event.value("stage", ""));
    if (tty) {
      int available =
          columns - 1 - int(("[YuE2] " + prefix + " : " + suffix.str()).size());
      if (int(label.size()) > std::max(0, available))
        label = label.substr(0, std::max(0, available - 3)) +
                (available >= 3 ? "..." : "");
    }
    output("[YuE2] " + prefix + " " + label + ": " + suffix.str(),
           status != nullptr);
  }
  void accept(const Json &event) {
    std::lock_guard lock(mutex);
    if (!enabled)
      return;
    auto type = event.value("type", "");
    if (!progress_enabled && type != "warning")
      return;
    if (type == "stage_started") {
      stages.push_back({event, monotonic_seconds()});
      render(stages.back(), nullptr, true);
    } else if (type == "progress" || type == "stage_completed") {
      auto found =
          std::find_if(stages.rbegin(), stages.rend(), [&](const Stage &stage) {
            return stage.event.at("stage") == event.at("stage");
          });
      if (found == stages.rend())
        return;
      *found = {event, monotonic_seconds()};
      if (type == "progress")
        render(*found);
      else {
        auto status = event.value("status", "completed");
        render(*found,
               status == "cancelled" ? "Cancelled"
               : status == "failed"  ? "Failed"
               : event.value("truncated", false)
                   ? "Finished (generation limit reached)"
                   : "Completed",
               true);
        stages.erase(std::next(found).base());
      }
    } else if (type == "warning") {
      output("lyra: " + ascii(event.value("message", "")), true);
    } else if (type == "batch_item") {
      output("Batch line " + event.at("line").dump() + ": " +
                 (event.at("id").is_null()
                      ? "invalid request"
                      : ascii(event.at("id").get<std::string>())),
             true);
    } else if (type == "generation_completed") {
      std::ostringstream text;
      text << "Generated " << event.at("audio_seconds").get<double>()
           << " seconds of audio in "
           << event.at("elapsed_seconds").get<double>() << " seconds";
      if (event.value("truncated", false))
        text << " (truncated)";
      output(text.str(), true);
    }
  }
};
TerminalProgress::TerminalProgress(bool enabled)
    : impl_(std::make_unique<Impl>(enabled)) {}
TerminalProgress::~TerminalProgress() = default;
void TerminalProgress::operator()(const Json &event) { impl_->accept(event); }
} // namespace lyra
