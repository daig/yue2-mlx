#include "CLyraCore/lyra.h"
#include "lyra/storage.hpp"
#include "lyra/workflow.hpp"
#import <Foundation/Foundation.h>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>

struct lyra_context {
  lyra::fs::path metallib;
  lyra_event_callback callback;
  void *user_data;
  std::atomic<bool> cancelled{false};
  std::mutex execution;
};

namespace {
char *copy_string(const std::string &text) {
  auto *result = static_cast<char *>(std::malloc(text.size() + 1));
  if (!result)
    throw std::bad_alloc();
  std::memcpy(result, text.c_str(), text.size() + 1);
  return result;
}

lyra_status error_result(char **output, const std::exception &error,
                         lyra_status status) noexcept {
  if (output) {
    try {
      *output = copy_string(lyra::Json({{"status", "failed"},
                                        {"type", lyra::exception_type(error)},
                                        {"reason", error.what()}})
                                .dump());
    } catch (...) {
      *output = nullptr;
    }
  }
  return status;
}

lyra_status unknown_error(char **output) noexcept {
  constexpr char message[] = "{\"status\":\"failed\",\"type\":\"RuntimeError\","
                             "\"reason\":\"Unknown native exception\"}";
  if (output) {
    *output = static_cast<char *>(std::malloc(sizeof(message)));
    if (*output)
      std::memcpy(*output, message, sizeof(message));
  }
  return LYRA_FAILURE;
}

lyra::Operation operation_value(lyra_operation operation) {
  switch (operation) {
  case LYRA_PREPARE:
    return lyra::Operation::Prepare;
  case LYRA_GENERATE:
    return lyra::Operation::Generate;
  case LYRA_PLAN:
    return lyra::Operation::Plan;
  case LYRA_RENDER_PLAN:
    return lyra::Operation::RenderPlan;
  case LYRA_REPLAY:
    return lyra::Operation::Replay;
  case LYRA_BATCH:
    return lyra::Operation::Batch;
  case LYRA_DOCTOR:
    return lyra::Operation::Doctor;
  }
  throw lyra::WorkflowInputError("Unknown operation");
}
} // namespace

extern "C" const char *lyra_core_build_identifier(void) {
  return lyra::core_build_sha256();
}

extern "C" lyra_context *lyra_context_create(const char *metallib_path,
                                             lyra_event_callback callback,
                                             void *user_data,
                                             char **error_json) {
  if (error_json)
    *error_json = nullptr;
  try {
    if (!metallib_path || !*metallib_path)
      throw lyra::WorkflowInputError("A bundled mlx.metallib path is required");
    auto context = std::make_unique<lyra_context>();
    context->metallib = metallib_path;
    context->callback = callback;
    context->user_data = user_data;
    return context.release();
  } catch (const std::exception &error) {
    error_result(error_json, error, LYRA_FAILURE);
  } catch (...) {
    unknown_error(error_json);
  }
  return nullptr;
}

extern "C" void lyra_context_destroy(lyra_context *context) { delete context; }
extern "C" void lyra_context_cancel(lyra_context *context) {
  if (context)
    context->cancelled.store(true, std::memory_order_relaxed);
}
extern "C" void lyra_string_free(char *string) { std::free(string); }

extern "C" lyra_status lyra_execute(lyra_context *context,
                                    lyra_operation operation,
                                    const char *options_json,
                                    char **result_json, char **error_json) {
  if (result_json)
    *result_json = nullptr;
  if (error_json)
    *error_json = nullptr;
  @autoreleasepool {
    try {
      if (!context || !options_json || !result_json || !error_json)
        throw lyra::WorkflowInputError(
            "Context, options, and output pointers are required");
      std::unique_lock lock(context->execution, std::try_to_lock);
      if (!lock.owns_lock())
        throw lyra::Error("RuntimeError",
                          "This context is already executing an operation");
      context->cancelled.store(false, std::memory_order_relaxed);
      auto operation_kind = operation_value(operation);
      lyra::Json options;
      try {
        options = lyra::Json::parse(options_json);
      } catch (const lyra::Json::parse_error &error) {
        throw lyra::WorkflowInputError(error.what());
      }
      lyra::configure_runtime(context->metallib);
      lyra::ExecutionContext execution;
      execution.cancelled = [context] {
        return context->cancelled.load(std::memory_order_relaxed);
      };
      if (context->callback) {
        execution.event = [context](const lyra::Json &event) {
          auto text = event.dump();
          context->callback(context->user_data, text.c_str());
        };
      }
      auto result = lyra::run_workflow(operation_kind, options, execution);
      *result_json = copy_string(result.value.dump());
      return result.succeeded ? LYRA_SUCCESS : LYRA_FAILURE;
    } catch (const lyra::WorkflowInputError &error) {
      return error_result(error_json, error, LYRA_INVALID_INPUT);
    } catch (const std::exception &error) {
      const auto *native = dynamic_cast<const lyra::Error *>(&error);
      return error_result(error_json, error,
                          native && native->type == "InterruptedError"
                              ? LYRA_CANCELLED
                              : LYRA_FAILURE);
    } catch (...) {
      return unknown_error(error_json);
    }
  }
}
