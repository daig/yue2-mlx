#ifndef LYRA_CORE_H
#define LYRA_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lyra_context lyra_context;
typedef enum lyra_operation {
  LYRA_PREPARE,
  LYRA_GENERATE,
  LYRA_PLAN,
  LYRA_RENDER_PLAN,
  LYRA_REPLAY,
  LYRA_BATCH,
  LYRA_DOCTOR
} lyra_operation;
typedef enum lyra_status {
  LYRA_SUCCESS = 0,
  LYRA_FAILURE = 1,
  LYRA_INVALID_INPUT = 2,
  LYRA_CANCELLED = 130
} lyra_status;

// JSON strings are UTF-8. Event JSON is borrowed until the callback returns.
// Callbacks run synchronously on the executing thread; do not throw or reenter.
typedef void (*lyra_event_callback)(void *user_data, const char *event_json);

// Borrowed build identity, valid for the lifetime of the process.
const char *lyra_core_build_identifier(void);

// The frontend supplies its bundled mlx.metallib. No GPU work starts here.
// After the first successful MLX initialization the resource path is immutable.
// On failure, returns NULL and an owned error JSON when allocation permits.
lyra_context *lyra_context_create(const char *metallib_path,
                                  lyra_event_callback callback, void *user_data,
                                  char **error_json);

// Destroy only after execute has returned and all cancellation calls have
// ended.
void lyra_context_destroy(lyra_context *context);

// Thread-safe cancellation of the current operation. Future operations start
// uncancelled. Execution itself is synchronous and never spawns the CLI.
void lyra_context_cancel(lyra_context *context);

// One call at a time per context. Result/error strings are owned by the caller.
// Partial batch failure or an unready doctor returns LYRA_FAILURE with a result
// and no error; exceptions return an error instead. Inputs use workflow options
// documented in docs/usage.md, not command-line arguments.
lyra_status lyra_execute(lyra_context *context, lyra_operation operation,
                         const char *options_json, char **result_json,
                         char **error_json);
void lyra_string_free(char *string);

#ifdef __cplusplus
}
#endif
#endif
