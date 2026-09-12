# Usage and API reference

Run shell commands from the repository root after completing the [native installation and model preparation](../README.md#quick-start). The shell examples assume `lyra` is on `PATH`, `MLX_ENABLE_TF32=0`, and `LYRA_VAE` was loaded with `plutil -extract vae raw -o - models/paths.json`. Production requires no Python environment. The seven commands are `prepare`, `doctor`, `generate`, `batch`, `plan`, `render-plan`, and `replay`. To create an editable request, copy `examples/full-song.json` to `request.json`; requests are UTF-8 JSON.

## Yueqin macOS app

Build and launch [`Yueqin/Yueqin.xcodeproj`](../Yueqin/Yueqin.xcodeproj) as described in the [app setup](../README.md#yueqin-macos-app). The sidebar presents every native workflow. **Activity & Results** stays available while navigating: below the request controls in **Plan score**, and in the right-hand inspector elsewhere. Run the selected workflow with the labeled toolbar button or **Command-Return**. Execution is serial and off the UI thread. Cancel from the toolbar, activity panel or **Command-Period**; quitting during execution offers to cancel and waits for the native worker to stop.

### Models and output

- **Prepare models** downloads or reuses the pinned source/VAE cache and converts the selected AR precision. Successful preparation updates the shared model paths. To reuse a CLI installation without another conversion, expand **Engine settings** and select its converted-model directory and local VAE directory.
- **Diagnostics** checks readiness without downloading. Optional hash verification requires local model and VAE directories. An unready result retains its errors and complete report; readiness is not a quality or performance certification.
- **Engine settings** exposes model/VAE identifiers or paths, converted directory, BF16/8-bit/4-bit precision, offline resolution, AC-power requirement and VAE core frames. Settings and per-workflow drafts persist locally. Blank numeric fields retain native defaults.
- A blank recording output creates a unique directory under `~/Music/Yueqin`, or the configured default output folder. An explicit output must be new or empty. **Resume matching output** requires selecting the prior directory and keeps the native identity/receipt checks; it does not append to or overwrite an unrelated recording. Planning, rendering and replay create new outputs rather than resuming.
- Filesystem fields accept absolute paths or `~/…`; native pickers avoid dependence on Xcode's or Finder's working directory. Preparation has its own converted-model output, and diagnostics has an optional JSON report file rather than a recording directory.

### Requests and raw controls

**Generate song** and **Plan score** can compose a request or submit a JSON file. Composed requests expose style, lyrics text/file, generated or supplied ABC, `cot` (`full`, `melody`, `off`), identifier, exact signed 64-bit seed and CFG scale. Advanced controls retain separate ABC/semantic temperature, top-p, top-k, repetition penalty/window and minimum/maximum tokens. Solver steps and the complete `generation_config` JSON object are available directly; a populated solver-steps field overrides `ode_steps` in that object. Composed requests can be exported as JSON.

JSON-file mode passes the file path to the core without importing or rewriting it. Relative `lyrics_path` and `abc_path` still resolve beside that file. Only explicitly enabled file overrides are submitted; edit nested sampling and generation configuration in the original JSON. Switching between file and composed mode retains the composed draft.

**Render plan** consumes a saved-plan directory. **Replay artifacts** consumes a saved-song directory and exposes `decode` and `synthesize`. The result inspector can hand a saved plan to rendering or a recording to replay, preserving the input and clearing the new output destination. **Batch** accepts JSONL, optional score-mode override and matching-output resume; failed rows remain visible alongside successful recordings.

### Results

The inspector shows native stages/counts, elapsed time, warnings, typed errors, truncation flags and batch-row outcomes. A token-limited run can complete without reaching the sequence's natural end. Files at a failed run's destination are not presented as proof of newly completed output.

Play/pause and seek use native audio playback. **Export FLAC…** copies the original recording without re-encoding, staging beside the destination before an atomic replacement. **Advanced output** is collapsed by default and resets when another run starts. It contains generated-file Finder links, output/model paths, exact submitted options, result JSON and any partial batch receipt. Playback, workflow actions, errors and truncation warnings remain outside it. Neither frontend adds memory caps or allocation guards.

GUI verification is bounded functional smoke with real checkpoints: all seven workflows, both replay stages, a short recording, resume, partial batches, cancellation/recovery, readiness/error presentation and playback were exercised in a temporary native SwiftUI host. Native numerical fidelity, listening quality and performance acceptance remain deferred. System-hosted save-dialog acceptance was not completed by session automation; request serialization and FLAC creation/replacement/failure preservation were exercised separately.

### Score workspace

**Plan score** reserves the large, persistent right-hand pane for notation, with request settings and activity in a narrower, resizable left-hand column. Both columns stay within the window when resized; the score does not move into the results inspector or disappear when the request form scrolls. The placement is a working score surface for future editing, but currently provides only viewing, zoom and scrolling.

After successful planning, the application reads the actual saved `score.abc` as UTF-8 on the worker and presents it directly. The last score remains available during the app session when navigating or running another workflow. Replanning keeps the previous score visible and interactive until the replacement is loaded. Cancellation, failed planning or a score-loading error retains it with an explicit previous-score message, never presenting it as the new result. Token-limited notation carries a warning beside the score.

An intentional `cot = off` plan has no notation and clears the previous score to the empty state; an unexpectedly missing or unreadable score reports a loading error instead. No plan files are rewritten. **Render this plan** still passes the intact plan folder to **Render plan**; file browsing belongs under **Advanced output**, not on the score surface.

Integration verification exercised the actual Xcode-built application in an isolated bundle: saved and model-generated ABC, cancellation, failed regeneration, retained-score navigation, valid score-off plans, hidden/expanded artifacts, render-plan handoff, zoom, appearances and window resizing. This is functional UI evidence, not numerical or performance acceptance.

### Standalone ABC score component

[`ABCScoreView`](../Yueqin/Yueqin/ABCScore/ABCScoreView.swift) is the reusable, read-only SwiftUI view embedded in the score workspace. It remains independent of the application's request and result models. Its only input is notation text:

```swift
ABCScoreView(abc: scoreText)
    .frame(minHeight: 360)
```

The host owns file loading, editing, generation and placement. The component has no `LyraCore`, workspace or model dependency. It supplies native zoom controls (50–300%), fit-width reset, light/dark appearance, scrolling, empty/error presentation and a collapsible plain-text parser-warning panel. Multiple tunes and renderable portions of malformed input remain viewable. Parser warnings are not validation of musical correctness or generation fidelity.

Engraving preserves the source's staff-system breaks at abcjs's natural layout width; responsive SVG scaling fits the available space without re-parsing on resize, theme or zoom changes. Automatic line wrapping is deliberately disabled: it can misalign multi-voice scores containing multi-measure rests. Zoom enlarges the scrollable notation rather than changing the score's line breaks.

The private `WKWebView` loads only bundled resources, uses a nonpersistent data store, blocks external navigation and network connections, and passes ABC as structured JavaScript arguments rather than interpolated HTML or executable code. Keep both Swift files in [`ABCScore/`](../Yueqin/Yueqin/ABCScore/) in the same target and package its four web/license resources in that target's bundle. Yueqin's existing filesystem-synchronized Xcode group includes them automatically. There is no notation editor, audio playback or online renderer in this component.

The unmodified bundled renderer is [abcjs 6.7.0](https://github.com/paulrosen/abcjs/tree/v6.7.0), distributed under its adjacent [MIT license](../Yueqin/Yueqin/ABCScore/abcjs-basic-min.js.LICENSE). `abcjs-basic-min.js` comes from the upstream tag's `dist/` directory; its SHA-256 is `b0cde4bc52bb33949181683a245005fff8a024a8c1f07ec6ce3222cd4bd72e51`.

## CLI generation

A request is JSON. The smallest useful form is:

```json
{
  "id": "first_song",
  "style": "English, warm piano pop, expressive vocal, 88 BPM",
  "lyrics": "[Verse]\nMorning finds the open road\n\n[Chorus]\nCarry every note back home",
  "cot": "full",
  "seed": 831001
}
```

Generate from the converted model and local VAE, with all model resolution forced offline:

```bash
lyra generate request.json \
  --model models/converted \
  --vae "$LYRA_VAE" \
  --precision bf16 \
  --offline \
  --output outputs/first-song
```

If the portable directory was exported, it contains both model paths:

```bash
lyra generate request.json --model models/offline --offline --output outputs/first-song
```

Output directories must be absent or empty; Lyra never mixes or silently overwrites recordings. Use `--resume` to reuse a matching completed result or retry a matching interrupted/failed output. `--quiet` disables display progress without changing RNG or generation. `--require-ac` rejects a run that starts off AC power or loses AC power. CLI runs also write sampled resource evidence beside the output as `<output>.resources.jsonl` and `<output>.resources.json`.

### Inline request overrides

The request path is optional when `--style` and `--lyrics` (or `--lyrics-file`) are supplied. `--id`, `--seed`, `--cfg-scale`, `--mode`, and `--abc`/`--abc-file` override matching JSON fields. A JSON request may use relative `lyrics_path` and `abc_path` values; they resolve relative to that request file. If `--output` is omitted, `generate` and `plan` use `runs/default/<id>`, while `batch` uses `runs/batch`.

CLI values take precedence over request JSON fields. `--style` (also `--tags`) replaces the JSON style/tags value; explicit lyric/score file options replace the corresponding inline value or JSON path. CLI file paths are relative to the working directory, whereas JSON paths are relative to the request file. Pass the request either positionally or with `--request`, not both.

```bash
lyra generate \
  --style "English, warm piano pop, expressive vocal, 88 BPM" \
  --lyrics-file lyrics.txt \
  --abc-file validation/score.abc \
  --mode full --seed 831001 \
  --model models/converted --vae "$LYRA_VAE" --offline \
  --output outputs/inline-song
```

`--resume` is identity-checked against the normalized request, effective generation configuration and model identities. A changed request or configuration requires a new output directory.

A supplied score is read as UTF-8 while preserving its bytes through tokenization. `--abc`/`--abc-file` and `--mode` override the corresponding request fields:

```bash
lyra generate request.json \
  --abc validation/score.abc \
  --mode full \
  --model models/converted \
  --vae "$LYRA_VAE" \
  --offline \
  --output outputs/supplied-score
```

The five valid paths are:

| `cot` | ABC | Behavior |
|---|---|---|
| `full` | generated | Generate melody and chord ABC, then audio |
| `full` | supplied | Use exact supplied melody/chord ABC |
| `melody` | generated | Generate melody-plan ABC; accompaniment remains free |
| `melody` | supplied | Use exact supplied melody ABC |
| `off` | none | Generate semantic music directly; supplied ABC is invalid |


### Serial batches and diagnostics

`batch` reads one JSON request object per nonempty JSONL line. Every row requires a unique, single-component `id`; malformed rows are recorded as failures without creating a song directory. Batch execution is deliberately serial (`--concurrency 1` is the only accepted value) because one process-wide GPU workload is allowed.

```bash
lyra batch --input requests.jsonl \
  --model models/converted --vae "$LYRA_VAE" --offline \
  --output outputs/batch
```

The batch receipt is `outputs/batch/batch.json`. Add `--resume` to identity-check and reuse completed per-song results, or retry matching failed/interrupted rows.

`doctor` reports native dependency versions, OS/architecture support, Metal backends and unsafe execution environment settings without downloading models. Add `--verify-hashes --model ... --vae ...` to verify local conversion and decoder identities; its exit status is nonzero when the environment is not ready. Readiness does not establish native quality or performance parity.

```bash
lyra doctor
lyra doctor --verify-hashes \
  --model models/converted --vae "$LYRA_VAE" --precision bf16
```

## Requests and generation controls

`SongRequest` fields are `style`, `lyrics`, `cot`, `seed`, `abc`, `cfg_scale`, and filename-safe `id`. `tags` is an alias for `style`, but the two cannot disagree. Seeds are integers in `[0, 2**63)` and belong to one request.

A CLI request may contain `generation_config`, `abc_sampling`, and `semantic_sampling`. In C++, set `PipelineOptions::generation` when constructing the pipeline and pass stage-local JSON sampling overrides to `plan`, `generate_semantic`, or `generate`. A sampling override changes only named fields.

| Control | ABC default | Semantic default |
|---|---:|---:|
| `temperature` | 0.7 | 1.0 |
| `top_p` | 0.9 | 0.95 |
| `top_k` | 30 | 100 |
| `repetition_penalty` | 1.005 | 1.2 |
| `penalty_window` | 100 | 50 |
| `min_tokens` | 32 | 200 |
| `max_tokens` | 4096 | 9000 |

`temperature=0` selects greedy decoding. The semantic CFG default is `1.0` for `full`/`melody` and the upstream historical `1.01` arithmetic for `off`; `cfg_scale` explicitly overrides it in `[0, 20]`. ABC planning has no CFG. The acoustic default is 32 midpoint steps, which means 64 velocity evaluations. `ode_steps` is configurable, while `ode_method="midpoint"` and context `24576` are fixed protocol fields. Prefix plus requested generation budget overflow fails explicitly. Exhausting `max_tokens` sets the corresponding truncation flag rather than pretending an end token was sampled.

Example fields to merge into a complete request JSON:

```json
{
  "generation_config": {"ode_steps": 32},
  "abc_sampling": {"top_k": 30},
  "semantic_sampling": {"top_k": 80, "top_p": 0.95},
  "cfg_scale": 1.2
}
```

## Shared workflows and Swift package

[`lyra::run_workflow`](../native/include/lyra/workflow.hpp) owns the same seven operations used by the CLI. Its inputs are an `Operation`, a JSON options object and an optional `ExecutionContext`; its result contains the operation's JSON payload and `succeeded`. The CMake `lyra_core` target supplies the native engine and build identity to C++ consumers. CLI argument aliases, terminal rendering, exit codes and signal handlers stay in the terminal frontend.

The root [`Package.swift`](../Package.swift) exposes the `LyraCore` library product for macOS ≥26.2. Build its native inputs before resolving the local package in Xcode or SwiftPM:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target lyra_swift_package -j 6
swift build -c release
```

Add the repository as a local package dependency and link `LyraCore`. CMake uses the same pinned MLX 0.32.2 sources as the CLI, merges the native static archives into `CLyraCore.xcframework`, and copies `mlx.metallib` and license notices into the Swift resource bundle. There is no installed Lyra dylib, CLI subprocess or Python runtime. `CSndFile` resolves the existing Homebrew libsndfile dependency through pkg-config; standalone app dependency bundling is deferred.

Regenerate `lyra_swift_package` after native changes, then rebuild the Swift client. The generated `NativeBuild.swift` identity is a Swift compiler input so native-only rebuilds also relink clients; initialization rejects mismatched Swift/native builds. Generated XCFrameworks, resources and that identity source are not checked in. Normal Swift-only changes use SwiftPM directly.

[`Engine`](../native/swift/LyraCore/Engine.swift) exposes `.prepare`, `.generate`, `.plan`, `.renderPlan`, `.replay`, `.batch` and `.doctor`. `execute` is synchronous; schedule generation on a worker, not the UI actor. For example, a non-UI caller can inspect readiness without downloading models:

```swift
import LyraCore

let engine = try Engine()
let report = try engine.execute(.doctor)
print(String(decoding: report.json, as: UTF8.self))
```

Pass options as UTF-8 JSON `Data`, not command-line arguments. Integer seeds remain exact; encode them as integers rather than converting them through `Double`. Request JSON, sampling controls, generation configuration and defaults are the existing CLI contract:

| Workflow | Operation options |
|---|---|
| `prepare` | `source`, `output`, `cache_dir`, `precision`, `offline` |
| `generate` | `request` object or `request_file` path, `overrides`, `lyrics_file`, `abc_file`, `output`, `resume` |
| `plan` | Same request inputs as `generate`, plus `output`; no resume |
| `renderPlan` | `input` saved-plan directory and `output` directory, both required |
| `replay` | `input` saved-song directory and `output` directory, both required; `stage` is `decode` (default) or `synthesize` |
| `batch` | Required `input` JSONL path, `output`, `resume`, and optional `overrides` for the batch's request fields, including `cot` |
| `doctor` | `model` (or `converted_dir`), `vae`, `precision`, `verify_hashes`, optional report-file `output` |

Generation, planning, rendering, replay and batch also accept `model`, `vae`, `converted_dir`, `precision` (`bf16`, `8bit`, `4bit`), `offline`, `require_ac` and `vae_core_frames`. These are native values: booleans and integers are not flag strings. `doctor` never downloads models; hash verification requires local converted-model and VAE directories.

`request` and `request_file` are mutually exclusive. `overrides` accepts `id`, `style`, `lyrics`, integer `seed`, numeric `cfg_scale`, and `cot` (`full`, `melody`, `off`). File overrides take precedence over request content. Relative `lyrics_path` and `abc_path` inside a request file or batch row resolve against that file's directory; inline requests use the working directory. Explicit `lyrics_file`/`abc_file` paths also use the working directory. Prefer absolute filesystem paths in GUI clients. Nested `generation_config`, `abc_sampling` and `semantic_sampling` stay in the request; they are not new derived controls.

`WorkflowResult.json` retains the operation-specific payload. An unready doctor or partially failed batch returns `succeeded == false` with its diagnostics/results intact. Execution errors throw `EngineError` with `type`, `message` and numeric `status`; input errors use status 2 and cancellation uses 130. The CLI maps these states to its existing exit codes. Batch execution remains serial; `--concurrency 1`, aliases and `--quiet`/`--no-progress` are CLI presentation/argument concerns, not extra engine options.

Optional `Engine(onEvent:)` receives owned JSON `Data` synchronously on the executing thread. Events include workflow start, stage start/progress/completion, generation completion, batch rows, failures and warnings. Dispatch UI updates to the UI actor; do not re-enter `execute` from a callback. Omitting the callback avoids native progress serialization. See [cancellation and ownership](#cancellation-and-ownership) for lifecycle rules.

The lower-level [C API](../native/include/CLyraCore/lyra.h) is also available. C++/C hosts must explicitly supply the `mlx.metallib` path through `configure_runtime`/context creation; the core never guesses from the host executable. Swift supplies its bundle resource automatically. C callbacks borrow their event string only for the callback; result/error strings belong to the caller and must be released with `lyra_string_free`.

## Native API and independent stages

The C++20 interface is [`lyra::Pipeline`](../native/include/lyra/pipeline.hpp), configured with `PipelineOptions`. It uses native MLX 0.32.2 for AR/NAR and native FP32 MPSGraph for decoding, with no PyTorch runtime.

| Method | Input → output |
|---|---|
| `plan` | `SongRequest`, optional ABC sampling JSON → `SymbolicPlan` |
| `generate_semantic` | `SymbolicPlan`, optional sampling JSON → `SemanticResult` |
| `synthesize` | `SemanticResult`, explicit FP32 `FloatMatrix` noise → latent `FloatMatrix` |
| `decode` | latent `FloatMatrix` → stereo audio `FloatMatrix` |
| `generate` | `SongRequest`, optional ABC/semantic sampling JSON → `SongResult` |
| `render` | `SymbolicPlan`, optional semantic sampling JSON → `SongResult` |

`PipelineOptions` includes model/VAE paths, converted directory, precision, offline/progress/AC settings, VAE core frames, query chunk size and generation configuration. `effective_config` reports the effective request configuration; `close` releases pipeline resources.

`SymbolicPlan` retains the request, ABC text and original token IDs/prefix, timing and ABC truncation state. `SemanticResult` retains that plan and codec-local IDs. Synthesis uses CPU FP32 `[T, 64]` matrices for noise and latents; decoding produces FP32 stereo audio. `SongResult` retains audio, semantic/latent/noise data, configuration, identities, timings and separate truncation flags. Audio export uses PCM-24 FLAC and float WAV.

The historical Python API remains under `oracle/port/lyra` for optional reference work. Root `pyproject.toml` and `uv.lock` retain its original package/import names, Python version and dependencies but no `project.scripts`. Only reference consumers need `uv sync --frozen` and `uv run python your_script.py`; they do not provide the production CLI.

## Exact plans, editing, and artifact replay

Save a plan without rendering it:

```bash
lyra plan request.json \
  --model models/converted --vae "$LYRA_VAE" --offline \
  --output outputs/plan
```

The CLI and **Plan score** GUI workflow normally produce the same five-file plan. An intentional `cot = off` plan omits `score.abc`:

| File | Meaning |
| --- | --- |
| `score.abc` | Human-readable ABC notation: voices, notes/rests, rhythm, key/meter and other score directives. The GUI renders this directly in the score workspace. Copy it before external editing. |
| `abc_tokens.npy` | The exact integer token IDs representing the planned score, stored as a NumPy-format array. Rendering reuses these IDs rather than re-tokenizing the text. |
| `prefix.npy` | The complete saved conditioning-token prefix for subsequent generation, also a NumPy-format array. It is not another score or an audio waveform. |
| `plan.json` | Structured plan data: request, ABC text and IDs, timing, truncation flags and effective generation configuration. |
| `plan_manifest.json` | SHA-256 integrity records for the other four files. Keep all five together for **Render plan** / `render-plan`. |

Planning alone does not create audio, semantic tokens or acoustic latents. When resource recording is enabled, sibling files such as `outputs/plan.resources.json` and `outputs/plan.resources.jsonl` contain the diagnostic resource summary and individual samples. They are not score content or required plan inputs.

Render the exact saved plan later. The saved generation configuration and original ABC IDs/prefix are restored directly; ABC is not decoded and re-tokenized:

```bash
lyra render-plan outputs/plan \
  --model models/converted --vae "$LYRA_VAE" --offline \
  --output outputs/from-saved-plan
```

A saved plan is immutable integrity-checked input. Do not edit `outputs/plan/score.abc` in place: its manifest will reject the change. To edit a composition, copy the score to a new file, edit it, and submit that file as a new external plan input:

```bash
cp outputs/plan/score.abc edited.abc
# Edit edited.abc, then generate a new recording:
lyra generate request.json \
  --abc edited.abc --mode full \
  --model models/converted --vae "$LYRA_VAE" --offline \
  --output outputs/edited-recording
```

Edit `request.json` as well when changing lyrics or style. Every such edit intentionally creates a new request and recording.

A full artifact directory can restart at either retained latents or retained semantic tokens:

```bash
# Reuse latents; run only the decoder.
lyra replay outputs/first-song --stage decode \
  --model models/converted --vae "$LYRA_VAE" --offline \
  --output outputs/redecoded

# Reuse semantic tokens and solver noise; rerun NAR and the decoder.
lyra replay outputs/first-song --stage synthesize \
  --model models/converted --vae "$LYRA_VAE" --offline \
  --output outputs/resynthesized
```

If an older artifact lacks retained noise, `--stage synthesize` regenerates the original request-local CPU noise from the saved seed and records that fact. Replay records the source identity, source weights/configuration/artifact hashes, truncation, and whether latents or noise were reused.

### Artifact contents and provenance

A saved plan contains `plan.json`, `plan_manifest.json`, `abc_tokens.npy`, `prefix.npy`, and `score.abc` when applicable. A saved song adds `request.json`, `config.json`, `semantic.npy`, `latent.npy`, optional retained `noise.npy`, `audio.flac`, and `result.json`.

`result.json` hashes every other artifact and records request/configuration/weight identity, stage timings, sample rate, duration, and separate truncation flags. `config.json` records effective sampling/CFG, backend and dtypes, geometry, RNG policy, runtime versions/hash, source commit, and decoder release. `weights` includes the complete conversion manifest for the generator and the verified VAE identity. A `status` of `complete` means the requested operation finished; it does not override `truncated.abc` or `truncated.semantic`.

Runtime identity now hashes the compiled native engine/MLX/JACCL archives (`core_build_sha256`) and Metal library (`metallib_sha256`), not the CLI or Swift host executable. Matching builds can resume each other's completed results. Resume remains strict: a changed native build or an older executable-based identity can require a new output directory. Saved plans, recordings and replay retain their existing formats.

## Cancellation and ownership

CLI interruption stops the current operation rather than treating partial artifacts as completed output. The CLI installs/restores its own SIGINT/SIGTERM handlers; embedding the core does not replace the host's handlers. Use matching `--resume` / `resume: true` for supported retries.

Swift's `Engine.cancel()` is thread-safe and cooperatively cancels the active execution at engine checkpoints; it does not forcibly interrupt a running GPU kernel. Cancellation is reset for the next execution, including after an interrupted attempt. Concurrent calls on one engine are rejected, and callbacks must not re-enter it. C++ workflow callers supply `ExecutionContext.event` and `ExecutionContext.cancelled`; direct stage callers can install the same operation-local context with `ExecutionScope`. C++ callback exceptions are contained and request cancellation.

The `workflow_started` event is delivered after the C/Swift per-call cancellation reset and before validation or workflow work. A frontend that allows cancellation while execution is still queued must retain that intent and reapply `cancel()` from this synchronous startup callback. Yueqin does this, so an immediate Cancel is not lost when the worker enters the core.

One pipeline can retain weights between serial requests; concurrent calls are unsupported, and GPU execution ownership prevents competing Lyra workloads. `Pipeline::close` or destruction releases resident resources. Workflow calls manage their own pipeline/resource lifetime.

## Faithful execution and memory geometry

- **AR:** MLX executes only the required Qwen3-compatible expert path with the pinned tokenizer, exact prompt/special-token layout, phase-local native-ID output projections, upstream minimum/end/repetition/top-k/top-p/CFG behavior, device-local sampling, absolute cache/RoPE positions, and 1024-token prefill chunks. BF16 mode keeps weights and caches in BF16. Each AR phase creates its own request-local MLX key from the full 63-bit seed and never mutates global MLX RNG state.
- **NAR:** native MLX preserves BF16 conditioning and solver arithmetic, two zero boundary latents, local learned latent positions, globally offset RoPE, the source timestep shift, and midpoint updates: 32 steps mean 64 velocity calls. The original chunk capacity is `min((24576 - len(prefix) - 3) // 2, 24576)` semantic frames. Each chunk prefills causal conditioning exactly once, then uses the original prefix plus only its local codec slice and `MUSIC_END`; earlier codec chunks are not accumulated. A native Torch-compatible CPU noise algorithm draws FP32 noise once for the whole song from the request seed, then slices it at those original cuts; it does not load Torch.
- **Attention:** `query_chunk_size=256` is the default query memory bound; every query still sees the full conditioning and acoustic key set. On the validated M5 runtime, noncausal BF16 acoustic attention uses a fused 64-query/16-key Metal kernel when the requested bound permits it. It reads BF16 Q/K/V directly, keeps softmax probabilities and matrix accumulators in FP32, and rounds only the result to BF16. It does not allocate a dense score matrix or expanded FP32 K/V buffers. Causal conditioning, smaller query bounds and unsupported devices/shapes retain the native precise paths; the short unmasked vector kernel already has FP32 opmath. MLX TF32 remains disabled. Query tiling is not local attention, streaming or a change to song chunks. Retained weights, KV and solver state stay BF16; quantized AR generation caches are discarded before BF16 NAR conditioning.
- **VAE:** the default `YuE2-Vae` decoder is native decoder-only FP32 MPSGraph. Default tiling is 256 latent-frame cores with a 16-frame halo and exact crop; there are no crossfades, shortened right context, or output padding. For `T` latent frames, output is 48 kHz stereo with `1920*T - 64` samples per channel. An alternate decoder must be passed explicitly; the benchmark `YuE2-Vae-legacy` is never silently substituted.

Resource monitoring is **observation-only**. The CLI records process footprint, system memory pressure/headroom, swap activity and native runtime memory counters in its resource sidecars. Memory usage, pressure and swapping do not automatically abort execution, and there is no application-imposed memory budget or decoder allocation preflight cap. Framework counters are not additive components of process footprint; system swap counters are not attributable to this process.

MLX uses its framework-default allocation policy, with the existing 128 MiB reusable-buffer cache setting. Existing prefill/query/VAE tiling is unchanged. Actual allocation failures, invalid inputs, explicit cancellation, single-owner GPU execution and opt-in `--require-ac` checks retain their normal behavior. Generation and resource metadata identify this policy as `memory_policy="observe_only"`.

These are native implementation invariants, not numerical or performance acceptance results. Retained BF16/performance/listening/numerical measurements describe the historical Python implementation. Initial native migration uses reasoning-first implementation and bounded smoke execution; comprehensive native fidelity/performance auditing is deferred to user guidance.

## Reproducibility contract

Within the supported MLX runtime and unchanged request/configuration/weights, AR sampling and acoustic noise are request-local and reproducible. ABC and semantic phases each reset their own MLX stream from the request seed, matching the source phase policy; the acoustic stage independently uses the same seed for one full-song CPU FP32 noise draw. Equal numeric seeds across PyTorch and MLX do **not** promise identical sampled songs because their categorical RNG implementations differ. Fidelity comparisons therefore reuse identical saved IDs and noise rather than comparing independently sampled songs.


## Portable offline models

Native generation can use the converted generator and pinned decoder directories directly. Copy both intact to an offline machine, retain their manifests, and pass their paths with `--model` and `--vae`; no Python environment is needed.

### Optional historical reference export

The following export API is available only in the optional Python reference environment (`uv sync --frozen`, then run a script with `uv run python`). It is not part of native installation or a native CLI command:

```python
import os
from lyra import YuE2Pipeline

with YuE2Pipeline.from_pretrained(
    "models/converted", vae=os.environ["LYRA_VAE"], local_files_only=True,
) as pipe:
    pipe.save_pretrained("models/offline")
```

An already exported directory can be used by native `lyra` with `--model models/offline --offline`, without `--vae`. Keep the native executable, its relative `../lib/mlx.metallib`, and the libsndfile runtime dependency on the offline machine; the Python export environment is not needed to render.

## Experimental AR quantization

BF16 is the MVP default. The optional variants below have not completed the matched listening assessment; do not treat them as equally validated defaults.

```bash
lyra prepare --cache-dir models/hf-cache --output models/converted --precision 8bit --offline
lyra prepare --cache-dir models/hf-cache --output models/converted --precision 4bit --offline
```

These use affine, group-size-64, linear-only AR quantization. Both preserve the BF16 partitions required for acoustic conditioning and synthesis; they add files rather than quartering the complete installation’s disk size. Pass the matching `--precision` to generation. Cached pinned source snapshots are required when preparing variants.

See [validation status](../validation/README.md) and [architecture/provenance](../PORTING.md) for the current limitations and source identities.
