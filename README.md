# yue2-mlx

Run [YuE2](https://huggingface.co/m-a-p/YuE2-3B) music generation locally on Apple Silicon. The production CLI is a C++20/Objective-C++ executable: native MLX 0.32.2 runs autoregressive planning and acoustic synthesis, and a native FP32 MPSGraph decoder renders audio. There is no Python entrypoint, subprocess fallback, or PyTorch runtime. After model preparation, generation works offline.

**Experimental, reasoning-first native migration.** Initial native checks are bounded smoke execution, not numerical or performance parity acceptance. Historical BF16 generation, listening and numerical measurements below belong to the prior Python implementation. A comprehensive native fidelity/performance audit is explicitly deferred to user guidance. The project is independent of the upstream YuE team.

The repository is named `yue2-mlx`; the native command is **`lyra`**. The historical `lyra-yue2` Python distribution and `lyra` import remain available only as optional reference tooling.

## Requirements

- **Currently validated hardware:** M5 MacBook Air with **32 GB unified memory**. Other Apple Silicon configurations are not yet validated. Intel Macs, Linux and Windows are not supported by this runtime.
- **macOS 26.2 or newer**, Xcode command line tools and the Metal toolchain, **CMake 3.25 or newer**, Ninja, and **libsndfile 1.2**. No Python environment is required.
- Internet access for initial build dependencies and approximately **7.8 GB of model downloads**. Model weights are downloaded from their pinned upstream Hugging Face repositories, not from this GitHub repository.
- **At least 20 GB of free disk space** for a BF16 model setup. The original generator, converted copy and decoder occupy approximately **15.1 GB**, before build dependencies, caches and generated recordings. Allow additional space for the native build.
- Connect AC power, close other memory-heavy workloads, and run only one generation at a time. The runtime enforces a sampled **16 GiB process budget**. Warning-level memory pressure alone is not fatal; critical pressure, insufficient memory headroom, or excessive new swapping stops the run.

The guard also preflights native decoder buffers and graph outputs; it is not a blanket MPSGraph hard allocation cap.

## Quick start

### 1. Install

Install the Xcode command line tools with `xcode-select --install` if absent, and ensure the selected Xcode installation includes the Metal toolchain. Keep `MLX_ENABLE_TF32=0` for the intended FP32 arithmetic.

```bash
git clone https://github.com/daig/yue2-mlx.git
cd yue2-mlx
brew install cmake ninja libsndfile
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target lyra -j 6
export PATH="$PWD/build/bin:$PATH"
export MLX_ENABLE_TF32=0
```


Optional user-local installation:

```bash
cmake --install build --prefix "$HOME/.local" --component lyra
export PATH="$HOME/.local/bin:$PATH"
```

Keep `bin/lyra` and its relative `../lib/mlx.metallib` together when moving an installation. libsndfile remains a native runtime dependency.

### 2. Download and convert once

This is fully automatic: it downloads the pinned generator and default decoder, checks source identities and tensor layouts, and losslessly repackages BF16 generator tensors for this runtime. No manual checkpoint editing, training or calibration is needed.

```bash
mkdir -p models &&
lyra prepare \
  --cache-dir models/hf-cache \
  --output models/converted \
  --precision bf16 > models/paths.json &&
LYRA_VAE="$(plutil -extract vae raw -o - models/paths.json)" &&
export LYRA_VAE
```

The returned model paths are saved in `models/paths.json`; the decoder path is captured automatically. Continue only after this command succeeds. Network, permissions or disk-space errors must be resolved before generation. Re-running preparation verifies and reuses a valid conversion; incomplete staged weights are not installed as a completed model.

Keep `models/converted` and `models/hf-cache` in place. In a **new terminal**, return to this repository and restore the decoder path:

```bash
export PATH="$PWD/build/bin:$PATH"
LYRA_VAE="$(plutil -extract vae raw -o - models/paths.json)" &&
export LYRA_VAE
export MLX_ENABLE_TF32=0
```

### 3. Generate a short first clip

```bash
lyra generate examples/quickstart.json \
  --model models/converted \
  --vae "$LYRA_VAE" \
  --precision bf16 \
  --offline \
  --require-ac \
  --output outputs/quickstart
```

The included English piano-pop request uses a supplied score and a **400-token semantic budget**, producing at most approximately **16 seconds** of audio. It intentionally permits truncation for a short installation check; it is not a complete-song quality example. It retains the normal **32 midpoint solver steps**.

The recording is `outputs/quickstart/audio.flac`. `result.json` records timing, identities and truncation flags; intermediate artifacts allow later replay. On macOS, listen with:

```bash
open outputs/quickstart/audio.flac
```

Output directories must be absent or empty. Use a new name such as `outputs/quickstart-2` to render again; recordings are never silently overwritten. `--quiet` disables progress display.

### 4. Generate a full song

```bash
lyra generate examples/full-song.json \
  --model models/converted \
  --vae "$LYRA_VAE" \
  --precision bf16 \
  --offline \
  --require-ac \
  --output outputs/full-song
```

This is the Chinese City Pop example from the [retained acceptance corpus](validation/corpus.json), attributed there to the upstream demo. It generates its own score and uses the normal generation budgets. Copy the JSON file and edit `style` and `lyrics` for your own request. Song duration is generated, not a fixed-length promise; inspect the truncation flags when a token budget is reached.

**Historical Python measurement on the 32 GB M5 Air:** three sequential BF16 runs of this request produced naturally ended **179.719-second** songs in **13.6–15.1 minutes each**, with **12.63 GiB sampled peak process footprint** and no additional swap-outs. These are not native timing or memory results. The songs miss the unchanged strict 180-second harness cutoff by 0.281 seconds; they were not padded. Historical listening review is complete; the duration gate remains failed. See the [measurement summary](validation/mvp-results.json) and [validation provenance](validation/README.md).

## Native stages and optional reference API

[`lyra::Pipeline`](native/include/lyra/pipeline.hpp) exposes `plan → generate_semantic → synthesize → decode`, plus end-to-end `generate` and saved-plan `render`. `PipelineOptions` controls model paths, precision, offline operation, generation configuration and resource/tiling settings. `synthesize` accepts explicit CPU FP32 noise for exact-input replay. The CLI exposes `prepare`, `doctor`, `generate`, `batch`, `plan`, `render-plan`, and `replay`; see the [usage reference](docs/usage.md).

The old Python API lives under `oracle/port/lyra`. Root `pyproject.toml` and `uv.lock` are retained solely for this optional historical reference, with unchanged package/import names, Python version and dependencies, and no `project.scripts` CLI entrypoint. Use `uv sync --frozen` only when working with that reference API or its tests/tools; it is not an installation or execution requirement for native `lyra`.

## Historical Python acoustic optimization

The Python implementation's precise M5 acoustic kernel reduced mean NAR-stage time from **524.2 to 453.1 seconds** on the same 179.72-second song: **13.6% less time** across two reversed-order pairs. The matched 400-frame stage improved **8.2%**. These are historical acoustic-only measurements, not native, end-to-end or PyTorch/MPS speedups; long-run timings varied substantially with run order.

The fused kernel keeps BF16 state/cache, FP32 probabilities and accumulation, full keys and all 64 velocity evaluations. Both 64-/400-frame source and FP32-anchor checks pass unchanged. Full-song output is not bit-identical to the old implementation; the completed source comparison retains one FP32-anchor cache failure, accepted after listening review. See the [measurement method and caveats](validation/README.md#precise-acoustic-attention-optimization) and [retained evidence](validation/nar-optimization.json).

## Compatibility and historical MVP boundaries

The native implementation preserves the Torch-compatible CPU noise algorithm, 32 midpoint steps / 64 velocity calls, the same MPP kernel and BF16 rounding, and PCM-24 FLAC / float WAV export. Preservation is an implementation contract, not a claim of measured native parity. The rendering, listening and numerical acceptance statements below describe the historical Python implementation only.

- All five generated/supplied-score paths across `full`, `melody` and `off` modes have rendered with real checkpoints. The default decoder is `YuE2-Vae`, not the legacy benchmark decoder.
- A matched 16-second reference/port acoustic pair from the original FP32-attention implementation was manually reviewed as sounding correct and identical. Subsequent current-kernel full-song reference and generated audio also received listening sign-off; numerical source comparison passes while one FP32-anchor cache bound fails and remains recorded.
- Four strict AR numerical comparisons remain outside their empirical bounds. They have not been waived for this release. Bounded NAR/VAE checks pass their recorded limits; the full-song NAR comparison is complete with one explicitly accepted FP32-anchor failure.
- **BF16 is the MVP default.** Optional 8-bit/4-bit AR modes remain experimental pending listening. They retain BF16 acoustic conditioning and add weight files rather than shrinking the whole installation to one quarter.
- The serial batch CLI, completed-result `--resume`, standalone `doctor`, and inline request overrides are available. Parallel batch execution is not supported.
- Editing a score, style or lyrics regenerates audio; it is not waveform-preserving inpainting. Source-audio transcription, streaming, real-time guarantees, best-of-N selection and a native Mac GUI are outside this release.

For diagnostics, include the command, macOS/chip/memory information, exception and relevant stage timings in a [GitHub issue](https://github.com/daig/yue2-mlx/issues). Redact private lyrics, local paths and credentials before sharing reports.

## Development and provenance

For the **optional Python reference only**:

```bash
uv sync --frozen
uv run pytest -q
uv run ruff check oracle/port/lyra tests tools
```

Do not co-install its vendored top-level `yue2` module with upstream `yue2-infer`. Historical model-based acceptance and the separately locked PyTorch oracle are described in [validation](validation/README.md). Native architecture invariants and pinned source/model revisions are in [PORTING.md](PORTING.md). Raw tensor captures, research outputs, environments and model weights are deliberately excluded from the repository and release packages.

## Licenses

- Project code: [Apache-2.0](LICENSE). Vendored upstream code retains its [Apache license](vendor/yue/LICENSE).
- **Model weights: [CC BY-NC 4.0](vendor/yue/MODEL_LICENSE)**, including the generator and default decoder. The code license does not grant unrestricted commercial model use.
- VAE-derived code retains the [upstream third-party notices](vendor/yue/THIRD_PARTY_NOTICES.md) and [MIT license texts](vendor/yue/licenses/). Native AR retains [Apple's MIT license](native/licenses/MLX_LM_MIT.txt); the [native NAR kernel](native/src/nar_attention.cpp) includes the MIT notice for its adapted MLX lane layout. Native CPU noise retains the [PyTorch and MT19937 BSD notices](native/licenses/PYTORCH-BSD.txt).

No model weights are bundled or re-hosted by this release.
