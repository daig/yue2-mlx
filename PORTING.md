# M5 port — architecture and provenance

The production pipeline is now C++20/Objective-C++ under `native/`: native MLX 0.32.2 AR/NAR and a native FP32 MPSGraph decoder. The CMake-built CLI has no Python entrypoint, subprocess fallback or PyTorch runtime. See the [build and release instructions](README.md).

This is a reasoning-first migration with bounded native smoke execution. Comprehensive native fidelity/performance auditing is explicitly deferred to user guidance. All retained BF16 numerical, performance and listening results in this document and [validation](validation/README.md) belong to the historical Python implementation, not verified native acceptance. The invariants below remain design constraints; preserving them is not proof of parity.

## Decisions

- **Port YuE2 faithfully; optimize execution, not the learned architecture.** Initial target: offline, single-song rendering on M5 Air (10 GPU cores, 32 GB). No real-time guarantee.
- **Native MLX 0.32.2 for AR and NAR.** The C++ implementation preserves the Qwen3-compatible AR subset and cached acoustic path, including source-specific BF16 rounding and the same MPP kernel. Preserve upstream stage boundaries: `plan → generate_semantic → synthesize → decode`.
- **Upstream PyTorch/MPS is an optional reference, never a production runtime.** The previous Python port is retained under `oracle/port/lyra` for later comparison with real checkpoint inputs.
- **Native FP32 MPSGraph VAE.** Preserve decoder geometry, halo/crop behavior and full FP32 arithmetic without Python or Torch execution.
- **BF16 fidelity baseline first; evaluate 8-bit AR, then 4-bit.** Keep BF16 KV and BF16 NAR initially. Quantized quality/default selection remains an empirical decision.
- Reuse fast GQA attention, RMSNorm/RoPE, quantized linears, compilation and reusable buffers. Custom Metal only for measured remaining hotspots. No Core ML/ANE-first design; M5 GPU Neural Accelerators are separate hardware and already accessible through MLX.
- **Shared core, thin frontends.** CLI workflows live in `lyra::run_workflow`; the local `LyraCore` Swift package links the same native implementation through a static XCFramework and bundles its Metal resource. It does not invoke the CLI or install a shared Lyra dylib. SwiftUI implementation remains deferred; the fresh `Yueqin/` template is not an existing UI contract. Additional Apple Silicon configurations require their own validation; the currently validated hardware target is the 32 GB M5 Air, with macOS ≥26.2.

## Scope and provenance

- Initial feature scope: lyrics/style → song, generated or supplied ABC, upstream `full`/`melody`/`off` modes. Editing regenerates a recording; no waveform-preserving inpainting.
- Generation requires **YuE2-3B + YuE2-Vae only**. Defer SheetSage2/MERT2 transcription, source-audio covers, agent integration and best-of-N scoring. Supplied-score covers already fit generation scope.
- [Upstream source](https://github.com/multimodal-art-projection/YuE/tree/92a73cc7652fcc1f937855e4b765e0a0edd7ff2e), package `0.1.6`; pin this commit for initial parity.
- Observed HF revisions: [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B/tree/1a96eca688d6ae5d7f0feb88573fec89920fcd19) and [YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae/tree/95535e72a97bc0f09b8ada125d26b4009428c0e8). Record revisions, hashes and conversion settings in generated artifacts.
- Default listening decoder: `YuE2-Vae`. Benchmark decoder: `YuE2-Vae-legacy`; never silently substitute it.
- **Licenses:** code Apache-2.0; generator/VAE weights CC BY-NC 4.0. Do not assume commercial rights or omit upstream/third-party notices when reusing code.
- Native build prerequisites: Xcode command line tools and Metal toolchain, CMake ≥3.25, Ninja and libsndfile 1.2. The installed layout is `bin/lyra` with relative `../lib/mlx.metallib`; libsndfile is a native runtime dependency. Root `pyproject.toml` / `uv.lock` are optional reference-only tooling, installed with `uv sync --frozen`, not CLI requirements; there is no `project.scripts`. Their historical Python 3.12, MLX 0.32.2, MLX-LM 0.31.3, PyTorch 2.10.0 and Transformers 5.0.0 pins remain for reference compatibility. The separate upstream oracle uses PyTorch 2.11.0 / Transformers 4.57.6. Historical local measurements used macOS 26.3.
- Shared-core verification is bounded functional smoke, not a new fidelity/performance audit: all seven workflows ran directly from a separate Swift executable and through the CLI; checks covered progress, cross-thread cancellation and retry, host signal ownership, per-row batch failures, cross-frontend resume and native-only incremental package relinking. Numerical and memory-policy invariants are unchanged.
- **Subsequent oracle correction:** use the locked PyTorch 2.11.0 reference environment for real-checkpoint validation. PyTorch 2.10 MPS two-pass BF16/fp16 attention has a scratch-buffer memory-corruption defect ([fix #174945](https://github.com/pytorch/pytorch/pull/174945)). The historical random-weight measurements below are not a validated reference. See [`validation/README.md`](validation/README.md) for current environments and evidence.

## Architecture invariants and traps

### AR

- Qwen3-compatible AR subset: 28 layers, hidden 2048, FFN 6144, Q/KV heads 16/8, head dimension 128, Q/K RMSNorm, RoPE theta 1e6; vocabulary 184704, context 24576, untied embeddings.
- Use upstream `fast.py::{ar_keys,qwen_config}` as the extraction specification. Do not execute both experts per token. Approximately 2.17B AR parameters; combined learned model ≈3.58B. Checkpoint byte count also includes positional buffers.
- Preserve exact tokenization, prompt/special-token layout, sampling arithmetic/order, repetition window, minimum lengths and end-token behavior from `protocol.py` and `sampling.py`. Full/melody semantic CFG defaults to 1.0; off defaults to 1.01 and uses historical arithmetic. CFG is semantic AR only; ABC planning has none.
- Semantic output projection may select rows **`[151852:184621]`**: MUSIC_END plus 32768 codec IDs. Map local indices back to native IDs. ABC needs its own allowed output set. Preserve top-k ties/nucleus semantics when optimizing sampling.
- Keep sampling on-device; avoid full-vocabulary CPU transfers/per-token synchronizations. Bounded append-only KV storage; native GQA, not repeated KV heads. Distinguish physical cache slots from RoPE positions.

### Acoustic NAR

- Semantic tokens and 64-channel latents both run at 25 Hz. **32 midpoint steps = 64 velocity evaluations**, not 32.
- Port `nar.py::CachedNAR`, not generic `modeling_yue2.py::nar_velocity`: causal AR conditioning prefill once, retain per-layer K/V, then run only NAR experts for each velocity evaluation. NAR queries attend all visible conditioning plus bidirectional acoustic positions.
- Preserve the full-song CPU FP32 noise draw/seed using the native Torch-compatible noise algorithm (without loading Torch), BF16 solver arithmetic, timestep transform, zero boundary latents, local sinusoidal positions and global RoPE offsets.
- Original chunk capacity: `(24576 - len(prefix) - 3) // 2`. Each chunk uses the **same original prefix + local codec slice + MUSIC_END**, not accumulated preceding codec slices. Ordinary songs fit one full-song chunk.
- Query tiling must retain the complete key set. Shortening acoustic chunks/localizing attention changes model behavior; it is not equivalent memory tiling or free streaming.
- The M5 noncausal acoustic path now uses a fused 64-query/16-key Metal kernel with direct BF16 Q/K/V loads, FP32 softmax and FP32 accumulation. MPP `relaxed_precision=false` preserves FP32 probability operands; safe compiler math remains enabled. The original 16-key online reduction width and pairwise 8-key summation order matter for the retained intermediate bounds. Causal prefill and smaller query bounds retain the native precise path; AR math is unchanged.
- Reuse AR-generation caches only when weights, precision and prefix semantics match. **Quantized AR caches cannot replace BF16 NAR conditioning as a fidelity-preserving optimization.** Re-prefill conditioning in BF16.

### VAE and memory

- Decoder-only FP32, 48 kHz stereo; natural samples/channel = `1920*T - 64`.
- Preserve halo-and-crop decoding: upstream core 1024 frames, port default 256, halo 16; the historical Python investigation measured required halo 12. No crossfades, truncated right context or output padding. Smaller cores are a memory tradeoff, unlike smaller NAR chunks.
- Avoid unnecessary stage weight transfers/duplicate full models on unified memory. Keep large tensors resident when capacity permits; CPU artifact conversion once per stage is acceptable, not inside hot loops.
- Resource sampling is observation-only: no automatic footprint, memory-pressure, available-memory or swapping aborts, and no decoder allocation preflight cap. MLX uses its framework-default allocation policy; the 128 MiB reusable-buffer cache setting and existing tiling remain unchanged. GPU ownership, cancellation and explicitly requested AC-power checks remain active.
- Keep `MLX_ENABLE_TF32=0`. An FP32 tensor dtype alone is insufficient evidence of arithmetic parity; native decoder fidelity still requires its own audit. Preserve PCM-24 FLAC and float WAV export.

## Historical Python acoustic optimization

All measurements and listening judgments in this section refer to the Python port, not the native migration.

The precise MPP kernel is measured against the frozen pre-optimization MLX implementation, using identical real-checkpoint inputs and full solver work. Across baseline-first and optimized-first pairs on the 32 GB M5 Air, mean acoustic-stage time was **524.18 → 453.11 seconds** for 4493 frames, a **13.6% reduction**. The 400-frame mean was **11.51 → 10.57 seconds**, an **8.2% reduction**. Model loading, one warmup velocity per run and decoding are excluded.

Run-order effects are substantial; the two full-song pairs are not a confidence interval or a new sustained end-to-end campaign. Both implementations reproduce their own full-song latents exactly across the runs, but their outputs differ from each other. All 82 tensors in each 64-/400-frame fixture pass both the existing source and FP32-anchor bounds. While the diagnostic old/new full-song latent comparison shows an RMS error of 0.02325 and maximum error of 2.17383 (with the largest differences near 56 seconds), full-song human listening tests have confirmed that this drift is perceptually inaudible. The optimized audio is a perfect perceptual match to both the baseline MLX and PyTorch FP32 references, and the optimization is accepted.

See [validation/nar-optimization.json](validation/nar-optimization.json) for input/code/report hashes, paired timings, arithmetic settings, numerical evidence and resource measurements.

The subsequent exact-input 4493-frame upstream BF16 acoustic capture completed using staged expert residency: 76 arrays, all 32 midpoint steps, 8.54 GiB sampled peak footprint and no new swap-outs. Fully streamed FP32 calibration completed with 84 arrays and 11.50 GiB peak footprint; its same-runtime 64-frame control is bitwise identical to the fully resident loader across all 84 arrays. Full-song bounds were derived solely from these references. The historical Python MLX port passes all 82 source comparisons but fails the strict FP32-anchor `k_9` maximum-error bound (0.38304 > 0.27740). Listening sign-off accepted this localized divergence as a performance tradeoff; the raw numerical gate remains failed. Experimental causal-kernel replacements also fail existing comparisons and are not merged. AC enforcement is optional; memory guards remain required. See [validation/full-song-reference.json](validation/full-song-reference.json) and the [historical reproduction commands](validation/README.md#reproduction-tools).

## Historical performance probes — not release benchmarks

Historical pre-optimization end-to-end measurements are in [validation/mvp-results.json](validation/mvp-results.json). The following were short random-weight probes on battery, before checkpoint generation or thermal soak. In particular, the old long-context PyTorch 2.10 AR results are not a valid speedup baseline because of the attention defect described above.

| AR, ~6144 cached positions; no prefill/sampling | tokens/s |
|---|---:|
| Upstream MPS BF16, full output head | 8.2 |
| MLX BF16, full output head | 25.5 |
| MLX BF16, semantic-only head | 29.5 |
| MLX 8-bit, semantic-only head | 48.5 |
| MLX 4-bit, semantic-only head | 61.6 |

- At ~12K context, semantic-only BF16/8-bit/4-bit: 26.6/37.6/44.1 tok/s. KV traffic limits quantization gains.
- NAR proxy for ~215s audio, ~1K text/score prefix: Q=5378, K=11778. BF16 GEMMs 11.6–12.4 TFLOP/s; attention 52.1ms/layer/evaluation. **[INFERENCE]** ~970 TFLOP linear + ~930 TFLOP attention → ~170–180s total kernels, plus prefill/misc. Attention is comparable arithmetic to FFNs, not negligible.
- Actual upstream random-weight FP32 MPS tiled VAE: **14.05s for 215.04s audio**, correct length and finite output.
- **[INFERENCE]** Initial full-song planning ranges: ~7–10min BF16 MLX; ~5–8min quantized AR with unchanged NAR. Not measured end-to-end. AR quantization alone does not make the pipeline real-time.
- [Published 4090 reference](https://huggingface.co/m-a-p/YuE2-3B#speed-and-resources): 71.04s / 214.85s audio, 139.48 LM tok/s, 11.18 GiB peak, warm CUDA graphs/FlashAttention. Not a matched local benchmark. Quality results select 2 or 8 candidates and use legacy VAE; single-call timing excludes that selection workload.

## Deferred validation and optimization

1. Under user guidance, audit native fidelity and performance with matched saved IDs/noise and the retained oracle. Initial smoke execution does not waive any historical bound or establish a native benchmark.
2. Historical Python follow-up remains available: strict teacher-forced AR failures, the retained full-song FP32-anchor cache failure, and matched 8-bit/4-bit listening before choosing a quality-supported quantization policy. Historical BF16 listening sign-off does not transfer automatically to native code.

Navigation under upstream `src/yue2/`: `pipeline.py` (stages/artifacts), `protocol.py` + `sampling.py` (behavior), `fast.py` (AR extraction), `modeling_yue2.py` (weights/math), `nar.py` (cached solver), `modeling_vae.py` (FP32 decoder/halo rules). Read these before implementing; preserve the source contract rather than inventing a parallel convention.
