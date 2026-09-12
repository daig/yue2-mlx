# Validation status — BF16 MVP

This release exposes the working generation pipeline with explicit open acceptance gates. It does not claim that all reference-equivalence or listening checks pass. No numerical threshold was relaxed to publish the MVP.

## Evidence included in the repository

- [mvp-results.json](mvp-results.json): curated three-song timings, process-memory and swap-out measurements, request/corpus hashes, upstream revision and retained source-report hash.
- [listening.json](listening.json): the recorded partial human assessment, scope and audio hashes. Recordings are not bundled.
- [nar-optimization.json](nar-optimization.json): precise MPP acoustic-kernel measurements, counterbalanced old/new timings, unchanged 64-/400-frame source and FP32-anchor gates, input/code/report hashes and full-song comparison caveats.
- [full-song-reference.json](full-song-reference.json): completed 4493-frame BF16 source and FP32 anchor, same-runtime streaming proof, independently derived bounds, full intermediate comparison and retained failures. The full-song source comparison passes; one FP32-anchor cache bound fails and is retained as an explicitly accepted listening/performance tradeoff.
- [corpus.json](corpus.json), [score.abc](score.abc) and [melody.abc](melody.abc): fixed requests and supplied-score fixtures.
- The `*-limits.json` files retain reference-derived numerical bounds and calibration provenance. AR files containing `torch211` name the corrected reference environment; earlier AR files are historical, not interchangeable acceptance limits.
- [integrity-smoke.json](integrity-smoke.json): retained rejection outcomes for invalid converted artifacts.

Large raw tensor captures, audio, environments, model weights and detailed machine logs are deliberately not included. The curated reports summarize retained local evidence, not a downloadable reproduction of every original capture. Generate new captures with the tools below; input/calibration identities must match before applying numerical limits.

## What has been exercised

- All five real-checkpoint generation paths: generated/supplied full score, generated/supplied melody score, and no score. BF16 and experimental 8-bit/4-bit AR paths have rendered; this is execution coverage, not equal quality acceptance.
- Default 32-step midpoint acoustic synthesis, retaining all 64 velocity evaluations and full attention visibility. The retained 64-/400-frame NAR checks pass their recorded historical reference-derived bounds; the corrected-runtime full-song source and anchor comparison is complete.
- Default FP32 MPS VAE output length, halo/crop boundaries and retained waveform checks. Decoder execution was audited for unintended CPU fallback.
- Offline loading, model/conversion identities, saved plans, artifact replay, cancellation/resource guards, export and request-local reproducibility.
- Fresh BF16 conversion and verified reuse. A retained warm-filesystem local run measured 6.87 seconds for conversion and 4.99 seconds for verified reuse; these exclude downloads and are not cold-install timing promises.

### Precise acoustic attention optimization

On the 32 GB M5 MacBook Air, on AC power, the frozen old MLX attention and the optimized kernel were run in two sequential paired processes with reversed implementation order. Each pair shared loaded BF16 weights and exact saved prefix, semantics and initial noise. Every run performed its own conditioning prefill and all **32 midpoint steps / 64 velocity evaluations**. Stage times include prefill plus solve, excluding model loading, one warmup velocity per run and decoding.

| Input | Old mean NAR stage | Optimized mean NAR stage | Time reduction |
|---|---:|---:|---:|
| 64 frames | 3.25 s | 3.15 s | 3.1% |
| 400 frames | 11.51 s | 10.57 s | 8.2% |
| 4493 frames / 179.72 s audio | 524.18 s | 453.11 s | 13.6% |

The raw full-song pairs were **519.87 → 504.65 s** with the old implementation first, and **528.49 → 401.58 s** with the optimized implementation first. Step times changed substantially with run position. Reversing order reduces that bias but does not control GPU temperature/clocks; two runs per implementation are not a confidence interval, a new three-run end-to-end campaign or a PyTorch/MPS speedup.

Both implementations reproduced their own full-song latent arrays bit-for-bit across runs. The optimized output differs from the old output: full-song latent RMS error **0.02325**, maximum **2.17383**, with the largest differences near **56 seconds**. The complete full-song source comparison is now complete: source bounds pass, while one FP32-anchor cache bound fails and remains recorded. Full-song reference and current-kernel generated audio have received listening sign-off.

Both **64- and 400-frame** exact-input checks pass all **82 tensors** against the source and independent FP32 anchor, with the existing limits unchanged. An earlier 32-key kernel failed an intermediate 400-frame velocity bound and was rejected; retaining the original 16-key reduction width and pairwise summation order resolved that failure. BF16 state/cache, FP32 softmax/accumulation, safe compiler math, TF32-off execution, full key visibility and original acoustic chunks remain intact.

Across both paired timing processes, sampled peak process footprint was **8.72 GiB**, within the 16 GiB guard, with **zero additional system swap-outs**. Historical swap and small swap-ins remained present. These acoustic-only resource figures are not comparable to the end-to-end decoder-inclusive peaks below. Detailed identities, raw timings and numerical/resource provenance are retained in [nar-optimization.json](nar-optimization.json).

### Full-song acoustic source capture

The exact saved 4493-frame prefix, semantics and CPU FP32 noise have now been replayed through the upstream PyTorch 2.11 BF16 acoustic implementation. Expert-staged loading retained all 32 midpoint steps and produced 76 reference arrays, including all conditioning K/V and selected solver intermediates. Capture wall time was **1008.85 seconds**, with **8.54 GiB** sampled peak footprint and no additional swap-outs. This includes loading and tensor capture, so it is not a matched speedup baseline.

Independent full-song bounds now use only the BF16 source and FP32 anchor, not port outputs. The production MLX path passes all **82 source comparisons**, but fails one FP32-anchor comparison: conditioning cache **`k_9` maximum error 0.38304 exceeds 0.27740**. Final latents pass both comparisons; their source-relative RMS error is **0.02027**, maximum **1.44678**. **The raw full-song numerical gate remains failed**; the localized discrepancy is explicitly retained and accepted after listening review, not hidden by the passing final latents.

FP32 calibration streams one layer's weights at a time during both upstream AR conditioning and every NAR velocity. On the same PyTorch 2.11 runtime, the 64-frame fully resident control and fully streamed capture are **bitwise identical across all 84 arrays**; sampled peak footprint falls from **9.49 to 4.46 GiB**. The complete fully streamed full-song anchor produced **84 arrays in 1738.19 seconds**, with **11.50 GiB** sampled peak footprint and no additional swap-outs. Earlier attempts stopped on AC disconnection and system memory pressure; neither is counted as completed calibration. Failures, identities and measurements are retained in [full-song-reference.json](full-song-reference.json).

The complete MLX comparison took **1026.14 seconds**, including model loading, all conditioning caches, eight fixed-state velocities, the 64-evaluation trajectory replay and a separate 64-evaluation production solve. Peak sampled footprint was **10.51 GiB**, with no additional swap-outs. These capture-heavy timings are not a matched acoustic speed benchmark.

A precise causal MPP experiment was also rejected: it fixes several AR comparisons but fails others; in NAR-only conditioning it clears `k_9` but fails the `k_27` FP32-anchor bound (**1.69901 > 1.37528**). Neither experimental causal path is in production. No numerical limits were loosened.

### Historical sustained BF16 rendering

The tested machine was an M5 MacBook Air with 32 GB unified memory, on AC power. The same complete-song request was rendered three times sequentially in one process:

| Run | Audio duration | Generation wall time |
|---|---:|---:|
| 1 | 179.718667 s | 818.90 s |
| 2 | 179.718667 s | 897.01 s |
| 3 | 179.718667 s | 903.13 s |

All three naturally sampled their end tokens; neither ABC nor semantics were truncated. Sampled peak process footprint was **12.63 GiB**, with **zero additional system swap-outs**. Historical swap remained present and small swap-ins occurred: this is not a claim that swap space was empty or that no swap I/O happened. The process lifetime peak is a distinct counter and is recorded separately in [mvp-results.json](mvp-results.json).

The retained harness requires at least 180 seconds. These approximately three-minute songs **fail that strict duration gate by 0.281 seconds**. No padding, truncation or changed cutoff was used to mark the gate passed. Full-song listening review is complete; the duration gate remains failed.

A separate `tonight_awake_second_verse` case adds exactly one four-line verse before the bridge, retaining the original style, seed and all other lyrics. It was declared before generating any output for that case. It is reserved for the next current-kernel sustained run, not a replacement for the historical failure. The strict 180–240 second gate and normal generation budgets remain unchanged; its generated duration is not yet known.

### Human listening

The maintainer manually reviewed a **15.998667-second** reference/port pair and reported correct, perceptually identical sound. That comparison reused identical prefix, ABC, semantic IDs and initial noise, isolating NAR plus the default decoder. The semantic sequence was intentionally capped. This is not a bit-identical waveform claim and does not cover independently sampled AR output, complete-song continuity or quantized quality.

This assessment predates the precise MPP acoustic kernel. Subsequent full-song reference and current-kernel generated audio received complete listening sign-off. The sign-off does not claim bit-identical waveforms or waive the retained FP32-anchor cache failure.

## Open acceptance gates

1. **Four strict AR comparisons remain outside current bounds:** a negative stress-cache case, long stress logits, a saved-song negative-cache case and a long valid-score-prefix cache case. The largest discrepancies were already present in prefill; the checked real-song logit argmax values agreed. This does not establish full-trajectory equivalence or an audible regression.
2. A native upstream CPU BF16 control also exceeded some MPS-derived bounds. That demonstrates that the empirical bounds are not universal across backends; it does **not** turn the MLX failures into passes. Bounds and production arithmetic remain unchanged.
3. Complete full-length matched teacher-forced AR comparisons and strict AR validation remain pending. Full-song acoustic comparison is complete with one explicitly accepted FP32-anchor failure; complete-song listening is signed off.
4. Matched BF16, then 8-bit, then 4-bit listening is incomplete. BF16 is the conservative MVP runtime default, not the result of a completed quality-selection gate.
5. Hardware configurations beyond the 32 GB M5 Air have not been validated. No full-song, matched PyTorch/MPS-versus-MLX speedup is claimed.

## Reproduction tools

Run from the repository root after the [quick start](../README.md#quick-start). Use new output directories and one GPU workload at a time. AC enforcement is optional: add `--require-ac` when a run must stop on disconnection. The current validation campaign permits battery operation at the maintainer's request; start/end power state remains recorded. Do not increase memory limits or shorten attention/solver work to hide a failed acceptance run, and do not compare timings as matched measurements across uncontrolled power conditions.

Ordinary regression checks do not require model downloads:

```bash
uv run pytest -q
uv run ruff check src tests tools
```

The mode workload exercises bounded clips; `full` and `sustain` use full-song generation and can take many minutes. Their strict duration failure is intentionally retained:

```bash
uv run python tools/acceptance.py modes \
  --model models/converted --vae "$LYRA_VAE" \
  --precision bf16 --output outputs/acceptance-modes
```

To select the predeclared extended case for three sequential renders in one pipeline process:

```bash
uv run python tools/acceptance.py sustain \
  --case tonight_awake_second_verse \
  --model models/converted --vae "$LYRA_VAE" \
  --precision bf16 \
  --output outputs/acceptance-sustain-second-verse
```

`--case` is supported only for `full` and `sustain`; omitting it preserves `tonight_awake`. The `modes` workload retains its fixed fixtures.

The reference is isolated because its dependencies differ from production:

```bash
uv venv --python 3.12 .oracle
uv pip sync --python .oracle/bin/python --require-hashes oracle/requirements.txt
PYTHONPATH=src:vendor/yue/src .oracle/bin/python tools/oracle.py --help
```

Use **PyTorch 2.11.0** from [oracle/requirements.txt](../oracle/requirements.txt) for AR reference captures. PyTorch 2.10 has an MPS two-pass BF16/fp16 attention scratch-buffer defect ([upstream fix](https://github.com/pytorch/pytorch/pull/174945)). Production keeps PyTorch 2.10 for the separate FP32 decoder, not for AR/NAR inference.

Tool responsibilities:

- [oracle.py](../tools/oracle.py): bounded reference generation, teacher-forced AR traces (including saved `--source` requests), NAR and VAE captures. Explicit `nar --full-song` reuses every saved semantic frame, prefix and original noise from a naturally completed source. It rejects truncated sources, missing noise and songs requiring more than one original acoustic chunk. Default NAR fixtures remain bounded.
- [fidelity.py](../tools/fidelity.py): compare MLX with saved exact-input reference tensors. Explicit `nar --full-song` requires a full-song source capture. Without `--limits`, it records measurements, not acceptance.
- [calibrate.py](../tools/calibrate.py) and [limits.py](../tools/limits.py): calibration evidence and bound validation. NAR calibration supports explicit `--full-song` with one resident FP32 expert layer during both AR conditioning and NAR velocities. Do not reuse limits with mismatched input or reference identities, or derive them from partial captures.
- [acceptance.py](../tools/acceptance.py): fixed modes, selectable recorded full-song cases and sustained execution/resource reports.

Inspect each tool's `--help` for inputs and supported bounds. FP32 calibration stages weights rather than loading both complete generator experts concurrently. Equal numeric seeds in PyTorch and MLX do not guarantee equal sampled sequences; fidelity comparisons must reuse saved IDs and noise.

For full-song acoustic validation, set `LYRA_GENERATOR` to the pinned original generator checkpoint directory, not the converted MLX directory, and `LYRA_SOURCE` to a naturally completed saved song. Run these steps sequentially, stopping on any failure:

```bash
export PYTHONPATH=src:vendor/yue/src
export HF_HUB_OFFLINE=1 MLX_ENABLE_TF32=0
export PYTORCH_MPS_FAST_MATH=0 PYTORCH_ENABLE_MPS_FALLBACK=0

.oracle/bin/python tools/oracle.py nar --full-song \
  --model "$LYRA_GENERATOR" --vae "$LYRA_VAE" --source "$LYRA_SOURCE" \
  --memory-budget-gib 16 --output outputs/full-song-check/source

.oracle/bin/python tools/calibrate.py nar --full-song \
  --model "$LYRA_GENERATOR" --oracle outputs/full-song-check/source \
  --memory-budget-gib 16 --output outputs/full-song-check/anchor

uv run python tools/limits.py nar \
  --oracle outputs/full-song-check/source --calibration outputs/full-song-check/anchor \
  --output outputs/full-song-check/limits.json

uv run python tools/fidelity.py nar --full-song \
  --model models/converted --precision bf16 --query-chunk-size 256 \
  --oracle outputs/full-song-check/source --limits outputs/full-song-check/limits.json \
  --memory-budget-gib 16 --output outputs/full-song-check/actual

uv run python tools/limits.py nar \
  --oracle outputs/full-song-check/source --calibration outputs/full-song-check/anchor \
  --actual outputs/full-song-check/actual/nar-actual.npz \
  --limits outputs/full-song-check/limits.json \
  --output outputs/full-song-check/actual/precision-anchor.json
```

Use fresh output directories and never overwrite existing bounds to accommodate a port result. The fidelity run checks fixed-state velocities, accumulated solver states and a separate production solve, so it does more work than one ordinary acoustic render. The retained full-song source can be reused; an incomplete FP32 calibration must be recaptured successfully before continuing.
