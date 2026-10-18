# Paired H100-to-A100 replay, 18 October 2026

[Experiment map](../../experiments/README.md) · [Archived evidence](cross_gpu_replay_20261018/) ·
[Verified summary](cross_gpu_replay_20261018/summary.json)

**Completed the first paired replay of the same frozen capture on H100 and A100.**
The A100 consumed the exact H100 prompt and output token IDs: 80 prompts × 256
positions, four schedules and four detectors. This fills the missing paired
comparison for this H100-origin capture. The original A100-origin capture remains
a separate, unfinished reverse-direction experiment.

For batch-1 eager replay, all three token-detector 90% intervals include zero.
Honest differences depend on the schedule: token-DiFR rises by **+0.0141 d′**
under batch 4 and **+0.0460 d′** under sequential replay, with per-comparison
intervals above zero. Sequential token-toploc also rises by **+0.0336 d′**.
These are measured positive honest shifts; they do not determine a deployment
false-positive rate or validate a quantization detector.

## Measurement and controls

The H100 baseline is the [18 October frozen replay](h100_followup_20261018.md).
The new run used NVIDIA A100-SXM4-80GB, driver 580.126.16; the H100
used driver 595.91.07. Hardware and driver both change. Both used
runtime commit `5b5667c60f478403a4132c2c9744c6b1ef21988b` and identical hashes for all
39 scoring source files. Python 3.12.3, PyTorch 2.8.0+cu128, CUDA runtime 12.8,
Transformers 4.57.6, NumPy 2.1.2, Accelerate 1.15.0, Hugging Face Hub 0.36.2,
Tokenizers 0.22.2 and Safetensors 0.8.0 match the archived environment. Matmul
TF32 was disabled, cuDNN TF32 enabled, and PyTorch/OMP/OpenBLAS threads set to one.

Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, BF16, offline model loading. The original
capture used eager sequential generation, temperature 1, top-k 50, top-p 0.95,
shared seed 42, prompt length capped at 32 tokens, and no injected logit or
activation noise. A100 replay did not regenerate or retokenize any response.
The archived model-file inventory records hashes of all 12 downloaded files.

| Check | Result |
|---|---|
| Full capture | Identical SHA-256 on H100 and A100; 20,480 positions |
| Runtime inputs | All 39 source hashes and recorded scoring conditions match |
| A100 smoke | First 4 frozen sequences × first 16 positions; all four exact-rerun controls pass |
| Full same-shape control | Raw baseline and rerun arrays match exactly for all four detectors on both GPUs |
| A100 replay process time | 22.68 minutes, including loading and scoring |

The smoke is a documented prefix of the full capture. It checks execution and
same-shape repeatability; it is not an independent capture or a second full
cross-GPU trial. Process time is a resource record, not a hardware speed benchmark.

The pre-experiment suite passed **186 tests**. The final suite passed
**198 tests**, including twelve new checks for pairing, artifact integrity,
bootstrap reproduction and zero-variance interpretation. The two existing
warnings concern a service dependency deprecation and an intentional batch/pool
ceiling test. Both test logs are included in the archive.

## Paired score differences

**Baseline H100, alternate A100.** Positive d′ means A100 assigns higher scores
to the same returned tokens. Batch-1 eager (`chunked_b1`) is the main comparison.
The other schedules are sensitivity checks. Each row compares the same schedule
across GPUs.
Scores use the H100 row's 99.9th-percentile winsorization cap and standard
deviation. Intervals are exploratory 90% paired **sequence** bootstrap intervals,
400 draws, seed 0; token positions are not independent resampling units.
The intervals are not simultaneous bounds.

| Replay schedule | Detector | d′ | 90% paired interval |
|---|---|---:|---|
| chunked_b1 | token_difr | +0.005909 | [-0.006403, +0.016629] |
| chunked_b1 | cross_entropy | -0.000242 | [-0.000724, +0.000251] |
| chunked_b1 | token_toploc | +0.004244 | [-0.005855, +0.012857] |
| chunked_b1 | activation_difr | +0.064020 | [+0.049546, +0.084018] |
| chunked_b4 | token_difr | +0.014068 | [+0.001851, +0.025704] |
| chunked_b4 | cross_entropy | +0.000064 | [-0.000690, +0.000903] |
| chunked_b4 | token_toploc | +0.009986 | [+0.000165, +0.018934] |
| chunked_b4 | activation_difr | +0.084870 | [+0.062008, +0.108246] |
| chunked_b1_sdpa | token_difr | -0.001516 | [-0.011674, +0.009257] |
| chunked_b1_sdpa | cross_entropy | -0.000264 | [-0.000662, +0.000152] |
| chunked_b1_sdpa | token_toploc | -0.001688 | [-0.010606, +0.006916] |
| chunked_b1_sdpa | activation_difr | +0.038527 | [+0.027980, +0.051819] |
| sequential | token_difr | +0.045962 | [+0.028359, +0.064139] |
| sequential | cross_entropy | +0.001367 | [+0.000730, +0.002131] |
| sequential | token_toploc | +0.033624 | [+0.021537, +0.045877] |
| sequential | activation_difr | undefined | zero baseline variance |

`activation_difr` requires internal projected activations. It is an additional
visibility control, not a public-token detector. `token_difr` requires the shared
RNG contract; cross-entropy and token-toploc use tokens plus reference recomputation.
Changed-score counts refer to scores; the returned token IDs remain fixed.

## Zero-variance activation control

Every H100 sequential activation score is zero. Its baseline cap and standard
deviation are therefore zero. The unchanged legacy comparison clips every
positive A100 activation score to zero and reports d′ = 0 with [0, 0]; that is
**not evidence of cross-GPU agreement**. The verified summary marks d′ and its
interval undefined for this row and retains the original statistic in
`legacy_statistic` for auditability.

Before clipping, the H100 mean is 0.000000 and the
A100 mean is 0.251817; the mean absolute difference
is 0.251817. Scores differ at
20,480 of 20,480 positions. These are descriptive
raw-score differences, not a standardized effect or a calibrated verdict.

## Scope and remaining work

This is one model, one built-in prompt pool and seed, one H100-origin capture,
and one pair of GPU/driver configurations. There is no calibration/test split,
attack condition, or fixed decision threshold in this replay. It measures honest
numerical shifts, not deployment false-positive rates or attack-detection power.
The historical simulated-quantization effects used a noisy reference arm; their
d′ values are not a matched attack control for this clean capture.

Further work needs independent prompt/seed blocks, additional hardware and
driver configurations, and matched real-quantization attacks with held-out
calibration. The A100-origin capture replay on H100 and L40S replication remain
separate experiments.

## Reproduction and evidence integrity

Frozen capture SHA-256:

```text
53985023de9acea63b76eaaa1c2183f4cc94917b95e49bf59597f6a6881e5bba
```

The capture and H100 scores remain in the tracked
[H100 transfer bundle](h100_followup_20261018/h100-replay-transfer.tar.gz).
The new archive includes A100 arrays, raw rerun controls, the smoke capture,
manifests, package and NVIDIA snapshots, model-file hashes, exact command queue,
logs, the runner, and an artifact hash inventory. It references the original
H100 arrays instead of duplicating them. No model weights or credentials are
included.

From the repository root, rebuild or verify without CUDA, model downloads, or
network access:

```sh
python -m experiments.analyze_cross_gpu_replay
python -m experiments.analyze_cross_gpu_replay --check
```

The checker validates file hashes, full and smoke capture identity, score keys,
dimensions and finite values, actual rerun equality, paired scoring conditions,
queue completion, and all sixteen paired bootstrap results. It tolerates only
float64 roundoff in recomputed summaries. The saved legacy comparison is retained
unchanged, including the zero-variance diagnostic described above.

For a fresh GPU replay, follow the transfer bundle's `README.txt`: prepare a
separate checkout at the runtime commit, the matching packages and pinned model
cache, then run its `replay` and `compare` commands into fresh output paths.
The [exact queue](cross_gpu_replay_20261018/queue.json) records this run's commands;
the [runner](cross_gpu_replay_20261018/run_paired_replay.py) also records the smoke
derivation and environment settings. The runner was stored under
`runs/cross-gpu-20261018/` alongside its isolated `source/` checkout and extracted
`h100-replay-transfer/` bundle.
