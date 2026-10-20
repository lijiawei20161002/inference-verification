# BF16 E0 discovery on L40S, 20 October 2026

[Experiment map](../../experiments/README.md) ·
[Implementation plan](../../planning/NEXT_EXPERIMENTS.md) ·
[Original summary](numerical_e0_l40s_20261020/summary.json) ·
[Review statistics](numerical_e0_l40s_20261020/review.json)

The campaign reproduced **49/49 historical capped events** in the archived
64-sequence BF16 evaluation pool. All 49 remain provider/reference token
disagreements under the legacy nucleus convention when filtering arithmetic is
matched. With the separately named crossing-inclusive convention, **6/49**
remain disagreements. This is frozen-logit discovery evidence, not a causal
mechanism result or a fresh confirmatory evaluation.

The run finished at 10:40:33 UTC in 1,095.07 seconds. Its `complete` state means
the implemented campaign finished; **M1 is not fully complete**, because
kernel-dispatch equivalence was checked only on a four-token smoke sequence.

## Counts and controls

| Quantity | Result |
|---|---:|
| Evaluation sequences | 64 |
| Generated positions per sequence | 128 |
| Evaluation positions | 8,192 |
| Historical capped events reproduced | 49/49 |
| Current capped positions | 49, across 32 sequences |
| Preselected control positions | 64 |
| Captured events including controls | 113 |
| Replay cells per event | 30 |
| Total recomputed replay cells | 3,390 |

Baseline and capture each ran twice in independent Python processes. Seven
comparisons, including smoke/profiler controls, passed. Full-pool claimed tokens
and hashes of provider/reference logits, reference hidden states, fingerprints
and scores matched across the corresponding workers. Actual RNG tensors,
support masks, native BF16 logits and KV states were retained at the 113
selected positions; they were not saved at every evaluation position.

All 49 capped positions have the claimed token excluded from the reference
support under the original protocol. Here the cap therefore reflects exclusion,
not a finite measured numerical margin. One of the 64 non-capped controls also
has a token disagreement under the matched legacy protocol; the other 63 have
`no_default_disagreement`. These controls are selected positions, not an
unbiased estimate of disagreement frequency.

## Frozen-logit replay

The same paired logits and actual shared Gumbel vectors were replayed under
three arithmetic settings, two nucleus conventions and five sampling modes.
`original` means float32 provider filtering and float64 reference filtering;
the other settings use matching float32 or matching float64 on both sides.

The following counts apply to the 49 historical capped events. Each count is
identical across all three arithmetic settings. Entries count different
provider/reference selected tokens, not capped scores or sequence rejections.

| Sampling mode | Legacy nucleus | Crossing-inclusive nucleus |
|---|---:|---:|
| Greedy, no filtering | 8/49 | 8/49 |
| Shared Gumbel, no filtering | 4/49 | 4/49 |
| Top-k only | 4/49 | 4/49 |
| Top-p only | 21/49 | 4/49 |
| Top-k followed by top-p | 49/49 | 6/49 |

Nucleus convention has no effect in modes without top-p. The inclusive variant
retains through the first strictly crossing token; it is an experimental
protocol and does not change production sampling or reuse calibrated detector
thresholds. Each cell recomputes the provider decision and compares the
reference decision on the identical prefix. These are not free-generation
trajectories under the alternative protocols.

The label `residual_under_matched_protocol` refers to matched arithmetic with
the **legacy** top-k/top-p protocol. It does not mean disagreement survives every
tested convention. The 43 agreements under the inclusive variant do not by
themselves prove protocol-only causes: changing the convention also changes the
decision boundary acting on unequal logits. No operation-level intervention or
hardware-fault attribution has been established.

## Review and reproducibility

The review checked the campaign artifact hashes, frozen source inventory,
all **742 worker artifact hashes across eight workers**, and worker source
snapshots. It compared the complete sequence records, recomputed the smoke
kernel sequence (9,441 kernels on each side, equal names/grid/block), and
recomputed all 113 events from their NPZ evidence. All 3,390 saved replay cells
matched exactly. The focused CPU suite passed **51 tests**. The CI NumPy-core
command passed **210 tests**, with one expected small-pool ratio warning; it
excludes the service and signed-audit test modules, as in that CI job.

Runtime sources started from commit `d31d2a85b20516766b8fe5ab9e66036d1b782d2b`
plus the then-uncommitted capture, replay, campaign and test modules. Those five
files match the frozen runtime snapshot byte for byte; their SHA-256 values
are in `review.json`. Documentation was updated after the run.

Environment: NVIDIA L40S, driver 595.91.07, CUDA runtime 12.8, Python 3.12.3,
PyTorch 2.8.0+cu128, Transformers 4.57.6, NumPy 2.1.2, Accelerate 1.15.0 and
bitsandbytes 0.50.2. Model: `Qwen/Qwen3-1.7B`, revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Matmul TF32 was disabled, cuDNN TF32
enabled, and PyTorch/OMP/OpenBLAS/MKL threads set to one. The preserved sampler
uses temperature 1, top-k 50 and top-p 0.95. Synthetic noise was disabled.

Run the focused checks from the repository root:

```sh
.venv/bin/python -m pytest tests/test_frozen_logits.py tests/test_numerical_diagnostics.py tests/test_real_quantization.py tests/test_e0_campaign.py -q
```

The [numerical tools README](../../experiments/numerical/README.md) describes
launching a new campaign. Use a new output directory and the pinned model
cache; do not overwrite this run. `replay_event` in `frozen_logits.py` validates
and recomputes an individual saved event without loading the model.

## Evidence retention and remaining gate

The full run remains in `runs/numerical-e0-20261020/` (4,764,868,231 bytes,
approximately 4.44 GiB, excluding the external model cache). Git contains only
the reviewed report, the unchanged original summary, aggregate review
statistics and [original artifact hashes](numerical_e0_l40s_20261020/run_artifacts_sha256.json).
Raw tensors, traces and source snapshots are not included in this commit;
retain the original directory before releasing the host. A Git clone alone
cannot independently replay these captured events.

The original summary's `m1_status` uses the phrase `full-pool`; its full-pool
coverage applies to output equivalence. Actual RNG/support evidence covers
the selected events and controls as described above. Kernel tracing covers
only the short smoke sequence. The original files and hashes are preserved.

Next, extend kernel-dispatch equivalence validation to the actual event shapes.
Only after that M1 gate passes should M3 test matched-input GEMM/reduction and
attention/reduction cases with state-consistent interventions. This discovery
corpus supplies no deployment error rate, causal witness, contract violation,
or evidence of an attack.
