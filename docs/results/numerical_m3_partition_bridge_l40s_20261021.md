# M3 partition coverage and native bridges on L40S, 21 October 2026

[Experiment map](../../experiments/README.md) ·
[Implementation plan](../../planning/NEXT_EXPERIMENTS.md) ·
[Original summary](numerical_m3_partition_bridge_l40s_20261021/summary.json) ·
[Review statistics](numerical_m3_partition_bridge_l40s_20261021/review.json) ·
[Per-case results](numerical_m3_partition_bridge_l40s_20261021/cases.json)

**The 64-case partition/bridge follow-up completed and its retained evidence
passed offline review. The full M3 mechanism gate remains pending.** Reversing
the same FP32 partial sums changed pre-cast outputs in every case. Final BF16
outputs changed in 7/32 GEMM and 2/32 attention cases, propagating to selected
logits in 2 and 1 cases respectively. The primary sampler had no support or
token changes. One attention case changed its legacy top-p-only sampled token.
Full-shape native operator replay, native sham and restoration controls passed.

## Protocol and results

The campaign ran from 08:44:54 to 08:58:18 UTC on 21 October, taking 804.14
seconds. It retained the [first pilot's](numerical_m3_pilot_l40s_20261021.md)
deterministic 32 cases per family: 16 historical-event and 16 control cases,
assigned across layers 0, 14 and 27. Three fresh processes and a fourth
saved-operand replay process produced identical records and selected arrays.
The separate four-case partition/bridge preflight also completed successfully.

GEMM uses K partitions of 64. Attention now uses KV partitions of 8, with
3–19 active partitions per selected query; allowed key counts are 18–147.
The same-input chunk-32 local control remains A/B-equal in all 32 attention
cases. Changing chunk size changes A's local output in 2 cases and B's in 3.
This improves active-partition coverage on the historical inputs; it does not
add new scientific samples or long-context evidence.

A sums identical FP32 partials in ascending order and B in descending order,
then each casts once to BF16. The saved evidence now includes FP32 partials and
pre-cast outputs. Every intervention changes one operation output row and
recomputes descendants from the original full token checkpoint with empty KV.
These are fixed-prefix replays, not new generated trajectories.

| Cases with a change between A and B | GEMM (32) | Attention (32) |
|---|---:|---:|
| FP32 pre-cast output | 32 | 32 |
| Local BF16 output | 7 | 2 |
| Full-sequence logits hash | 2 | 1 |
| Saved logits at the selected position | 2 | 1 |
| Primary top-k/top-p support | 0 | 0 |
| Primary sampled token | 0 | 0 |
| Support in any sampling cell | 2 | 1 |
| Sampled token in any sampling cell | 0 | 1 |

The primary sampler uses float32 arithmetic, legacy top-k 50 followed by top-p
0.95, and the actual saved Gumbel vector. Each comparison has 20 sampling cells:
two arithmetic dtypes, two nucleus conventions and five sampling modes.
The legacy summary field `any_sampling_cell_changed` counts cases with a
token change only; explicit support counts appear under `diagnostics`.

In `attention_p60_t102_l27`, one of 2,048 local BF16 elements changes by
0.03125. The legacy top-p-only cell changes token 11 to 323 and has a support
symmetric difference of 2, in both float32 and float64. This is one case and
two matrix cells, not two independent discoveries. The primary top-k/top-p
and crossing-inclusive configurations do not change. The other local case,
`attention_p50_t120_l27`, changes one BF16 element by approximately 1.12e-8
without changing selected logits. GEMM's propagated cases remain
`gemm_p74_t13_l14` and `gemm_p90_t121_l0`, with auxiliary support changes and
no sampled-token changes.

## Native bridges and limits

Full-shape BF16 linear operations and the pinned Qwen eager attention
arithmetic are reconstructed from saved operands, including tensor strides.
Their full outputs match the captured native operator outputs exactly in all
64 cases. Native-output sham and restoration after computing recipe A reproduce
the native full-model output, hidden-state and KV hashes. Capture-only,
A/B shams, forward/reverse swaps and ancestor checks also pass. Selected native
logits and KV hashes match the parent M1 capture.

Native-to-recipe effects are a separate contrast. A and B each differ from
native local outputs in all 32 cases of both families:

| Cases with a change from native | GEMM → A | GEMM → B | Attention → A | Attention → B |
|---|---:|---:|---:|---:|
| Primary support | 21 | 21 | 20 | 20 |
| Primary token | 6 | 6 | 8 | 8 |
| Support in any cell | 27 | 28 | 28 | 29 |
| Token in any cell | 9 | 9 | 12 | 13 |

These effects include differences between the native arithmetic and the FP32
diagnostic recipe. They cannot be assigned to the A/B reduction order alone.
Exact native-output reconstruction does not identify undocumented cuBLAS
split-K or native attention merge policy, or show that either recipe explains
the historical provider/reference discrepancy. The full M3 scientific gate
therefore remains pending. No fault, attack or deployment-rate claim follows.
Cases share prompts, and technical repeats are not independent scientific
samples. Native operator output tensors are retained; full-model logits and
hidden states are represented by hashes plus saved selected logits.

## Review and reproduction

The offline review checked 5 campaign artifact hashes, all 345 frozen source
files and 335 matching current Python files, 12 parent campaign artifacts,
357 parent capture artifacts, and **4,100 worker artifact hashes**. All 256
case executions passed the record, tensor and control checks. It independently
recomputed signatures from **768 raw traces / 168,320 kernel launches**,
**1,280 A/B sampling cells** and **2,560 native-to-recipe sampling cells**.
The recomputed family counts agree exactly with the original summary/report.

For every case execution, review checks identical FP32 partials, saved ordered
sums, pre-cast difference metrics and exact BF16 casts, plus full native output
equality. It verifies all 128 attention chunk-32 control pairs and active
partition coverage. Repeated selected logits, noise and input IDs are identical.
Full-model state hashes are compared across recorded controls/repeats; the
review does not rerun the GPU model, native operators or attention division.

The CI NumPy-core regression command passed **233 tests in 110.81 seconds**,
with one expected small-pool ratio warning. The campaign's focused checks
also passed 74 tests before the GPU workers ran. Service and signed-audit
tests were excluded, matching the NumPy-core CI job:

```sh
.venv/bin/python -m pytest tests/ -q \
  --ignore=tests/test_audit_protocol.py \
  --ignore=tests/test_audit_conformance.py \
  --ignore=tests/test_service.py
```

With the retained run, its preflight and parent M1 capture available, run from
the repository root in the pinned Python environment (PyTorch is used for CPU
BF16 casts; no GPU is required):

```sh
.venv/bin/python docs/results/numerical_m3_partition_bridge_l40s_20261021/review_retained_run.py
```

Do not use Python's `-O` option: assertions are the audit checks. The script
writes `/tmp/m3_partition_bridge_review.json` and
`/tmp/m3_partition_bridge_cases.json`. The plan retains absolute parent and
preflight paths; restore them when moving evidence. Current-source matching
requires the frozen run's Python sources. New GPU campaigns must use fresh
output directories and repeat the [preflight procedure](../../experiments/numerical/README.md#second-m3-pilot-partition-coverage-and-native-bridges).

Environment: NVIDIA L40S, driver 595.91.07, CUDA runtime 12.8, Python 3.12.3,
PyTorch 2.8.0+cu128, Transformers 4.57.6, NumPy 2.1.2, Accelerate 1.15.0 and
bitsandbytes 0.50.2. Model: `Qwen/Qwen3-1.7B`, revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Matmul TF32 was disabled, cuDNN TF32
enabled, and computation threads set to one. Peak allocated CUDA memory was
3,642,030,080 bytes (3.39 GiB).

## Retained evidence

The complete local run at `runs/numerical-m3-partition-bridge-20261021/`
contains 4,509 files and 6,065,204,176 bytes (5.65 GiB), excluding parent runs,
preflight and model cache. Git contains this reviewed report, unchanged
original report/summary and campaign hashes, derived review/per-case statistics,
and the offline reviewer. Raw operands, FP32 intermediates, logits, native
outputs, traces and source snapshots remain in the ignored local run directory.
A Git clone alone cannot reproduce the raw evidence checks; retain the run,
preflight and parent capture before releasing the host.
