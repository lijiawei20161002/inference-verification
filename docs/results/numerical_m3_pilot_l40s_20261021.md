# M3 controlled reduction-order pilot on L40S, 21 October 2026

[Experiment map](../../experiments/README.md) ·
[Implementation plan](../../planning/NEXT_EXPERIMENTS.md) ·
[Original summary](numerical_m3_pilot_l40s_20261021/summary.json) ·
[Review statistics](numerical_m3_pilot_l40s_20261021/review.json) ·
[Per-case results](numerical_m3_pilot_l40s_20261021/cases.json)

**The 64-case pilot completed and its retained evidence passed review. The full
M3 mechanism gate remains pending.** Reversing the declared FP32 partial-sum
order changed GEMM local BF16 outputs in 7/32 cases and propagated to selected
logits in 2/32. No sampled token changed in either family. Attention had no
A/B output differences; its negative result has limited partition coverage.
These are historical discovery inputs and explicit diagnostic implementations,
not an attribution of native cuBLAS or attention behavior.

## Protocol and results

The campaign ran from 02:16:02 to 02:24:58 UTC on 21 October, taking 536.13
seconds. Each family has 16 historical-event and 16 control cases, frozen before
interventions and assigned across layers 0, 14 and 27. Three fresh processes
reconstructed the model inputs; a fourth loaded saved operands and reproduced
all 64 records. The earlier four-case preflight also passed.

GEMM replaces one MLP down-projection output row using K partitions of 64.
Attention replaces one output row using KV partitions of 32 and a common
maximum to merge partial softmax statistics. A sums the same FP32 partials in
ascending order; B uses descending order. Both cast once to BF16 at the end.
Every model forward starts from the full original token checkpoint with empty
KV and recomputes descendants. This is fixed-prefix replay, not free generation.

| Cases with a change between A and B | GEMM (32) | Attention (32) |
|---|---:|---:|
| Local BF16 output | 7 | 0 |
| Full-sequence logits hash | 2 | 0 |
| Saved logits at the selected position | 2 | 0 |
| Primary top-k/top-p support | 0 | 0 |
| Primary sampled token | 0 | 0 |
| Support in any auxiliary sampling cell | 2 | 0 |
| Sampled token in any sampling cell | 0 | 0 |

The primary sampler uses matched float32 arithmetic, legacy top-k 50 followed
by top-p 0.95, and the actual saved Gumbel vector. The matrix has 20 cells per
case: two arithmetic dtypes, two nucleus conventions and five sampling modes.

The two propagated cases are `gemm_p74_t13_l14` and `gemm_p90_t121_l0`. Each
changed one of 2,048 local BF16 elements, by 0.0078125 and 0.00390625 respectively.
Their top-k-only support symmetric differences are 1 and 3 across both dtypes
and convention labels. No token changed, and subsequent top-p filtering removed
the support differences in the primary configuration. **The original summary's
`any_sampling_cell_changed` field counts token changes only.** Its zero does not
mean all auxiliary support masks were unchanged. The original report and
summary are preserved byte for byte; this review supplies that clarification.

## Controls and scientific limits

Native selected logits and full KV states matched the parent
[M1 capture](numerical_m1_dispatch_l40s_20261021.md). Capture-only execution,
both shams, forward A-to-B replacement, reverse B-to-A replacement, unchanged
preceding logits and ancestor KV checks passed. Worker records, selected arrays,
and ordered kernel-name/grid/block signatures matched across all four processes.

Both A and B differed from the native local output in **all 32 cases of each
family**. Consequently, A/B equality or a successful replacement control does
not establish a bridge to the historical native provider/reference discrepancy.
Full-model output and hidden-state hashes are retained; raw full output tensors
are not part of this M3 capture. Review compared the recorded full-state hashes
and independently inspected the saved selected logits and parent KV tensors.

Attention had 18 cases with one unmasked partition, 1 with two, 3 with three,
6 with four, and 4 with five. Allowed key counts ranged from 18 to 147. Reversing
the merge order of a single contributing partition cannot test a multi-partition
effect. The 14 cases with multiple contributing partitions also had no final
BF16 A/B difference; this does not rule out FP32 changes erased by the final
cast or effects on longer contexts. A follow-up needs meaningful active-partition
coverage and native execution evidence before advancing the full M3 gate.

There is no historical mechanism attribution, deployment rate estimate, fault
finding or attack claim. Cases share prompts and technical repeats are not new
independent scientific samples. No fresh GPU measurement was made during review.

## Review and reproduction

The offline review checked all 5 campaign artifact hashes, 342 frozen source
files and 332 matching current Python files; 12 parent campaign artifacts and
357 parent capture artifacts; and all **2,052 worker artifact hashes**. It
recomputed signatures from **512 raw traces / 81,664 kernel launches**, local
BF16 difference metrics, native bridges, selected input/noise/logit equivalence,
parent KV continuity, and **1,280 unique sampling cells**. Repeated saved arrays
were identical, and every worker's corresponding sampling records matched.
The reviewed counts agree with the original summary.

The CI NumPy-core command passed **227 tests in 108.22 seconds**, with one
expected small-pool ratio warning. Service and signed-audit tests were excluded,
matching that CI job:

```sh
.venv/bin/python -m pytest tests/ -q \
  --ignore=tests/test_audit_protocol.py \
  --ignore=tests/test_audit_conformance.py \
  --ignore=tests/test_service.py
```

With the retained run, parent M1 capture and preflight directories available,
run the archived offline reviewer from the repository root:

```sh
.venv/bin/python docs/results/numerical_m3_pilot_l40s_20261021/review_retained_run.py
```

It writes `/tmp/m3_pilot_review.json` and `/tmp/m3_pilot_cases.json`. The historical
plan contains absolute parent paths; restore those paths when moving evidence.
Its current-source equality check requires the original runtime Python sources.
New GPU runs use the [runner and preflight instructions](../../experiments/numerical/README.md#detached-m3-controlled-reduction-order-pilot)
with new output directories and the pinned environment.

Environment: NVIDIA L40S, driver 595.91.07, CUDA runtime 12.8, Python 3.12.3,
PyTorch 2.8.0+cu128, Transformers 4.57.6, NumPy 2.1.2, Accelerate 1.15.0 and
bitsandbytes 0.50.2. Model: `Qwen/Qwen3-1.7B`, revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Matmul TF32 was disabled, cuDNN TF32
enabled, and computation threads set to one. Peak allocated CUDA memory was
3,639,166,976 bytes (3.39 GiB). Runtime code began at `b3503e5` plus the then
uncommitted M3 files; reviewed experimental Python sources match the frozen run.

## Retained evidence

The complete local run at `runs/numerical-m3-20261021/` contains 2,456 files and
4,726,098,935 bytes (4.40 GiB), excluding the model cache, preflight and parent
runs. Git contains the reviewed report, unchanged original report/summary and
campaign hashes, derived review/per-case statistics, and the offline reviewer.
Raw operands, logits, traces and frozen source snapshots remain local. A Git
clone alone cannot recompute the raw evidence checks; preserve these directories
before releasing the host.
