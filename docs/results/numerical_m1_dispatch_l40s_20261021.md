# M1 event-shape dispatch validation on L40S, 21 October 2026

[Experiment map](../../experiments/README.md) ·
[Implementation plan](../../planning/NEXT_EXPERIMENTS.md) ·
[Original summary](numerical_m1_dispatch_l40s_20261021/summary.json) ·
[Review statistics](numerical_m1_dispatch_l40s_20261021/review.json)

**M1 passed for the historical evaluation corpus and the selected complete
forwards on the pinned single-GPU HF path.** Two independent profiled
baseline/capture pairs produced identical ordered kernel names, grid dimensions
and block dimensions. Full trajectories matched the unprofiled baseline/capture
controls and parent E0 outputs. Captured event arrays and KV states also matched
with and without profiling. All **49/49 historical capped events** reproduced.

This closes the event-shape dispatch prerequisite left open by the
[20 October E0 discovery campaign](numerical_e0_l40s_20261020.md). It establishes
instrumentation equivalence within the tested scope; it does not identify a
numerical mechanism. **M3 has not started.** The original E0 report and summaries
retain their contemporaneous, narrower conclusions.

## Counts and comparisons

The campaign started at 23:43:07 UTC on 20 October and finished at 00:11:42 UTC
on 21 October, taking 1,714.96 seconds (28 minutes 35 seconds). Six fresh Python
processes regenerated all 64 complete 128-token trajectories: unprofiled
baseline and capture, then two profiled baseline/capture pairs.

| Quantity | Result |
|---|---:|
| Evaluation sequences | 64 |
| Generated positions per sequence | 128 |
| Evaluation positions per worker | 8,192 |
| Historical events reproduced | 49/49 |
| Selected provider forwards per profiled worker | 113 |
| Full reference forwards per profiled worker | 64 |
| Traces per profiled worker | 177 |
| Kernel launches per profiled worker | 336,299 |
| Profiled workers | 4 |
| Captured event/state artifacts compared per profiled capture | 290 |

Provider tracing covers the union of historical capped positions, current caps
and one preselected non-historical-cap control per prompt. In this run the
union contains 49 event positions and 64 controls. Each trace covers a complete
selected forward, after naturally regenerating its prefix and KV state. All
128 tokens are generated, including the untraced calls. Each reference trace
covers the full prompt plus claimed tokens, rather than a shortened event prefix.

Five full-trajectory comparisons against the unprofiled baseline passed. These
compare the full sequence records, including claimed tokens, capped positions,
historical reproduction identities, and hashes of provider/reference logits,
reference hidden states, fingerprints and scores. The parent E0 baseline also
matched exactly, with unchanged model assets, production sources and recorded
environment. Diagnostic sources were updated for this campaign.

All four dispatch comparisons passed, each over 177 calls and 336,299 launches:

| Left | Right | Result |
|---|---|---|
| Profiled baseline 1 | Profiled capture 1 | Exact |
| Profiled baseline 1 | Profiled baseline 2 | Exact |
| Profiled baseline 2 | Profiled capture 2 | Exact |
| Profiled capture 1 | Profiled capture 2 | Exact |

Each profiled capture also matched all 290 raw NPZ artifacts from the unprofiled
capture: 113 event arrays and 177 provider/reference state files. This includes
saved native BF16 logits, actual Gumbel vectors, support masks, prefix tokens,
attention/position metadata and post-forward KV tensors. These arrays were saved
at selected calls, not at every one of the 8,192 generated positions.

## Review and reproducibility

The review independently reread the retained files and checked:

- All 12 campaign artifact hashes, 337 frozen source files, 2,044 artifact
  hashes across the six workers, and each worker's 50 source snapshot files.
- The parent summary hash and baseline artifacts, corpus identity, sequence
  completeness, environment/model/production-source continuity, and all five
  full-trajectory comparisons.
- All 708 raw traces, recomputing signatures for 1,345,196 kernel launches and
  matching every saved per-call signature and all four dispatch comparisons.
- Both raw capture comparisons and every numeric count in the original report
  against the sequence records and required call set.

No discrepancy was found. The existing dispatch tests cover missing/duplicate
calls, invalid shapes, changed launch dimensions, missing kernel metadata,
state drift, parent environment/production-source drift and subset gate limits.
The review reran the NumPy-core regression command below: **220 tests passed**
in 108.75 seconds, with one expected small-pool ratio warning. This includes
the focused numerical tests and follows the repository CI job; it excludes the
service and signed-audit modules:

```sh
.venv/bin/python -m pytest tests/ -q \
  --ignore=tests/test_audit_protocol.py \
  --ignore=tests/test_audit_conformance.py \
  --ignore=tests/test_service.py
```

Runtime sources started from commit `b2b7e9e` plus the then-uncommitted dispatch
adapter, validation, campaign and test changes. The runtime Python sources match
the frozen snapshot byte for byte; relevant hashes are in `review.json`.
Documentation was updated after the run and review. No fresh GPU measurements
were made during this review.

Environment: NVIDIA L40S, driver 595.91.07, CUDA runtime 12.8, Python 3.12.3,
PyTorch 2.8.0+cu128, Transformers 4.57.6, NumPy 2.1.2, Accelerate 1.15.0 and
bitsandbytes 0.50.2. Model: `Qwen/Qwen3-1.7B`, revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. The preserved sampler uses temperature
1, top-k 50 and top-p 0.95; synthetic noise was disabled. Matmul TF32 was disabled,
cuDNN TF32 enabled, and PyTorch/OMP/OpenBLAS/MKL threads set to one.

The [numerical tools README](../../experiments/numerical/README.md#detached-m1-dispatch-follow-up)
gives the reproduction command and preflight procedure. A subset run cannot pass
the full gate. Use a new output directory and the pinned environment.

## Evidence retention and limits

The full run remains in `runs/numerical-m1-dispatch-20261020/`: 2,745 files,
11,281,195,321 bytes (approximately 10.51 GiB), excluding the external model
cache and parent E0 directory. Git contains this reviewed report, the unchanged
original report and summary, review statistics, original campaign hashes, and
the four compact per-call dispatch signatures. Raw tensors, traces and source
snapshots remain local. A Git clone alone cannot independently recompute the
raw trace or tensor comparisons; preserve both run directories before releasing
the host.

The gate is `passed_for_historical_evaluation_corpus`. Kernel-name/grid/block
equality applies to the traced complete forwards; full output equality does not
prove kernel equality on untraced calls. This is historical discovery data, not
a fresh confirmatory sample or a claim about other GPUs, contexts or backends.
Timing under profiling is not a latency result. Matching dispatch does not
identify a reduction tree, a faulty operation, a contract violation or an attack.

The next implementation step is M3: identical-operand operation replay and
state-consistent forward/reverse interventions for GEMM/reduction and
attention/reduction candidates. No mechanism or causal witness has yet been
established.
