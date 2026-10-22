# Next experiments: implementation and execution plan

**Updated: 22 October 2026.** This is the active work queue. The
[full experimental design](../docs/NUMERICAL_CAUSAL_DIAGNOSIS_EXPERIMENTS.md)
specifies E0–E5, statistical units, blinded evaluation, and paper claims.
The [previous verification agenda](LEGACY_VERIFICATION_EXPERIMENTS.md) is
preserved for historical context and is not the current execution order.

## Objective

Build **numerical causal diagnosis with reproducible witnesses**: explain which
operation or interacting operation set changes a numerical quantity, how that
change reaches a sampling boundary, and whether the execution violates an
explicit contract. Explanatory correctness, state-consistent intervention, and
generalization are the candidate contributions. Lower cost is a secondary metric.

BF16 divergence is still unresolved. A different token or a large verification
score does not by itself identify a bug, an attack, or a hardware mechanism.
The proposed algorithm and its novelty remain to be validated against prior work.

## What is ready, and what is not

| Component | Status | Entry point / next action |
|---|---|---|
| Real BF16/NF4/INT8 generation and reference replay | Existing, archived pilot | [Quantization runner](../experiments/exp_real_quantization_gpu.py) and [validated loader](../experiments/analyze_real_quantization.py) |
| Evaluation-only score-tail analysis and capped-event ledger | Implemented, CPU only | [Numerical tools](../experiments/numerical/README.md) |
| Legacy sampler boundary illustrations and regression checks | Implemented, synthetic examples only | [Sampler checks](../experiments/numerical/sampler_contract.py); not a GPU conformance result |
| Paired full-logit, support, RNG, and state capture | M1 passed for the historical evaluation corpus and selected complete forwards on L40S | [Reviewed M1 report](../docs/results/numerical_m1_dispatch_l40s_20261021.md) |
| Dtype / filtering factorial replay | Discovery matrix complete; 49 residuals under the matched legacy protocol | Milestone M2 |
| State-consistent operator interventions | Two 64-case pilots reviewed; native-shape preflight passed, but the full native-shape round stopped on missing GPU trace evidence; full mechanism gate pending | [Reviewed pilot](../docs/results/numerical_m3_partition_bridge_l40s_20261021.md) and [22 October status](../docs/results/numerical_native_shape_status_l40s_20261022.json) |
| Automatic localizer and witness reduction | Pending; a standalone checker prototype has passed the four-case native-shape preflight only | Milestone M4 |
| Locked comparison, benign controls, transfer, natural-case evaluation | Pending | Milestone M5 |

The archived BF16 evaluation pool has **49 capped token-DiFR positions among
8,192 positions**, accounting for approximately **98.4% of score mass**.
Across BF16/NF4/INT8, the new ledger identifies 395 capped positions
(49/252/94). These are discovery observations. The archive does not contain
the complete paired logits or intermediate states needed to attribute them.

## Repository boundaries

- Keep production/library behavior in `ivgym/`. The top-p comment now matches
  the legacy code; no sampling arithmetic was changed by this cleanup.
- Put new diagnostic tooling in `experiments/numerical/`. Keep GPU imports
  inside GPU entry points so archive inspection remains NumPy-only.
- Reuse the existing archive validator; do not introduce a second weaker loader.
- Keep existing runner names and archive paths stable. The slides analyzer is
  a thin adapter; reusable analysis no longer lives in presentation code.
- Put fresh results under a new directory in ignored `runs/`. Preserve
  original captures, hashes, failures, and source snapshots. Never overwrite
  `docs/results/` during a reproduction.
- Add tests for evidence contracts and interventions. Do not add empty GPU
  runner/localizer stubs that appear executable before they are implemented.

## M0 — CPU preparation (implemented)

Run the commands in the [numerical tools README](../experiments/numerical/README.md)
to validate the archive, export capped-event identities, and record the legacy
sampler examples. Use fresh output paths for every invocation.

Acceptance criteria:

- The archive validator accepts all hashes, shapes, corpus partitions, captures,
  and exact reference-rerun controls.
- Only evaluation observations enter the tail summary and ledger.
- Each event maps to a corpus row, prompt ID, sequence seed, and zero-based
  generated-token position.
- A zero total score reports an undefined mass fraction, not a fabricated zero.
- The regression suite preserves top-k ties, the legacy top-p boundary, and
  float32/float64 input dtype. Synthetic examples stay labeled as synthetic.

This finishes preparation only. It does not establish provider/reference
equivalence, reproduce a GPU discrepancy, or explain a BF16 event.

## M1 — E0 capture and instrumentation equivalence (passed for the tested scope)

The capture adapter is implemented under `experiments/numerical/`, reusing the
generation/reference split of the real-quantization runner. The 20 October run
used the archived BF16 corpus and sampling seeds as a discovery set: 49/49
historical events reproduced across 64 sequences and 8,192 positions. Two
baseline and two capture processes agreed on the recorded output hashes.
The [reviewed E0 report](../docs/results/numerical_e0_l40s_20261020.md) originally
covered kernel-name/grid/block equality only on a four-token smoke sequence.
The [21 October follow-up](../docs/results/numerical_m1_dispatch_l40s_20261021.md)
passed dispatch validation at the selected event/control shapes: two fresh
profiled baseline/capture pairs each traced 113 provider forwards and 64 full
reference forwards, with equal ordered kernel names/grid/block. Full trajectories
matched unprofiled controls and parent E0; captured event and KV artifacts also
matched with and without profiling. M1 is passed for this historical corpus and
the pinned HF path, not for untraced calls, other environments or backends.

Capture, for each selected event and its controls:

- Full unfiltered provider/reference logits with original dtype and exact bits.
- Actual top-k and top-p support masks, sorted boundary values, cumulative
  probabilities, uncapped score/margin, and the actual shared random tensor.
- Exact prefix tokens, masks, position IDs, KV lineage, model/contract hashes,
  operation shapes/strides, and dispatch metadata.
- Instrumentation status, synthetic-noise-disabled assertions, output hashes,
  environment metadata, and independently reloadable artifact references.

First compare an uninstrumented run with a capture-enabled run at identical
shape and state. Also repeat each fixed-shape run in a fresh process.
Confirm dispatch and outputs before interpreting natural events. If hooks
change fusion or outputs, report the failure and use isolated operator replay;
do not present the altered path as the original execution.

**Exit gate:** actual random tensors and same-input sampler decisions agree
under a declared common protocol; capture preserves the tested computation;
all event identities are traceable. Historical events that do not reproduce
remain explicitly unmatched. Newly observed events receive new IDs.

**First GPU deliverable:** an E0 report containing the reproduction rate for the
49 historical BF16 events, a sampling/instrumentation failure table, and raw
paired evidence for every successfully reacquired event. No causal claim yet.

## M2 — separate protocol effects from numerical effects

The CPU replay matrix is implemented and complete on the 20 October discovery
captures (113 events, 30 cells each). It evaluates these factors:

| Factor | Conditions |
|---|---|
| Logit source | Provider / reference, on the identical prefix |
| Filtering arithmetic | Original dtype / explicit float32 / explicit float64 |
| Nucleus convention | Preserved legacy / explicitly named crossing-inclusive variant |
| Sampling | Greedy; shared Gumbel without filtering; top-k only; top-p only; both |

Hold the random tensor fixed. Record changes in support membership, uncapped
margin, selected token, and capped score. Apply each condition to the same
logits; separately generated trajectories are a different experiment.

The inclusive variant is an experimental protocol, not an in-place fix.
Freeze its filtering order, tie handling, dtype, and version before using it in
generation, and recalibrate any detector thresholds. Preserve the legacy path.

**Exit gate:** every observed event is labeled protocol-explained,
residual-under-matched-protocol, non-reproduced, or unresolved. Do not infer that
all residuals are hardware faults. If matched sampling removes the interesting
phenomenon, narrow the paper to that evidence or reconsider the direction
before building a general causal system.

## M3 — E1 mechanism proof of concept

On 21 October, the explicit reduction-order pilot was implemented with full
token-checkpoint replay, fresh KV reconstruction, capture/sham controls and
forward/reverse swaps. Its four-case preflight passed two independent repeats
and saved-operand replay. The [reviewed 32-case-per-family campaign](../docs/results/numerical_m3_pilot_l40s_20261021.md)
completed three independent repeats and saved-operand replay. GEMM changed
local outputs in 7 cases and selected logits in 2; no sampled tokens changed.
Two cases changed top-k-only support. Attention had no A/B output changes,
but 18/32 cases had only one unmasked partition. Both diagnostic recipes
differed from native outputs in all cases. These results do not pass the full
M3 gate or attribute the historical BF16 discrepancies. The follow-up below
addresses partition coverage and observable native-output equivalence. See the
[runner instructions](../experiments/numerical/README.md#detached-m3-controlled-reduction-order-pilot).

The [reviewed partition/bridge round](../docs/results/numerical_m3_partition_bridge_l40s_20261021.md)
completed the same 64 cases, three repeats and saved-operand replay. Attention
chunks of 8 exercised 3–19 active partitions per case; the same-input chunk-32
local controls remained A/B-equal. FP32 pre-cast outputs changed in all 32 cases
of each family. Final BF16 outputs changed in 7 GEMM and 2 attention cases;
selected logits changed in 2 and 1 respectively. The primary sampler had no
support or token changes. One attention case changed the legacy top-p-only
token in both arithmetic dtypes. Full-shape native operator replay, native sham
and restoration controls passed, with operand strides preserved. Native-to-A/B
sampling changes are reported separately from the A/B order contrast.
The [second-round instructions](../experiments/numerical/README.md#second-m3-pilot-partition-coverage-and-native-bridges)
give reproduction commands. These remain historical discovery experiments;
the native bridge checks output equivalence and does not identify kernel internals.

Start with one GEMM/reduction case and one attention/reduction case on L40S.
Do not begin with a full model-wide dependency engine.

For each case, save identical operands and verify the actual implementation
used. A shape change alone does not prove split-K or a particular reduction
tree. Compare operation outputs, then replay their effect through the model to
the selected logit/support/token decision.

Implement checkpoint restore and descendant recomputation. After changing an
ancestor, recompute affected residual paths, normalization, attention, and KV
state; retain only state proven independent of the intervention. Use forward
and reverse replacements, sham replacements, and repeated controls.

**Exit gate:** two non-protocol mechanisms have matched-input evidence,
state-correct forward/reverse interventions, and independently replayable
witnesses. A legal numerical variation must remain labeled allowed variation
unless a separately declared contract is demonstrably violated.

Failure to observe two mechanisms is a research result, not permission to
rename correlated traces as causal evidence.

## M4 — implement and freeze the proposed algorithm

Once M3 works, implement a dependency-aware localizer:

1. Build operation/state dependencies from valid captures and checkpoints.
2. Propose candidate regions from numerical and decision-boundary changes.
3. Test candidates through state-consistent interventions.
4. Search interacting sets when single-site replacement is insufficient.
5. Reduce a successful witness while preserving its stated property.
6. Replay the bundle with an independent checker; otherwise return unresolved.

Do not assume causal effects are monotone across layers. Cancellation,
reinforcement, and KV dependencies can invalidate naive binary search.
A reduced witness is only locally minimal under the tested reduction moves;
do not claim a globally minimal cause without evidence.

Build a fault harness that hides injection sites and labels from the diagnoser.
Use development cases to choose search policies and tolerances. Include
first-divergence, norm ranking, layer bisection, and single-site replacement
baselines, with the same observable inputs and intervention budget. Include
state-recomputation, interaction-search, and witness-validation ablations.

**Exit gate:** freeze algorithm/configuration/source hashes, candidate budget,
failure taxonomy, splits, and checker before locked evaluation. Recheck the
primary literature and implementation versions listed in the full design;
do not treat a local proxy as an evaluation of an external system.

## M5 — confirmatory evaluation and paper decision

Follow E2–E5 in the full design, in this order:

1. Blinded faults and interaction cases on L40S.
2. Legal BF16 execution changes as negative controls.
3. Transfer to H100, then A100; a separate pinned vLLM backend and a second
   model only after capture equivalence passes for each.
4. Fresh natural discrepancies with a frozen method and independent replay.

Use the proposed fresh development / calibration / locked evaluation split of
128 / 199 / 256 prompt-seed pairs and a preselected 64-pair transfer subset;
finalize power and resource estimates from E0 before collecting locked data.
Split source documents/template families, not just seeds. Treat a prompt as
the statistical cluster; multiple positions or injected variants are not
independent samples.

Primary outputs: validated-witness recall, false-witness rate, unresolved
coverage, independent replay success, conformance errors, and transfer results.
Report all denominators and prompt-cluster uncertainty, including capture and
replay failures. Runtime/storage are supporting measurements.

The main paper is viable only if explanations survive independent replay,
state-correctness controls, and blinded comparisons. If success is limited to
hand-picked cases or one backend, narrow the claim accordingly. The optional
trajectory-consistency and identifiability directions remain separate projects.

## Next machine and launch gate

Use **one dedicated NVIDIA L40S 48 GB**, preferably the original 18 October host
if still available. Live availability has not been checked. Suggested host
provisioning: at least 16 vCPUs, 64 GB RAM (128 GB preferred), and 1 TB persistent
SSD. Measure actual storage and memory needs during M1.

Reconstruct the archived HF baseline before changing libraries:

| Item | Archived baseline |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| Transformers | 4.57.6 |
| NumPy | 2.1.2 |
| bitsandbytes | 0.50.2, needed for later quantized arms |
| NVIDIA driver | 595.91.07 on the archived host |
| Model | Qwen/Qwen3-1.7B |
| Revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |

Keep the later vLLM environment separate. If the original driver cannot be
reproduced, record that change and treat the run as a new environment.
E0 does not need Nsight Compute counters; verify counter permissions before
adding counter-dependent E1 claims. Timeline traces alone do not establish
a reduction mechanism.

Launch the E0 GPU job only after the capture runner exists, the CPU checks pass,
the model cache and environment are verified, source/plan hashes are frozen,
and exclusive GPU use is confirmed. The 20 October E0 campaign completed; a new
run must repeat these checks and use a fresh output directory.

## Immediate handoff

The 22 October native-shape implementation has passed preflight, but the full
campaign is **incomplete**. This workspace did not retain the previous ignored
`runs/` directories, so E0 and the complete M1 dispatch controls were reacquired in
`runs/numerical-e0-reacquire-20261022/` and
`runs/numerical-m1-reacquire-20261022/` directories. These are new acquisitions,
not restored copies of the earlier evidence. Both completed, with 49/49
historical events reproduced. The latest four-case native-shape preflight
passed two independent repeats and its standalone checker verified 4/4 cases.
The full core suite passed 249 tests and the preflight focused suite passed 90.
See the
[new runner protocol](../experiments/numerical/README.md#third-m3-round-native-execution-shapes-and-independent-witnesses).

Both 64-case attempts stopped during the first repeat after four completed
case records: `gemm_p60_t60_l14/provider_trace.json` contained CPU launch
events but no GPU kernels. The second attempt added explicit CUDA
synchronization, which did not resolve the missing trace. Neither attempt
produced a full summary or completed independent review. Partial outcomes
must not be aggregated as a completed campaign or used to advance M3.
The [status and integrity review](../docs/results/numerical_native_shape_status_l40s_20261022.json)
records the evidence. Preserve the original failed queue at
`runs/numerical-round-20261022/`, the original native-shape attempt, and the
fresh `runs/numerical-m3-native-shape-preflight-sync-20261022/` and
`runs/numerical-m3-native-shape-sync-20261022/` directories. No campaign is
currently running. The immediate execution task is to repair and validate
trace collection, then repeat preflight and the full campaign in new
directories; the next scientific task below remains unchanged.

The next task is **M3: independently replayable witnesses and evidence connecting
the controlled operator contrast to the historical native discrepancy**.
The two reviewed pilots now cover FP32 intermediates, multi-partition attention,
full-shape native output replay and native restoration. The full scientific gate
remains pending: native-to-recipe differences alone do not identify the historical
mechanism. Keep M4 deferred until that gate is met. The 20 October
campaign completed the M2 discovery
matrix (113 captured events, 30 cells each); all 49 historical capped events
remain residuals under matched filtering arithmetic with the legacy nucleus
convention. The crossing-inclusive variant leaves 6/49 disagreements on the
same frozen logits. These observations justify further investigation, but are
not hardware faults or causal witnesses.

The [reviewed E0 report](../docs/results/numerical_e0_l40s_20261020.md) and
[M1 follow-up](../docs/results/numerical_m1_dispatch_l40s_20261021.md) record the
results and evidence hashes. The full local evidence remains in
`runs/numerical-e0-20261020/` and `runs/numerical-m1-dispatch-20261020/`, including
raw captures, traces and source snapshots. Both directories are intentionally
ignored by Git; preserve them before releasing the host. Both M3 runs
(`runs/numerical-m3-20261021/` and
`runs/numerical-m3-partition-bridge-20261021/`, with their preflights) must remain
available for offline review. The dispatch prerequisite
now permits M3 investigation within the validated scope; no mechanism has been
attributed. A new detector, automatic localizer, or MLSys contribution should
not be claimed before the later intervention and evaluation gates pass.
