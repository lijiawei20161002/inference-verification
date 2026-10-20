# Explaining and Verifying Numerical Divergence in LLM Inference

**Experimental design, version 0.1 — 20 October 2026**

**Status:** proposed experiments; no new GPU measurements are reported here.
**Primary direction:** numerical causal diagnosis with reproducible witnesses.
**Contribution priority:** explanatory correctness, execution conformance, and
generalization. Lower verification cost is not a primary contribution.

[Research directions](../planning/NON_COST_RESEARCH_DIRECTIONS.md) ·
[BF16 technical review](../planning/BF16_TECHNICAL_NOVELTY.md) ·
[Current implementation plan](../planning/NEXT_EXPERIMENTS.md) ·
[Existing experiment map](../experiments/README.md)

## 1. Research objective and claims

Develop a method that turns an observed inference discrepancy into an
experimentally validated explanation:

> A particular operation, under a particular execution condition, changes a
> numerical quantity, changes a sampling decision, and accounts for the observed
> discrepancy under controlled intervention.

A diagnosis should produce an executable witness containing the relevant input,
execution configuration, operation or interacting operation set, numerical
difference, sampling-boundary effect, and intervention outcomes.

The method must distinguish **causal attribution** from **contract violation**.
A legal change in reduction order can cause a token change without being a bug.
Replacing its result with a canonical reference is an explanatory intervention,
not necessarily a repair. Conversely, an injected fault may be present without
affecting the selected output.

| ID | Proposed claim | Evidence required | Result that defeats the claim |
|---|---|---|---|
| C1 | Numerical divergence can be attributed to specific execution mechanisms | Matched-input interventions, reverse interventions, and negative controls | Explanations rely only on correlated traces or the first differing tensor |
| C2 | The method produces valid witnesses for previously unseen cases | Frozen algorithm; blinded faults; independent witness replay | Manual per-case tuning or poor witness reproduction |
| C3 | The method separates legitimate BF16 variation from contract violations | Predeclared contracts; legal execution changes as negative controls | Correct attribution is routinely misreported as misconduct |
| C4 | Dependency-aware diagnosis handles state and interacting causes | KV-consistent replay and cancellation/reinforcement cases | Results disappear when descendants are recomputed correctly |
| C5 | The method transfers beyond one model/backend combination | Held-out GPU/backend/model evaluations | Only the original implementation is diagnosed successfully |

These are hypotheses, not novelty claims already established. The minimal
paper requires C1–C4; C5 determines the breadth of the final claim.

## 2. Existing evidence and what must be collected

The [L40S real-weight pilot](results/real_quantization_l40s_20261018.md)
used Qwen3-1.7B, 32 calibration and 64 evaluation sequences per arm, with 128
generated tokens per sequence. Token-DiFR rejected 0/64 BF16, 34/64 NF4, and
5/64 INT8 sequences. These are detection results, not localization results.

The [offline score-tail diagnostic](../experiments/numerical/README.md)
found 49 scores equal to the cap of 30 among 8,192 BF16 evaluation positions;
those scores contribute 98.4% of total BF16 token-DiFR score mass. All 49 also
reach the local token-rank cap. This motivates investigating support boundaries,
but does not identify their cause: a large finite margin can also reach the cap.

The archive contains claimed tokens, fingerprints, scores, and aggregate logit
differences. It does **not** preserve the complete paired logits, support masks,
or intermediate tensors needed for causal attribution. Reacquisition is required.
Historical capped positions are a discovery set, never a confirmatory test set.

Two known controls must be resolved first:

- The pilot filters provider logits as float32 and reference logits as float64.
- The legacy top-p implementation excludes the token that first crosses the
  cumulative-probability threshold, except when it is first. Its misleading
  comment has been corrected without changing code behavior. Comparison with
  a crossing-inclusive convention remains a separately versioned experiment.

The local NumPy random stream also does not reproduce vLLM's stream merely by
using the same integer seed. None of these observations establishes the cause
of the historical 49 events.

## 3. Scope, access, and execution contract

### 3.1 Access model

The main study concerns **owned or cooperative inference systems**. The
diagnoser can capture operation inputs/outputs, inspect execution metadata,
replay kernels, and restore trusted checkpoints. It is not a token-only audit
of an uncooperative public API.

Trace collection and interventions are performed in a controlled diagnostic
run. The benchmark's injection manifest is available only to the evaluator.
The diagnoser must not receive fault labels, injection coordinates, diagnostic
file names encoding the fault, or evaluation-only annotations.

Self-reported provider traces are not authenticated execution evidence.
Extending to an adversarial provider requires a separate trust mechanism.

### 3.2 Initial platform

| Item | Initial choice | Extension |
|---|---|---|
| Model | Qwen/Qwen3-1.7B, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | A second dense architecture, e.g. Llama; freeze its available checkpoint hash before collection |
| GPU | L40S, connecting to the existing archive | A100 and H100 in separate, pinned environments |
| Engine | Explicit HF eager and supported SDPA paths | A pinned vLLM release, after instrumentation equivalence is established |
| Execution | Single GPU; no offload; evaluation mode | Dynamic batching and graph execution as separately identified conditions |
| Precision | BF16 weights/activations/KV where applicable; record actual mixed-precision operations | Real NF4/INT8 as broad interventions, not single-site faults |
| Generation | Fixed 128-token diagnostic traces | EOS-respecting generation and longer-context transfer |

**Next campaign machine:** one exclusively allocated L40S 48 GB, preferably
the original 18 October host if it is still retained. Its current availability
has not been checked. Suggested host provisioning is at least 16 vCPUs,
64 GB RAM (128 GB preferred for captured tensors), and 1 TB of persistent SSD
storage. These are planning recommendations, not measured peak requirements.
Use staged snapshots rather than retaining every layer at every token.

Run E0 and the first E1 mechanism cases on this machine. Reproduce the archived
HF environment first: Python 3.12.3, PyTorch 2.8.0+cu128, Transformers 4.57.6,
NumPy 2.1.2, and bitsandbytes 0.50.2; the archived host driver was 595.91.07.
The later vLLM environment is separate and must not replace this baseline.
After valid witnesses exist, use H100 for the first hardware-transfer study,
followed by A100. Hardware selection here prioritizes experimental continuity.

The previous L40S host could collect timelines but lacked permission for some
Nsight Compute hardware counters. E0 does not require those counters. Confirm
host-level profiling permissions before adding counter-dependent E1 analyses;
see [the profiling setup](PROFILING_SETUP.md).

Matching a GPU name does not match the environment. Archive driver, runtime,
library and compiler versions, GPU UUID, kernel dispatch, actual padded shapes,
attention backend, and relevant environment variables. A cross-GPU comparison
that also changes driver or kernel implementation is a compound deployment
comparison unless matched-input replay isolates a particular cause.

### 3.3 Contract and labels

Each experiment declares model/tokenizer hashes, masks and position IDs,
weight/KV formats, permitted operator implementations, rounding/cast points
when specified, cache semantics, sampling order, filtering/tie conventions,
random-number mapping, and allowed state transitions.

Use two independent labels:

1. **Explanation:** validated single mechanism; validated interacting set;
   multiple possible explanations; no propagated effect; unresolved.
2. **Conformance:** allowed variation; demonstrated contract violation;
   insufficient contract/evidence.

A generic BF16 declaration does not prescribe every accumulation order.
An alternative legal kernel must not be labeled faulty merely because it differs
from the reference. A demonstrable mask, position, or cache-semantics error has a
different status from an allowed rounding difference.

## 4. Experimental units, splits, and controls

### 4.1 Separate discovery from confirmation

| Partition | Proposed size | Purpose |
|---|---:|---|
| Legacy discovery | The archived 96 BF16 sequences; focus on the 64 evaluation sequences containing the 49 capped positions | Reproduce and investigate old observations |
| Fresh development | 128 new prompt/seed pairs | Implement instrumentation, choose candidate search and tolerances |
| Benign calibration | 199 new prompt/seed pairs per declared evaluation condition | Freeze any discrepancy/violation thresholds |
| Locked evaluation | 256 new prompt/seed pairs | All main paired method comparisons |
| Transfer subset | 64 preselected pairs from locked evaluation | Frozen-method cross-GPU/backend/model transfer |

Use eight document/template families in locked evaluation, 32 pairs each; at
least four families must be absent from development and calibration. Include
code, structured text, arithmetic, multilingual prose, and long-context
retrieval. Split at the original document/template level, not by changing
numbers in the same template. Archive exact texts and token IDs. Do not append
adversarially selected prompts to the locked test set.

Use initial context strata of approximately 128, 512, and 2,048 tokens,
balanced as closely as possible within each family. Freeze exact tokenized
lengths before running. The transfer subset adds 8,192-token contexts where
supported, reported separately. Do not silently truncate.

The same prompt/seed may be paired across conditions and methods. It remains
one statistical cluster. Multiple tokens, layers, fault variants, and technical
reruns of that prompt are not independent samples.

Distribute locked runs over at least eight fresh process sessions with randomized,
balanced condition order. Report session effects. These sessions test
reproducibility on the specified deployments; they are not eight independent
GPU populations.

### 4.2 Sampling controls

Freeze a documented sampler contract after E0. Evaluate:

| Setting | Temperature | Filtering | Purpose |
|---|---:|---|---|
| S0 | 0 | None | Greedy decision boundaries |
| S1 | 1 | None | Shared-Gumbel margin without support discontinuities |
| S2 | 1 | Top-k = 50 only | Rank-boundary effects |
| S3 | 1 | Top-p = 0.95 only | Cumulative-probability boundary effects |
| S4 | 1 | Top-k = 50, then top-p = 0.95 | Connection to the archived pilot |

For local paired experiments, save and reuse the actual random tensor or a
verified byte-identical reconstruction. Sampling settings can be applied to
the same frozen logits for a local mechanism experiment; free generation under
each setting creates different prefixes and is a separate evaluation.

Preserve the legacy sampler as a versioned condition. Revised semantics require
new calibration. Never reuse old thresholds after changing the protocol.

### 4.3 Instrumentation validity

Before using a capture, compare it with an uninstrumented run at the same shape
and state. Hooks, extra outputs, synchronization, or profiling can change fusion
and dispatch. Record the actual kernel/shape sequence and output hashes.

If capture changes execution, first use non-mutating metadata collection and
replay extracted operators separately. Label the latter as an isolated mechanism
experiment. Do not attribute a discrepancy introduced by the capture tool to the
original serving run.

Disable all synthetic benign-noise paths. In particular,
`ShapeBackend` defaults to `ref_noise=True`, and `HFGPUBackend` exposes nonzero
verifier/activation noise defaults. The new capture manifest must assert that
such paths are disabled. Synthetic faults are explicitly labeled experimental
interventions, never real quantization or natural GPU noise.

## 5. Proposed diagnostic procedure

The method under evaluation is a design to implement, not an existing feature.

1. Detect a discrepancy under a frozen trigger, preserving uncapped measurements.
2. Construct a dependency graph linking operations, tensors, sampling decisions,
   and KV writes/reads.
3. Slice backward from the affected sampling decision.
4. Generate candidate semantic changes: reduction topology, output cast,
   normalization, position/mask handling, or state provenance.
5. Execute matched-input interventions and recompute affected descendants.
6. Test reverse interventions, unrelated-site controls, and multi-site effects.
7. Reduce the witness while preserving its stated property.
8. Emit the witness and a separate conformance judgment, or return unresolved.

For a stateful operator, replacing its output while retaining incompatible
downstream KV is invalid. Restore a checkpoint before the intervention and
recompute the dependent state. Log every preserved and recomputed object.

### 5.1 Two estimands

**Local effect:** fixed prefix, fixed random draw, identical operation input;
measure how a specific implementation change affects logits, membership, and
the next-token decision.

**Trajectory effect:** start from a valid earlier checkpoint, apply the change,
then generate or teacher-force the declared continuation while rebuilding
dependent state. Measure effects on later decisions separately.

Do not compare unaligned free-generation logits after the first token divergence
as if they shared the same input. Teacher forcing isolates a conditional
computation; it does not prove that the same continuation would be generated.

### 5.2 Witness validation

A witness must contain the original event, intervention specification, matched
inputs, resulting state, before/after numerical measurements, and an independently
replayable predicate. A second process must verify it without the diagnosis
search history.

For controlled single-fault cases, require both removal and reintroduction of the
fault to change the declared outcome when applicable. For natural differences,
report intervention-supported influence; do not claim a unique root cause from
one successful replacement. An interacting set can be a valid explanation.

Reduction must preserve legal masks, positions, shapes, and state dependencies.
State whether the reduced witness preserves the original dispatch mechanism or
only the same symptom. A reduction algorithm generally establishes only
deletion-minimality under its tested transformations, not global minimality.

## 6. Experiment E0 — sampler conformance and historical event attribution

**Question:** how much apparent BF16 disagreement is explained by sampling
arithmetic or conventions before invoking hardware mechanisms?

**Design:**

1. Reacquire legacy BF16 runs without changing the original environment or
   protocol where feasible. Match claimed-token and score hashes first.
   If reproduction fails, record the failure and treat new events as new data.
2. Save both sides' full pre-filter logits for the relevant positions, candidate
   membership, sorted boundary values, cumulative probability around top-p,
   shared noise, post-noise competitors, and uncapped margins.
3. On identical saved logits/noise, cross float32 versus float64 filtering with
   the legacy versus documented crossing-token convention. This is a CPU
   factorial experiment after capture, with no GPU arithmetic changing.
4. Fix sampling arithmetic on both sides and compare their separately computed
   logits. Run S0–S4 to separate support changes from changes within fixed support.
5. Run explicit tie, near-threshold, excluded-token, empty/degenerate-input
   handling, and RNG-mapping conformance fixtures.

**Outputs:** an event ledger for all 49 historical capped positions, if
reproduced; counts of protocol-only, support-boundary, finite-margin, mixed, and
unresolved explanations; both token-level descriptions and sequence-level rates.

**Controls:** identical logits/dtype/protocol/noise must produce identical
decisions; uncapped and capped scores must reconcile; do not infer membership
from a score equal to 30.

**Gate:** stop hardware attribution until the sampler and capture controls pass.
If protocol alignment explains the phenomenon, fix and rerun the baseline.
A sampler correction alone does not support the proposed paper.

## 7. Experiment E1 — controlled numerical mechanism atlas

**Question:** which operator-level execution differences actually propagate to
sampling boundaries?

Collect 32 development input snapshots for each supported mechanism family,
distributed across early/middle/late layers and the context strata. Include
routine and boundary-near inputs; report them separately. This is a stratified
design, not a full Cartesian sweep.

| Family | Controlled intervention | Measurements and mandatory control |
|---|---|---|
| GEMM reduction | Change a verified split-K/reduction implementation while preserving operands | Raw output differences and downstream effects; verify actual dispatch |
| Epilogue/cast | Compare documented legal cast/fusion choices; separately inject a declared cast violation | ULP/absolute changes at the first affected output; no unrelated operator changes |
| Attention merge | Change supported split-KV/merge order | Per-partition and merged outputs; identical Q/K/V, mask, scale, and positions |
| RMSNorm | Change supported reduction/accumulation strategy | Input norm and output changes; fixed epsilon and parameters |
| RoPE/nonlinearity | Compare supported implementations on identical operands | Position/frequency or function input alignment; approximation contract recorded |
| State construction | Sequential versus prefill reconstruction of the same prefix | Intermediate K/V differences and subsequent attention effects; label this as a compound intervention |

For each pair, collect three same-configuration technical repeats in separate
processes. Apply S0–S4 to resulting logits. A fixed-input kernel microbenchmark
and its downstream model replay are separate artifacts.

Use Hawkeye or an available exact emulator only within its documented support;
do not assume it covers the entire model or current kernel version. FlashInfer
split controls require a compatible backend and are not switches for an
unrelated attention implementation.

**Outputs:** a causal-chain table with operator change, local numerical effect,
support change, and token effect; representative executable witnesses; effect
sizes for both propagated and non-propagated changes.

**Gate:** identify at least two distinct non-protocol mechanism families with
intervention-supported sampling effects before claiming a general hardware
diagnosis contribution. Unsupported kernel controls are explicitly unsupported;
a runtime flag alone is not evidence that a mechanism changed.

## 8. Experiment E2 — blinded single-fault localization

**Question:** does the frozen method identify a causal mechanism on new cases
better than simple localization rules?

Prepare 64 cases per family, six families, using locked prompts. Balance two
predeclared severities per family. Reuse prompts across families only as paired
clusters. Injection locations and severities are selected before observing
whether the detector or diagnoser succeeds.

| Fault family | Example intervention | Contract defining the fault |
|---|---|---|
| Epilogue | Incorrect scale application or output conversion | Pinned operator scale/cast semantics |
| Normalization | Incorrect epsilon or normalization factor | Model configuration and operator formula |
| Position handling | Incorrect RoPE position offset | Prefix and position-ID contract |
| Attention mask | Read a forbidden location or omit a required valid location | Declared causal/full-context semantics |
| KV state | Replace a cache entry with a stale entry or drop required history | Cache lineage and retention contract |
| Sampling | Wrong random counter or filtering convention | Frozen E0 sampling protocol |

Choose bounded, executable perturbations on development data. Include subtle
faults and no-ops, not only failures causing NaNs or grossly corrupted text.
Match simple baselines on the same captured information.

Evaluate two routes:

- **Oracle-triggered diagnosis:** the evaluator supplies the discrepancy
  position, but not its cause. This isolates diagnosis quality.
- **End-to-end diagnosis:** the method must trigger, localize, and validate.
  This includes missed discrepancies and unresolved cases.

Report all injected cases. Separate faults with no observable effect from
triggered faults; do not label an inert injection as a missed causal explanation.
Conditional localization results must be accompanied by the trigger rate and
unconditional end-to-end witness yield.

**Primary outputs:** validated-witness recall, false-witness rate, operation and
mechanism-family localization, and the fraction unresolved. Top-1/top-3
localization is secondary to actually validating the explanation.

**Gate:** a useful method must improve validated-witness recall over the strongest
same-information simple baseline, without achieving that gain by making more
unsupported causal claims.

## 9. Experiment E3 — interacting causes and state-consistent interventions

**Question:** does the dependency-aware procedure handle cases where the
failure predicate is non-monotonic?

Preselect 96 locked prompt cases, 24 in each stratum:

1. Two changes that reinforce an observable discrepancy.
2. Two changes that partially or completely cancel.
3. A change whose effect depends on a previous KV-state alteration.
4. A real change paired with an unrelated/no-op intervention.

Freeze pairs and severities after development. If a proposed cancellation pair
does not cancel in evaluation, retain it and report the realized interaction.
Do not keep searching the evaluation set for favorable examples.

For each case run the full 2×2 intervention table: neither change, A only,
B only, A+B. Repeat with state-consistent descendant replay. For continuous
responses report `Y_AB - Y_A - Y_B + Y_0`; also report the complete discrete
support/token outcomes, for which that contrast alone is insufficient.

Compare the full method with:

- First-different-layer and independent single-site replacement.
- A binary-search localizer that assumes a monotonic failure predicate.
- The proposed method with dependency edges removed.
- The proposed method with KV descendants incorrectly retained, as an explicitly
  invalid-intervention control, never as a deployable baseline.

For small candidate sets, exhaustively enumerate subsets to check witness
minimality. Report deletion-minimality for larger sets.

**Gate:** joint explanations must survive valid state reconstruction. If benefits
depend on inconsistent activation/KV splicing, reject the result.

## 10. Experiment E4 — legitimate variation and transfer

**Question:** can the method explain numerical changes without falsely accusing
legal executions, and can its explanation procedure transfer?

Use the 256 locked prompt/seed pairs in each supported, preregistered benign
condition. Freeze the supported set before evaluation:

| Condition | Permitted variation | Interpretation control |
|---|---|---|
| B0 | Exact same execution repeated | Capture/replay validity |
| B1 | Actual batch size 1 versus 4 | Keep the focal input/mask identical; API concurrency is insufficient |
| B2 | Supported eager versus SDPA attention | Validate identical mathematical masks, positions, and scaling |
| B3 | Sequential decode versus prefill replay | Same token prefix; acknowledge different arithmetic paths |
| B4 | Correct cache reuse versus valid reconstruction | Check lineage and positions; allow lawful numerical differences |
| B5 | Pinned vLLM eager versus graph execution | Confirm instrumentation and actual kernel shapes |

Reserve separate calibration for each condition that uses a learned threshold.
Return contract/evidence-insufficient when required metadata is absent; measure
that rate. Report both false-violation rate and unsupported-witness rate.
Explaining an allowed numerical cause is a success, not a false alarm.

Run the frozen algorithm on the 64-pair transfer subset on A100/H100, an unseen
backend/kernel family, and the second dense model as resources permit. Do not
combine “new hardware” and “new model” into the only transfer result.
Report zero-retuning results first. If calibration is subsequently permitted,
use an independently collected calibration partition and label it separately.

**Gate:** the claim of cross-condition diagnosis requires valid witnesses on
unseen conditions and acceptable false-violation rates. Failed transfer narrows
the claim; it cannot be repaired by tuning on the transfer evaluation set.

## 11. Experiment E5 — natural cases and real quantization

**Question:** does the method explain naturally occurring discrepancies rather
than only faults tailored to the injection harness?

Collect a discovery ledger of natural discrepancies from E0/E1 and fresh
deployments. Freeze a collection window and inclusion rule before examining
confirmatory cases. Include unresolved cases and benign differences.

For each case produce the same independent witness bundle as for injected
cases. Natural cases have no oracle root-cause label: use intervention evidence,
reverse tests where meaningful, and reproduction by a second investigator or
independent runner. A persuasive narrative is not a ground-truth annotation.

On 64 preselected evaluation prompts, additionally collect actual BF16/NF4/INT8
executions with matched sampling semantics. Compare fixed-prefix replay and
free generation separately. Packed weights, compute dtype, implicit casts,
kernel implementation, and KV format must be verified.

Whole-model quantization is a broad, multi-operation intervention. Do not assign
it an artificial single-layer ground truth. Test whether the method identifies
supported causal subsets or correctly returns a broad/ambiguous explanation.
Transplanting a BF16 tensor may restore a decision without proving that a
particular layer is the unique cause.

**Gate:** a paper about inference systems needs multiple independently
reproducible natural mechanisms/cases. If all successes are trivial injected
errors or the local sampler mismatch, present the work as a narrower tool or
continue research; do not claim a general solution to BF16 verification.

## 12. Baselines, ablations, and information matching

| Baseline | Purpose | Fair comparison |
|---|---|---|
| First differing layer/operator | Simple chronological localization | Same tensor visibility and numerical tolerances |
| Largest normalized numerical difference | Test whether causal reasoning beats magnitude ranking | Normalize using development data only |
| Independent single-operation replacement | Strong simple intervention baseline | Same replay and state-correctness requirements |
| Linear/backward scan; binary search | Search baselines | Expose failure on non-monotonic cases |
| Exact replay/emulator on supported operators | Mechanism oracle/reference control | Report supported coverage; no full-model assumption |
| DiFR margin/likelihood/Monte Carlo | Detection-only context | No fabricated localization accuracy for a detector |
| TELLER, if reproducible and compatible | Trace/log diagnosis comparison | Match available modalities; label any adaptation |

If an external implementation is unavailable or incompatible, document that
fact and use a clearly named proxy; do not label a reimplementation-inspired
baseline as the original method. The repository's `token_toploc` is neither a
TOPLOC reproduction nor RUT.

Required ablations remove, one at a time: dependency slicing; sampling-support
information; reverse interventions; multi-cause handling; KV-aware replay; and
witness minimization. Evaluate **validity and coverage** as well as localization.
More compute is not the claimed benefit, but budgets and timeouts must be
reported consistently so that one method is not silently granted unlimited
search. Timeout/unresolved cases remain in the denominator.

## 13. Metrics, statistical analysis, and decision rules

### 13.1 Endpoints

- **Validated-witness recall:** cases receiving a replay-validated causal
  witness divided by all eligible discrepancy cases, including unresolved cases.
  For injected cases, require consistency with the hidden ground-truth
  intervention and its realized effect.
- **False-witness rate:** issued witnesses that fail the independent predicate,
  intervention checks, or known ground-truth consistency, divided by all issued
  witnesses. Also report this count over all eligible cases.
- **False-violation rate:** legally executed cases labeled a contract violation,
  divided by all legal cases.
- **Witness reproduction:** unchanged witness passes in a fresh process under
  its declared environment. Cross-environment transfer is a different endpoint.
- **Localization:** operation/family top-1 and top-3; set precision/recall for
  multi-cause cases. Do not demand a unique label when multiple causes are valid.
- **Coverage:** triggered, explained, unresolved, unsupported, timeout, and
  non-propagating fractions.

Measure memory, trace volume, and runtime as feasibility metrics. They do not
replace the explanatory and conformance endpoints.

### 13.2 Analysis

Freeze algorithms, budgets, trigger rules, primary comparisons, contracts, and
the evaluation manifest before unlocking results. Select the strongest simple
baseline on development data, not by inspecting test outcomes.

Use paired prompt-cluster bootstrap intervals for differences between methods,
retaining all conditions and interventions for each sampled prompt. Use at
least 2,000 draws with a recorded seed. Report session-stratified results and a
session-level sensitivity analysis. Eight sessions provide limited evidence
about deployment-to-deployment variability.

Report family-specific results and a macro-average giving each fault family
equal weight. Use Holm correction for multiple confirmatory superiority tests.
Use simultaneous one-sided bounds for the preregistered benign-condition
false-violation claims, e.g. Bonferroni-adjusted exact binomial bounds when
prompt cases can reasonably be treated as independent within a fixed condition.
If residual clustering invalidates that assumption, use cluster-level analysis
and collect more independent sessions instead of citing the binomial guarantee.

The proposed 256 evaluation pairs are a planning allocation, not a completed
power analysis. Before locking, estimate paired disagreement rates and cluster
variation from development only; size the comparison for a 10-percentage-point
improvement in validated-witness recall, 80% power, and the planned multiplicity
correction. Increase the allocation before test collection if needed.

For scale: with zero false violations in 256 independent cases, a one-sided
Bonferroni bound across six conditions is approximately 1.9%. This calculation
does not cover new deployments, threshold fitting uncertainty, or correlated
cases.

### 13.3 Proposed go/no-go criteria

These are research decision targets to freeze before confirmation, not measured
results or universal standards:

1. All E0 identity and sampler controls pass; instrumentation changes are
   eliminated or explicitly isolated.
2. At least two non-protocol numerical mechanism families have valid
   intervention-supported witnesses.
3. Validated-witness recall improves by at least 10 percentage points over the
   development-selected simple baseline, and the adjusted interval excludes
   zero for the preregistered comparison.
4. False-witness and benign false-violation rates each have appropriate
   one-sided 95% upper bounds below 5%; report coverage so abstention cannot
   manufacture this result.
5. At least 95% of issued witness bundles reproduce under their declared
   environment; report its interval and every failure.
6. Multi-cause improvements survive correct KV reconstruction and held-out
   evaluation. A broader transfer claim additionally requires E4 transfer.

A failed gate is a result to report. Do not select a new threshold or exclude a
fault family using locked results and continue calling the evaluation held out.

## 14. Optional extension A — whole-trajectory execution consistency

This is a separate claim and is not required for the diagnosis paper.

**Hypothesis:** per-position compatibility can miss inconsistencies that become
visible when legal state transitions are enforced jointly.

Begin with a finite, explicitly declared two-backend state model and 64 fresh
prompt/seed pairs. Test fixed A, fixed B, legal A→B with valid state transfer or
reconstruction, and contract-violating state changes such as stale-KV reuse.
Do not declare ordinary kernel switching illegal just to obtain separation.

Compare per-position best-match scoring, pooled sequence scores, and a
dependency-constrained path checker. Use exhaustive path enumeration on short
traces as an oracle. Validate a state abstraction by checking that every legal
trace in the bounded test model is represented.

Report detection of inconsistent traces, legal-path false rejection, spurious
abstract paths, and minimal inconsistent witness sets. The important result is
joint consistency beyond marginal scores, not faster path search.

Run with full trusted diagnostic state, reduced metadata, and tokens/shared
noise only. If partial observations do not constrain paths sufficiently, report
that limitation. A provider's unverified trace must not become trusted input.
This extension must differentiate itself from IMMACULATE's commitments and
Bit-Exact Verification's replay model.

## 15. Optional extension B — observability and identifiability

This is a theory-plus-systems study, not an inference from low detection rates.

**Question:** which guarantees become possible when the observer receives
tokens only, tokens plus a verified shared random stream, or trusted internal
state?

Construct finite-vocabulary, bounded-length examples with exactly enumerable
transcript distributions under multiple allowed BF16-like arithmetic paths and
a quantized alternative. Include top-k/top-p discontinuities. Compute optimal
tests and exact total variation in these toy models.

The elementary bound `TPR <= FPR + TV(P, Q)` concerns the **entire observed
transcript** and is background, not a new theorem. When shared randomness is
observed, use the joint distribution of randomness and outputs. For adaptive
queries, include the interaction history. A marginal next-token distance is
insufficient.

Then perform matched real-model evaluations under the same visibility tiers,
including actual quantization and legitimate execution changes. Use them to
test predictions, not to claim exact high-dimensional TV bounds. Shared-seed
agreement and optimal-coupling agreement are different quantities.

A contribution requires a tight system-specific boundary, a constructive
distinguishing method, or a justified minimal-observation requirement. Low
INT8 detection power alone proves none of these.

## 16. Implementation tasks and artifact contract

Existing reusable components:

- [Real quantization runner](../experiments/exp_real_quantization_gpu.py):
  packed-weight checks, independent reference model, generation/replay split,
  and manifest capture.
- [Shape replay backend](../experiments/exp_benign_shape_dprime_gpu.py):
  schedule controls; disable synthetic noise explicitly.
- [Sampling](../ivgym/sampling.py) and [verifiers](../ivgym/verifiers.py):
  legacy behavior to preserve; GPU conformance and uncapped diagnostic
  instrumentation still need implementation.
- [Score-tail analysis](../experiments/numerical/score_tails.py):
  validated evaluation-only summary and event ledger, not a causal localizer.
- [Sampler examples](../experiments/numerical/sampler_contract.py):
  synthetic CPU boundary checks; no GPU inference or cross-engine RNG claim.

New components, **not yet implemented**:

| Component | Required behavior |
|---|---|
| Capture adapter | Observe actual operations/shapes and preserve logits/support/state evidence |
| Intervention runner | Match inputs; restore checkpoints; recompute dependent state |
| Blinded fault generator | Separate evaluator-only labels from diagnoser inputs |
| Causal localizer | Search candidates and interacting sets; return unresolved when necessary |
| Witness reducer/checker | Preserve the witness property and independently replay it |
| Causal-study analyzer | Extend the existing score-tail preparation to enforce new campaign splits, denominators, paired analysis, and witness provenance |

Each future campaign should write a new immutable result directory containing:

```text
plan.json                    # hypotheses, contracts, split hashes, fixed gates
environment.json             # model/software/GPU/kernel provenance
corpus.json                  # source-family IDs, token IDs, split membership
cases.jsonl                  # public case inputs; no injection labels
evaluator/faults.jsonl        # hidden intervention labels and realized effects
events.jsonl                 # raw trigger and support/margin measurements
snapshots/                   # exact tensor bits, metadata, checkpoint lineage
interventions.jsonl          # candidates, replacements, descendants, outcomes
witnesses/                   # independently executable diagnosis bundles
summary.json                 # all cases, denominators, intervals, failure counts
SHA256SUMS
```

An event record includes case/prefix/step IDs; model and contract hashes;
operation/layer/kernel identifiers; shapes/strides/dtypes; mask and position
hashes; logits and support references; random-stream identity; uncapped margin;
KV lineage; and whether the run is natural, legal variation, or controlled
injection. Store NaN/Inf classification separately from finite numeric fields.

Preserve the full logits for included diagnostic events: a saved top-k list
alone may not reconstruct a top-p boundary. To limit storage, use staged
capture and rerun selected frozen cases, but verify the event and dispatch are
unchanged before accepting the detailed capture. Report capture failures.

## 17. Execution order and paper figures

| Stage | Work | Exit condition |
|---|---|---|
| A | E0 and capture equivalence | Sampler and instrumentation controls pass |
| B | E1, development faults, initial witnesses | Two non-protocol mechanisms; independent replay works |
| C | Freeze method, manifests, power calculation, and budgets | No unresolved protocol or label leakage issues |
| D | E2/E3 and benign E4 | Locked paired evaluation and state-correctness checks |
| E | E4 transfer and E5 natural cases | Claim breadth determined by observed transfer |
| Optional | Extension A or B | Separate hypothesis and evidence; no automatic scope expansion |

Estimate run time and storage from Stage A before scheduling larger campaigns.
Do not invent a GPU-hour estimate from the old timing experiments: full tensor
capture and intervention replay are different workloads.

Suggested main-paper figures:

1. A real discrepancy traced from an operation to a sampling boundary.
2. The diagnostic dependency graph and state-consistent intervention procedure.
3. Validated-witness recall versus false-witness rate, with unresolved coverage.
4. Multi-cause cancellation/reinforcement cases and ablations.
5. Benign conformance and held-out transfer.
6. Independently reproducible natural cases with explicit scope limitations.

The paper should not lead with a speedup plot. Its central question is whether
the system can produce explanations and conformance judgments that simpler
methods cannot substantiate.

## 18. Related-work controls

Primary sources checked during the accompanying technical review:

- [DiFR](https://arxiv.org/abs/2511.20621): token/activation verification,
  pooled legitimate calibration, and alternative score constructions. Compare
  detection separately from diagnosis.
- [Hawkeye](https://arxiv.org/abs/2603.20421): Tensor Core numerical modeling
  and replay. Use as a supported operator-level reference.
- [TELLER](https://arxiv.org/abs/2608.01975): cross-layer trace/log root-cause
  analysis. General fault localization is not a new contribution by itself.
- [Bit-Exact Verification](https://arxiv.org/abs/2606.00279): precise execution
  reconstruction. Distinguish causal witnesses from reproducing a known path.
- [Taming Bitwise Behavior](https://arxiv.org/abs/2609.11356): reduction-order
  descriptions and compiler/static checks. Kernel equivalence is prior work.
- [ModelMeta](https://arxiv.org/abs/2507.04354): model-level metamorphic testing.
  Metamorphic transformations alone do not establish novelty.
- [INT8 Conformance](https://arxiv.org/abs/2609.00363): tolerance blind spots
  for specific numerical faults. Include subtle faults and explicit contracts.
- [IMMACULATE](https://arxiv.org/abs/2602.22700): discrete-state commitments
  and verifiable auditing. Required comparison for trajectory extensions.
- [Statistically-Lossless Quantization](https://arxiv.org/abs/2605.02404):
  task/distribution fidelity and optimal coupling. Required distinction for
  observability extensions.
- [FlashInfer attention documentation](https://docs.flashinfer.ai/api/attention.html):
  supported fixed-split controls; verify applicability to the pinned backend.
- [vLLM batch invariance documentation](https://docs.vllm.ai/en/latest/features/batch_invariance/):
  execution control, not a general cross-hardware guarantee.

Paper-specific support and implementation versions must be checked again when
the actual reproduction environments are pinned.
