# Verifier benchmark and verdict design: borrowed from ContinuousBench and SentryWard

[Project home](../README.md) · [Research plan](NEXT_EXPERIMENTS.md) ·
[Research guide](../docs/RESEARCH_GUIDE.md) · [Experiment map](../experiments/README.md)

Two sources, two parts.

**Part I** borrows benchmark-design methodology from *ContinuousBench: Can
Differentially Private Synthetic Text Transfer Knowledge?* (ICLR 2027 submission
14589). The paper's subject, DP text synthesis, is unrelated to inference
verification. Its benchmark-design methodology is not. Six moves in that paper
map onto open items in [NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md), and Part I
turns each into a runnable experiment with a claim at risk, a protocol, a cost,
and the outcome that would falsify it.

**Part II** borrows verdict packaging from
[SentryWard](https://github.com/gpu-verifier/sentryward), a static formal
verifier for warp-specialized PTX kernels on Blackwell. It proves race-freedom,
barrier correctness and real-arithmetic equivalence to a PyTorch reference. None
of that machinery runs against a black-box endpoint. What transfers is how it
packages verdicts, assumptions, evidence and reference bindings, which are the
parts of IVGym that are currently prose. Part II items are infrastructure with
acceptance criteria rather than experiments with falsifiers, except S6.

---

# Part I: benchmark design (ContinuousBench)

The correspondence, in one table:

| ContinuousBench | IVGym analogue | Experiment |
|---|---|---|
| Tasks must be unsolvable without the corpus (*access dependency*) | Attacks must be undetectable without recompute (*recompute dependency*) | B1 |
| Three reference runs bracket every number | Same-shape null, benign floor, full-recompute ceiling on every cell | B2 |
| Singleton split: a track the method must fail | Must-not-flag track: legitimate serving changes the verifier must pass | B3 |
| Utility as a function of support count, not a binary | Detection as a function of deviation support, not one AUC per attack | B4 |
| Quarterly regeneration with per-release validation | Versioned calibration releases with validation gates and a measured half-life | B5 |
| Freshness against pretraining contamination | Probe freshness against a provider that recognizes the probe set | B6 |

The paper's MAUVE-versus-QA finding (distributional similarity stays high while
specific facts vanish) is the same shape as the adversarial-temperature result
in this repo (cross-entropy blind, `token_difr` at 1.0). That is a citation, not
an experiment, and belongs in the paper's related work.

Nothing below needs the DP machinery, the training recipes, or the
generator/evaluator matching argument. Those do not transfer.

---

## Ordering and total cost

| # | Experiment | GPU cost | Needs | Reuses |
|---|---|---|---|---|
| B2 | Reference triple on every cell | 0 (numpy over cached scores) | nothing | `benign_shape_dprime.json`, `pool_scaling.json`, `baseline_headroom.json` |
| B1 | Recompute-dependency screen | ~0.5 GPU-day (real NF4/INT8 cells) | one H100 | `exp_gpu`, `exp_robustness_gpu` score arrays |
| B3 | Must-not-flag track | single-card arms ~2 GPU-h; cross-GPU needs A100 + L40S | the multi-machine plan | `exp_benign_shape_dprime_gpu` |
| B4 | Deviation-support curve | ~1 GPU-day | one H100 | `infogain.py`, `rescore_at_budget` |
| B5 | Calibration releases + half-life | ~0.5 GPU-h per collection, repeated over days | one host kept for a week | `ivgym calibrate`, audit protocol |
| B6 | Probe freshness | (a),(b) CPU only; (c) ~0.5 GPU-day | a traffic-like prompt corpus | `audit/protocol.py` commitment + selection |

B2 first because it is free and every later table must carry it. B1 second
because it decides which cells the other four are run on. B3 and B5 are the two
that decide deployment, in the same sense that item 1 of NEXT_EXPERIMENTS does.

---

## B1. Recompute-dependency screen: which cells belong in the benchmark

**ContinuousBench move.** A task is admitted only if the base checkpoint scores
near chance on it and training on the corpus solves it. Saturated tasks (IMDB at
94% before any training) are excluded because they cannot distinguish methods.

**Claim at risk.** That the current attack suite ranks verifiers at all. The
pilot flagged temperature override 10/10 and real NF4 0/10. A suite where every
verifier passes the first and fails the second ranks nothing; the ordering of
verifiers lives entirely in the cells between those two, and nobody has listed
them.

**Protocol.**

1. Fix one pool configuration under the `EvalConfig` ceiling (pool/batch ratio
   ≤ 10%, batch sized by `signal.batch_for_pauc` from a pilot d′).
2. For every attack in the registry (`attacks.py`: `quant_4bit`, `quant_2bit`,
   `kv_fp8`, `temp_1.1`, `seed_43`, `bug_k2`, `bug_k32`, `adv_quant_temp`) plus
   three real-weight cells (NF4, INT8, AWQ via `bitsandbytes`/AutoAWQ) and one
   real-sampler cell (served top-p 0.9 against claimed 1.0), score the pool with:
   - the cheapest Tier-0 verifier for that attack family (`surface_stat`,
     `surface_rank`, `surface_tokens`, `accept_rate`), and
   - full-budget Tier-1 (`token_difr`, `cross_entropy`, `activation_difr`,
     `token_toploc`).
3. Classify each attack by two numbers, Tier-0 best pAUC and Tier-1 best pAUC,
   both at `max_fpr = 0.005`:

   | class | rule | role in the benchmark |
   |---|---|---|
   | saturated | Tier-0 best ≥ 0.90 | excluded from ranking; kept as a smoke test |
   | recompute-dependent | Tier-0 best ≤ 0.55 and Tier-1 best ≥ 0.75 | **the ranking set** |
   | unresolved | Tier-1 best < 0.60 at the affordable pool | reported as unresolved with the pool it would need |
   | floor-limited | d′(attack) ≤ d′(benign floor from B3) | excluded; the honest null covers it |

4. Score each verifier on the ranking set only, by **ceiling fraction
   recovered per reference-GPU-second**: `pAUC(verifier, budget) / pAUC(best
   full-recompute)` divided by `reference_gpu_seconds` from the audit cost
   record. That is the leaderboard metric.

**Deliverable.** `docs/results/recompute_dependency.json` and a table in the
research guide: one row per attack, its class, and the Tier-0/Tier-1 numbers
that put it there. Script: `experiments/exp_recompute_dependency_gpu.py`,
numpy-only after generation.

**Falsifies the benchmark if:** the ranking set is empty at any pool this repo
can afford. Then the suite is either saturated or unresolved, and "run them
together" has never been tested on a cell where it matters. That outcome is
itself the result and should be published as such.

**Cost.** Cached score arrays cover the simulated attacks. The four real cells
are ~0.5 GPU-day on one H100.

---

## B2. The reference triple on every cell

**ContinuousBench move.** Every configuration reports three reference runs: no
training (floor), train on real (ceiling), and non-private synthesis (the
best the method family can do). No number is reported alone.

**Claim at risk.** Every AUC table in `docs/results/`. The batch/pool ratio
artifact and the five reversed cells in `exp_reversal_check_gpu` were both
numbers published without a bracket. The bracket exists in the repo
(`benign_shape_dprime.json`, `baseline_headroom.json`) but is not attached to
the results it would have caught.

**Protocol.** Extend `harness.EvalResult` with three fields that `evaluate`
fills on every call:

| field | definition | expected value |
|---|---|---|
| `dprime_null` | d′ of the honest pool against a replay of itself at the same forward-pass shape | exactly `0.0`; anything else is a determinism bug |
| `dprime_floor` | largest *positive* d′ across the benign schedules of B3 (batched, kernel swap, token-by-token) for this verifier and model | the number the attack must clear |
| `dprime_ceiling` | d′ of the best full-recompute verifier on this attack at the pool `batch_for_pauc` nominates | what any budgeted verifier is recovering a fraction of |

and one derived field, `headroom = (dprime_attack - dprime_floor) /
dprime_ceiling`, clipped at 0. `paper/make_tables.py` prints the triple in
every table or refuses to build. Cells with `headroom == 0` are labelled
floor-limited and fed back to B1.

**Deliverable.** The field additions, a migration of the committed JSON results
to carry them where the inputs exist, and a test in `tests/` asserting
`dprime_null == 0.0` on the synthetic backend.

**Falsifies nothing by itself.** It is discipline, not a measurement. The
acceptance criterion is that no table in `paper/tables/` lacks the triple.

**Cost.** Zero GPU. One afternoon of harness work.

---

## B3. The must-not-flag track

**ContinuousBench move.** The Geminon singleton split contains facts that appear
once in the corpus. A DP method is *supposed* to fail on it, and the split is
included so that a method cannot score well by doing the wrong thing.

**Claim at risk.** The deployment claim that a 5% per-audit threshold yields a
5% false-positive rate. The honest null in every committed result is "the same
model on the same H100". A provider legitimately changes batch shape, kernels,
serving stack, and hardware without changing the model, and a verifier that
flags those is a verifier that accuses honest providers.

**Protocol.** A fixed list of legitimate serving changes, each a condition under
which the *claimed* model is served honestly:

| condition | single card? | status |
|---|---|---|
| batch of 4 and 16 with unrelated rows | yes | measured (`exp_benign_shape_dprime_gpu`) |
| attention kernel `sdpa` / `flash` / `eager` | yes | measured |
| token-by-token through a KV cache | yes | measured, d′ = −0.034 |
| chunked prefill at 64 / 128 / 512 | yes | partly (`exp_specdec_shape_gpu`) |
| same-family speculative decoding with correct rejection sampling | yes | not measured as a benign condition |
| vLLM continuous batching + paged KV vs. HF eager | yes | not measured |
| bf16 vs. fp16 reference weights | yes | not measured |
| A100 and L40S serving, H100 reference | no | the open arm of NEXT_EXPERIMENTS item 1 |
| different CUDA / torch / vLLM version on the same SKU | no | not measured |

For each condition, N = 100 fresh honest blocks (the pilot's block format,
96 generated tokens, two selected responses) are scored against the frozen
H100 calibration from the audit pilot, at the deployed operating point
(`alpha = 0.05`, Bonferroni across the two scores). Two pass criteria, both
required:

1. **Flag rate.** Flags on the N blocks ≤ α with the Clopper–Pearson 95%
   interval not excluding α. At N = 100 and α = 0.05 the upper bound of a pass
   is about 10 flags.
2. **Effect size.** Positive-signed `d′(condition)` ≤ 0.25 · d′ of the weakest
   attack in the B1 ranking set, measured on the same `evaluate` scale.

A verifier passes the track only if it passes every condition. A verifier that
fails a condition is not disqualified; it is labelled "requires per-condition
calibration", and B5 then measures what that costs.

**Deliverable.** `docs/results/must_not_flag.json`, a per-verifier pass/fail
grid, and a section in the research guide replacing "the honest null is not one
thing" with the track and its criteria. Script:
`experiments/exp_must_not_flag_gpu.py`, taking `--condition` so the
cross-hardware arms run from `scripts/run_research_job.py` on the A100 and L40S
hosts in the multi-machine plan.

**Falsifies the deployment story if:** any single-model condition produces a
positive d′ at or above `d′(quant_4bit) = 0.075`. Then the audit protocol must
carry the serving configuration in `deployment.json` as a *calibration key*,
not just a manifest, and the cost of a verdict in `cost_of_a_verdict.json`
multiplies by the number of configurations a provider legitimately runs.

**Cost.** Single-card conditions ~2 GPU-hours total. Cross-hardware arms need
the A100 and L40S hosts for ~1 hour each, which the multi-machine plan already
budgets.

---

## B4. Detection as a function of deviation support

**ContinuousBench move.** Instead of one accuracy per method, utility is
plotted against the number of independent records supporting each fact. The
threshold where DP synthesis separates from no-training (≈600 records) is the
paper's most informative figure, and it falls out of stratification, not of a
new method.

**Claim at risk.** Two of them. First, that selective (information-directed)
verification is dead: `infogain.py` was anti-correlated with the deviation on
Gaussian logit noise, which spreads the deviation over every position. Second,
that per-attack AUC is the right unit of report. A verifier's power depends on
how many tokens in a block carry the deviation, and no table in the repo
stratifies by that.

**Protocol.**

1. **Define support.** For a block scored by full recompute, the deviation
   mask is the set of positions where per-token `KL(M ‖ M̂)` exceeds τ, with τ
   set so that honest replay yields support ≈ 0 (99.9th percentile of the
   same-shape null from B2). Support `s` is the count of masked positions in
   the block.
2. **Controlled arm (synthetic shape).** Apply `attacks.Quantization` noise to
   a fraction `f ∈ {0.01, 0.03, 0.10, 0.30, 1.0}` of positions, scaling σ so
   the *mean* KL per block is matched across `f`. Same divergence budget,
   different concentration. This is the sparse-versus-diffuse question from
   NEXT_EXPERIMENTS item 2, run as a sweep rather than a single Gini number.
3. **Real arm (measured support).** Real NF4, real INT8, and mixed-precision
   (quantize the 4, 8, 16 layers with the smallest logit effect per FLOP
   saved, the weight-side adversary of item 3). Support is measured, not set.
4. **Score** each pool with every Tier-1 verifier at budgets
   `{0.05, 0.1, 0.25, 0.5, 1.0}` through `harness.verify(..., budget=b)` and
   `rescore_at_budget`, selection driven by proxy entropy (the default) and by
   `infogain`.
5. **Plot** pAUC against support, one line per verifier × budget, with the
   reference triple from B2 as horizontal rules. Bucket the real arm by
   measured support the way the paper buckets News by support count.

**Deliverable.** `docs/results/support_curve.json`, `plot_support_curve.py`,
and the number the paper currently lacks: the minimum support at which each
verifier reaches pAUC 0.90 at the deployed FPR, per budget. Script:
`experiments/exp_support_curve_gpu.py`.

**Falsifies the retirement of `infogain.py` if:** at `f ≤ 0.03` with matched
mean KL, selective recompute at budget 0.1 beats uniform recompute at budget
0.1 by more than the bootstrap interval. Then the earlier negative result was
scoped to diffuse deviations, and the information-directed story returns for
the sparse ones.

**Falsifies the current reporting unit if:** the support-stratified curves for
two verifiers cross. Then a single pAUC per attack hides which verifier to
deploy against which deviation shape, and the tables must stratify.

**Cost.** ~1 GPU-day on one H100, most of it generating the real-arm pools.

---

## B5. Calibration releases with validation gates and a measured half-life

**ContinuousBench move.** The benchmark is a versioned package regenerated on a
schedule, and every release passes two automated validation checks before
publication: the base model is near chance on it, and training on the corpus
solves it. Contamination is treated as something that accumulates, so the
answer is a pipeline, not a one-time cleaning.

**Claim at risk.** That the frozen H100 calibration from the audit pilot still
describes the honest null a week later, or after a driver, torch, or vLLM
update on the same host. B3 shows that forward-pass shape moves the logits as a
step; software versions change shape silently. A stale calibration raises the
honest flag rate without any attack and without any notice.

**Protocol.**

1. **Bundle format.** `ivgym calibrate` already writes a calibration file for
   the service. Extend it to a *release*: model id and weight digest, hardware
   SKU, software manifest (driver, CUDA, torch, vLLM/HF versions), serving
   configuration from B3's condition list, N fresh honest blocks, creation
   time, and a `validation` record.
2. **Validation gates**, run by a new `ivgym validate-calibration`:
   - *null gate*: same-shape replay of 20 of the bundle's own blocks gives
     `dprime_null == 0.0`;
   - *held-out gate*: 40 fresh honest blocks not in the bundle flag at ≤ α
     with a Clopper–Pearson interval containing α;
   - *canary gate*: two saturated attacks from B1 (`temp_1.1`,
     `quant_2bit`) score within the bootstrap interval of their B1 d′.
   A bundle that fails any gate is not installable through
   `calibration_files`; the service returns `no_applicable_calibration`.
3. **Half-life measurement.** On one H100 host, collect a release on day 0.
   Collect 100 fresh honest blocks on days 0, 1, 3, 7, and once more after a
   deliberate torch or vLLM minor-version bump. Score every later set against
   the day-0 release and record the flag rate. Repeat with the day-7 release
   as the reference to separate drift from day-0 sampling noise.
4. **Report** flag rate against calibration age, per verifier score
   (`unfiltered_nll_v1`, `capped_log_rank_v1`), and the age at which it exceeds
   2α. That age is the release cadence.

**Deliverable.** The release format and gate command, `docs/results/
calibration_half_life.json`, and a one-line rule in the service docs: "a
release older than *T* days, or from a different software manifest, is not
applicable". Scripts: `ivgym/service/calibrate.py` extension plus
`experiments/exp_calibration_half_life_gpu.py`.

**Falsifies the cost model if:** the day-7 flag rate against the day-0 release
exceeds 2α on the same host with no software change. Then calibration must be
regenerated per audit, and `cost_of_a_verdict.json` must add the calibration
collection cost to every verdict rather than amortizing it.

**Falsifies the manifest-as-metadata assumption if:** the post-upgrade flag
rate exceeds 2α while same-day collections do not. Then the software manifest
is part of the calibration key, as hardware already is in B3.

**Cost.** ~0.5 GPU-hour per collection, six collections, plus keeping one
instance for a week. Dollars, not GPU-days.

---

## B6. Probe freshness against a provider that recognizes the probe set

**ContinuousBench move.** Freshness is enforced by post-cutoff collection or
procedural regeneration, and *validated* by checking the base model is near
chance. The threat is a model that already knows the test.

**Claim at risk.** The threat model's silence on selective honesty. Every
experiment in the repo uses one prompt domain, and the audit pilot sends
probes the provider could, in principle, recognize. A provider that serves
`M` on recognized probes and `M̂` on everything else passes every verifier at
zero detection cost to itself. This is the cheapest adaptive attack of
NEXT_EXPERIMENTS item 3 and needs no query access to the verdict.

**Protocol.** Three parts, the first two without a GPU.

1. **Distinguishability.** Assemble a traffic-like prompt sample (a public chat
   prompt corpus, deduplicated, ≥ 5,000 prompts) and the repo's current probe
   prompts. Train a cheap classifier on features a provider sees before
   generation: prompt length, token count, perplexity under a 0.6B proxy,
   presence of templating artifacts, request timing regularity. Report its
   held-out AUC. That number is how easy the current probes are to spot.
2. **Regeneration.** Write a probe generator that draws probes per audit from
   the traffic distribution (sample a real prompt, apply a seeded paraphrase
   and entity swap from the audit's `selection_commitment` seed so the probe is
   unique and committed), and rerun step 1. Target: classifier AUC ≤ 0.60.
   The commitment already exists in `audit/protocol.py`; this adds the probe
   source.
3. **Selective-honesty attack.** In the HF backend, implement a provider that
   runs the step-1 classifier and serves `M` when it predicts "probe", `M̂`
   (real NF4 or `quant_4bit`) otherwise. Sweep the classifier's operating
   point so its probe-recall runs from 0.5 to 0.99, and for each point measure
   detection pAUC of the full Tier-1 portfolio under (a) the old fixed probes,
   (b) the fresh probes from step 2, and (c) post-closure selection over real
   traffic transcripts, which the protocol supports but the pilot did not use.

**Deliverable.** A power-versus-distinguishability curve, three lines, in
`docs/results/probe_freshness.json`, and the probe generator wired into
`AuditRunner`. Scripts: `experiments/exp_probe_distinguishability.py` (CPU),
`experiments/exp_selective_honesty_gpu.py`.

**Falsifies the fresh-probe fix if:** line (b) sits near line (a) when the
provider's classifier has AUC 0.70. Then paraphrased probes are still
recognizable and the only robust design is (c): verify a committed random
sample of the provider's actual traffic, which changes the protocol from
"send probes" to "audit transcripts" and must be stated as such.

**Falsifies the threat if:** line (a) already holds up at classifier AUC 0.9,
which would mean verification of even recognized probes is enough because the
provider cannot afford to serve `M` on them. That would be surprising and
would narrow the adaptive-attacker item.

**Cost.** Parts 1 and 2 are CPU only. Part 3 is ~0.5 GPU-day.

---

---

# Part II: verdict packaging (SentryWard)

What SentryWard gets right, and IVGym currently leaves to prose or to free-text
`reason` strings in `ivgym/service/engine.py`:

| SentryWard | IVGym today | Item |
|---|---|---|
| 14 closed result categories in `config/result-categories.json`, per-check status, precedence rules | `inconclusive` / `inconsistent` / `no_deviation_detected` / `unsupported` + ad-hoc reasons, assigned inline | S1 |
| Assumption pool + defaults file + per-assumption enabled/disabled behavior; disabling never yields a silent pass | Exchangeability, single serving config, no mixed routing stated in docs only | S2 |
| Every finding carries a witness (the two unordered accesses); diagnostics have `code`, `stage`, `details` | Verdicts carry p-values and scores; evidence is the whole signed transcript | S3 |
| `reference_binding.json`: explicit input/output correspondence, "dtype is never guessed", ambiguity is an error | Raw `/completions` only; template and tokenizer handling is the open `/chat/completions` roadmap item | S4 |
| `examples/results.json`: schema version, per-category counts, artifact paths, a pointer to the field each number came from; `make rerun-examples` | `docs/results/*.json` with per-experiment shapes; claim-to-artifact index is a prose section | S5 |
| Greedy removal of synchronization, kept if the kernel still verifies | "Fewer tokens for the same verdict" measured by budget sweep, not by minimal set | S6 |
| `skills/prepare-kernel-input` (with an audit script) and `skills/generate-kernel-report` | No agent-facing skills | S7 |

Order: S1 and S2 first because S3, S4 and S5 all write into the schema they
define. S6 is the only GPU-free experiment and can run any time. S7 last.

| # | Item | Cost | Touches |
|---|---|---|---|
| S1 | Closed verdict schema | one day | `service/engine.py`, `audit/protocol.py`, new `config/result_categories.json`, `docs/VERIFICATION_SERVICE.md` |
| S2 | Assumption pool | half a day | new `config/assumption_pool.json`, `config/default_assumptions.json`, verdict JSON |
| S3 | Witness objects | one day | `verifiers.py` evidence → verdict; `audit/protocol.py` replay |
| S4 | Reference binding | two days | new `audit/binding.py`, `deployment.json` contract, live adapter |
| S5 | Results index + rerun target | one day | `docs/results/index.json`, `tools/rerun_examples.py`, CI |
| S6 | Minimal-witness study | 0 GPU, numpy over cached scores | `experiments/exp_minimal_witness.py` |
| S7 | Agent skills | half a day | `skills/prepare-audit/`, `skills/explain-verdict/` |

---

## S1. A closed verdict schema with per-check status and precedence

**SentryWard move.** One result field drawn from a closed list, a `checks`
object recording each check separately as `passed`, `failed`, `incomplete`,
`not_run` or `not_requested`, a legacy alias map so old names resolve to
current categories, and three rules: a safety failure dominates the numerical
result; a budget stop retains the checks that already completed; a missing
field is never evidence that a check passed or a count was zero. The README
warns that exit code 0 can accompany findings, so callers read `result` and
`checks`, never the exit code.

**Gap.** `engine.py` assigns `status` strings in at least six places with
reasons such as `token_evidence_missing`, `mixed_timing_units` and
`incomplete_or_unpaired_clock_probes`. `protocol.py` computes an `outcome` on
its own. There is no single list, no per-channel status, and no stated
precedence between an artifact-consistency failure and a statistical verdict.

**Change.**

1. `config/result_categories.json` with the closed list:

   | result | meaning |
   |---|---|
   | `no_deviation_detected` | every requested channel passed at the calibrated operating point |
   | `inconsistent` | at least one requested channel rejected the null at the operating point |
   | `artifact_invalid` | signature, digest, commitment or replay check failed; dominates any statistical result |
   | `inconclusive` | channels ran but no applicable calibration, or resolution below the requested FPR |
   | `unsupported` | a required input form, evidence type or endpoint feature is not handled |
   | `invalid_input` | malformed capture, spec, deployment or binding |
   | `missing_input` | a required file, calibration or binding is absent |
   | `reference_budget_exceeded` | the reference-GPU budget expired before the verdict |
   | `timeout` | the wall-clock budget expired |
   | `tool_failure` | crash, invalid native output, I/O failure |

   plus a `legacy` map from every string currently emitted in `engine.py` and
   `protocol.py` to one of these.
2. `checks: {token, clock, artifact_consistency, binding}` each in the
   five-state status vocabulary, with `not_requested` for channels the caller
   did not ask for.
3. Precedence, written in `docs/VERIFICATION_SERVICE.md`: `artifact_invalid`
   > `invalid_input` / `missing_input` > `unsupported` > statistical result;
   a budget stop keeps completed channels in `checks` and reports the stop as
   `result`.
4. A `schema` field (`ivgym.verdict.v1`) on every verdict and audit JSON.

**Acceptance.** A test in `tests/` loads every verdict JSON committed under
`docs/results/` and the audit pilot's `audit_protocol_h100.json`, maps each
through the legacy table, and asserts every `result` is in the closed list and
every channel in `checks` is in the five-state vocabulary. The service refuses
to start if `engine.py` emits a string not in the list.

---

## S2. Assumptions as a configurable pool that every verdict records

**SentryWard move.** `config/assumption-pool.json` lists each assumption with a
key, label and one-sentence text; `config/default-assumptions.json` sets
defaults; `docs/assumptions.md` states the enabled and disabled behavior for
each. Disabling an assumption turns affected checks into `unsupported` with a
suggestion to enable it. It never silently passes.

**Gap.** IVGym's deployment claims rest on assumptions that appear only as
prose: calibration traffic is exchangeable with audited traffic, the provider
serves one configuration, no mixed routing across models, the provider is not
adapting to the verifier, the clock sees no traffic shaping. The pilot report
says "they assume" once, in a footnote-level sentence.

**Change.** `config/assumption_pool.json`:

| key | text | default | if disabled |
|---|---|---|---|
| `calibration_exchangeable` | Audited blocks are exchangeable with the installed calibration blocks | true | statistical channels return `inconclusive` |
| `single_serving_config` | The provider served one hardware and software configuration for the audit epoch | true | token channel requires per-condition calibration from B3; else `inconclusive` |
| `no_mixed_routing` | Every audited request was served by the claimed model, or none were | true | verdict is per-block, no epoch-level aggregation |
| `verifier_blind_provider` | The provider did not adapt its serving to the probe set | true | probes must come from the B6 fresh generator; else `unsupported` |
| `clock_no_traffic_shaping` | Timing intervals reflect device execution plus a stationary network term | true | clock channel `not_run` |
| `reference_is_claimed_checkpoint` | The reference weights match the claimed model digest | true | cannot be disabled; recorded only |

Every verdict carries `assumptions: {key: bool}` as in force at scoring time,
and `deployment.json` may override defaults. The B3 and B5 experiments are the
measurements of `single_serving_config` and `calibration_exchangeable`
respectively, and their results should update this file's text, not just the
research guide.

**Acceptance.** `docs/ASSUMPTIONS.md` generated from the pool file, one section
per key with enabled and disabled behavior; the verdict test from S1 also
asserts the `assumptions` object is present and keyed by the pool.

---

## S3. Witness-carrying findings

**SentryWard move.** A race finding names the two unordered accesses. Each
diagnostic has a `code`, a `stage`, and `details` holding operands, source sites
and resource evidence. The docs instruct consumers to preserve original
structured witnesses and never to read only the first diagnostic.

**Gap.** An `inconsistent` verdict says which score rejected and at what
p-value. The evidence is the entire signed transcript bundle. A provider who
wants to dispute the flag has nothing smaller than "replay everything" to
dispute, and a third party cannot check the flag without the full reference
run.

**Change.** Each flagged block carries a `witness`:

```json
{
  "block_id": "fresh-17",
  "score": "capped_log_rank_v1",
  "statistic": 3.41, "p_value": 0.004, "threshold": 0.025,
  "positions": [
    {"index": 41, "token": 1172, "reference_logprob": -0.31,
     "observed_rank": 9, "contribution": 0.62},
    {"index": 57, "token": 318,  "reference_logprob": -2.80,
     "observed_rank": 214, "contribution": 0.41}
  ],
  "positions_truncated": true, "positions_complete": false,
  "selection_path": "commitment/fresh-17/response-1"
}
```

`positions` holds the top-k contributors by per-token evidence (the `evidence`
array every `Verifier` already produces), with `contribution` as the share of
the block statistic. `positions_truncated` and `positions_complete` follow the
SentryWard rule that a short list is not evidence of a short cause. A
`witness_replay` mode in `audit/protocol.py` recomputes only the witnessed
positions against the reference and checks they reproduce the listed
statistics, which costs a prefill of each witnessed sequence rather than the
whole audit.

**Acceptance.** `ivgym audit replay --witness-only` reproduces every witness in
the pilot artifact bundle within floating-point tolerance, and the research
guide shows one witness end to end.

---

## S4. A reference binding for endpoints

**SentryWard move.** `reference_binding.json` maps the kernel's allocations,
parameters and outputs to the Python reference's arguments, with a schema
version, explicit layouts, and two rules: dtype is never guessed from byte
width, and a missing or ambiguous value is an error, not a default.

**Gap.** The live adapter speaks raw `/completions`. The roadmap lists
`/chat/completions` "with explicit handling of templates and hidden
preprocessing" as open. Without a binding, a chat-template mismatch, a
tokenizer revision drift, an injected system prompt, a stop-sequence
difference or a logprob post-processing step all surface as *deviation
evidence* rather than as a binding error. That is a false-positive class that
no calibration fixes.

**Change.** `audit/binding.py` and a `binding` section of `deployment.json`
with schema `ivgym.endpoint-binding.v1`:

| field | meaning | rule |
|---|---|---|
| `endpoint_kind` | `completions` or `chat_completions` | required |
| `chat_template_sha256` | digest of the template the reference applies | required for chat; never inferred from the model name |
| `tokenizer_revision` | tokenizer files digest | required; a mismatch is `invalid_input` |
| `special_tokens` | bos/eos/pad ids and whether bos is added | required |
| `system_prompt` | provider-injected prefix, or explicit `null` | required; absence is `missing_input`, not "none" |
| `stop_sequences` | what terminates the provider's generation | required |
| `logprob_transform` | `raw`, `top_k_renormalized`, `temperature_scaled` | required for logprob-bearing captures |
| `request_fields_ignored` | provider fields the reference does not model | listed, so they cannot hide a deviation |

The audit runner builds the reference input from the binding and the capture,
and `checks.binding` is `failed` with `result = invalid_input` when any required
field is absent or the applied template does not reproduce the provider's
echoed prompt tokens. A binding failure is reported before any token is scored,
so it can never be confused with `inconsistent`.

**Acceptance.** The chat endpoint arm of the audit protocol runs against a
local vLLM serving Qwen3 with its own chat template; a deliberately wrong
template digest returns `invalid_input` with `checks.token = not_run`, and the
correct one returns the same token statistics as the raw-completions path on
the same prompts.

---

## S5. A results index with provenance, and a rerun target

**SentryWard move.** `examples/results.json` has a schema version, a
`verification_results_source` with repository, revision, path and sha256, a
`counts` object over result categories, and one record per example with
artifact paths and `analysis_time_source` pointing to the exact field the
number was read from. `make rerun-examples` re-runs the corpus and diffs
categories. Narrative reports are generated on demand, not committed.

**Gap.** `docs/results/` holds about forty JSON and text artifacts with
per-experiment shapes. The claim-to-artifact index lives in a prose section of
the research guide, and `paper/make_tables.py` knows the mapping implicitly.
Nothing checks that a committed number still reproduces from its artifact.

**Change.**

1. `docs/results/index.json` with schema `ivgym.results-index.v1`: one record
   per experiment giving the script, the commit it ran at, the model and
   hardware, the artifact paths, the result categories it produced (from S1),
   and for every number quoted in the paper a `{path, field}` pointer.
2. `tools/rerun_examples.py`: re-executes the CPU-replayable experiments
   (`quickstart`, clock replay, `analyze_h100_campaign`, every `plot_*`,
   the S3 witness replay) from committed artifacts and diffs the produced
   categories and headline numbers against the index. Runs in CI on CPU.
3. `paper/make_tables.py` reads the index rather than hard-coded paths, and
   refuses to build on a pointer that does not resolve.

**Acceptance.** CI job green on a clean checkout with no GPU; a deliberately
edited number in one artifact turns it red.

---

## S6. Minimal-witness study: greedy removal while the verdict holds

**SentryWard move.** Remove one synchronization operation at a time; keep the
removal if the kernel still verifies. The residue is the set of operations the
proof actually needs, and the caveat is explicit: it holds for the tested
shapes.

**Claim at risk.** The "fewer tokens for the same verdict" section of the
research guide measures power at fixed budgets. It does not say how small the
sufficient evidence is for a given attack, nor whether the information-directed
selector picks the tokens that a minimal set contains.

**Protocol.** Numpy only, over the cached per-token score arrays of every B1
ranking-set cell.

1. Start from a block that is flagged at the deployed operating point under
   full recompute.
2. Greedily remove the token with the smallest per-token evidence; recompute
   the block statistic and p-value from the remaining tokens with the
   calibration null re-derived for the smaller count; keep the removal if the
   block is still flagged.
3. Stop when no further removal keeps the flag. The residue is the minimal
   witness for that block under that order. Repeat with a random removal order
   to bound order dependence.
4. Report, per cell and verifier: minimal witness size as a fraction of the
   block, its overlap with the top-k positions that `infogain` and proxy-entropy
   selection would have chosen at that budget, and the overlap with the B4
   deviation mask.

**Falsifies the selector if:** the minimal witness overlaps the
entropy-selected set no better than a random set of the same size on the
sparse cells of B4. Then selective recompute is not finding the evidence that
exists, which is a different failure from the evidence not existing.

**Falsifies the block size if:** minimal witnesses are consistently below 10%
of the block on the recompute-dependent cells. Then the audit protocol's block
length is set by calibration, not by evidence, and the cost model should say
so.

**Cost.** Zero GPU. The caveat from SentryWard carries over verbatim: a minimal
witness holds for the tested pool and calibration; another calibration may need
tokens that were removable here.

---

## S7. Agent-facing skills

**SentryWard move.** `skills/prepare-kernel-input/` includes a `SKILL.md`, a
reference folder and an `audit_input.py` script that validates the launch
configuration before the expensive run. `skills/generate-kernel-report/`
produces the narrative from `output.json` on demand, which is why narrative
reports are not committed.

**Gap.** Preparing an audit requires reading several docs to assemble a spec,
a deployment record, a calibration bundle and now a binding, and the only
validation is the audit runner failing at `authenticate`. Verdict JSON is
explained in prose by hand.

**Change.** Two skills, mirroring the structure:

- `skills/prepare-audit/`: `SKILL.md` plus `scripts/audit_inputs.py`, which
  checks `spec.json`, `deployment.json`, the S4 binding and the installed
  calibration against the S1/S2 schemas and the B5 validation gates, and prints
  what would be `missing_input` or `invalid_input` before any GPU time is spent.
- `skills/explain-verdict/`: turns a verdict JSON (result, checks, assumptions,
  witnesses) into a narrative that states the result category, which checks
  ran, which assumptions were in force, and what the witness shows, in that
  order. Committed results stay JSON; narratives are generated.

**Acceptance.** `audit_inputs.py` run on the pilot bundle reports clean;
removing `calibration.json` reports `missing_input` with the field name.

---

## A framing point for the paper, from the equivalence engine

SentryWard proves equivalence *in real arithmetic* and states that
floating-point is treated as real numbers. That is the exact boundary IVGym
measures from the other side. The benign floor of B3 is floating-point
non-equivalence among kernels that are real-arithmetic equivalent: batch
shape, attention kernel, chunked prefill. The attacks are real-arithmetic
non-equivalence: NF4 weights, fp8 KV, a different sampler compute a different
function. Their corpus contains both kinds of kernel (f16 TMA GEMMs next to
mxfp4 and fp8 variants), which is the same split.

Two consequences, neither an experiment:

- B3's membership rule for "legitimate serving change" has a principled form:
  the kernel is real-equivalent to the reference. Anything else is a candidate
  attack, whatever the provider calls it.
- The tools are complementary, not competing. A provider can publish a static
  certificate that its serving kernels are real-equivalent to the reference
  implementation; an outside auditor still needs the statistical test, because
  the certificate says nothing about which weights were loaded or how the
  sampler was configured. That sentence belongs in related work.

---

# Closing

## How the results feed the paper

- B1's table replaces the current flat attack × verifier grid with a classed
  one, and gives the leaderboard its inclusion rule.
- B2's triple appears in every table, which closes the pool-ratio class of
  error for good.
- B3 and B5 together are the false-positive side of the deployment claim: B3
  across configurations, B5 across time. Item 1 of NEXT_EXPERIMENTS becomes
  one row of B3.
- B4 either revives selective verification for sparse deviations or retires it
  with a stated scope, and adds the "minimum evidence for a verdict" figure.
- B6 is the first adaptive adversary in the repo and the cheapest one.
- S1 and S2 turn the deployment caveats into a schema the API enforces, so the
  paper can say "under assumptions {…}" and point at a file.
- S3 and S4 remove two false-positive classes that no calibration fixes:
  uncontestable flags and binding errors scored as deviations.
- S5 makes every number in the paper reproducible from a committed artifact
  on CPU, in CI.
- S6 adds the minimal-evidence number next to B4's support curve.
- The real-arithmetic framing gives B3 its membership rule and the related
  work a sentence on static kernel certification.

## What this document does not propose

- No DP, no training, no generator/evaluator matching. The reference model is
  already required to be the exact claimed checkpoint.
- No SMT, happens-before or PTX machinery. Nothing in SentryWard's verifier
  runs against a black-box endpoint.
- No new verifier. Every experiment above scores the existing registry; the
  benchmark and the verdict contract are the contributions.
- No leaderboard hosting until B1 shows the ranking set is non-empty.
