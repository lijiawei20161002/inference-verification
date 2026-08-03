# Learned triage and the real cost of a selective audit

> **STATUS (2026-07-30). Both parts are measured; the study is complete.** One
> port transfers and one does not.
>
> * **Part 2 — the prefix scheduler: a clean win.** A "5%" top-k audit really costs
>   76% of a full audit and *more wall-clock than auditing everything*; the prefix
>   schedule spends exactly its budget, 16× faster at a 5% budget, and at matched
>   real cost it detects 0.12–0.20 AUC better (2–4 sd) than top-k points that cost
>   *more*, in the band where top-k has already paid for every prefill. Geometry
>   and a stopwatch, reproduced at two scales.
> * **Part 1 — the learned confidence head: a clean negative.** With real headroom
>   (full recompute 0.867), the head does not beat `entropy`, the hand-crafted
>   default it was ported to replace — not as a ranking key, and not as the value
>   term inside the scheduler it was ported alongside. Its calibration stage buys
>   nothing out of sample either.
> * **The detour that reframed everything (§ 3).** Chasing Part 1's flat baseline
>   found that this repo's published AUCs — including the README headline — were
>   inflated by resampling batches from too small a token pool. 0.977 and 0.530 are
>   the same measurement at two batch/pool ratios. That finding outlived the
>   experiment that prompted it.
>
> Two earlier readings are **withdrawn** and kept on the record with their logs: the
> 32 × 96 run in which the head won, and the pre-winsor-fix table in which a
> selective audit beat full recompute. See "Measured results" and "Where this
> stands" at the bottom.

*Two things ported out of DSpark (arXiv:2607.05147) into the verification game: a
learned, calibrated **confidence head** replacing the hand-crafted triage value
signals, and a **prefix scheduler** built on the cost model a selective audit
actually pays. Ties together `ivgym/triage.py`, `ivgym/harness.py`
(`select_prefix_scheduled`, `prefill_cost`), `ivgym/backends/hf_gpu.py`
(`lazy_reference`), and the two experiments below.*

## Why DSpark at all

DSpark verifies to go **faster** against a model it trusts; `ivgym` verifies to
**catch a provider that lied**. The accept/reject step is correctness-preserving
for them and a hypothesis test for us, so nothing transfers wholesale. Two
components do transfer, because they are about *scheduling a scarce verification
budget* and are indifferent to why you are verifying:

1. the **confidence head** — a learned, BCE-trained, post-hoc-calibrated estimate
   of per-position value, computed from cheap draft-side features; and
2. the **hardware-aware prefix scheduler** — greedy admission over a globally
   sorted pool, keyed on value per unit of *marginal cost*, committing to
   contiguous prefixes.

One component does **not** transfer: DSpark's DFlash backbone injects key-value
features from the *target* model's layers into the draft. That requires running
`M`, or trusting the provider to hand over KV features — which contradicts the
premise in `ivgym/spec_decode.py` that the proxy is a *client-owned* anchor and
nothing trusts a provider self-report. The lightweight sequential (Markov/RNN)
head on a client-owned small model is fine; the KV injection is not.

---

## 1. The confidence head as a triage value function

### What it replaces

`harness.verify(budget<1)` audits the top-`budget` fraction of tokens by a cheap
per-token `value`. Those signals were four hand-written guesses
(`verifiers._VALUE_FNS`): `uniform`, `entropy` = `H(q)`, `tie_margin` =
`-(q1-q2)`, `surprisal` = `-log q(x)`. Each encodes a hunch about which positions
carry evidence that `M` was really run.

### The target is NOT DSpark's target

DSpark predicts `P(token is accepted)`. Copying that would be a mistake here. The
batch statistic is a mean of per-token evidence over the audited tokens, so what
decides detection power at a position is the **standardized per-token effect
size**

```
Delta(t) / sigma_h(t),    Delta(t) = E[score | deviating] − E[score | honest]
```

— the per-token d-prime, whose square is the Fisher information the position
contributes. `entropy` is a *surrogate* for this (uncertain positions tend to be
sensitive). The head is trained to predict the thing itself.

### Labels without labeled attacks

`triage.surrogate_sensitivity` needs **honest data only**. For each honest token
it perturbs the reference logit row with a generic zero-mean Gaussian probe — a
stand-in deviation committed to no particular attack — and measures how much the
Tier-1 score moves. That is a Monte-Carlo estimate of `Delta(t)` for the whole
logit-perturbation family, which is how quantization, fp8 KV cache, and a swapped
checkpoint all enter. It reads `ref_logits`, but only during a **one-time offline
fit** — the same trusted-`M`-run amortization `spec_decode.ProxyReference.fit`
already assumes. At inference the head reads the cheap proxy only, so it stays
Tier-0.

`triage.paired_effect_size` is the **oracle** label: the realized per-token
`|s_attack − s_honest| / sigma_honest` from labeled pairs. It is not deployable.
Fitting the *same* nine-feature head on it answers the non-circular question —
how much better would triage be with real deviation labels instead of the
surrogate probe? — and the cross-attack columns show whether that label transfers.

### Features (all Tier-0)

`entropy`, `renyi2`, `tie_margin`, `log_top1`, `top5_mass`, `surprisal`,
`log_rank`, `log_support`, `rel_position`. The first and third reproduce the
existing `entropy` / `tie_margin` signals exactly (asserted in
`tests/test_triage_and_cost.py`), so the head strictly generalizes them.

### Calibration buys something top-k cannot have

`select_triaged` takes a fraction an operator has to pick, and spends it whether
or not the tokens are worth it. `triage.SequentialTemperatureScaling` is DSpark's
post-hoc fit ported over — a global temperature plus per-position-bucket offsets,
fit by NLL on a **held-out honest** split. DSpark calibrates sequentially because
acceptance decays with depth in a draft block; the analogue here is depth in the
generated sequence. Within a bucket the map `z -> z/T + c_k` is strictly monotone,
so calibration cannot reorder tokens — it only makes the number a probability, at
which point admission can be a threshold on expected value rather than a fraction.

Run: `experiments/exp_confidence_head_gpu.py` →
`docs/figures/fig_confidence_head_pareto.png`,
`docs/figures/fig_confidence_head_diagnostics.png`,
`docs/results/confidence_head.json`.

---

## 2. The audit cost model, and what the old number was measuring

### The gap

`TokenScores.recompute_ratio` is the fraction of **tokens** a Tier-1 verifier
scored. It reads like a cost. It is not one. Reading `M`'s logits at generated
position `j` requires a prefill of `[prompt + claimed[:j]]`, so:

* a sequence's audit cost is set by the **deepest** position audited in it, and
* every sequence touched **at all** pays for its prompt again.

`select_triaged` ranks tokens globally and takes the top fraction, which scatters
the audit across essentially every sequence. Measured on 32 sequences × 96 tokens
with 12-token prompts, a **5% token budget costs 84% of a full audit**; 10% costs
92% (`tests/test_triage_and_cost.py::test_topk_token_budget_hides_a_near_full_prefill_cost`).

Worse, the old `HFGPUBackend` prefilled `M` over every full sequence *during
generation* (`_populate_ref_cache`), before any budget was known, so
`reference_logits` was a cache read and no budget could have been measured at all.

### The fix

* `HFGPUBackend(lazy_reference=True)` prefills nothing during generation and
  **raises** if the verifier reads a row it has not paid for. `prefill_reference
  (prompt_id, depth)` does the minimal prefill (`L + depth − 1` tokens) and charges
  marginal tokens on extension, as a KV-cache-extending verifier would.
* `harness.prefill_cost` / `full_prefill_cost` price a mask.
* `TokenScores.prefill_ratio` reports the measured physical cost next to the
  notional `recompute_ratio`.
* `harness.select_prefix_scheduled` is the DSpark scheduler port: greedy admission
  by value-per-marginal-cost, where admitting a sequence to depth `d` costs
  `prompt_len + d − 1` and makes all `d` of its tokens auditable for free. Depth is
  cheap, breadth is expensive, so the schedule concentrates the audit — behaviour a
  per-token top-k cannot express. Reachable as `verify(..., scheduler="prefix")`,
  where `budget` then means a fraction of the real prefill cost.

Run: `experiments/exp_prefix_cost_gpu.py` → `docs/figures/fig_prefix_cost.png`,
`docs/results/prefix_cost.json`.

---

## 3. Where the detection baseline went: the batch/pool ratio

Part 1's first run was inconclusive for a reason that had nothing to do with the
head: full recompute itself scored only **AUC 0.570** on `quant_4bit`, against the
0.85–1.00 the same detector scores elsewhere in this repo. With the ceiling at
0.57 there is no signal for a triage policy to allocate, so every value signal
lands inside noise. `experiments/exp_baseline_headroom_gpu.py` was written to
explain that gap before touching the head, and it separates four candidate causes
by sweeping each one with the *same* cached per-token scores (117 prompts × 192
tokens = 22 464 tokens per configuration, 5 protocol seeds per AUC).

Run: `experiments/exp_baseline_headroom_gpu.py` →
`experiments/plot_headroom.py` → `docs/figures/fig_baseline_headroom.png`,
`docs/results/baseline_headroom.json`, log in
`docs/results/logs/baseline_headroom_n117_t192.log`.

### The answer: it was an accounting artifact, and 0.570 is the honest number

`harness.batch_means` resamples `batch_size` tokens **without replacement from a
fixed token pool**. As `batch_size` approaches the pool size every batch mean
converges to the pool mean, honest variance collapses, and the AUC stops measuring
"would a fresh batch be flagged" and starts measuring "do these two particular
pools have different means" — which is nearly deterministic. `EvalConfig`'s
docstring already warned about this and gave a rule of thumb (keep the batch well
under ~10% of the honest eval split). The sweep shows what happens when you don't,
on `M=Qwen3-1.7B`, `quant_4bit`, batch fixed at 200:

| honest pool (tokens) | batch / null-split ratio | AUC @ FPR ≤ 0.5% |
|---|---|---|
| 576    | **69.4%** | **0.977 ± 0.010** |
| 1 152  | 34.7% | 0.577 ± 0.042 |
| 2 304  | 17.4% | 0.679 ± 0.099 |
| 4 608  | 8.7%  | 0.523 ± 0.021 |
| 9 216  | 4.3%  | 0.524 ± 0.012 |
| 18 432 | 2.2%  | 0.522 ± 0.008 |
| 22 464 | 1.8%  | 0.530 ± 0.011 |

Same model, same attack, same per-token scores — only the pool grows. **0.977 and
0.530 are the same measurement at two batch/pool ratios.** The repo's own
published configurations sit at 69% (`exp_gpu.py`'s 12 × 48 default) and 78% (the
README headline's 20 × 128 at batch 1000); the triage run sat at 4.9%, inside the
documented ceiling. So the triage experiment's 0.570 was not a regression — it was
the first properly-ratioed measurement of this detector, and the numbers it was
being compared against were inflated.

### Legitimate power, separated from the artifact

Batch size *does* buy real power — a batch of `b` independent tokens separates by
`d' · sqrt(b)`. Growing the pool along with the batch to pin the ratio at 10%
isolates that effect:

| batch | pool grown to (tokens) | AUC @ 10% ratio |
|---|---|---|
| 50   | 1 000  | 0.511 ± 0.012 |
| 100  | 2 000  | 0.552 ± 0.022 |
| 200  | 4 000  | 0.556 ± 0.044 |
| 400  | 8 000  | 0.583 ± 0.013 |
| 800  | 16 000 | 0.598 ± 0.042 |

The measured per-token effect size is **d' = 0.0716** on Qwen3-1.7B (token-level
AUC 0.514) and 0.0873 on Qwen3-0.6B. Solving `d'·sqrt(b) = 3.767` — the separation
a Gaussian pair needs for standardized pAUC@FPR≤0.5% = 0.90 — predicts **batch
2 766**, which needs a **≥ 55 000-token honest pool** to stay inside the ratio
ceiling. That prediction is confirmed on the *un*pinned axis: at the full 22 464-token
pool, batch 1 600 reaches 0.715 and batch 3 200 reaches 0.925 — but those sit at
14% and 28% ratios, i.e. they are already paying for the number with overlap.

Model choice and attack strength were the other two candidates. The model barely
matters (d' 0.072 vs 0.087). **Attack strength dominates**, and it is the cheap
way to buy headroom:

| quantization σ | d' | batch needed for AUC 0.90 | registered as |
|---|---|---|---|
| 0.09 | 0.0178 | 44 958 | — |
| 0.18 | 0.0716 | 2 766 | `quant_4bit` |
| 0.36 | 0.1864 | **409** | `quant_2bit` (added) |

A batch of 409 needs only an ~8 200-token pool at a 10% ratio — which is exactly
the 64 × 128 configuration Part 1 already runs. So `attacks.CoarseQuantization`
(`quant_2bit`, every σ of `quant_4bit` scaled by 2, so it is a strength rung of the
same family and not a differently-shaped attack) is registered, and Part 1 is
re-run against it at batch 400. That is the configuration with real headroom, and
it is honest about where the headroom comes from: a deviation twice as large, not a
smaller token pool.

---

## Incidental fixes this work surfaced

* **Winsorization deleted the signal at small budgets — and it deleted the control
  arm hardest.** `evaluate` caps per-token scores at the 99.9th percentile of the
  honest *calibration* split, so one `delta_max` outlier cannot carry a batch mean.
  Under a **selective** audit that array is mostly the verifier's `neutral`
  placeholder, which is not a score at all — and `token_difr` is *itself* exactly 0
  wherever the sampled token agrees with the verifier's argmax, which is the common
  case (~98% of honest tokens). At a 5% budget, then, ~99.9% of the padded
  calibration split is exactly 0.0, the 99.9th percentile **is** 0.0,
  `np.minimum(scores, 0)` flattens honest and attack alike to a constant, every
  batch mean ties, and `partial_auc` returns **exactly 0.5000 ± 0.0000** — read as
  "no signal" when the metric had deleted a signal that was plainly there (0.573
  unwinsorized on the same data). It bit `uniform` hardest, because a concentrating
  value signal audits tokens likelier to be nonzero and keeps its cap alive: the
  artifact was **inflating the measured advantage of triage over random
  allocation**, in the one experiment whose entire purpose is to measure that
  advantage. `TokenScores` now carries the `audited` mask, and the cap is taken over
  audited honest tokens only. A full audit has no padding, so `audited=None` keeps
  every non-selective number in the repo bit-identical — asserted, along with the
  bug itself, in
  `tests/test_triage_and_cost.py::test_winsor_cap_ignores_the_unaudited_padding`.
  How far it reached is measured directly, because Part 1 was run once on each
  side of the fix with nothing else changed (`quant_2bit`, 64 × 128, batch 400;
  both logs kept, the pre-fix one as `..._prewinsorfix.log`):

  | budget | uniform | entropy | tie_margin | surprisal | learned | oracle |
  |---|---|---|---|---|---|---|
  | 5%   | 0.500 → **0.778** | 0.737 → 0.848 | 0.853 → 0.769 | 0.870 → 0.791 | 0.868 → 0.766 | 0.837 → 0.731 |
  | 10%  | 0.702 → 0.605 | 0.946 → **0.711** | 0.927 → 0.698 | 0.851 → 0.767 | 0.891 → 0.775 | 0.854 → 0.719 |
  | 20%  | 0.931 → 0.741 | 0.994 → **0.829** | 0.851 → 0.697 | = | 0.791 → 0.757 | 0.891 → 0.808 |
  | 35%  | 0.922 → 0.749 | 0.898 → 0.824 | = | = | = | = |
  | ≥50% | = | = | = | = | = | = |

  (`=` is bit-identical.) Every cell at a budget ≥ 50% is untouched, and the
  distortion grows as the budget shrinks — the signature the mechanism predicts,
  since the cap survives as long as the audited honest tokens supply the top 0.1%
  of the padded split. It was **not** a uniform bias: it inflated some cells by up
  to +0.17 (`entropy` at 20%: 0.994, above full recompute, which is what prompted
  the audit of the metric in the first place) and deflated others by up to −0.28.
  Every headline number the pre-fix run produced is therefore withdrawn.

  The other selective-budget results that were on disk at the time
  (`exp_selective_verify_gpu` and the `tie_triage` figures) ran at budgets ≥ 10%
  and were affected at the milder end of that scale. They were never re-run, so
  rather than ship stale numbers they have been **deleted** from the repo; only
  the prefix-scheduler results below, measured after the fix, remain.
* **`prefill_reference` redrew ~20M benign-noise normals on every call.** The
  verifier's benign noise has to be drawn at the sequence's full length and sliced,
  so a position's noise cannot depend on how deep the audit went — but it is a pure
  function of `(model_seed, prompt_id)`, and a cost sweep re-pays prefills on
  purpose (`drop_reference_cache`), so the same draw was being regenerated hundreds
  of times at ~200 ms a call for `V = 151936`. Memoized per prompt; it cannot change
  a score (same seed, same shape) or a reported cost (`prefill_tokens` counts
  model-forward input tokens, and the timed region wraps only the forward pass).
  This was item 6 on the old next-steps list, resolved without touching the noise
  stream.
* **`generate` argsorted the full vocabulary every step for nothing.** The
  descending top-id ordering exists only for `Attack.sample_override`, whose base
  implementation returns `None` unconditionally *and without drawing from `rng`* —
  so for the 7 of 9 registered attacks that do not hijack the sampler, the argsort
  plus a second `filtered_logits` (~13% of a generation step) was dead work, and
  skipping it leaves the RNG stream and therefore every sampled token
  bit-identical. Pinned by
  `tests/test_smoke.py::test_top_id_ordering_is_skipped_only_when_unused`, which
  asserts both the predicate and the no-draw property a new attack could break.
* **`select_triaged` tie-breaking.** With a stable sort, the `uniform` value signal
  (all values equal) degenerated to "audit the first `k` tokens of the first
  sequences" — concentrated in a few sequences, not the equal-cost *random
  subsample* control it is documented to be. Ties now break under a fixed-seed
  permutation, so it stays reproducible and is actually random.
* **`sampling.apply_top_p` sorted the whole vocabulary.** After `top_k`, at most
  `k` entries are finite and the rest are `NEG_INF`, which contribute exactly 0 to
  the softmax and can never be kept — so restricting the sort to the finite
  candidates is output-identical and turns `O(V log V)` into `O(k log k)`.
  `filtered_logits` is the hot loop of every Tier-1 verifier: **5.6 ms → 0.31 ms**
  per call at `V = 151936`, an 18× speedup of the whole audit path. (One caveat,
  present in the original too: a token whose cumulative mass lands *exactly* on `p`
  in float32 can fall on either side depending on summation order. Observed once in
  1200 randomized comparisons, on uniform-random logits with 676 kept tokens; real
  LM logits keep a handful.)
* **The head's sigmoid clipped its logit at ±30**, collapsing the tail to one value.
  Harmless for a probability, not harmless for a *ranking key* fed to
  `select_triaged`. Replaced with an overflow-free sign-branched logistic.

---

## Measured results

All on an H100-80GB, `M=Qwen3-1.7B`, `proxy=Qwen3-0.6B`, `attack=quant_2bit` (the
§ 3 headroom rung), standardized `EvalConfig` (AUC @ FPR ≤ 0.5%, `n_batches=2000`),
64 prompts × 128 tokens at batch 400 — the one configuration in this document with
both a properly-ratioed pool and a detectable deviation. Raw logs in
`docs/results/logs/`, payloads in
`docs/results/{confidence_head,confidence_head_params,prefix_cost}.json`.

### Part 2 — the cost model. Solid.

Full-audit cost 8662 reference-forward tokens (64 prompts × (8.3 avg prompt + 127
generated)), `value_fn=tie_margin`, `lazy_reference=True`:

| nominal budget | tokens audited | **real prefill cost** | prefills | measured sec |
|---|---|---|---|---|
| 5%  | 5.0%  | **75.8%** | 64 | 1.75 |
| 10% | 10.0% | **86.4%** | 64 | 1.53 |
| 20% | 20.0% | **92.4%** | 64 | 1.51 |
| 50% | 50.0% | **97.1%** | 64 | 1.48 |
| 100%| 100%  | 100%      | 64 | 1.48 |

The reported `recompute_ratio` was overstating the saving by up to **15×**. The
wall-clock is the sharpest form of it: **top-k at a "5%" budget takes 1.75 s, more
than auditing every single token (1.48 s)** — it re-prefills all 64 sequences and
the per-call overhead dominates. The old selective tier at small budgets was not
saving anything.

Prefix-scheduled, same masks priced the same way:

| nominal budget | tokens audited | real prefill cost | prefills | measured sec |
|---|---|---|---|---|
| 5%  | 4.9%  | 5.0%  | 4  | 0.11 |
| 10% | 9.8%  | 10.0% | 8  | 0.21 |
| 20% | 19.7% | 20.0% | 18 | 0.42 |
| 50% | 49.4% | 50.0% | 40 | 0.97 |
| 100%| 100%  | 100%  | 64 | 1.49 |

It spends exactly what it is budgeted, and buys **the same token count for ~1/15 of
the cost** at a 5% budget (**16× less wall-clock**: 0.11 s vs 1.75 s). This part is
geometry plus a stopwatch, not statistics — it does not depend on the detection
numbers being strong, and it reproduced unchanged when the configuration doubled
(the same table at 32 × 96 read 75.2% / 84.2% / 94.1% and 0.73 s vs 0.63 s).

### Part 2b — the same comparison at matched *real* cost

Nominal budget is the wrong x-axis for comparing the two schedulers, because the
two spend differently at the same nominal number. Panel B of the figure re-plots
both against realized prefill cost, which is the axis a client actually pays on.

The comparison is cleanest stated as **dominance**, which needs no interpolation
between sampled points: one
prefix-scheduled operating point — 75% of a full audit, `tie_margin`, AUC
0.788 ± .043 — costs strictly less *and* detects more than four separate top-k
points, every one of which the current library would report as the cheaper option:

| top-k operating point | its real cost | its AUC | prefix @ 0.75 cost, 0.788 ± .043 | sd of the difference |
|---|---|---|---|---|
| 5% budget  | 0.758 | 0.650 ± .082 | cheaper, +0.138 | 1.5 sd |
| 10% budget | 0.864 | 0.584 ± .015 | cheaper, +0.204 | 4.4 sd |
| 20% budget | 0.924 | 0.652 ± .050 | cheaper, +0.136 | 2.1 sd |
| 35% budget | 0.956 | 0.671 ± .044 | cheaper, +0.117 | 1.9 sd |

(Only the 5% row is a near-tie in cost; the other three the prefix schedule wins
on both axes at once. The prefix curve is not sampled between its 75% and 100%
budgets, and it is not monotone elsewhere — 0.811 at 35%, 0.793 at 50% — so no
claim is made about the unsampled interval.)

Against top-k at 0.86–0.96 realized cost the prefix schedule is ahead by
0.12–0.20 AUC, 2–4 sd, for less money — the clearest detection-side result in this
document. The mechanism is not subtle: in that band top-k has already paid for all
64 prefills and is buying only extra
*depth* on sequences it has bought anyway, while the prefix schedule is still
converting budget into audited tokens one-for-one. Two honest caveats: top-k's own
curve is non-monotonic in this range (0.650 at 5% but 0.584 at 10%), which is more
noise than the 5-seed AUC axis can resolve, and the gap does **not** reproduce for
every value signal — `entropy` + top-k at a 5% budget happens to cost only 56% and
scores 0.815 ± .096, the best cheap cell in the run, but it is a single point on a
curve that then drops to 0.646 at the next budget. The cost claim (panels A, C) is
exact; this Pareto claim is directional.

### Part 1 — the learned head. Now measured with headroom: it does not win.

`attack=quant_2bit` (the § 3 headroom rung), 64 prompts × 128 tokens (8192 eval
tokens), **batch 400 = 9.8%** of the honest null split, head fit on 16 honest
prompts disjoint from all three other splits, **AUC as mean ± sd over 5 protocol
seeds**. Full recompute scores 0.867 here, so there is a real 0.37 of headroom for
a triage policy to allocate — the thing the first run did not have.

| budget | uniform | entropy | tie_margin | surprisal | learned | oracle |
|---|---|---|---|---|---|---|
| 5%  | 0.778 ± .126 | **0.848 ± .063** | 0.769 ± .110 | 0.791 ± .069 | 0.766 ± .112 | 0.731 ± .082 |
| 10% | 0.605 ± .057 | 0.711 ± .065 | 0.698 ± .083 | 0.767 ± .055 | **0.775 ± .088** | 0.719 ± .050 |
| 20% | 0.741 ± .081 | **0.829 ± .029** | 0.697 ± .048 | 0.783 ± .063 | 0.757 ± .074 | 0.808 ± .056 |
| 35% | 0.749 ± .101 | 0.824 ± .061 | 0.815 ± .040 | **0.841 ± .045** | 0.769 ± .063 | 0.840 ± .062 |
| 50% | 0.702 ± .034 | 0.809 ± .062 | 0.828 ± .050 | **0.860 ± .041** | 0.828 ± .053 | 0.859 ± .049 |
| 75% | 0.794 ± .045 | 0.856 ± .044 | **0.871 ± .040** | 0.865 ± .043 | 0.865 ± .043 | 0.833 ± .046 |
| 100%| 0.867 ± .042 | 0.867 | 0.867 | 0.867 | 0.867 | 0.867 |

Four things this now says, in decreasing order of confidence:

1. **Nothing beats a full audit.** The best selective cell in the table
   (`tie_margin` at 75%, 0.871 ± .040) is inside noise of full recompute. Triage
   buys *cost*, not detection power — which is the honest framing, and what makes
   Part 2's cost result the load-bearing one.
2. **The learned head does not beat the hand-crafted signals it was ported to
   replace.** It leads exactly one row (10%, 0.775 ± .088), by well under 2 sd over
   `surprisal`; `entropy` — the library's existing default — leads two rows and is
   never worse than the head by more than ~1 sd. On the weaker `quant_4bit`
   deviation at a 20% budget the head is 0.551 ± .029 against `entropy` 0.573 ± .051.
   The DSpark confidence head, as ported, is not an improvement here.
3. **Triage does beat spending the same tokens at random**, modestly: `uniform`
   is last or near-last at 10% (0.605 ± .057 vs `learned` 0.775 ± .088, ~2 sd) and
   at 20% (0.741 ± .081 vs `entropy` 0.829 ± .029). So allocation is doing
   *something* — this is the one place the properly-powered run is more positive
   than the old inconclusive one, which could not separate `uniform` from anything.
4. **No two non-uniform signals separate from each other by 2 sd anywhere in the
   table.** With five seeds and this pool, the experiment can rank triage against
   random but cannot rank triage signals against each other. Saying which cheap
   signal is best would need a bigger pool, not a cleverer head.

The `oracle` head — same nine features, trained on *real* labeled `quant_2bit`
pairs rather than the honest-only surrogate probe — does not beat the deployable
head by more than noise either (its best row is 0.840 at 35%). So the ceiling is
not the surrogate label: it is either the feature set or the per-token-d′ target
itself. Note that both heads put a **negative** weight on entropy while `entropy`
is the strongest allocator in the table, which is direct evidence that per-token
surrogate sensitivity is not the quantity that maximizes batch-level detection.

Two secondary findings:

* **STS buys nothing, now measured out of sample.** T = 0.975 (near-identity), and
  on the fourth held-out honest split ECE gets *worse*, 0.0190 → 0.0246. (In-sample,
  where the first run measured it, 0.0108 → 0.0114 — the same "no improvement"
  verdict, so that conclusion survives the hygiene fix.) The head is trained with
  BCE against the same label the reliability curve scores, so it is already
  calibrated; DSpark needs sequential calibration because a *parallel drafter's*
  confidence decays with block depth, and that failure mode has no analogue in a
  logistic head over per-position proxy features.
* **What the head learned** (standardized coefficients): `tie_margin` +0.86,
  `surprisal` +0.80, `log_top1` +0.46, `log_support` +0.34, `entropy` −0.28,
  `log_rank` −0.27, `top5_mass` −0.23, `renyi2` +0.05, `rel_position` −0.03. It
  rediscovers near-tie-ness and proxy surprisal as dominant and *disagrees* with the
  library's default value function on the sign of entropy — and, per the table
  above, the library's default is the one that allocates better.

#### Two reversals are on the record here

Both are kept deliberately, with their logs, because each was a plausible-looking
finding that did not survive a fix to the measurement:

* The **32 × 96 run** (batch 200 = 13% of the null split, above the ratio ceiling)
  showed `learned` 0.684 vs `entropy` 0.612 vs `tie_margin` 0.552, i.e. the head
  winning clearly. That ordering did not survive a properly-ratioed pool.
* The **pre-winsor-fix run of this very table** showed `entropy` 0.994 at a 20%
  budget — *above* full recompute — and `uniform` at exactly 0.500. Both were
  artifacts of the winsorization bug in "Incidental fixes" below; the "selective
  audit beats full recompute" reading it invited is withdrawn.

### Parts 1 × 2 — admitting against the calibrated head. It does not beat a guess.

The two ports are independent until you ask what the scheduler should rank by.
DSpark's admission rule is value per unit of marginal cost, and its value term is
the confidence head — so the combined claim is that a *calibrated* value estimate
makes value-density admission pay off in a way a hand-guessed signal cannot.
`exp_prefix_cost_gpu.py` now runs the prefix scheduler against all three by
default (`IVGYM_VALUE_FNS=tie_margin,entropy,learned`, same masks, same pricing,
same 5 protocol seeds), which tests exactly that. The schedule spends its budget
exactly, so for these rows nominal budget *is* realized prefill cost:

| realized prefill cost | `tie_margin` | `entropy` | `learned` (head) |
|---|---|---|---|
| 5%   | 0.622 ± .109 | **0.738 ± .066** | 0.695 ± .085 |
| 10%  | 0.614 ± .048 | **0.716 ± .079** | 0.672 ± .076 |
| 20%  | 0.707 ± .068 | **0.729 ± .087** | 0.702 ± .066 |
| 35%  | **0.811 ± .069** | 0.745 ± .066 | 0.717 ± .073 |
| 50%  | 0.793 ± .068 | 0.778 ± .051 | **0.811 ± .060** |
| 75%  | 0.788 ± .043 | 0.788 ± .061 | 0.748 ± .063 |
| 100% | 0.823 ± .050 | 0.823 ± .050 | 0.823 ± .050 |

**No.** The head leads exactly one row (50%), `entropy` three, `tie_margin` one,
with the 75% row a dead tie — and no pair separates by 2 sd of the difference
anywhere in the table — at
the 5% row, where the spread is widest, `entropy` − `tie_margin` = 0.116 against a
paired sd of ~0.13. This is the same verdict Part 1 reached for top-k selection,
now reached again under the admission rule the head was ported *for*, so the two
negative results are not one measurement counted twice: the head fails both as a
per-token ranking key and as the value term in a cost-aware schedule.

What survives the cross is the split between the two ports. **The scheduler is the
part that transfers**: it is a statement about prefill geometry, it holds for every
value signal, and it is the reason a 5% budget can cost 5%. **The confidence head
is the part that does not**: the choice of value signal moves the result by less
than the noise, so there is nothing for a better-calibrated signal to buy here.
Anyone porting DSpark into a verification setting should take the scheduler and
skip the head.

## Where this stands / next steps

**All six items on the previous list are closed.** Kept here because two of them
changed what an earlier version of this document claimed:

1. ~~Reproduce the `token_difr` baseline and find a configuration with headroom.~~
   Done — § 3. The answer was not a missing configuration: the baseline was fine and
   the numbers it was being compared against were inflated by the batch/pool ratio.
   Headroom now comes from `quant_2bit`, a stronger deviation, not a smaller pool.
2. ~~Re-run `exp_confidence_head_gpu.py` there.~~ Done — Part 1, and it is a clean
   negative: the head does not beat the signals it was ported to replace.
3. ~~Fit STS on a fourth held-out honest split.~~ Done — out of sample ECE gets
   *worse* (0.0190 → 0.0246); the "calibration buys nothing" verdict survives the
   hygiene fix.
4. ~~Re-run `exp_prefix_cost_gpu.py` at 64 × 128.~~ Done — Parts 2 / 2b. The cost
   result reproduced unchanged at double the size, and the matched-real-cost Pareto
   is the strongest detection-side finding here.
5. ~~Schedule against the calibrated head.~~ Done — Parts 1 × 2. Also negative, and
   it is the result that separates the two ports: take the scheduler, skip the head.
6. ~~`prefill_reference` perf.~~ Done — memoized per prompt, no change to any score
   or cost ("Incidental fixes").

Both experiments now **default** to the configuration reported here
(64 × 128, batch 400, `quant_2bit`, 5 seeds), so a plain
`python -m experiments.exp_confidence_head_gpu` reproduces the tables rather than
the superseded run. What is genuinely open, in descending order of what it buys:

1. **The AUC axis is still the weak one.** Five seeds over a 64 × 128 pool can rank
   triage against random but cannot rank two triage signals against each other
   (Part 1, finding 4), and it is why Part 2b's Pareto gap is directional rather
   than exact. The fix is pool size, not a cleverer estimator: § 3's d′ arithmetic
   says a ≥ 55 000-token honest pool would let `quant_4bit` itself be measured at
   AUC 0.90 inside the ratio ceiling. That is ~7× this run's generation cost, and
   the most useful place to spend the next block of GPU-hours.
2. ~~**Re-run the other selective-budget results on disk.**~~ **Resolved by
   deletion.** `exp_selective_verify_gpu` and the `tie_triage` figures predated
   the winsorization fix and were never re-measured, so they have been removed
   rather than left on disk to be cited. The claim they carried — that triage can
   beat a full audit — is withdrawn; triage buys cost, not detection power.
3. ~~**The ratio artifact is not confined to this document.**~~ **Resolved.**
   § 3 showed the README headline table (78%) and `exp_gpu.py`'s default (69%)
   both sat above the ceiling. Both have since been re-measured inside it
   (`exp_headline_ratio_gpu`), and the ceiling is now enforced in code:
   `EvalConfig.max_pool_ratio` warns (or raises) on any over-ratio measurement and
   `EvalResult.pool_ratio` records the ratio on every number the repo produces.
4. **Does the Pareto gap survive a different attack family?** Part 2b is one
   deviation (`quant_2bit`). The scheduler's advantage is geometric and so should be
   attack-independent — a cheap, falsifiable prediction: run `kv_fp8` or `bug_k2` at
   the same pool and the panel-A/C curves should be identical and panel B should
   keep its ordering.

## Sources

- DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive
  Generation — https://arxiv.org/abs/2607.05147
- DiFR: Inference Verification Despite Nondeterminism — https://arxiv.org/pdf/2511.20621
