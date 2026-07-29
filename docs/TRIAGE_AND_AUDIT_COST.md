# Learned triage and the real cost of a selective audit

> **STATUS (2026-07-29).** Part 2 (cost model + prefix scheduler) is **done and
> measured**: a clean, reproducible win, numbers below. Part 1 (learned confidence
> head) is **built, tested, and inconclusive** — at properly-powered scale no triage
> value signal separates from any other, because the *baseline* `token_difr` AUC in
> this configuration is only 0.570 and there is no headroom for triage to allocate.
> Diagnosing that baseline is the next step, not more head tuning. See
> "Measured results" and "Where this stands" at the bottom.

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

## Incidental fixes this work surfaced

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

All on an H100-80GB, `M=Qwen3-1.7B`, `proxy=Qwen3-0.6B`, `attack=quant_4bit`,
standardized `EvalConfig` (AUC @ FPR ≤ 0.5%, `n_batches=2000`, batch 200). Raw logs
in `docs/results/logs/`, payloads in `docs/results/{confidence_head,prefix_cost}.json`.

### Part 2 — the cost model. Solid.

32 prompts × 96 tokens, full-audit cost 3303 reference-forward tokens,
`value_fn=tie_margin`, `lazy_reference=True`:

| nominal budget | tokens audited | **real prefill cost** | prefills | measured sec |
|---|---|---|---|---|
| 5%  | 5.0%  | **75.2%** | 32 | 0.73 |
| 10% | 10.0% | **84.2%** | 32 | 0.66 |
| 20% | 20.0% | **94.1%** | 32 | 0.65 |
| 50% | 50.0% | **97.8%** | 32 | 0.63 |
| 100%| 100%  | 100%      | 32 | 0.63 |

The reported `recompute_ratio` was overstating the saving by up to **15×**. The
wall-clock is the sharpest form of it: **top-k at a "5%" budget takes 0.73 s, more
than auditing every single token (0.63 s)** — it re-prefills all 32 sequences and
the per-call overhead dominates. The old selective tier at small budgets was not
saving anything.

Prefix-scheduled, same masks priced the same way:

| nominal budget | tokens audited | real prefill cost | prefills | measured sec |
|---|---|---|---|---|
| 5%  | 4.9%  | 5.0%  | 2  | 0.06 |
| 10% | 9.9%  | 10.0% | 4  | 0.09 |
| 20% | 19.5% | 20.0% | 10 | 0.20 |
| 50% | 48.7% | 50.0% | 21 | 0.41 |

It spends exactly what it is budgeted, and buys **the same token count for ~1/15 of
the cost** at a 5% budget (**12× less wall-clock**: 0.06 s vs 0.73 s). This part is
geometry plus a stopwatch, not statistics — it does not depend on the detection
numbers being strong.

### Part 1 — the learned head. Inconclusive.

64 prompts × 128 tokens (8192 eval tokens; batch = 4.9% of the honest null split,
inside the ≤10% ceiling `EvalConfig` documents), head trained on 16 **disjoint**
honest prompts:

| budget | uniform | entropy | tie_margin | surprisal | learned | oracle |
|---|---|---|---|---|---|---|
| 5%  | 0.500 | 0.520 | 0.531 | 0.548 | 0.553 | 0.531 |
| 10% | 0.628 | 0.573 | 0.528 | 0.549 | 0.566 | 0.536 |
| 20% | 0.600 | 0.645 | 0.517 | 0.539 | 0.534 | 0.523 |
| 35% | 0.555 | 0.579 | 0.546 | 0.546 | 0.532 | 0.536 |
| 50% | 0.531 | 0.565 | 0.532 | 0.553 | 0.549 | 0.551 |
| 100%| 0.570 | 0.570 | 0.570 | 0.570 | 0.570 | 0.570 |

**Read this as a null result.** Every signal sits in 0.50–0.65 with no stable
ordering; `uniform` (the random control) wins at 10%, `entropy` at 20%, `learned`
at 5%, and the curves are non-monotonic in budget. The `oracle` head — same
features, trained on *real* labeled `quant_4bit` pairs — does no better, which
rules out "the surrogate probe is the wrong label" as the explanation.

The diagnostic that matters: **full recompute itself scores only 0.570** here,
against the 0.87–1.0 `docs/SPEC_DECODING_AND_PROXY_DETECTION.md` reports for
`token_difr` on forward-pass attacks. With the ceiling at 0.57 there is essentially
no signal for a triage policy to allocate, so this experiment was comparing
allocation strategies inside noise. An earlier, *smaller* run (32×96, batch 200 =
13% of the null split — above the documented ceiling) showed `learned` 0.684 vs
`entropy` 0.612 vs `tie_margin` 0.552 and near-oracle performance; that ordering did
not survive the properly-powered rerun, so it was noise. Both logs are kept in
`docs/results/logs/` precisely so that reversal is on the record.

Two secondary findings that are solid:

* **STS is near-identity here** (T = 0.975; ECE 0.0108 → 0.0114, i.e. no
  improvement). The head is trained with BCE against the same label the reliability
  curve scores, so it is already calibrated. DSpark needs sequential calibration
  because a *parallel drafter's* confidence decays with block depth; that failure
  mode has no analogue in a logistic head over per-position proxy features. The
  calibration component of the port buys nothing. (Caveat: this reliability is
  measured **in-sample** — STS is fit and scored on the same honest split. Fixing
  that to a fourth held-out split is on the list below.)
* **What the head learned** (standardized coefficients): `tie_margin` +0.86,
  `surprisal` +0.80, `log_top1` +0.46, `log_support` +0.34, `entropy` −0.28,
  `log_rank` −0.27, `rel_position` −0.03. It rediscovers near-tie-ness and proxy
  surprisal as the dominant signals and puts a *negative* weight on entropy — i.e.
  it does not agree with the library's current default value function.

## Where this stands / next steps

1. **Reproduce the repo's own `token_difr` baseline first.** Run `exp_gpu.py` on this
   box and find the configuration where quant_4bit reaches AUC ≈ 0.9 (larger batch?
   stronger attack sigma? different M?). Triage comparisons are only meaningful with
   headroom. Until then Part 1's table says nothing about the head.
2. Then re-run `exp_confidence_head_gpu.py` in that configuration.
3. Fit STS on a fourth held-out honest split so the reliability panel is out-of-sample.
4. Re-run `exp_prefix_cost_gpu.py` at 64×128 so panel B's AUC axis has the same power
   as panel A/C already have (A and C are exact geometry and measured time, and do not
   need it).
5. Combine the two parts: the prefix scheduler currently schedules against
   `tie_margin`. Once the head is validated, schedule against the *calibrated head*
   and check whether value-density admission beats value-density-with-a-guessed-signal.
6. Possible perf follow-up: `prefill_reference` draws benign verifier noise at full
   sequence length and slices, which wastes work at shallow depths (it dominated
   exp2's runtime). Per-position seeding would fix it but changes `activation_difr`'s
   noise stream, so it needs care.

## Sources

- DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive
  Generation — https://arxiv.org/abs/2607.05147
- DiFR: Inference Verification Despite Nondeterminism — https://arxiv.org/pdf/2511.20621
