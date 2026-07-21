# The inference-verification game (and how `ivgym` scores it)

This document answers four questions about what `ivgym` actually models:

1. What code this is built on (provenance vs. DiFR and the exfil repo).
2. Who the attacker and defender are, and what each one controls.
3. How either side *wins* — the explicit decision rule and win conditions.
4. How a per-token DiFR score becomes a per-batch **classification** — i.e. the
   aggregation across tokens and the threshold rule, which DiFR itself leaves to
   the integrator.

It also states how this game relates to the security game in the stego work.

---

## 1. Provenance — what this is built on

`ivgym` is a **clean-room reimplementation that follows the *DiFR paper*
methodology** (*DiFR: Inference Verification Despite Nondeterminism*, Karvonen
et al., 2025), not a fork of either codebase:

- It does **not** vendor or import
  [`adamkarvonen/difr`](https://github.com/adamkarvonen/difr). The seed-synced
  Gumbel-Max sampler (`ivgym/sampling.py`), the per-token margin score
  (`ivgym/defenses.py`), and the generate→verify→calibrate→evaluate loop
  (`ivgym/harness.py`) are reimplemented in pure numpy from the paper's
  Algorithm 1 / metric definitions, and run against a real model on a GPU via
  the HF-GPU backend (`ivgym/backends/hf_gpu.py`).
- It does **not** build on top of
  [`RoyRin/inference_verification_for_model_weight_exfil`](https://github.com/RoyRin/inference_verification_for_model_weight_exfil).
  The exfiltration *threat model* (a provider serving different/cheaper/stolen
  weights than the spec it claims) is the motivation, and the framing here is
  meant to be compatible with that repo's, but no code is shared.
- The one genuine shared lineage is **in this repository's own history**: the
  earlier `e4_stego/` (GLS verifier + stego codec) and `bitexact_a100/`
  experiments. The stego "security game" referenced below is that work. `ivgym`
  is the restructuring of that line into a pluggable attack-vs-defense gym.

> If we did pull any specific code from either external repo, say so here and
> cite the file — this section should be edited to match ground truth, not left
> as a claim. As the tree stands, it is an independent implementation.

---

## 2. The players and what they control

Two parties, one specification ϕ.

- **Specification ϕ** (`SamplingSpec` in `ivgym/core.py`): the sampling contract
  the provider publicly commits to — `temperature`, `top_k`, `top_p`, `seed`.
  Both sides agree on ϕ up front.

- **Provider = attacker** (`Attack` in `ivgym/attacks.py`). Serves
  `(prompt, claimed_tokens)` pairs and *claims* it produced them under ϕ on the
  reference model. It may secretly deviate. The deviation knobs are exactly the
  `Attack` hooks:
  - `provider_spec(ϕ)` — silently run a *different* sampling spec (wrong
    temperature `temp_1.1`, wrong seed `seed_43`).
  - `logit_bias_sigma()` / `perturb_logits()` — perturb the forward pass, the
    way 4-bit quantization (`quant_4bit`) or an fp8 KV cache (`kv_fp8`) would.
  - `sample_override()` — hijack the sampler outright (`bug_k2`, `bug_k32`).
  - Compose them: `adv_quant_temp` quantizes **and** lowers temperature to hide
    the quantization from a cross-entropy detector.

  An honest provider is just `Attack()` with only benign numerical noise.

- **Verifier = defender** (`Defense` in `ivgym/defenses.py`). Trusts ϕ and the
  reference model, **not** the provider. For each claimed token it recomputes
  the reference logits (and, optionally, a projected activation fingerprint) and
  emits a **per-token divergence score**, higher = more divergent. The verifier
  never observes the attack; it only ever recomputes under ϕ. Built-ins:
  `token_difr` (post-Gumbel margin), `cross_entropy` (NLL of the claimed token),
  `activation_difr` (L2 on projected activations).

The provider moves first (generates), the verifier responds (scores + decides).
The provider does **not** see the verifier's noise draws or threshold; the
verifier does **not** see the attack — this asymmetry is the game.

---

## 3. From a per-token score to a decision (the part DiFR leaves open)

> Roy's point, made explicit: **DiFR defines a score for a *single token*. The
> aggregation across tokens and the classification rule are separate choices and
> must be specified.** Here is exactly what `ivgym` does.

### 3a. Per-token score `s_t`
For each claimed token `t`, the defense returns one scalar (`Defense.score`,
`ivgym/defenses.py`). For the default **Token-DiFR**
(`defenses.py:74`):

```
filt = top_p(top_k(ref_logits))          # filter under ϕ
z    = filt + temperature * gumbel        # SHARED per-position Gumbel noise
v*   = argmax(z)                          # the verifier's own sampled token
s_t  = 0                       if v* == claimed_token        (agreement)
     = min(z[v*] - z[claimed], Δ_max)     otherwise          (post-Gumbel margin)
     = Δ_max (=30)             if claimed_token was filtered out by top-k/top-p
```

Because provider and verifier draw the **same** Gumbel noise from a shared
per-position seed (`position_seed` in `ivgym/sampling.py`), an honest token gives
`s_t = 0` up to numerical noise; any real deviation pushes `s_t > 0`.

### 3b. Aggregation across tokens — the batch statistic `S`
A single token is far too noisy to classify on. `ivgym` aggregates exactly as
the paper's batch-size sweep does (`harness.batch_means`, `harness.py:72`), under
the single standardized protocol `harness.EvalConfig` fixes for every
experiment:

1. Split the honest tokens into a **calibration** split and a disjoint **eval**
   split (`EvalConfig.calib_frac`, default 0.5). The calibration split supplies
   both the winsorization cap and the decision threshold `τ` (3c below); the
   eval split is the honest null the attack is actually scored against. Keeping
   these disjoint means `τ` is never fit on the same honest data it is then
   tested on.
2. **Winsorize** each per-token score (eval-honest and attack alike) at the
   99.9th percentile of the *honest calibration* split (`harness.winsorize`,
   `harness.py:64`) so one filtered-out token (`s_t = Δ_max`) can't dominate a
   batch mean.
3. Draw a **batch of `b` tokens** and take the **mean**:
   `S = mean(s_t for t in batch)`.

`S` is the batch-level statistic — the quantity that is actually classified.
Detection power is a function of `b`: tiny per-token signals separate cleanly
once averaged over a large enough batch, which is why the sweeps vary `b` and
why real-model AUCs sharpen as `IVGYM_TOKENS`/`IVGYM_BATCH` grow.

> Mean is the current aggregator because it matches the paper's batched
> hypothesis test and is what the AUC numbers in the README are computed with.
> Sum / trimmed-mean / max-of-windows are all valid alternative aggregators; if
> we change it, this is the line to change and to report.

### 3c. Classification rule
The decision is a one-sided threshold test on `S`:

```
H0: provider is honest (ran ϕ on the reference model)
H1: provider deviated

decide DEVIATION  iff  S > τ
```

The threshold **τ is calibrated to a false-positive budget α = `EvalConfig.max_fpr`
(fixed at 0.5%)**, not picked by hand: τ = the `(1 − α)` empirical quantile of the
honest **calibration**-split batch statistics (`metrics.tpr_at_fpr`,
`metrics.py:25`; the calibration split, not the eval split — see 3b). So the
verifier commits to "I will wrongly flag an honest provider at most α of the
time," then τ follows, and it is then tested **out of sample** against the
disjoint honest eval split.

We report three numbers (`harness.evaluate`, `harness.py`; `EvalResult`):
- **AUC — the headline metric.** The **standardized partial AUC at
  FPR ≤ 0.5%** (`metrics.partial_auc`, McClish-standardized): threshold-free
  separability of honest-eval vs. attacked `S`, but restricted to the strict
  low-false-positive region a verifier actually has to operate in (an inference
  provider is not a one-shot bet — it is checked continuously, so it cannot
  tolerate more than a sliver of honest false flags). Standardization keeps the
  scale identical to full AUC: a random verifier scores 0.5, a perfect one
  scores 1.0, regardless of the FPR window. Reported as `EvalResult.auc`.
- **AUC (full-range)** — the threshold-free separability over *all* τ
  (`metrics.roc_auc`), kept as `EvalResult.auc_full` for context / comparison
  to legacy numbers; it sweeps FPR regions no real verifier would operate at,
  so it is not the metric used to declare a win.
- **TPR @ FPR = 0.5%** — the operationally meaningful catch rate at the τ fixed
  by the same 0.5% honest-flag budget, evaluated out of sample. `EvalResult.tpr`.

`EvalConfig` (`harness.py`) is the *one* place all of this is fixed — FPR
budget, batch-resampling count (`n_batches`), winsorization percentile, and the
calibration/eval split fraction — so every experiment in the repo scores under
an identical, standardized protocol instead of each picking its own `n_batches`
or FPR target. Resolving `FPR ≤ 0.5%` at all needs enough resampled honest
batches to land points inside that thin a slice; `EvalConfig` raises
`ValueError` rather than silently returning a noisy estimate if `n_batches` is
too small for the configured `max_fpr` (in practice, `n_batches >= 2000` for the
default 0.5% budget).

That floor only guarantees enough resampled *batches* fall in the region — it
does not guarantee those batches are independent evidence. `batch_means`
resamples without replacement from a fixed, finite token pool; if `batch_size`
is a large fraction of that pool (a small prompt/token count), the batches
overlap heavily and the extreme tail can look artificially close to the
diagonal (i.e. AUC@FPR≤0.5% underestimates true separability) no matter how
large `n_batches` is. What actually resolves the tail is a bigger honest token
pool (more prompts / tokens), not more resampling of the same pool — keep
`batch_size` well under ~10% of the honest eval-split token count.

---

## 4. How either side wins

It is a one-shot hypothesis test with a fixed FPR budget α:

- **Defender (verifier) wins** when, at a threshold τ honoring `FPR ≤ α`
  (α = 0.5%), it flags the deviating provider — i.e. **TPR@α → 1** (equivalently
  AUC@FPR≤α → 1). The README grid is exactly the scoreboard: `token_difr` wins
  every column on the real model.
- **Attacker (provider) wins** when it deviates from ϕ (cheaper/different/
  exfiltrated weights) yet keeps `S` statistically indistinguishable from honest
  at that same τ — **TPR@α ≈ α / AUC@FPR≤α ≈ 0.5**. The canonical attacker win in the
  repo is `adv_quant_temp` vs. `cross_entropy`: quantize for the compute saving,
  then retune temperature so mean NLL matches honest, collapsing the
  cross-entropy detector (paper Fig. 2). Both the attack (`adv_quant_temp`) and
  the defense (`cross_entropy`) are built-ins, so the matchup runs via
  `experiments/run.py --attacks adv_quant_temp --defenses cross_entropy token_difr`.

The interesting result is that an attack that beats one defense need not beat
another: `adv_quant_temp` defeats `cross_entropy` but **not** seed-synchronized
`token_difr`, because the Gumbel margin sees the forward-pass change regardless
of the temperature cover. The whole point of the gym is to make "who beats whom,
and at what batch size" a measurable grid rather than an anecdote.

---

## 5. Relationship to the stego paper's security game

**Same game, dual objective.** The stego/GLS work (this repo's `e4_stego/`)
defines a `SAFE` set: outputs the verifier cannot distinguish from an honest
sample. There, the stego *encoder* wins by pushing bits through the channel
while staying inside `SAFE` (low BER **and** undetectable), and the metric of
interest is **channel capacity subject to undetectability**.

Inference-verification is the **same indistinguishability test with the roles
relabeled and the objective flipped to the defender's side**:

| | Stego game (encoder's view) | Inference-verification (verifier's view) |
|---|---|---|
| Adversary's goal | embed bits, stay in `SAFE` | run cheaper/other weights, stay below τ |
| "stay hidden" means | output ∈ `SAFE` | batch statistic `S ≤ τ` |
| Headline metric | capacity / BER at fixed detectability | detection power (TPR@FPR, AUC) |
| Same underlying object | H0-honest vs H1-deviated test | H0-honest vs H1-deviated test |

So `S ≤ τ` here is the operational definition of "the output is in `SAFE`," and
an attacker that wins the inference-verification game (AUC ≈ 0.5) is exactly one
whose deviations live entirely inside the stego `SAFE` set. The difference is
which side's quantity you optimize/report: the stego paper maximizes the
attacker's capacity inside `SAFE`; `ivgym` maximizes the defender's detection
power for pushing attacks *out* of `SAFE`.
