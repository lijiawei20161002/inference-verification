# What to run next

Ranked by *how much a negative result would hurt*, not by how likely a positive
is. Each entry states the claim at risk, the experiment, the cost, and — most
importantly — **what outcome would falsify it**, because several results in this
repo's history looked strong until someone asked that question.

Three things bound the current record, and the list below is organized around
them:

- **A** — the attacks are simulated at the logit level, so the headline
  quantization numbers are a *model of* the attack, not a measurement of it.
- **B** — the verifier is passive; no adversary has ever adapted to it.
- **C** — one GPU, one prompt domain, so the false-positive side of the ROC has
  never been stressed by anything except run-to-run nondeterminism on a single
  accelerator.

Of these, **C is the one that decides whether any of this deploys**, and it is
the cheapest to run. It is first for that reason.

---

## Tier 1 — the results that are load-bearing and untested

### 1. Cross-GPU benign variation: the false-positive side of the ROC

**Claim at risk:** every AUC in the repo. The honest null is currently "the same
model, on the same H100, run twice". A real verifier calibrates on one machine
and audits a provider on another, so the honest distribution it must not flag
includes a *different accelerator, driver, kernel selection and batch shape*.
If that benign shift is comparable to `quant_4bit`'s d′ = 0.077, the entire
quantization result collapses into false positives and the deployment
recommendation inverts: you would need to calibrate per-datacenter, or not at
all.

**Experiment.** Generate one honest pool on GPU A. Score it against a reference
recomputed on GPU B (different SKU: H100 vs A100 vs L40S), and against the same
reference recomputed on GPU A. Report `d′(benign cross-GPU)` on exactly the scale
`evaluate` uses, and compare it directly to `d′(quant_4bit) = 0.0775` and
`d′(kv_fp8) = 0.015` from `pool_scaling.json` / the report.

Sweep, in increasing order of nuisance: same GPU different batch shape → same GPU
different attention kernel (`sdpa` / `flash` / `eager`) → different GPU same
family → different vendor generation. The first two are runnable *today* on one
card and are a strict lower bound on the last two.

**Falsifies the deployment story if:** `d′(benign) ≳ d′(quant_4bit)`. Then no pool
size fixes it, because growing the pool sharpens the false positive exactly as
fast as the true positive.

**Cost:** hours on one card for the kernel/batch arm; a second card for the rest.
**Why first:** it is the only item that can invalidate results rather than extend
them, and half of it needs no new hardware.

---

### 2. Genuinely quantized weights, end to end

**Claim at risk:** every `quant_*` number, i.e. the headline detectability of the
economically most plausible cheat.

The attacks apply an i.i.d. Gaussian logit perturbation. A real NF4/INT8 quantized
checkpoint produces a *deterministic, sparse, heavy-tailed* logit shift —
structurally different even when matched on mean divergence. The repo's one
genuine nf4 datapoint (in the proxy-panel study) read chance, which is at least
consistent, but "consistent with" is not "measured".

**Experiment.** Serve `M̂` = a real `bitsandbytes` NF4 (and INT8, and AWQ/GPTQ)
load of `M`, generate honestly *under `M̂`*, and verify against fp16 `M`. Report
per-token d′ for all four Tier-1 verifiers, plus the batch each nominates for AUC
0.90 — the pool-independent statement that survives whatever pool you can afford.
Then the diagnostic that matters: **is the shift concentrated?** Compute the
Gini coefficient of `|Δ logit|` across positions for real NF4 versus the Gaussian
model at matched mean divergence.

**The interesting prediction, and it cuts against the current negative results.**
If real quantization is sparse — a large shift at few positions rather than a
small shift everywhere — then `Δ²/v` is *concentrated*, selective verification
should work far better than it did against the Gaussian model, and the
information-directed allocation of `ivgym/infogain.py` may have been tested
against the one deviation shape that guarantees it nothing to find. That would
not overturn the negative (the estimator really was anti-correlated), but it
would sharply narrow its scope, from "sensitivity is the wrong target" to
"sensitivity is the wrong target *for diffuse deviations*".

**Falsifies the current record if:** real NF4 has d′ substantially different from
the matched Gaussian model in either direction. Both directions are informative.

**Cost:** ~1 GPU-day. `bitsandbytes` is already a documented optional dependency.

---

### 3. An adaptive attacker with query access to the deployed detector

**Claim at risk:** the whole threat model. Every attacker here is fixed and
verifier-blind. The one place adaptivity *was* tried — searching a
temperature × top-p grid in the proxy study — already found evasions, which is
the tell.

**Experiment.** Give the attacker black-box query access to the deployed verdict
(flag / no-flag, or the batch statistic `S`) and a budget of `Q` queries. Optimize
a served configuration to minimize `S` subject to a compute saving of at least
`c`. Two attacker classes, and the difference between them is the result:

- **Sampler-side:** temperature, top-k, top-p, and a per-position mixture between
  `M` and `M̂`. Cheap to search, and the report's own evasions suggest it works.
- **Weight-side:** mixed-precision, i.e. quantize only the layers whose logit
  effect is smallest per FLOP saved. This is the realistic adversary — it is what
  a provider optimizing margin would actually do.

Report the **evasion frontier**: compute saved versus detection AUC at a fixed FPR
budget, one curve per verifier and one for the portfolio. The portfolio curve is
the deliverable — the repo's central recommendation is "run them together", and
that recommendation has never been attacked.

**Falsifies the portfolio recommendation if:** a single configuration drives every
verifier to chance simultaneously at a meaningful compute saving. Mechanistically
this should be hard — `activation_difr` and `token_difr` read disjoint parts of
the pipeline — but "should be hard" is exactly the kind of claim this repo has
been wrong about before.

**Cost:** ~2–3 GPU-days. Highest scientific value on the list.

---

## Tier 2 — finishing what the corrections started

### 4. Re-measure the cross-family sweep inside the ceiling

**Claim at risk:** "no single recompute detector is uniformly robust", and the
"portfolio lifts the floor to 0.78" number.

The 63-cell sweep ran at a **52%** batch/pool ratio. `exp_reversal_check_gpu`
re-measured five of those cells and none survived as published. The other 58 have
not been re-run, and the report says so. The mechanism argument stands
independently, but the *levels* are unsupported and are still being quoted.

**Experiment.** Re-run `exp_robustness_gpu` with the pool sized from d′ rather
than guessed: measure d′ per cell on a cheap pilot, then generate the pool each
cell needs (`signal.batch_for_pauc`). Cells whose required pool is unaffordable
should be **reported as unresolvable**, not measured badly — that is itself the
honest result, and it is what the current sweep hid.

**Cost:** ~1–2 GPU-days, most of it generation.
**Note:** the ceiling is now enforced (`EvalConfig.max_pool_ratio`), so this sweep
cannot silently repeat the error.

### 5. Does `token_toploc` really cost nothing?

**Claim at risk:** the most deployable finding in the repo, and the least
examined. `token_toploc` tracks `token_difr` closely while needing **no seed
synchronization** — a much weaker requirement on the provider, and therefore the
detector most likely to actually ship.

But it has only ever been measured where the provider cooperates on everything
*except* the seed. Two open questions: (a) how does its rank statistic degrade
when the provider's top-p/top-k filtering is not exactly reproducible on the
verifier's hardware (which is item 1's failure mode, hitting the one detector
that depends on set membership); and (b) what is its d′ against a provider that
deliberately keeps the claimed token's *rank* honest while moving its
probability — a SAFE-set attack aimed specifically at rank-based verification?

**Cost:** ~half a GPU-day. Cheapest item with a real deployment consequence.

### 6. Communication cost: `activation_difr` versus `token_toploc`

Still open from the original roadmap, and now well-posed because the cost model
exists. `activation_difr` is the strongest detector on forward-pass attacks and
sits at exactly chance on sampler attacks; `token_toploc` is the reverse trade.
The Pareto between them is `k × J` (projection dimension × number of layers)
against detection, at fixed communication bytes. Pure sweep, no new machinery.

---

## Tier 3 — worth doing, lower risk

### 7. Frontier-scale, or the closest affordable proxy for it

Every result is 0.13B–8B. Two things change with scale and both matter: honest
nondeterminism may not scale the same way as the deviation signal (if it shrinks
faster, verification gets *easier* at scale, which would be the most important
positive result available); and the proxy-to-target ratio collapses, which should
*improve* the acceptance-rate detector, since `KL(M‖q)` grows.

Even without a frontier model, the trend is measurable: run `exp_pool_scaling_gpu`
at 0.6B / 1.7B / 4B / 8B and fit d′ versus parameter count. An extrapolation from
four points is weak evidence, but it is the difference between having a slope and
having none.

### 8. Sequential verification instead of fixed-batch

Every verdict here is a fixed-`b` batch. A real verifier watches a provider
continuously, which is a **sequential** test: a SPRT or e-value accumulates
evidence across an ongoing relationship and stops as soon as the posterior clears
a threshold, at typically 2–3× fewer samples than a fixed-`n` test of the same
power. Given that the binding constraint identified throughout is *how many
tokens a verdict needs*, this is the one methodological change that could move the
headline cost without needing a better detector.

It also fits the deployment story exactly: the report's own framing is that "any
protocol promising a verdict on a single completion is not measuring
quantization" — a sequential test is what replaces it.

### 9. Multi-provider / cross-checking

If a client uses several providers for the same spec, honest providers agree with
each other up to nondeterminism and a cheating one does not — without anyone
running `M`. This is a genuinely different mechanism from everything in the repo
(it replaces the trusted anchor with an assumption of non-collusion), and it is
cheap to simulate with the existing backend.

---

## Two things I would *not* spend GPU-hours on

- **A better per-token value signal.** Three attempts, three losses: the confidence
  head, the derived matched filter, and the oracle matched filter. The diagnosis in
  `ivgym/infogain.py` is specific and it is about *estimability*, not about
  insufficient cleverness — the cheap features predict the local noise well
  (R² = 0.73) and the signal not at all (R² = 0.002). Item 2 above is the only
  thing that should reopen this, and only if real quantization turns out to be
  sparse.
- **Another cheap black-box statistic for quantization.** The budget is
  `KL(M‖q)` and it is measured. A new surface statistic cannot exceed it. Effort
  belongs on the *cost* of recomputation, where the prefix scheduler already showed
  16× is available, not on avoiding it.
