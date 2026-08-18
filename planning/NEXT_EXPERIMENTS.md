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

**Status: the single-card arm is run, and it is not small.** The four
`exp_specdec_*_gpu` experiments hold the model, the weights, the prefix and the
arithmetic fixed and change only the *shape* of the forward pass. A rerun at a
fixed shape is bitwise identical (`max |Δ logit| = 0.0`, every dtype). Changing
the shape is not: chunked-vs-sequential leaves 13.6% of logit values bitwise
identical on ~190-token sequences, a batch of 4 leaves 9.5%, and **0.88% of
generated tokens change their argmax** with no attack anywhere in the system.
Both knobs act as a *step* — sequence length between 64 and 128, batch size
between 2 and 4 — which is worse for a verifier than a trend, because a
calibration set can sit on the other side of a switch from the traffic it audits
without anything announcing it.

**The single-card arm is now closed, and it comes back negative — which is the
good direction.** `exp_benign_shape_dprime_gpu` scores a replayed honest pool
(80 × 256, Qwen3-1.7B) through `harness.evaluate` under four verifier-side
schedules, paired on the same tokens and the same memoized noise draw. Replaying
the *same* shape returns `d′ = 0.000000` exactly, so the other rows are the
schedule and nothing else:

| replay shape | `d′(benign)`, token_difr |
|---|---:|
| batched with 3 unrelated rows | −0.0061 [−0.0143, +0.0041] |
| a different attention kernel | +0.0019 [−0.0081, +0.0111] |
| token-by-token through a KV cache | −0.0339 [−0.0416, −0.0256] |

The largest positive floor is **+0.0019** against `d′(quant_4bit) = 0.0754`
[0.0493, 0.1044] measured in the same run. Batching audits or swapping attention
kernels does not manufacture false accusations at this scale.

**What that leaves.** Token-by-token replay is *significantly* non-zero and 45% of
the attack in magnitude — it just happens to point away from the attack, and for
`activation_difr` it is −0.61. The finding is therefore "the single-card floor is
small and, here, helpfully signed", not "there is no floor". Nothing in the
mechanism argument makes that sign a property of the hardware rather than a
coincidence of this detector and this deviation.

**Still open, and now the whole of the item:** the cross-GPU arm. Different SKU,
different driver, different kernel selection. The shapes above are a lower bound
on it, and they are the part that needed no new hardware.

**Falsifies the deployment story if:** `d′(benign) ≳ d′(quant_4bit)` *with a
positive sign*. Then no pool size fixes it, because growing the pool sharpens the
false positive exactly as fast as the true one.

**Cost:** the single-card arm is done (~1.3 GPU-hours). A second card for the rest.

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

### 8. Sequential verification — run, and now a pool problem

`exp_sequential_verdict` re-analyses the cost-of-a-verdict score arrays under a
truncated SPRT and an anytime-valid mixture e-process. It works: **1.13–1.98×
fewer tokens**, median 1.31×, against a fixed design bisected to genuinely reach
90% power. Two things came out of it that matter more than the ratio.

**The ceiling applies to a bootstrap over streams.** The first run drew streams of
up to 886% of its own token pool and reported a median 1.40× saving off the back
of them. Enforcing `EvalConfig.max_pool_ratio` leaves **3 of 22 cells resolvable**
at a 20 480-token pool; the other 19 are now reported as unresolvable with the
pool each would need (173 160 honest tokens for the cheapest, 201 M for the
worst). `tests/test_claims.py` fails if an over-ceiling cell is ever reported
again.

**`b*(d′)` is optimistic.** Inside the ceiling, the batch `signal.batch_for_pauc`
nominates realizes 0.05–1.00 power against a 0.90 target. `kv_fp8 /
activation_difr` is the sharp case: `d′ = 2.76` nominates 2 tokens, which deliver
4.8% power, where 7 deliver 94%. Every "tokens per verdict" in the repo is a lower
bound.

**What to run.** A deep-pool arm: 400 prompts × 256 tokens over `honest` +
`quant_4bit` only puts the three headline `quant_4bit` cells inside the ceiling
for ~3.5 GPU-hours (`IVGYM_TAG=deep400`, then `IVGYM_SOURCE=cost_of_a_verdict_deep400
IVGYM_MAXMULT=2`). Beyond that the question is whether the saving ratio is
scale-free in `b` — if it is, the resolvable cells license the rest; if it is not,
every cell needs its own pool and sequential verification is only affordable where
`d′` is already large.

### 9. The clock channel on a production stack

**Claim at risk:** everything in `docs/CLOCK_MEASURED.md`, which is now the measured
replacement for the four clock figures' modelled floors.

`exp_clock_channel_gpu` timed 126 cells on an H100 PCIe in two HF stacks and found
(a) observed ITL = 4.15 ms + bytes / 1.22 TB/s, so the floor is a *stack* property,
(b) the context term is per-position and per-layer, not per-byte — KV bytes vary 6.9x
across five architectures and the slope per layer varies 1.26x — and (c) real NF4
weights deliver 14% of their predicted time saving and are *slower* than bf16 for
every B >= 4. All of that is HF eager and HF + CUDA graphs.

**Experiment.** Rerun the identical grid behind vLLM (paged attention,
flash-decoding, fused kernels, `kv_cache_dtype=fp8`, `quantization=awq/gptq`). Two
outcomes and both are informative: if the context term becomes bandwidth-bound, KV
*precision* re-enters the channel and the differential verifier gets a second
deviation to read; if it does not, the position-not-bytes result generalizes and the
context-slope floor test is the whole channel.

**Falsifies the current record if:** the stack constant is small enough that an
absolute floor test is viable, in which case `fig_clock_basic` panel 3 is right and
`CLOCK_MEASURED` section 1 is an artifact of HF.

**Cost:** ~half a GPU-day, most of it vLLM startup per config.

### 10. The wire: what a client actually observes

**Claim at risk:** the only unmeasured input left in the clock channel. Every jitter
number in `CLOCK_MEASURED` is device-side (sd 0.11-0.8 ms) and is a strict lower
bound on a client's. The differential verifier's price table sweeps sigma from 0.1 to
100 ms rather than assuming one, so the sweep is honest, but nobody knows where on it
a real endpoint sits.

**Experiment.** Time ~50 streaming completions against two or three public endpoints.
Report the inter-token gap distribution, its sd, the coalescing quantum, and the
fraction of *zero* gaps -- section 5 predicts the last one is large, because
speculation and SSE framing both deliver tokens in clumps.

**Cost:** dollars, not GPU-hours. Cheapest load-bearing item on this list.

### 11. Multi-provider / cross-checking

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
