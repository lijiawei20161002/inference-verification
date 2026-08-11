# Does the *fixed* Token-DiFR generalize? A settings sweep

`exp_specdec_difr_gpu` established on **one** configuration that Token-DiFR's honest
null collapses under speculative decoding with a provider-local RNG, and that making
the provider's speculation **seed-coupled** (common random numbers: the draft proposes
`argmax(log q/T + G_t)`, accepted iff it equals `argmax(log p/T + G_t)`) restores it.
This sweep asks whether the fix holds when the deployment varies.

`experiments/exp_specdec_difr_sweep_gpu.py` -> `docs/results/specdec_difr_sweep.jsonl`,
`specdec_difr_sweep_summary.json`, `specdec_difr_identity.json`. H100, fp16 provider,
**bf16 verifier** (a genuinely different implementation of the same weights, so the
cross-stack numeric channel is real and not injected), 16 chat prompts x 192 tokens =
**3 072 tokens per cell**, 58 cells. Standardized pAUC @ FPR <= 0.5% at batch
245 (8.0% of pool, inside the repo's 10% batch/pool ceiling).

Two numbers decide it, both on `ivgym.signal.per_token_stats`' scale (so directly
comparable to `d'(quant_4bit) = 0.0775` elsewhere in the repo):

- **`d'_benign`** -- honest speculative provider vs the honest incremental-decode
  baseline. The false-positive side. The fix works only if it is ~0.
- **`d'_attack`** -- int4 provider vs honest provider *in the same regime*. The power
  side, which must not be paid for the fix.

Per-token scores: `mismatch` = the replay disagrees with the claimed token (DiFR's match
rate), and `margin` = the post-Gumbel deficit of the claimed token in nats (what
`verifiers.token_difr` aggregates). Both are reported; they do not always agree, and
where they disagree is informative.

## 1. The fix holds on every axis swept

Seed-coupled speculation, `d'` and pAUC against the matched incremental-decode null:

| axis | setting | honest mismatch | exact-decode null | `d'_benign` (mism / marg) | pAUC_benign | int4 `d'_attack` | int4 pAUC |
|---|---|---:|---:|---:|---:|---:|---:|
| speculation depth | K = 1 | 0.0101 | 0.0101 | +0.000 / +0.000 | 0.500 | 1.96 / 39 | 1.000 |
| speculation depth | K = 2 | 0.0120 | 0.0101 | +0.020 / +0.019 | 0.503 | 1.79 / 34 | 1.000 |
| speculation depth | K = 4 | 0.0104 | 0.0101 | +0.003 / +0.004 | 0.500 | 1.97 / 37 | 1.000 |
| speculation depth | K = 8 | 0.0094 | 0.0101 | -0.007 / -0.004 | 0.500 | 2.07 / 39 | 1.000 |
| temperature | T = 0.4 | 0.0059 | 0.0078 | -0.022 / -0.019 | 0.499 | 1.94 / 28 | 1.000 |
| temperature | T = 0.7 | 0.0094 | 0.0085 | +0.011 / -0.009 | 0.501 | 1.82 / 25 | 1.000 |
| temperature | T = 1.0 | 0.0104 | 0.0101 | +0.003 / +0.004 | 0.500 | 1.97 / 37 | 1.000 |
| truncation | top-p 0.9 | 0.0101 | 0.0107 | -0.006 / +0.025 | 0.499 | 1.63 / 2 | 1.000 |
| truncation | top-k 50 | 0.0091 | 0.0085 | +0.007 / -0.015 | 0.501 | 2.18 / 93 | 1.000 |
| provider shape | batch of 4 | 0.0107 | 0.0104 | +0.003 / +0.010 | 0.500 | 1.90 / 35 | 1.000 |
| model pair | Qwen2.5-1.5B <- 0.5B | 0.0104 | 0.0101 | +0.003 / +0.004 | 0.500 | 1.97 / 37 | 1.000 |
| model pair | Qwen2.5-3B <- 0.5B | 0.0137 | 0.0133 | +0.003 / +0.000 | 0.502 | 2.28 / 42 | 1.000 |
| model family | Qwen3-1.7B <- 0.6B | 0.0065 | 0.0068 | -0.004 / -0.001 | 0.500 | 2.44 / 56 | 1.000 |
| cross-family draft | 1.5B <- Coder-0.5B | 0.0114 | 0.0101 | +0.013 / +0.013 | 0.502 | 1.83 / 34 | 1.000 |

Across all 12 cells the benign shift is **indistinguishable from zero** --
`|d'_benign|` <= 0.022 on the mismatch scale and
pAUC_benign in [0.499, 0.503]
(chance = 0.500) -- while int4 stays at pAUC 1.000 in every one of them. The fix costs
no power and it does not depend on the depth, the temperature, the truncation rule, the
model, the family, the draft's provenance, or the provider's batch shape.

## 2. What it is a fix *for*: the same cells with a provider-local RNG

| model pair | honest mismatch, exact | honest mismatch, spec | `d'_benign` (mism) | pAUC_benign | int4 `d'_attack` re-calibrated *within* the regime |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B <- 0.5B | 0.0101 | 0.4294 | 4.195 | 1.000 | 0.251 (pAUC 0.948) |
| Qwen2.5-3B <- 0.5B | 0.0133 | 0.3467 | 2.905 | 1.000 | 0.233 (pAUC 0.887) |
| Qwen3-1.7B <- 0.6B | 0.0068 | 0.1488 | 1.722 | 1.000 | 0.341 (pAUC 0.989) |
| 1.5B <- Coder-0.5B | 0.0101 | 0.4066 | 3.967 | 1.000 | 0.331 (pAUC 0.996) |

The benign shift is **larger than the attack signal** in every pair: a verifier
calibrated on incremental decoding flags an honest speculative provider at pAUC 1.000.
Note the last column, which is the more precise diagnosis: if the verifier could
re-calibrate its null *on the provider's actual decoding algorithm*, int4 is still
visible (`d'` 0.23-0.34, pAUC 0.89-1.00) -- so speculation does not destroy the
evidence, it destroys the **portability of the calibration**, and costs 6-8x of the
per-token effect size (2.0 -> 0.25 on the 1.5B pair). Re-calibrating requires the
provider to attest which decoding algorithm it ran, which is exactly the claim under
audit.

## 3. The fix is not statistical -- it is output-identical

Same prompt, same seed, coupled speculation vs plain incremental decoding, comparing
**token sequences directly** (8 prompts x 128 tokens; `--identity`):

| arm | token-identical | sequences fully identical |
|---|---:|---:|
| coupled speculation vs incremental | 0.9668 (min 0.8906) | 66/72 |
| incremental rerun (control) | 1.0000 (min 1.0000) | 72/72 |
| provider-local RNG vs incremental | 0.0583 (min 0.0449) | 0/72 |

Coupled speculation reproduces the reference decode **token for token** -- exactly, all 8
sequences, in 4 of the 9 (K, T) cells (K=1 at both temperatures, K=4 at T=0.7 and T=1.0) --
and where it does not, the cause is a single early flip that then diverges the remainder of
an autoregressive sequence: 6-7 of 8 sequences are still identical end-to-end in those cells. That residual is the chunked-verify-vs-sequential-
decode numeric floor already measured in `exp_specdec_fliprate_gpu` (0.88% of positions),
not an algorithmic gap; it is also why DiFR must score per-token against a teacher-forced
replay of the *claimed* sequence rather than compare sequences. With a provider-local RNG
the sequences share ~5% of tokens, i.e. they diverge at the first sampled position and
never re-converge.

## 4. What the fix costs, and what it does not fix

| | accepted / drafted | tokens per target pass |
|---|---:|---:|
| standard speculation, K=4 | 0.532 | 3.11 |
| coupled speculation, K=4 | 0.477 | 2.88 |
| coupled, K=1 / K=8 | 0.707 / 0.293 | 1.70 / 3.30 |

Coupling costs **7.5% of the speculative speedup**
(2.88 vs 3.11 tokens per target forward pass at K=4);
the speedup itself still scales with K as usual. That is the price of an auditable
provider, and it is small.

Deviations at the base coupled setting (16x192 tokens, honest mismatch 0.0104):

| provider deviation | mismatch | `d'` (mism) | pAUC (mism) | `d'` (marg) | pAUC (marg) |
|---|---:|---:|---:|---:|---:|
| int3 RTN-g128 weights | 0.7607 | 7.390 | 1.000 | 469.73 | 1.000 |
| int4 RTN-g128 weights | 0.2100 | 1.965 | 1.000 | 37.46 | 1.000 |
| int8 RTN-g128 weights | 0.0163 | 0.058 | 0.518 | 0.07 | 0.530 |
| fp8 KV cache | 0.1475 | 1.350 | 1.000 | 45.77 | 1.000 |
| serves the 0.5B, bills the 1.5B | 0.3076 | 2.927 | 1.000 | 89.42 | 1.000 |
| temperature 1.05 instead of 1.0 | 0.0319 | 0.212 | 0.736 | 0.75 | 0.970 |
| typical (lossy) acceptance, claiming coupled | 0.3428 | 3.274 | 1.000 | 145.27 | 1.000 |

Two things the fix does **not** buy, and they are the repo's existing boundary rather than
anything new:

- **int8 is at chance** (`d'` = 0.058, pAUC 0.518). A deviation that moves the
  distribution less than the honest cross-stack floor is invisible however well the seed
  is synchronized. This is the same recompute-dominant wall as `kv_fp8` in the headline grid.
  `int8_gbt_aggregation.md` establishes that this is a property of the *channel*, not of
  the statistic: no aggregator of the cross-stack per-token features (including
  gradient-boosted trees over both replays) clears pAUC 0.537, while replaying in the
  provider's own dtype takes int8 to 0.875 -- coupling the numerics is to int8 what
  coupling the seed is to speculation.
- **A 5% temperature retune** is weak on the match rate (0.736) but strong on the
  margin (0.970). Where a deviation shifts *how badly* tokens miss rather than
  *whether* they miss, the binary match rate throws the signal away -- an argument for
  aggregating the margin, not the match indicator, which is what `verifiers.token_difr` does.

## 5. One real fragility found: nucleus truncation punctures the support

Under top-p 0.9 the honest match rate is unchanged (0.0101) and int4 detection is
unaffected (pAUC 1.000), but the *margin* statistic loses most of its effect size
(`d'_marg` 37.5 -> 1.5). Cause: 4 of 3 072 honest tokens land **outside the verifier's
own nucleus**, because provider and verifier compute the truncation set from logits that
differ in the last bits. Those positions have infinite margin (capped at 30 nats), the
honest margin sd jumps from 0.005 to 1.08, and the standardized effect collapses.
Top-k 50 does not do this (its set membership is far more stable than a cumulative-mass
threshold). This is the failure mode `NEXT_EXPERIMENTS.md` item 5 predicted for
support/rank-based verification, now measured: **it is the truncation rule, not the
seed, that decides whether a continuous DiFR statistic is usable.**

## Caveats

- 3 072 tokens per cell and one prompt bank; the honest mismatch rates (~1%) carry a
  binomial SE of ~0.2 pp, so the `d'_benign` column resolves "is it zero?" and not more.
- The identity arm is 8 sequences per cell. Its `coupled` numbers below 1.000 are
  dominated by *which* sequences suffered one early numeric flip, so the spread across K
  (0.89-1.00) is sampling noise on a per-sequence event, not a trend in K.
- Provider and verifier are the same weights on the same GPU in two dtypes. Cross-GPU
  benign variation (`NEXT_EXPERIMENTS.md` item 1) is still untested and remains the thing
  that could move the honest floor.
- The int4/int8/int3 arms are genuine RTN weight quantization (deterministic, sparse),
  unlike the Gaussian logit model used by the `quant_*` attacks in `ivgym/attacks.py`.
  They are therefore *not* directly comparable to `d'(quant_4bit) = 0.0775`.

