# The MVP inference-verification protocol: mechanism, performance, cost

This document explains the **end-to-end MVP** that lives in the four top-level
files — [`model.py`](../model.py), [`commit.py`](../commit.py),
[`protocol.py`](../protocol.py), and the measurement harness
[`experiment.py`](../experiment.py) — and reproduced in
[`docs/results/exp_protocol_measurement.txt`](results/exp_protocol_measurement.txt).

Where `ivgym/` studies the **detector** in isolation (given a claimed transcript
and a re-run, how well does a per-token score separate honest from cheating —
see [GAME.md](GAME.md)), the MVP closes the *whole loop*: a provider that
actually saves compute by cheating, a cryptographic commitment that stops it
from lying after the fact, a client that audits only a random sample, and a
FLOP-exact accounting of what both sides pay. It answers the question the
detector alone cannot: **how much does verification cost, and how little of it
do you need?**

Everything below is *measured*, not estimated. Every FLOP count comes from a real
matmul counter ([`model.py:26`](../model.py)); every detection number comes from
really serving, committing, and auditing a 6-layer transformer.

---

## 1. The threat model in one paragraph

A **provider** claims it served your requests with model `M` under a fixed
sampling spec. It is paid per request but its cost is dominated by the forward
pass, so it has a standing incentive to quietly run something cheaper — a
quantized copy, half the layers, a stubbed sampler — and pocket the difference.
The MVP models the cleanest version of this: the honest provider runs all
`n_layer` layers; the **cheating** provider early-exits after `n_layer // 2`
layers, a real approximation that **saves 48.7% of generation FLOPs**
([`protocol.py:32`](../protocol.py), measured in Phase 1). The **client** wants
to catch that deviation while (a) paying far less than re-running everything and
(b) almost never falsely accusing an honest provider.

---

## 2. The protocol

Three moving parts, each in its own file.

### 2.1 Serve — a real, cost-asymmetric forward pass (`model.py`)

The model is a small GPT-style transformer in pure numpy so that **every**
multiply-add is counted exactly (`mm()` increments a global FLOP counter). It
runs in two modes that mirror production:

- **`generate`** — autoregressive greedy decoding with a KV cache: an `L`-token
  prefill plus `T` sequential single-token steps. This is what the provider does
  to *serve* a request, and it is what gets cheapened when the provider early-exits.
- **`forward_full`** — one teacher-forced parallel pass over `prompt + tokens`,
  causal-masked. This is what the client does to *re-score* a committed
  transcript in a single shot.

A `Transcript` records the prompt, the emitted tokens, the per-token logits, and
the real `gen_flops` the provider spent ([`protocol.py:22`](../protocol.py)).

### 2.2 Commit — bind the transcripts before the audit (`commit.py`)

The provider hashes every transcript into a leaf and builds a **SHA-256 Merkle
tree**, publishing only the root. The logits are quantized to a grid (`quant =
1e-3`) before hashing so the committed value is one the client can re-derive; the
quantization step is the floor of the tolerance band (§2.4).

The Merkle root is the entire soundness mechanism. Because the provider must
publish the root **before** it learns which requests will be audited, it cannot
see the audit set and then swap a cheated transcript for an honest one — any such
swap changes a leaf, hence the root, and fails the authentication-path check.
This is what pins the per-request catch probability to exactly the audit rate `q`
rather than something the provider can drive toward zero by guessing the audit
set ([`commit.py:1`](../commit.py)).

### 2.3 Audit — re-score a random sample (`protocol.py`)

The client picks each request independently with probability `q` (its private
coins, flipped *after* the root is fixed). For each audited request it does two
things ([`protocol.py:46`](../protocol.py)):

1. **Binding check** — verify the Merkle authentication path against the
   published root. A tampered or non-committed transcript fails here.
2. **Re-score check** — run one full teacher-forced `forward_full` over
   `prompt + committed tokens`, and compare the recomputed logits to the
   committed ones. Flag the batch if the per-token logit `L∞` distance exceeds a
   tolerance `eps`.

A cheat is **caught** iff it is both *audited* and *flagged*.

### 2.4 Calibrate — tolerance vs. hardware nondeterminism

Real inference is not bit-exact across hardware, so the client's re-score is
modeled as the true logits plus Gaussian noise `sigma`
([`protocol.py:82`](../protocol.py)). The tolerance `eps` must sit **above** the
honest noise floor (so honest providers aren't flagged, keeping the
false-positive rate `α` low) and **below** the deviation a real cheat induces (so
cheats aren't missed, keeping the miss rate `δ` low). Phase 1 of the experiment
sweeps `eps` and reports both rates from real audits.

---

## 3. Performance

All numbers from `docs/results/exp_protocol_measurement.txt` (6-layer model,
`d_model=128`, `L=16`, `T=32`, `sigma=0.01`, `quant=1e-3`).

### 3.1 The honest/cheat gap is enormous, so calibration is trivial

| quantity | value |
|---|---|
| honest re-score `L∞` (noise only) | ≈ **0.040** |
| cheat re-score `L∞` (half the layers) | ≈ **0.70** |
| cheat token-argmax mismatch | **100%** of transcripts |

The honest signal and the cheat signal are **~18× apart**, so any `eps` in the
whole band `[0.05, 0.50]` gives `α = 0.000` false positives and `δ = 0.000`
misses. The MVP operates at `eps = 0.1`, where **per-cheat detection `(1−δ) =
1.000`**. Early-exit is a coarse cheat by design — it corrupts the forward pass
enough to flip the sampled token every time — so this is the *easy* regime; the
subtler forward-pass and sampler cheats studied in `ivgym/` are where detector
choice starts to matter.

### 3.2 The batch catch probability matches the closed form

With per-cheat detection `(1−δ)`, audit rate `q`, and `fN` cheats in a batch of
`N`, the probability that *at least one* cheat is caught is

```
P = 1 − (1 − q·(1−δ))^(fN)
```

Measured against theory (`N = 500`, `f = 0.02`, so 10 cheats/batch), every audit
being a *real* re-score plus a *real* Merkle-proof check:

| audit rate `q` | audits/batch | P empirical | P theory |
|---|---|---|---|
| 0.05 | 25 | 0.395 | 0.401 |
| 0.10 | 50 | 0.650 | 0.651 |
| 0.20 | 100 | 0.910 | 0.893 |
| 0.35 | 175 | 1.000 | 0.987 |
| 0.50 | 250 | 1.000 | 0.999 |

Empirical tracks theory to within Monte-Carlo noise, with **zero false alarms and
zero binding failures** across the sweep.

---

## 4. The cost tradeoff

This is the heart of the MVP: verification is only interesting if the client pays
much less than the provider it is policing.

### 4.1 The three FLOP quantities

| quantity | FLOPs/request | ratio |
|---|---|---|
| honest generation (full 6 layers) | 120,004,608 | 1.00× |
| **cheat** generation (3 layers) | 61,575,168 | **0.51×** (saves 48.7%) |
| **verify** (one teacher-forced pass) | 123,469,824 | β = **1.03×** |

Two facts drive everything:

- **`β ≈ 1.03`.** Re-scoring *one* request costs about the same as generating it
  — the parallel teacher-forced pass over `L+T` tokens is comparable to the
  prefill+decode the provider ran. So auditing *every* request would cost ~100%
  of the provider's compute: pointless. The win must come from **sampling**.
- **The cheat saves ~half.** That is the provider's incentive, and the size of
  what the client is trying to deter.

### 4.2 The key invariant: audit *count* is independent of batch size

To hit a target batch catch probability `P` you need audit rate
`q = ln(1/(1−P)) / (fN·(1−δ))`. Multiply by `N` and the `N` cancels:

```
audit count  qN = ln(1/(1−P)) / (f·(1−δ))
```

The number of requests you must re-score depends only on the **cheat fraction
`f`** and the **target `P`** — **not on how many requests you served.** Phase 3
confirms this: `f = 0.01` needs 460 audits whether `N = 10,000` or `N =
1,000,000`, and a Monte-Carlo check hits `P = 0.99` at that fixed count for both
`N = 5,000` and `N = 50,000`. This is what makes verification *scale*: the client
cost is flat in traffic volume.

### 4.3 The deliverable: client cost as a fraction of full re-execution

Client FLOPs `= audit_count × verify_flops`. Compared against the cost of naively
re-running an `N = 1,000,000`-request batch:

| cheat fraction `f` | target `P` | audits needed | client GFLOPs | **fraction of full re-exec** |
|---|---|---|---|---|
| 0.10 | 0.99 | 46 | 5.7 | **0.005%** |
| 0.10 | 0.999 | 69 | 8.5 | 0.007% |
| 0.01 | 0.99 | 461 | 56.9 | **0.047%** |
| 0.01 | 0.999 | 691 | 85.3 | 0.071% |
| 0.001 | 0.99 | 4,605 | 568.6 | **0.474%** |
| 0.001 | 0.999 | 6,908 | 852.9 | 0.711% |

**The takeaway.** Even against a rare cheater (0.1% of requests) the client
catches 99% of cheating batches for **under half a percent** of the cost of
re-running the workload; against a 1%-cheater it is **~0.05%**. The tradeoff is
governed entirely by two knobs:

- **Lower `f` (rarer cheating) ⇒ more audits.** Cost scales as `1/f`, because you
  must sample enough requests to *hit* a cheat. This is the fundamental tension:
  the stealthier the cheater, the more you pay — but the provider that cheats on
  fewer requests also steals proportionally less.
- **Higher `P` (more confidence) ⇒ more audits,** but only logarithmically
  (`ln(1/(1−P))`): going from 99% to 99.9% confidence costs just ~50% more audits.

Because `β ≈ 1` and `(1−δ) = 1` here, the cost is almost purely `ln(1/(1−P)) /
f` re-scores. A cheaper detector (a black-box I/O test, or the speculative-decode
*trace* check that needs no recompute — see the README and
[SPEC_DECODE_TRACE_VERIFICATION.md](SPEC_DECODE_TRACE_VERIFICATION.md)) lowers the
per-audit `β`, but the `1/f · ln(1/(1−P))` **count** is a property of the
inspection game, not the detector.

---

## 5. Assumptions and where the MVP stops

- **The cheat is coarse.** Half-the-layers flips the argmax on 100% of tokens, so
  `(1−δ) = 1`. A cheat tuned to stay just inside `eps` would raise `δ` and thus
  the audit count by `1/(1−δ)`; measuring that harder regime is exactly what the
  attack/defense registries in `ivgym/` are for.
- **Nondeterminism is modeled as i.i.d. Gaussian** on the re-score with a fixed
  `sigma`. Real cross-hardware divergence is heavier-tailed; `eps` would be set
  from a measured honest-null distribution rather than assumed.
- **One detector, one score.** The MVP uses a raw-logit `L∞` band. It does not
  yet compose detectors or use the seed-synced Gumbel machinery from `ivgym`.
- **Commitment is honest-size.** The Merkle tree binds transcripts but the MVP
  does not model a provider that commits to fewer requests than it served, or
  withholds the root — those are availability concerns layered on top of the
  soundness the tree provides.

### Reproduce

```bash
# from inference-verification/  (pure numpy + matplotlib; no GPU, ~1 min)
python experiment.py
# prints Phases 1–4 and writes docs/figures/fig_protocol_measurement.png
```
