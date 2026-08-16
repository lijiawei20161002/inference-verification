# The clock channel, measured

`fig_clock_basic.png` and its three companions say the same thing in their headers:
*nothing in this repo has ever timed a provider, so read this as the design of
`exp_clock_channel_gpu` and not its result.* This document is that result.

`experiments/exp_clock_channel_gpu.py` times one: a **114-cell** main grid, an
**18-cell** out-of-sample architecture arm and an **8-cell** quantized-KV-cache arm
(126 distinct configurations) of real batch-1..64 decode on an **H100 PCIe** (1.85
TB/s measured copy bandwidth, 15 µs kernel launch), across two stack modes, five
model architectures, real `bitsandbytes` NF4 weights, a real quanto int4 KV cache,
and real greedy speculative decoding. `experiments/plot_clock_measured.py` reads the artifact
(`docs/results/clock_channel.json`, `..._arch.json`, `..._kvq.json`) and produces
`docs/figures/fig_clock_measured.png`. Every latency below is measured; the single
modelled quantity is the client-side jitter in §7, and it is swept rather than
assumed.

**The headline is a swap.** The channel is real, but it is not the channel the
figures drew. The weight row — 4-bit quantization, priced there at 6 tokens of
stream off a 3.9× byte ratio — delivers **14% of its predicted time saving** and
inverts under batching. The context row survives everything, and it is the row the
returned-token channel reads worst.

---

## 1. The floor is a stack property, not a spec sheet

Observed ms/token at B=1, shortest context, CUDA-graph stack:

| config | weight bytes | roofline | observed | observed/roofline |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 1.19 GB | 0.66 ms | 5.37 ms | 8.1× |
| Qwen3-1.7B | 3.44 GB | 1.87 ms | 6.66 ms | 3.6× |
| Qwen3-4B | 8.04 GB | 4.36 ms | 10.86 ms | 2.5× |

A least squares fit over the three gives

```
observed  =  4.15 ms  +  bytes / 1.22 TB/s
```

so the weight read itself runs at **66% of this card's copy bandwidth** — the
roofline is a decent model of the *slope* — but it sits on a **4.15 ms additive
constant** that is pure stack. The figures' floor is the second term with the first
term set to zero.

Two consequences, and they are the whole problem with an absolute latency test:

- **The constant is not on any spec sheet.** In the plain eager-HF stack the same
  GPU, the same weights and the same contexts give **~30 ms per token, flat in
  bytes**. Identical hardware, identical model, and the clock channel does not exist
  at all. A client that does not know which stack it is talking to cannot convert
  "1.04 ms" into a claim about anyone.
- **This card is not the card the figures assume.** H100 *PCIe*, 1.85 TB/s measured,
  against the H100 SXM5 spec sheet's 3.35 TB/s. That alone moves every floor in
  `fig_clock_basic` by 1.8×, which is more than the gap between honest bf16 and the
  4-bit deviation it is supposed to detect.

## 2. Fewer bytes is not proportionally less time

Predicted speedup is the byte ratio; measured is the time ratio; the last column is
the fraction of the predicted *time saving* that actually arrived (graph stack, B=1):

| deviation | bytes saved | time saved | saving realised |
|---|---:|---:|---:|
| real NF4 weights (1.7B, 256 ctx) | 2.55× | **1.10×** | **14%** |
| 0.6B served as 1.7B | 2.83× | 1.24× | 30% |
| 1.7B served as 4B | 2.33× | 1.63× | 68% |
| half the context attended (32k→16k) | 1.35× | **1.81×** | **171%** |

Dequantization spends the bytes it saves. Dropping context over-delivers, because
attention — not the weight read — is the inefficient part of the read.

One subtlety worth stating precisely, because it cuts the other way. The cost law
prices a verdict off the **absolute** gap in ms, not off the ratio — and the absolute
gap survives: `fig_clock_basic` predicts 1.04 → 0.28 ms for 4-bit weights (0.76 ms),
and the measured gap is **0.58 ms**. At the figure's own σ = 0.50 ms that is 10
tokens per verdict against its published 6. So the figure's *price* for the weight
row is roughly right, for the wrong reason: the ratio collapses from 3.9× to 1.10×
because both arms sit on the same 4.15 ms stack constant, and the constant cancels in
a difference. What does not survive is the *impossibility* framing — panel 3's "3.7×
under a floor it cannot cross". There is no visible factor of four here, only a
0.58 ms shift, and detecting it requires knowing the honest floor to better than
0.58 ms, which §1 says a client cannot.

## 3. The context slope reads positions, not bytes

`d(ITL)/d(ctx)`, B=1, graph stack, five architectures chosen for KV geometry:

| model | layers | KV heads | KV B/token | slope µs/tok | ×roofline | ns/tok/layer |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 28 | 8 | 114,688 | 1.649 | 26.7× | 58.9 |
| Qwen3-1.7B | 28 | 8 | 114,688 | 1.648 | 26.6× | 58.9 |
| Qwen3-1.7B NF4 | 28 | 8 | 114,688 | 1.654 | 26.7× | 59.1 |
| Qwen3-4B | 36 | 8 | 147,456 | 2.470 | 31.1× | 68.6 |
| Qwen2.5-1.5B | 28 | **2** | 28,672 | 1.528 | 98.8× | 54.6 |
| SmolLM2-1.7B | 24 | **32** | 196,608 | 1.457 | 13.7× | 60.7 |

Fits are near-perfect: residual 0.03–0.12 ms on a 5–90 ms range.

- **KV bytes vary 6.9× across these models; the slope per layer varies 1.26×.** At a
  fixed 28 layers, bytes vary 4.0× and the slope varies **1.08×**.
- Single-query attention runs at ~4% of peak bandwidth, so the context term is
  per-position and per-layer, **not** per-byte.
- bf16 and NF4 share a slope to 0.4%: the context channel and the weight channel are
  orthogonal *by measurement*.

**Tested directly, with a real quantized cache.** The `kvquant` arm runs the same
model over a genuine quanto **int4 KV cache** — identical positions, 4× fewer cache
bytes — against a bf16 cache in the same eager stack (`clock_channel_kvq.json`,
minima quoted to dodge the one-off quanto JIT compile):

| KV cache | context slope | ITL at 8 192 ctx |
|---|---:|---:|
| bf16 | 0.031 µs/tok | 24.99 ms |
| int4 (quanto) | **0.051 µs/tok** | **31.60 ms** |

Four times fewer cache bytes makes the stream *slower*, by 6.6 ms per token and a
1.6× steeper slope. The sign is the finding; the magnitude is stack-specific, and the
eager stack's own context slope is 50× flatter than the graph stack's, so this arm
settles the direction and not the size. A byte-saving KV deviation is not a latency
deviation.

This is the time-domain twin of the repo's finding #2. There, deviations that change
*what the model can do* are cheap to catch and deviations that change *how precisely
it does it* are not. Here: deviations that change **which positions are read** are
loud, and deviations that change **how those positions are stored** are silent. A
context-truncating provider is maximally visible; a genuine fp8 KV cache is close to
invisible on this stack, and the earlier "half the bytes, half the slope" reasoning
does not survive the cross-architecture arm.

## 4. Batching hides the weight signal in noise, not in arithmetic

Per-request ITL, ctx 1024, graph stack:

| B | Qwen3-1.7B | Qwen3-0.6B | gap | ratio | NF4 vs bf16 |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.95 | 6.64 | 1.31 ms | 1.197× | 1.094× |
| 8 | 11.88 | 10.54 | 1.33 ms | 1.127× | **0.827×** |
| 64 | 43.02 | 41.65 | 1.37 ms | 1.033× | **0.936×** |

The footer of `fig_clock_basic` says a weight attack's floor drops as 1/B. Measured,
that is right about throughput and wrong about the client's observable: the
**absolute** gap in a client's own inter-token time is flat in B (1.31 → 1.37 ms),
because every decode step still reads one copy of the weights. What decays is the
*relative* gap, and what grows is the noise (device-side sd 0.11 ms at B=1, 0.76 ms
at B=64). Batching buries the weight signal; it does not amortize it away.

**NF4 is slower than bf16 for every B ≥ 4** (0.83× at B=8), and at B=64 it delivers
1 392 tok/s against bf16's 1 488. The economic reading matters for the whole channel:
at these sizes, weight quantization does not buy the provider *time*, it buys
*capacity* — fewer or cheaper GPUs for the same model. A latency channel cannot see a
capital saving. (§6 proposes the channel that can.)

## 5. An honest server already lives in the "impossible" region

Real greedy speculative decoding, Qwen3-0.6B drafting for Qwen3-1.7B, k = 4. Greedy
acceptance makes the emitted sequence **exactly** the target's own greedy
continuation, so this is an honest provider by construction. It emits 3.65 of a
possible 5 tokens per verification block, and the client sees them arrive together:

> **73% of an honest provider's client-visible inter-token gaps are 0 ms.**

`fig_clock_basic` panel 3 argues that jitter is additive and positive, so the minimum
gap is clean and one gap left of the floor is physics rather than a p-value. That is
true of *compute* and false of *arrival times*: speculation, multi-token prediction,
SSE frame coalescing and edge buffering all deliver tokens in clumps, and every clump
is a sub-floor gap. Any min-statistic verifier accuses ~every real provider. Mean and
slope statistics are unaffected, which is the argument for §6.

## 6. What survives: differential verifiers

The three measured obstacles — an unknown additive stack constant (§1), unknown and
time-varying co-tenancy (§4), and sub-floor honest gaps (§5) — all attack *absolute*
latency tests and none of them attack a *difference*. Three designs follow, in
descending order of how well the measurements support them.

### 6a. The context-slope floor test (strongest)

**Statistic.** `D = ITL(ctx_hi) − ITL(ctx_lo)` over two probe requests, matched in
everything but prompt length; equivalently the fitted slope `d(ITL)/d(ctx)`.

**Why it survives.** The stack constant, any constant network offset and any constant
padding cancel exactly. Co-tenancy only *raises* the slope — measured, the
per-request slope goes 1.68 → 4.42 µs/tok as the client's own B goes 1 → 8 — so the
honest slope has a **floor at B=1**, and a measured slope *below* that floor is not
attributable to load. That makes it a one-sided test against the most
provider-favourable honest configuration, which is the only kind of latency test that
does not require knowing the provider's hardware, stack or batch size.

**What it detects.** Context truncation, sliding-window eviction, stale prefix-cache
reuse — every deviation that returns fewer attended positions than it charged for.
Not KV precision (§3).

**Price, on the repo's own cost law** (`δ* = 3.767`, honest separation between 256
and 32 768 ctx measured at 53.6 ms):

| client-side jitter σ | half the context | full truncation |
|---:|---:|---:|
| 0.12 ms (measured, device-side) | <1 token | <1 token |
| 1 ms | <1 token | <1 token |
| 10 ms | 8 tokens | 2 tokens |
| 50 ms | 198 tokens | 49 tokens |
| 100 ms | 791 tokens | 198 tokens |

against the returned-token channel's **35 066 tokens** for `kv_fp8` and 2 013 for
`quant_4bit`. Even at 100 ms of wire jitter — implausibly bad — the differential
verifier costs 791 tokens for half the context and 198 for full truncation, i.e.
44× and 177× cheaper than the cheapest returned-token verdict on a KV deviation.

**Limits.** Two probe requests must be served under comparable conditions, so a
provider that fingerprints audit traffic can treat them differently; the probe costs
one long-context prompt; and the slope's *absolute* value is stack-dependent, so the
floor must be calibrated once against a provider-run the client is willing to treat
as honest, or bounded from below by the client's own hardware.

### 6b. The contention slope

**Statistic.** `d(ITL)/dB_self` — how the client's own inter-token time responds to
concurrency the client itself creates.

Measured, this reads the *per-request* work: at ctx 8192 both Qwen3-1.7B and
Qwen3-0.6B degrade at 3.37 ms per added concurrent request, identical to three
digits, so it does **not** discriminate model size. It discriminates context and KV
residency, i.e. the same family as 6a, through a second physical mechanism. Its value
is as an independent check on 6a rather than as new coverage — and it is intrusive,
because it means deliberately loading the provider.

### 6c. The capacity frontier (proposal, not yet run)

The deviation that §4 shows a latency channel cannot see — quantizing weights to fit
the model on fewer GPUs — is visible in *capacity*. A provider's aggregate HBM bounds
`B × ctx × kv_bytes_per_token + weight_bytes`, so the region of (concurrency, context)
it can hold at bounded latency is a physical fingerprint of what it loaded.

Measured anchor for the shape of the observable, ctx 1024, at an ITL target τ:

| τ | Qwen3-0.6B | Qwen3-1.7B | Qwen3-1.7B NF4 |
|---:|---:|---:|---:|
| 12 ms | B ≤ 10.4 | B ≤ 8.2 | B ≤ 5.1 |
| 16 ms | B ≤ 17.0 | B ≤ 14.6 | B ≤ 10.6 |
| 32 ms | B ≤ 46.6 | B ≤ 44.1 | B ≤ 37.8 |

Note the sign: NF4 supports **fewer** concurrent streams at a latency target, not
more. Compute, not bandwidth, is what binds once the weights are small — so a
capacity probe reads quantization with the *opposite* sign to the naive byte
argument, which is exactly the kind of thing that has to be measured rather than
derived. The honest version of this experiment needs a provider one is authorised to
load to saturation; against a commercial endpoint it is indistinguishable from abuse,
and it should not be run without permission.

## 6d. The surviving verifier, examined

`experiments/exp_slope_verifier_gpu.py` (+ `plot_slope_verifier.py`,
`docs/figures/fig_slope_verifier.png`) puts 6a through the four tests a verifier has
to pass: 42 new cells, five architectures that fitted nothing, a real truncating
provider scored through `harness.evaluate`, and a fresh-process re-measurement of the
honest null.

**The floor is NOT predictable from a model card — a negative result.** Fit
`slope = -0.85 + 0.0902 x layers` on the five Qwen configs of §3 and it misses five
unseen architectures by **26% mean, 48% worst**:

| model | layers | KV B/tok | measured µs/tok | predicted | error |
|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B | 16 | 32,768 | 1.155 | 0.596 | −48% |
| TinyLlama-1.1B | 22 | 22,528 | 1.580 | 1.137 | −28% |
| Pythia-1.4B (full MHA) | 24 | 196,608 | 1.119 | 1.318 | +18% |
| OLMo-2-1B | 16 | 131,072 | 0.746 | 0.596 | −20% |
| Qwen2.5-0.5B | 24 | 12,288 | 1.604 | 1.318 | −18% |

Adding KV bytes as a second predictor makes it worse (37% mean). So there is no
absolute physical floor a client can compute off a config file; an absolute test
would have to forgive a quarter of the slope, i.e. forgive a provider that keeps only
three quarters of the context.

**But the endpoint's own null is stable, which rescues it.** Re-measured in a fresh
process an hour later, the same cells drift by **0.01% in slope** and 0.028 ms in the
worst cell — 2 000× tighter than the cross-architecture law. A client calibrating the
slope on its own honest traffic to one endpoint has a floor good to 0.01%, so
sensitivity is bounded by the wire, not by calibration. **The verifier is
self-calibrating and relative, not absolute** — which also means it detects a
provider that *starts* truncating, the realistic threat, rather than one that always
did.

**It detects a real truncating provider on the house protocol.** A provider billing
for 32 768 tokens and holding W, scored through `harness.evaluate` (standardized pAUC
@ FPR ≤ 0.5%, honest calibration split, ceiling enforced at 4.2%), with a pessimistic
50 ms of client-side jitter:

| holds | d′ per probe pair | pAUC @ 32 pairs | tokens of stream for pAUC 0.90 |
|---:|---:|---:|---:|
| 512 | 0.89 | 0.996 | 36 |
| 2 048 | 0.82 | 0.987 | 42 |
| 8 192 | 0.68 | 0.940 | 62 |
| 16 384 (half) | 0.43 | 0.714 | 152 |

Against the returned-token channel's 35 066 tokens for `kv_fp8` and 2 013 for
`quant_4bit`. Measured pAUC tracks `signal.predicted_pauc(d′, b)` along the diagonal,
so the channel is priced by the repo's existing cost law with nothing added.

**Two protocol findings fall out.** Co-tenancy is a *one-sided* nuisance — it only
pushes the honest statistic away from the deviation — so fluctuating load costs power
and cannot manufacture a false accusation; sending the two probes together recovers
1.4× of the verdict price (d′ 2.48 paired vs 2.08 unpaired, simulated over the
measured ITL(B) grid). And the protocol's **winsorization does not transfer** to a
null with no wire jitter: that null is hard-bounded above, the honest 99.9th
percentile cap lands inside the deviation, and d′ reads 1.05 capped against 2.48
uncapped. With a jittered wire the cap never bites (< 0.004 pAUC), but the failure
mode is the same one `evaluate`'s own docstring describes for selective audits.

**Evasion costs the provider its own currency.** To hide from this test a truncating
provider must pad the long request by exactly what truncation saved — up to 53 ms per
output token, 87% of the honest per-token time. It keeps the 3.7 GB of KV memory and
gives back all of the speed. That is what a differential test buys: not a proof, but
a conversion of the cheat from free money into a real cost.

## 7. Latency constraints: four separate ideas, only two of which bind

The question "can a latency contract help?" has four distinct answers.

1. **An SLA alone does not bind at all.** A cheating provider's observable must sit
   below the contracted ceiling and above the honest floor. An honest provider
   satisfies both by assumption, so padding a deviating stream up to the honest
   distribution is always feasible, and it is *free*: padding costs latency, not
   compute, and at B > 1 the GPU spends the wait on other tenants. The repo's
   "concealment has a price" panel prices foregone *product speed*, which a provider
   billing per token does not sell.

2. **A latency floor on the price sheet does bind, and costs nothing to check.**
   This is the latency twin of `fig_price_floor_principle`. A provider advertising a
   per-token latency *faster* than the roofline of the model and hardware it claims
   has published an impossibility, before anyone measures anything. §1 is the caveat
   that makes this usable rather than naive: the check must use the *slope* term
   only, since the stack constant is unknown and strictly adds. The test is therefore
   one-sided and conservative — "no honest configuration of the claimed model on the
   claimed hardware can beat X ms/token" — and it needs no cooperation.

3. **Two-sided φ is what actually squeezes the provider.** Make the spec bind
   latency *and* content: the client contracts a model, a sampling spec, an
   accelerator class, and a latency band. The cheater must then satisfy the band
   with a stream whose *differential* structure (§6a) matches the claimed context —
   and the differential structure is what padding cannot fake, because padding a
   constant leaves the slope where it was. The value of the SLA is not that it
   detects anything; it is that it removes the provider's freedom to hide inside
   "we were slow that day".

4. **The audit's own latency budget is where the clock actually wins.** A recompute
   verdict costs GPU time and adds queueing; the differential clock verdict is read
   off a stream the client already bought, with zero audit FLOPs and zero added
   latency. For an online verifier under a deadline, that is the difference between
   a channel that can run on every request and one that can run on a sample.

## 7b. Reading the same premise somewhere else

Everything above reads the clock in the inter-token gap of a decode step. That is one
observable, and `docs/CLOCK_ALGORITHMS.md` reads the same premise in three others: the
PREFILL clock (compute-bound, and 25 000x cheaper than decode on the substitution row
at matched jitter), the INTRA-STREAM clock (free, and drift-limited on this stack),
and two estimator changes -- a low quantile instead of the mean or the minimum
(2.0x cheaper clean, and the minimum is 4 300x worse on a bursty wire), and a
pseudorandom probe schedule instead of a fixed one.

## 8. What is still unmeasured

- **The wire.** Every jitter number here is device-side (sd 0.11–0.8 ms), which is a
  strict *lower* bound on a client's. §7's table sweeps σ instead of assuming it, but
  the real distribution — including the coalescing quantum that §5 predicts will
  produce sub-floor gaps on any streaming API — has not been measured. This is the
  cheapest and most load-bearing experiment left: time ~50 streaming completions
  against two or three public endpoints, and report σ, the gap histogram, and the
  fraction of zero gaps. It costs dollars, not GPU-hours.
- **A production stack.** Everything here is HF eager and HF + CUDA graphs. vLLM or
  TensorRT-LLM with flash-decoding, paged attention and fused kernels will lower the
  4.15 ms constant and may move the context term back toward bandwidth-bound — which
  would restore fp8-KV detectability and change §3's conclusion. The experiment is a
  rerun of the same grid behind a vLLM backend.
- **A quantized KV cache in a bandwidth-bound stack.** §3 settles the direction with
  a real quanto int4 cache, but only in the eager stack. vLLM's `kv_cache_dtype=fp8`
  with paged attention is the case where the KV term might actually be
  bandwidth-bound, and it is the one that decides whether KV precision is *ever* a
  latency channel.
- **An adaptive provider.** Nothing here is padded, shaped, or audit-aware. The
  slope test's one-sidedness holds against additive nuisance; it has not been
  attacked by a provider that deliberately inflates short-context requests to fake a
  slope, which costs it real latency on its cheapest traffic and is the obvious
  counter-move.

## Reproducing

```bash
python -m experiments.exp_slope_verifier_gpu                       # ~6 min, the 6d arms
IVGYM_SLOPE_TAG=window IVGYM_SLOPE_ARMS=window \
    IVGYM_SLOPE_STEPS=384 IVGYM_SLOPE_REPS=8 \
    python -m experiments.exp_slope_verifier_gpu                   # ~8 min, deep pool
python -m experiments.plot_slope_verifier                          # fig_slope_verifier.png

python -m experiments.exp_clock_channel_gpu                       # ~21 min, writes clock_channel.json
IVGYM_CLOCK_TAG=arch IVGYM_CLOCK_ARMS=arch \
    python -m experiments.exp_clock_channel_gpu                   # ~3 min, the out-of-sample arm
IVGYM_CLOCK_TAG=kvq IVGYM_CLOCK_ARMS=kvquant IVGYM_CLOCK_STEPS=48 IVGYM_CLOCK_REPS=2 \
    python -m experiments.exp_clock_channel_gpu                   # ~2 min, the real int4 KV cache
python -m experiments.plot_clock_measured                         # table + fig_clock_measured.png
```
