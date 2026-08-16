# Same premise, better observables: three more clocks and two better estimators

`fig_clock_basic.png` makes one claim — *a token cannot arrive before its bytes have
moved* — and reads it in one place: the inter-token gap of a batch-1 decode.
`docs/CLOCK_MEASURED.md` measured that place and found it to be the worst one
available: an unknown 4.15 ms stack constant sits on it, real NF4 weights deliver 14%
of their predicted time saving and go *slower* at B ≥ 4, and 73% of an honest
speculative stream lands inside the region the figure labels IMPOSSIBLE.

The premise was never the problem. A decode step is the one part of serving that is
neither compute-bound nor proportional to what the client asked for. This document
keeps the premise and changes where it is read.

`experiments/exp_clock_algos_gpu.py` (+ `plot_clock_algos.py`,
`docs/figures/fig_clock_algos.png`, `docs/results/clock_algos*.json`), same H100 PCIe,
same models.

---

## 1. The prefill clock: the deviation the decode channel lost, recovered

Time to the first token is one prefill forward: `2 × params × prompt` FLOPs of weight
math plus attention. It is **compute-bound**, so unlike a decode step it does not sit
on a launch-latency floor — and it is the client's own prompt length multiplying the
provider's own parameter count.

TTFT in ms (min of 5 reps; relative sd 1.21% at prompts ≥ 4k, tighter than the decode
channel manages):

| prompt | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B | Qwen3-1.7B NF4 |
|---:|---:|---:|---:|---:|
| 256 | 24.7 | 31.6 | 31.0 | 60.8 |
| 4 096 | 45.9 | 76.5 | 162.8 | 78.1 |
| 32 768 | 841.2 | 1 076.7 | 2 516.8 | 1 084.3 |

**The same deviation, seen by three channels** (honest time / deviating time; 1.00× is
a channel that sees nothing):

| deviation | decode ITL, B=1 | decode ITL, B=64 | prefill TTFT |
|---|---:|---:|---:|
| 0.6B served as 1.7B | 1.20× | 1.03× | **1.67×** (at a 4k prompt) |
| real NF4 weights | 1.09× | 0.94× | **0.52×** (at a 256 prompt) |

And priced at a common one-sided wire jitter, per probe pair:

| deviation | channel | Δ | σ = 50 ms | σ = 500 ms |
|---|---|---:|---:|---:|
| 0.6B as 1.7B | decode ITL, B=1 | 1.31 ms | 25 813 pairs | 2 581 303 |
| 0.6B as 1.7B | **prefill, 32k prompt** | **235.6 ms** | **1 pair** | **80 pairs** |
| real NF4 | prefill, 256 prompt | 29.2 ms | 52 pairs | 5 210 |

The substitution row costs **25 813 probe pairs in decode and one in prefill** at
matched jitter; even against a 500 ms queue, prefill is ~32 000× cheaper. That row is
the repo's motivating threat (serving cheaper weights than contracted), and the decode
clock had effectively lost it.

**Two things follow that the original figure does not say.**

- *A deviation does not have to be **fast** to be visible; it has to be **different**.*
  Real NF4 weights **prefill 1.92× slower** at a short prompt, because dequantization
  is compute and prefill is compute-bound. `fig_clock_basic` panel 4 sorts deviations
  into "reads fewer bytes → the clock sees it" and "same bytes → the clock is blind".
  The correct partition is "changes the time signature" vs "does not", which is
  strictly weaker and admits more rows.
- *Where you probe matters as much as what you probe.* The substitution signal peaks
  at a 4k prompt (1.67×) and decays to 1.28× at 32k, because attention overtakes the
  parameter term. The NF4 signal is the reverse: 1.92× at 256, 1.007× at 32k. A probe
  suite should span prompt lengths rather than pick one.

**What this needs from the wire, stated plainly.** These are device-side prefill
times. Real TTFT also contains queueing, which is the dominant and most variable part
of it — far larger than the ITL jitter the decode channel fights. The prefill clock's
advantage is that its *effect* is hundreds of ms rather than ~1 ms, so it tolerates
proportionally more nuisance; its costs are prompt tokens rather than output tokens,
and a constant queue delay cancels only if the client reads the **slope** of TTFT
against prompt length, exactly as in the decode channel's context-slope test.

## 2. The intra-stream clock: the only free probe, and it does not work here

During the client's own generation, the KV cache grows one position per output token,
so an honest ITL must **rise with the output index** and a windowed provider's must
not. No probe requests, no pairing, no absolute floor, no traffic the client was not
already buying — the closest thing in the channel to the figure's promise of "evidence
that arrives with a stream the client already bought".

Measured over 4 096-token generations, honest against a real sliding-window provider
that crops its KV cache to 1 024 every step:

| arm | fitted growth |
|---|---:|
| honest, cache grows | +0.009 µs/token |
| sliding window 1 024 | +0.043 µs/token |
| run-to-run spread of the fit | **±0.059 µs/token** |

The two arms are indistinguishable. This stack's context term is 0.031 µs/token
(eager, measured in `CLOCK_MEASURED` §3), which is *below* the run-to-run spread of
the fitted slope — that spread is GPU clock ramp, not the KV cache. A CUDA-graph
stack's context term is 1.65 µs/token, 30× this noise floor, but a padded static cache
does not grow by construction, so the growth is only readable on a server that attends
valid positions only. **The arm is a negative result with a threshold attached:** the
intra-stream clock needs a stack whose per-position term exceeds ~0.06 µs/token of
drift, and it should be re-run against vLLM's paged attention, where it would be the
cheapest verifier in the repo.

## 3. Read it with a low quantile — not the mean, and not the minimum

`fig_clock_basic` panel 3 gets the nuisance model right (additive, positive,
one-sided) and draws the wrong conclusion from it (take the **minimum**). The
one-sidedness is real and it does mean the *low order statistics* of a probe block
beat its mean. The minimum is simply the wrong one to pick.

Tokens of stream per pAUC-0.90 verdict, 32-pair blocks, same measured samples:

| σ (ms) | mean | 20% trimmed | median | 25th pct | 10th pct | minimum |
|---:|---:|---:|---:|---:|---:|---:|
| 25 | 15 | 13 | 17 | 7 | 5 | 2 |
| **50** | **61** | 47 | 62 | **31** | 17 | **9** |
| 100 | 229 | 227 | 252 | 136 | 79 | 38 |

The same, with 15% of arrivals collapsed into bursts (what speculation and SSE frame
coalescing do — measured at 73% zero gaps in `fig_clock_measured` (E)):

| σ (ms) | mean | 20% trimmed | median | 25th pct | 10th pct | minimum |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 58 | 50 | 55 | **40** | 382 | **38 823** |

The minimum is 6.9× cheaper than the mean on a clean wire and **4 300× worse** on a
bursty one: one collapsed gap per block is enough to destroy it. The 25th percentile
is 2.0× cheaper than the mean clean and 1.4× cheaper bursty. **Ship the quantile.**
This is an estimator change — no extra data, no extra probes — and it is the single
cheapest improvement to the channel in this document.

## 4. Choose the probe schedule, not just the statistic

Paired probes fix co-tenancy (`CLOCK_MEASURED` §6d) but not a *drift*, and a strictly
alternating probe aliases onto a period-2 scheduler. With the nuisance realization
unknown to the client — a drift of random slope and sign, or a period-2 interferer of
random phase — d′ per 64-probe block:

| nuisance | two blocks | alternating | pseudorandom |
|---|---:|---:|---:|
| load drifts during the probe | 2.64 | 3.53 | 3.45 |
| load alternates (period-2 scheduler) | 548.59 | **1.55** | 12.48 |
| **worst case** | 2.64 | 1.55 | **3.45** |

No fixed schedule is robust to both. A pseudorandom context schedule correlated
against the returned latencies — lock-in detection — is never the worst: within 1.02×
of the best under drift and 8× better than alternating under a period-2 interferer. It
costs nothing but choosing the schedule.

## 5. What this amends in `fig_clock_basic`

| the figure says | measured |
|---|---|
| read the inter-token gap | read the **prefill** too; it is 25 000× cheaper on the substitution row |
| the saving is the signal | the *difference* is the signal — NF4 is visible by being **slower** |
| take the minimum gap | take a **low quantile**; the minimum is 4 300× worse on a bursty wire |
| one probe design | the **schedule** is a design variable; pseudorandom wins the worst case |
| four rows the clock sees, three it cannot | the row-to-channel map depends on prompt length and on which clock |

Unchanged, and now measured in four places: bytes are time, so a provider that does
less work than it billed for has a time signature. Every deviation in this document is
visible somewhere in the time dimension; none of them is visible *everywhere*, and the
verifier's job is picking where to look.

## 6. Still open

- **The wire.** Everything here is device-side. The prefill clock in particular
  competes with queueing delay, which nothing in this repo has measured.
- **A paged-attention stack.** It decides the intra-stream arm (§2) and would move the
  estimator table (§3) by changing the burst rate a client actually sees.
- **An adaptive provider.** None of these probes has been attacked. The prefill clock's
  obvious counter-move is to pad short prompts, which costs the provider TTFT on its
  cheapest traffic; that trade has not been priced.

## Reproducing

```bash
IVGYM_ALGO_ARMS=ttft  python -m experiments.exp_clock_algos_gpu     # ~2 min
IVGYM_ALGO_ARMS=intra IVGYM_ALGO_OUT=4096 IVGYM_ALGO_WARMUP=256 \
    IVGYM_ALGO_REPS=6 python -m experiments.exp_clock_algos_gpu     # ~11 min
python -m experiments.plot_clock_algos                              # table + figure
```
