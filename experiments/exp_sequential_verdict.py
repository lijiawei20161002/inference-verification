"""Fewer tokens for the same verdict: a sequential test against the fixed batch.

Every verdict in this repo is a FIXED-`b` test: draw `b` tokens, average, compare
against a threshold calibrated to FPR <= 0.5%. `signal.batch_for_pauc` says how big
`b` must be, and `exp_cost_of_a_verdict_gpu` multiplies that by the measured price
of a token to get what a verdict costs. This experiment attacks the OTHER factor:
a fixed-`b` test spends all `b` tokens even when the first 200 already settled it.

A verifier watching a provider is in a sequential situation, not a fixed-sample
one. The classical result (Wald) is that a sequential probability ratio test
reaches the same error pair `(alpha, beta)` at 2-3x fewer samples in expectation,
because most streams are decided early and only the ambiguous ones run long.
Nothing about the detector changes -- this is the same per-token evidence, spent
better.

Three designs, all at FPR <= 0.5% and power 0.90, all on the SAME per-token score
arrays measured by `exp_cost_of_a_verdict_gpu`:

  fixed        b = (delta*/d')^2 tokens, threshold calibrated out of sample. The
               repo's current protocol.
  sprt         Wald's SPRT on the Gaussian log-likelihood ratio, which needs the
               alternative `d'` -- an ORACLE design, and an upper bound on what
               sequential testing can buy.
  e_value      a mixture e-process (`E_n = (1+n tau^2)^-1/2 exp(tau^2 S_n^2 /
               (2(1+n tau^2)))`, reject when `E_n >= 1/alpha`). Anytime-valid by
               Ville's inequality with NO knowledge of `d'`, which is the honest
               deployment design: the verifier does not know how badly it is being
               cheated before it starts.

Calibration is out of sample exactly as `harness.evaluate` does it: the boundary
is set on an honest CALIBRATION split and the false-alarm rate is measured on a
disjoint honest split. The design is then TRUNCATED at the smallest `N_max` whose
power is still 0.90, so the sequential and fixed designs are compared at matched
error rates rather than at "whatever the boundary happened to give".

Streams are i.i.d. draws from the token pools -- the same independence assumption
`ivgym/signal.py` makes to turn `d'` into a batch size, so fixed and sequential are
compared on identical footing. Tokens inside one sequence are correlated in
reality; both numbers inherit that optimism, and their RATIO is the result.

**The batch/pool ceiling applies here and it binds hard.** A stream is drawn WITH
replacement from a finite token pool, so once its length approaches the pool the
draw stops being a fresh sample of the score distribution and becomes a re-shuffle
of the same tokens: the honest variance collapses onto the pool mean and the
realized false-alarm rate is set by the gap between the calibration and evaluation
splits rather than by the threshold. That is the failure `EvalConfig.max_pool_ratio`
(10%) exists to prevent, documented in the README under "The batch/pool ceiling",
and a bootstrap over streams is exactly as subject to it as `harness.evaluate` is.
The FIRST version of this experiment did not check it and simulated batches of up
to 886% of its pool; the realized false-alarm rates it printed ran from 0.0000 to
0.8065 against a 0.005 budget, which is what an over-ceiling measurement looks like
from the inside. So a cell is only reported if BOTH designs stay inside the
ceiling -- the fixed batch `b`, and the sequential design's truncation `n_max`.
Everything else is reported as **unresolvable at this pool**, with the pool it
would need, which is the honest output and not a missing one.

Because the ceiling binds, the fixed design is also matched rather than assumed:
`b*(d')` is what `signal.batch_for_pauc` NOMINATES, and the realized `(FPR, power)`
at that `b` is measured, not asserted. `fixed_matched` then bisects for the
smallest `b` that truly reaches `POWER` out of sample, so the sequential saving is
a ratio between two designs that both hit the spec.

    python -m experiments.exp_sequential_verdict        # no GPU, cells run in parallel

Env: IVGYM_SIMS(4000), IVGYM_CALMULT(4), IVGYM_BLOCKSPERB(32), IVGYM_MAXMULT(6),
     IVGYM_MAXB(200000), IVGYM_WORKERS(16), IVGYM_RATIO(from EvalConfig, 0.10),
     IVGYM_SOURCE(cost_of_a_verdict -- the artifact stem to re-analyse).
Reads `docs/results/<IVGYM_SOURCE>{.json,_scores.npz}`; writes
`docs/results/sequential_verdict[_<tag>].json`.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import harness, signal

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results"

N_SIMS = int(os.environ.get("IVGYM_SIMS", 4000))
CAL_MULT = int(os.environ.get("IVGYM_CALMULT", 4))   # calibration streams per eval stream
BLOCKS_PER_B = int(os.environ.get("IVGYM_BLOCKSPERB", 32))  # stopping granularity
MAX_MULT = float(os.environ.get("IVGYM_MAXMULT", 6))  # truncation, in units of b_fixed
MAX_B = int(os.environ.get("IVGYM_MAXB", 200_000))   # largest cell worth simulating
# the same ceiling `harness.evaluate` enforces, read from the library so the two
# cannot drift apart
MAX_RATIO = float(os.environ.get("IVGYM_RATIO", harness.EvalConfig.max_pool_ratio))
# which cost run to re-analyse: the wide 5-attack grid by default, or a deep-pool
# run written under a tag by `exp_cost_of_a_verdict_gpu`
SOURCE = os.environ.get("IVGYM_SOURCE", "cost_of_a_verdict")
OUT_NAME = ("sequential_verdict.json" if SOURCE == "cost_of_a_verdict"
            else f"sequential_verdict_{SOURCE.replace('cost_of_a_verdict_', '')}.json")
ALPHA = 0.005
POWER = 0.90
TAU = 0.05          # e-value mixture width: prior sd on the standardized shift


def winsorized(h: np.ndarray, a: np.ndarray, pct: float = 99.9):
    """The scale `harness.evaluate` scores on: winsorize both sides at the honest
    99.9th percentile, then standardize by the honest mean and sd."""
    cap = np.percentile(h[np.isfinite(h)], pct)
    h, a = np.minimum(h, cap), np.minimum(a, cap)
    mu, sd = h.mean(), h.std() + 1e-12
    return (h - mu) / sd, (a - mu) / sd


def block_sums(pool: np.ndarray, n_sims: int, n_blocks: int, block: int,
               rng, budget: int = 40_000_000) -> np.ndarray:
    """`[n_sims, n_blocks]` sums of i.i.d. token draws, `block` tokens per block.

    Drawn in chunks of streams: the full draw array is `n_sims x n_blocks x block`
    and runs to billions of entries for a weak cell (`b` large -> long streams),
    while the sums it collapses to are three orders of magnitude smaller. The
    chunk is sized from a fixed element BUDGET rather than a fixed number of
    streams, so a cell with 20x the stream length uses 20x fewer streams per chunk
    instead of 20x the memory.
    """
    out = np.empty((n_sims, n_blocks))
    chunk = max(1, min(n_sims, budget // max(n_blocks * block, 1)))
    for i in range(0, n_sims, chunk):
        m = min(chunk, n_sims - i)
        idx = rng.integers(0, len(pool), size=(m, n_blocks, block))
        out[i:i + m] = pool[idx].sum(axis=2)
    return out


def batch_mean(pool, n_sims, b, rng, budget: int = 40_000_000):
    out = np.empty(n_sims)
    chunk = max(1, min(n_sims, budget // max(b, 1)))
    for i in range(0, n_sims, chunk):
        m = min(chunk, n_sims - i)
        out[i:i + m] = pool[rng.integers(0, len(pool), size=(m, b))].mean(1)
    return out


def fixed_design(h_cal, h_ev, a, b, rng, n_sims):
    """The repo's protocol: threshold from the honest CALIBRATION split at
    1 - alpha, false-alarm and power measured on disjoint splits."""
    cal = batch_mean(h_cal, n_sims * CAL_MULT, b, rng)
    tau = float(np.quantile(cal, 1 - ALPHA))
    ev = batch_mean(h_ev, n_sims, b, rng)
    at = batch_mean(a, n_sims, b, rng)
    return {"tokens": int(b), "threshold": tau,
            "fpr": float(np.mean(ev > tau)), "power": float(np.mean(at > tau))}


def matched_fixed_design(h_cal, h_ev, a, b_hint, rng, n_sims, b_cap):
    """Smallest fixed `b` that really reaches `POWER` out of sample, by bisection.

    `b*(d') = (delta*/d')^2` is a NOMINATION from a Gaussian model of the batch
    mean. On real winsorized scores the realized power at that `b` came in
    anywhere from 0.05 to 1.00, so comparing a sequential design tuned to exactly
    0.90 power against a fixed design that happens to deliver 0.64 is not a
    comparison of costs -- it is a comparison of two different tests. This finds
    the fixed design that delivers what was asked, so the ratio means something.

    Returns None if no `b` at or below `b_cap` (the ceiling) reaches `POWER`.
    """
    hi = min(int(MAX_MULT * b_hint), b_cap)
    if hi < 1:
        return None
    top = fixed_design(h_cal, h_ev, a, hi, rng, n_sims)
    if top["power"] < POWER:
        return None
    lo, best = 1, {**top, "b": hi}
    while hi - lo > max(1, int(0.02 * hi)):
        mid = (lo + hi) // 2
        r = fixed_design(h_cal, h_ev, a, mid, rng, n_sims)
        if r["power"] < POWER:
            lo = mid
        else:
            hi, best = mid, {**r, "b": mid}
    return best


def sequential_design(kind, h_cal, h_ev, a, d_prime, n_max, block, rng, n_sims):
    """Calibrate a stopping boundary on honest calibration streams, then measure
    false-alarm rate, power and stopping time on disjoint honest / attack streams.

    Returns None if the design cannot reach `POWER` within `n_max` tokens.
    """
    n_blocks = max(int(np.ceil(n_max / block)), 1)
    nb = np.arange(1, n_blocks + 1) * block           # tokens after each block

    def statistic(sums):
        """Running evidence after each block, `[n_sims, n_blocks]`."""
        S = np.cumsum(sums, axis=1)                   # sum of standardized scores
        if kind == "sprt":
            # Gaussian LLR for N(0,1) vs N(d,1): d*S_n - n*d^2/2
            return d_prime * S - nb * d_prime ** 2 / 2.0
        # log of the one-sided normal-mixture e-value (log for numerical range)
        denom = 1.0 + nb * TAU ** 2
        return -0.5 * np.log(denom) + (TAU ** 2 * np.maximum(S, 0) ** 2) / (2 * denom)

    cal = statistic(block_sums(h_cal, n_sims * CAL_MULT, n_blocks, block, rng))
    # boundary = the (1-alpha) quantile of each honest stream's RUNNING MAXIMUM,
    # which is what a stream must exceed at some point to raise a false alarm
    boundary = float(np.quantile(cal.max(axis=1), 1 - ALPHA))

    def run(pool):
        st = statistic(block_sums(pool, n_sims, n_blocks, block, rng))
        crossed = st >= boundary
        any_cross = crossed.any(axis=1)
        first = np.where(any_cross, crossed.argmax(axis=1), n_blocks - 1)
        return any_cross, (first + 1) * block

    fp, _ = run(h_ev)
    hit, stop = run(a)
    if hit.mean() < POWER:
        return None
    return {"kind": kind, "boundary": boundary, "n_max": int(n_max),
            "block": int(block),
            "fpr": float(fp.mean()), "power": float(hit.mean()),
            "mean_tokens_under_attack": float(stop[hit].mean()) if hit.any() else float("nan"),
            "median_tokens_under_attack": float(np.median(stop[hit])) if hit.any() else float("nan"),
            "p90_tokens_under_attack": float(np.quantile(stop[hit], 0.9)) if hit.any() else float("nan")}


def tightest_truncation(kind, h_cal, h_ev, a, d_prime, b_fixed, block, rng, n_sims):
    """Smallest truncation `N_max` whose design still reaches `POWER`.

    Matching the designs matters: an untruncated sequential test with a
    conservative boundary can beat the fixed test on tokens while quietly
    delivering more power than was asked for, and the comparison would then be
    between two different questions. Bisection over `N_max` in blocks.
    """
    lo, hi = block, int(MAX_MULT * b_fixed)
    best = sequential_design(kind, h_cal, h_ev, a, d_prime, hi, block, rng, n_sims)
    if best is None:
        return None
    while hi - lo > max(block, 0.02 * hi):
        mid = int((lo + hi) / 2)
        r = sequential_design(kind, h_cal, h_ev, a, d_prime, mid, block, rng, n_sims)
        if r is None:
            lo = mid
        else:
            hi, best = mid, r
    return best


def analyze_cell(atk: str, det: str, b: int, d: float, price: float,
                 h_raw: np.ndarray, a_raw: np.ndarray) -> dict:
    """Fixed and sequential designs for one (attack, verifier) cell.

    Self-contained and seeded from the cell's own name, so cells are independent
    of each other and of the order they are evaluated in -- which is what lets
    them run in parallel (a weak cell simulates ~1e9 token draws and there is no
    reason to wait for it) without the result depending on the pool's scheduling.
    """
    rng = np.random.default_rng(zlib.crc32(f"{atk}/{det}".encode()))
    h, a = winsorized(h_raw, a_raw)
    cut = len(h) // 2
    perm = rng.permutation(len(h))
    h_cal, h_ev = h[perm[:cut]], h[perm[cut:]]

    # the ceiling, in tokens: no design may draw a stream longer than this share
    # of the smallest pool it resamples
    b_cap = int(MAX_RATIO * min(len(h_cal), len(h_ev), len(a)))

    # stopping granularity scales with the cell: a test that can only stop
    # every b/32 tokens cannot report a saving finer than 3%
    block = int(np.clip(b // BLOCKS_PER_B, 1, 512))
    fx = fixed_design(h_cal, h_ev, a, b, rng, N_SIMS)
    fm = matched_fixed_design(h_cal, h_ev, a, b, rng, N_SIMS, b_cap)
    sp = tightest_truncation("sprt", h_cal, h_ev, a, d, b, block, rng, N_SIMS)
    ev = tightest_truncation("e_value", h_cal, h_ev, a, d, b, block, rng, N_SIMS)
    # of the two sequential designs, prefer the anytime-valid one; it is the design
    # a verifier could actually deploy, and sprt is only the oracle upper bound
    best = ev or sp
    # A design is only reported if the longest stream it draws stays inside the
    # ceiling. `b` is the fixed design's; `n_max` is the sequential design's, and
    # it is the truncation, not the mean stopping time -- a stream that runs to the
    # horizon has drawn that many tokens whether or not the average one did.
    b_ratio = b / min(len(h_ev), len(a))
    seq_ratio = (best["n_max"] / min(len(h_ev), len(a))) if best else float("inf")
    within = (b <= b_cap) and (best is not None) and (best["n_max"] <= b_cap)
    saving = (b / best["mean_tokens_under_attack"]) if best else None
    saving_matched = (fm["b"] / best["mean_tokens_under_attack"]
                      if (best and fm) else None)
    return {"d_prime": d, "fixed": fx, "fixed_matched": fm,
            "sprt": sp, "e_value": ev, "block": block,
            "b_cap": b_cap, "batch_pool_ratio": b_ratio,
            "seq_pool_ratio": seq_ratio, "within_ceiling": bool(within),
            "usd_per_token": price,
            "usd_fixed": b * price,
            "usd_sequential": (best["mean_tokens_under_attack"] * price
                               if best else None),
            # the nominal saving is against b*(d'), which may not deliver POWER;
            # the matched one is against a fixed design that does. Only the matched
            # one is a like-for-like cost ratio.
            "saving_x": saving, "saving_x_matched": saving_matched}


def main() -> None:
    cost = json.loads((RES / f"{SOURCE}.json").read_text())
    z = np.load(RES / f"{SOURCE}_scores.npz")

    print("=" * 92)
    print("Sequential vs fixed-batch verdicts   "
          f"(alpha = {ALPHA:.1%}, power = {POWER:.0%}, boundary checked "
          f"{BLOCKS_PER_B} times per fixed batch)")
    print(f"  scores: {cost['M']} pool of {cost['pool_tokens_per_config']} tokens "
          f"per config, {N_SIMS} simulated streams per design")
    print("=" * 92, flush=True)

    pool_tokens = int(cost["pool_tokens_per_config"])
    split = pool_tokens // 2                 # evaluate splits honest 50/50
    b_cap = int(MAX_RATIO * min(split, pool_tokens))
    out = {"alpha": ALPHA, "power_target": POWER, "blocks_per_b": BLOCKS_PER_B,
           "n_sims": N_SIMS, "cal_mult": CAL_MULT,
           "tau_mixture": TAU, "max_tokens_per_verdict": MAX_B,
           "max_pool_ratio": MAX_RATIO, "pool_tokens_per_config": pool_tokens,
           "b_cap": b_cap, "max_mult": MAX_MULT,
           "source": SOURCE, "cells": {}, "skipped": [],
           "unresolvable": []}
    print(f"  batch/pool ceiling {MAX_RATIO:.0%} on a {split:,}-token honest split "
          f"=> any design drawing more than {b_cap:,} tokens per stream is "
          f"unresolvable at this pool")

    jobs, skipped, unres = [], [], []
    for atk, per_det in cost["cells"].items():
        for det, cell in per_det.items():
            if not cell["reachable"]:
                continue
            b = cell["tokens_per_verdict"]
            if b > b_cap:
                # NOT measured badly and NOT silently dropped. Resampling streams
                # of b tokens out of a pool of `split` collapses the honest
                # variance onto the pool mean; the number that comes back is a
                # property of the two splits, not of the test. Report the pool it
                # would take -- MAX_MULT*b so the sequential arm fits too.
                unres.append({"attack": atk, "verifier": det,
                              "tokens_per_verdict": int(b),
                              "batch_pool_ratio": b / split,
                              "honest_pool_needed": int(
                                  np.ceil(2 * MAX_MULT * b / MAX_RATIO))})
                continue
            if b > MAX_B:
                skipped.append({"attack": atk, "verifier": det,
                                "tokens_per_verdict": int(b),
                                "reason": f"b > IVGYM_MAXB={MAX_B:,}"})
                continue
            price = cost["price"][det]["sec_per_token"] / 3600.0 * cost["usd_per_gpu_hour"]
            jobs.append((atk, det, int(b), float(cell["d_prime"]), price,
                         z[f"honest__{det}"], z[f"{atk}__{det}"]))
    out["skipped"], out["unresolvable"] = skipped, unres

    print(f"\n{len(jobs)} cells inside the ceiling, {len(unres)} unresolvable at "
          f"this pool" + (f", {len(skipped)} unaffordable" if skipped else "") + ":")
    for s in sorted(unres, key=lambda r: -r["tokens_per_verdict"]):
        print(f"    unresolvable {s['attack']:>12} / {s['verifier']:<16} "
              f"b = {s['tokens_per_verdict']:>9,} = {s['batch_pool_ratio']:>7.1%} of "
              f"the pool; would need {s['honest_pool_needed']:,} honest tokens")
    for s in skipped:
        print(f"    skipped {s['attack']:>12} / {s['verifier']:<16} "
              f"b = {s['tokens_per_verdict']:,} tokens  ({s['reason']})")
    if not jobs:
        (RES / OUT_NAME).write_text(json.dumps(out, indent=1))
        print(f"\nno cell is resolvable at this pool -- wrote {RES/OUT_NAME} "
              f"saying so")
        return

    # Cells are independent; a weak one simulates ~1e9 draws while a strong one
    # takes seconds, so run them in a pool and print in a fixed order afterwards.
    workers = min(len(jobs), int(os.environ.get("IVGYM_WORKERS", 16))) or 1
    print(f"  simulating on {workers} worker processes ...", flush=True)
    recs: dict[tuple[str, str], dict] = {}
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(analyze_cell, *j): (j[0], j[1]) for j in jobs}
        for fut in cf.as_completed(futs):
            recs[futs[fut]] = fut.result()

    print(f"\n{'attack':>12} {'verifier':>16} {'d prime':>9} {'b*(d′)':>9} "
          f"{'matched b':>10} {'e-value E[N]':>13} {'saving':>9} {'ceiling':>9}")
    for atk, det, b, d, _price, _h, _a in jobs:
        rec = recs[(atk, det)]
        sp, ev, fx, fm = rec["sprt"], rec["e_value"], rec["fixed"], rec["fixed_matched"]
        out["cells"].setdefault(atk, {})[det] = rec
        ev_s = f"{ev['mean_tokens_under_attack']:,.0f}" if ev else "--"
        fm_s = f"{fm['b']:,}" if fm else "--"
        sav = (f"{rec['saving_x_matched']:.2f}x" if rec["saving_x_matched"] else "--")
        ok = "ok" if rec["within_ceiling"] else "OVER"
        print(f"{atk:>12} {det:>16} {d:>+9.4f} {b:>9,} {fm_s:>10} {ev_s:>13} "
              f"{sav:>9} {ok:>9}", flush=True)
        print(f"{'':>12} {'':>16} b*(d′) realizes FPR {fx['fpr']:.4f} / power "
              f"{fx['power']:.3f} against the {ALPHA:.3f}/{POWER:.2f} spec"
              + (f"  |  matched FPR {fm['fpr']:.4f} / power {fm['power']:.3f}" if fm else "")
              + (f"  |  e-value FPR {ev['fpr']:.4f} / power {ev['power']:.3f}" if ev else ""))

    # only cells inside the ceiling contribute a number; the rest are reported as
    # unresolvable above and must not be averaged into a headline
    good = [c for per in out["cells"].values() for c in per.values()
            if c["within_ceiling"]]
    savings = [c["saving_x_matched"] for c in good if c["saving_x_matched"]]
    out["n_within_ceiling"] = len(good)
    if savings:
        out["median_saving_x"] = float(np.median(savings))
        print(f"\n  median token saving across the {len(savings)} cells inside the "
              f"ceiling: {np.median(savings):.2f}x  "
              f"(range {min(savings):.2f}-{max(savings):.2f}x)")
        print("  The detector is unchanged. This is the same evidence, stopped when it "
              "is enough.")
    else:
        out["median_saving_x"] = None
        print("\n  no cell inside the ceiling yields a matched saving: at this pool "
              "the sequential-vs-fixed question is unresolved.")
    # the b*(d') spec check is measurable even where the saving is not
    miss = [(a, v, c) for a, per in out["cells"].items() for v, c in per.items()
            if c["within_ceiling"]]
    if miss:
        print(f"\n  what b*(d′) actually delivers, inside the ceiling "
              f"(target FPR <= {ALPHA:.3f}, power >= {POWER:.2f}):")
        for a, v, c in miss:
            print(f"    {a:>12} / {v:<16} FPR {c['fixed']['fpr']:.4f}  "
                  f"power {c['fixed']['power']:.3f}")

    (RES / OUT_NAME).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {RES/OUT_NAME}")


if __name__ == "__main__":
    main()
