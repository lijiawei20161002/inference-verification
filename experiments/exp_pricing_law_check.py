"""Is the price of a verdict even predictable? -- falsifying b*(d') = (delta*/d')^2.

Every price in `exp_cost_of_a_verdict_gpu` is a *prediction*. The measured half is
`d'`, a per-token effect size; the tokens-per-verdict half comes from the CLT model
in `ivgym.signal` -- batch `b` of independent tokens separates by `d' sqrt(b)`, so
reaching standardized pAUC 0.90 at FPR <= 0.5% takes `b* = (delta*/d')^2` tokens.
If that model is wrong, every cost on the poster is wrong by the same factor, and
the two documented reasons it could be wrong both bite here: tokens inside a
sequence are correlated (so the effective `b` is smaller than the nominal one), and
a heavy-tailed per-token score reaches the Gaussian limit slowly (so small batches
under-perform the prediction).

So test it, on the same score arrays the prices were computed from
(`cost_of_a_verdict_scores.npz`), with no new GPU time: for every cell and a grid
of batch sizes under the 10% pool ceiling, measure the standardized pAUC the way
`harness.evaluate` measures it (held-out honest calibration split, winsorized cap,
batches resampled without replacement) and compare it to `signal.predicted_pauc`.

    measured pAUC(b)   vs   predicted pAUC(d' sqrt(b))          [the calibration]
    b_hat / b*         at the cells whose b* fits under the ceiling   [the price]

`b_hat` is the batch at which the MEASURED curve crosses 0.90, obtained by
interpolating log b against measured pAUC -- so the residual is reported in the
unit the poster spends, tokens, and not only in pAUC.

Cells whose `b*` sits above the ceiling cannot be checked at `b*` itself: their
verdict is an extrapolation from the range that fits, which is exactly what this
artifact is for -- it reports how far the law holds where it CAN be checked, and
labels the rest extrapolation instead of quietly averaging the two.

    python -m experiments.exp_pricing_law_check

Env: IVGYM_BOOT_LAW(200) bootstrap resamples for the CI on the measured pAUC,
     IVGYM_NBATCH(4000) batches per point (the pAUC region needs >= 10 points at
     FPR <= 0.5%, so more batches is a real precision knob here).

Writes `docs/results/pricing_law_check.json`.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import metrics, signal
from ivgym.harness import EvalConfig, RatioCeilingWarning, batch_means

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results"

N_BATCH = int(os.environ.get("IVGYM_NBATCH", 4000))
TARGET = 0.90


def measured_pauc(h_cal: np.ndarray, h_ev: np.ndarray, a: np.ndarray, b: int,
                  cfg: EvalConfig, rng: np.random.Generator) -> tuple[float, float]:
    """(standardized pAUC, TPR at the calibrated tau) for one batch size.

    Same three-way split as `harness.evaluate`: the cap and tau come from the
    honest CALIBRATION half, the negatives from the disjoint eval half. Inlined
    rather than called because `evaluate` wants `TokenScores` and a `Verifier`
    object, and here the scores arrive as a bare array off an npz.
    """
    if cfg.winsor_pct is not None:
        cap = np.percentile(h_cal, cfg.winsor_pct)
        h_ev, a = np.minimum(h_ev, cap), np.minimum(a, cap)
    cb = batch_means(h_cal, b, cfg.n_batches, rng)
    hb = batch_means(h_ev, b, cfg.n_batches, rng)
    ab = batch_means(a, b, cfg.n_batches, rng)
    tau = np.quantile(cb, 1.0 - cfg.max_fpr)
    return (metrics.partial_auc(hb, ab, max_fpr=cfg.max_fpr),
            float((ab > tau).mean()))


def crossing(bs: list[int], aucs: list[float], target: float = TARGET) -> float | None:
    """Batch at which the measured curve first reaches `target`, interpolated in
    log b. `None` if the curve never gets there inside the ceiling -- which is a
    fact about this pool, not about the detector, and is recorded as such."""
    for i in range(1, len(bs)):
        if aucs[i - 1] < target <= aucs[i]:
            f = (target - aucs[i - 1]) / (aucs[i] - aucs[i - 1])
            return float(np.exp(np.log(bs[i - 1]) + f * (np.log(bs[i]) - np.log(bs[i - 1]))))
    return None


def main() -> None:
    t0 = time.time()
    CV = json.loads((RES / "cost_of_a_verdict.json").read_text())
    Z = np.load(RES / "cost_of_a_verdict_scores.npz")
    cfg = EvalConfig(n_batches=N_BATCH, over_ratio="raise")
    delta_star = CV["delta_star"]

    print("=" * 88)
    print("Does the pricing law hold?   measured pAUC vs (delta* / d')^2 prediction")
    print(f"  {CV['M']}, pool {CV['pool_tokens_per_config']:,} tokens/config, "
          f"{cfg.n_batches} batches/point, ceiling {cfg.max_pool_ratio:.0%}")
    print("=" * 88, flush=True)

    rows, per_cell = [], {}
    n_h = len(Z["honest__token_difr"])
    cut = int(n_h * cfg.calib_frac)
    b_max = int(cfg.max_pool_ratio * (n_h - cut))          # the ceiling, in tokens
    grid = [b for b in (8, 16, 32, 64, 128, 256, 512, 1024) if b <= b_max]
    print(f"  honest eval split {n_h - cut} tokens -> largest legal batch {b_max}; "
          f"grid {grid}\n")

    for atk in CV["attacks"]:
        if atk not in CV["cells"]:
            continue
        for v, cell in CV["cells"][atk].items():
            key = f"{atk}__{v}"
            if key not in Z.files:
                continue
            rng = np.random.default_rng(0)
            idx = np.random.default_rng(cfg.seed).permutation(n_h)
            h = np.asarray(Z[f"honest__{v}"], float)
            h_cal, h_ev = h[idx[:cut]], h[idx[cut:]]
            a = np.asarray(Z[key], float)
            d = cell["d_prime"]
            aucs, tprs, preds = [], [], []
            with warnings.catch_warnings():
                warnings.simplefilter("error", RatioCeilingWarning)
                for b in grid:
                    auc, tpr = measured_pauc(h_cal, h_ev, a, b, cfg, rng)
                    aucs.append(auc)
                    tprs.append(tpr)
                    preds.append(signal.predicted_pauc(d, b, cfg.max_fpr))
            b_star = cell["tokens_per_verdict"] if cell["reachable"] else None
            b_hat = crossing(grid, aucs)
            resid = [m - p for m, p in zip(aucs, preds)]
            # The sharpest single test: the poster sells `b*` tokens as a
            # pAUC-0.90 verdict, so measure the pAUC of exactly `b*` tokens. Only
            # possible where b* fits under the ceiling; elsewhere the price is an
            # extrapolation and is labelled one.
            at_star = None
            if b_star is not None and 1 <= b_star <= b_max:
                auc_s, tpr_s = measured_pauc(h_cal, h_ev, a, int(b_star), cfg, rng)
                at_star = {"batch": int(b_star), "measured": auc_s, "tpr": tpr_s,
                           "shortfall": TARGET - auc_s}
            per_cell[key] = {
                "attack": atk, "verifier": v, "tier": CV["tier"][v], "d_prime": d,
                "batches": grid, "measured": aucs, "predicted": preds, "tpr": tprs,
                "b_star_predicted": b_star, "b_hat_measured": b_hat,
                "at_b_star": at_star,
                "checkable_at_b_star": bool(b_star is not None and b_star <= b_max),
                "max_abs_residual": float(max(abs(r) for r in resid)),
                "mean_residual": float(np.mean(resid)),
            }
            rows.append(per_cell[key])
            flag = "" if b_star is None or b_star <= b_max else "  (b* above ceiling)"
            print(f"  {atk:>10} {v:>16}  d'={d:+.4f}  "
                  f"pAUC@{grid[-1]}: meas {aucs[-1]:.3f} pred {preds[-1]:.3f}  "
                  f"resid {resid[-1]:+.3f}{flag}", flush=True)

    all_m = np.array([m for r in rows for m in r["measured"]])
    all_p = np.array([p for r in rows for p in r["predicted"]])
    resid = all_m - all_p
    # A calibration plot is only honest with a spread, and pAUC is bounded, so
    # report both the raw bias and the bias restricted to the informative band
    # (predicted pAUC in (0.55, 0.99)) where the metric is not saturated.
    band = (all_p > 0.55) & (all_p < 0.99)
    checked = [r for r in rows if r["checkable_at_b_star"] and r["b_hat_measured"]]
    price_err = [r["b_hat_measured"] / r["b_star_predicted"] for r in checked]
    at_star = [r for r in rows if r["at_b_star"]]
    if at_star:
        print(f"\n  pAUC actually delivered by b* tokens, at the "
              f"{len(at_star)} cells whose b* fits under the ceiling:")
        for r in sorted(at_star, key=lambda r: r["at_b_star"]["measured"]):
            s = r["at_b_star"]
            print(f"    {r['attack']:>10} {r['verifier']:>16}  b*={s['batch']:>5}  "
                  f"measured pAUC {s['measured']:.3f}  "
                  f"(priced for {TARGET:.2f}: {s['shortfall']:+.3f})")

    print(f"\n  {len(rows)} cells x {len(grid)} batch sizes = {len(all_m)} points")
    print(f"  residual (measured - predicted): mean {resid.mean():+.4f}, "
          f"rms {np.sqrt((resid**2).mean()):.4f}, max |.| {np.abs(resid).max():.4f}")
    if band.any():
        print(f"  unsaturated band (pred 0.55-0.99, n={int(band.sum())}): "
              f"mean {resid[band].mean():+.4f}, rms "
              f"{np.sqrt((resid[band]**2).mean()):.4f}")
    if price_err:
        print(f"  cells whose b* is checkable inside the ceiling: {len(checked)}; "
              f"measured/predicted tokens per verdict "
              f"{min(price_err):.2f}-{max(price_err):.2f}x "
              f"(median {np.median(price_err):.2f}x)")
    else:
        print("  no cell reaches pAUC 0.90 inside the ceiling: every b* on the "
              "poster is an extrapolation from the checkable range.")

    out = {
        "M": CV["M"], "delta_star": delta_star, "target_pauc": TARGET,
        "max_fpr": cfg.max_fpr, "n_batches": cfg.n_batches,
        "pool_tokens_per_config": CV["pool_tokens_per_config"],
        "honest_eval_tokens": n_h - cut, "max_legal_batch": b_max,
        "batch_grid": grid, "cells": per_cell,
        "residual": {
            "mean": float(resid.mean()), "rms": float(np.sqrt((resid ** 2).mean())),
            "max_abs": float(np.abs(resid).max()), "n": int(len(all_m)),
            "band_mean": float(resid[band].mean()) if band.any() else None,
            "band_rms": float(np.sqrt((resid[band] ** 2).mean())) if band.any() else None,
            "band_n": int(band.sum()),
        },
        "at_b_star": {f"{r['attack']}__{r['verifier']}": r["at_b_star"]
                      for r in rows if r["at_b_star"]},
        "price_check": {
            "n_checkable": len(checked),
            "ratios": {f"{r['attack']}__{r['verifier']}":
                       r["b_hat_measured"] / r["b_star_predicted"] for r in checked},
            "median": float(np.median(price_err)) if price_err else None,
            "min": float(min(price_err)) if price_err else None,
            "max": float(max(price_err)) if price_err else None,
        },
        "elapsed_s": time.time() - t0,
    }
    (RES / "pricing_law_check.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {RES / 'pricing_law_check.json'}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
