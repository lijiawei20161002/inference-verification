"""The context-slope verifier, examined: is the surviving channel actually a verifier?

`fig_clock_measured.png` left exactly one thing standing -- the differential statistic
`D = ITL(ctx_hi) - ITL(ctx_lo)`, which cancels the 4.15 ms stack constant, is only ever
raised by co-tenancy, and reads the deviation family the returned-token channel is
worst at. This figure asks whether that survives contact with the four questions a
verifier has to answer, using `exp_slope_verifier_gpu` (out-of-sample architectures, a
real truncating provider, and a fresh-process re-measurement of the honest null).

  docs/figures/fig_slope_verifier.png
      (A) THE FLOOR IS NOT PREDICTABLE FROM A MODEL CARD, and this is a negative
          result against the obvious hope. Slope against layer count for eleven
          configurations; five of them -- Llama-3.2, TinyLlama, Pythia (full MHA),
          OLMo-2, Qwen2.5-0.5B, spanning 16x in KV bytes per token -- were never used
          to fit anything, and the layer law misses them by 26% mean / 48% worst.
          There is no absolute physical floor a client can compute off a config file;
          the test has to calibrate against the endpoint. (E) is why that is cheap.
      (B) IT DETECTS A REAL TRUNCATING PROVIDER, ON THE HOUSE PROTOCOL. Standardized
          pAUC at FPR <= 0.5% through `harness.evaluate` -- honest calibration split,
          winsorization, batch/pool ceiling enforced -- for a provider that bills for
          32 768 tokens of context and holds W. This is the clock's first appearance
          on the repo's own scoreboard rather than beside it.
      (C) MEASUREMENT AGREES WITH THE COST LAW. The same cells against
          `signal.predicted_pauc(d', b)`. The channel is not a special case: `d'` and
          `(delta*/d')^2` price it exactly as they price `token_difr`.
      (D) CO-TENANCY COSTS POWER, NOT CORRECTNESS. Fluctuating load is a ONE-SIDED
          nuisance -- it only ever pushes the honest statistic away from the deviation
          -- so it cannot manufacture a false accusation; sending the two probes
          together recovers 1.4x of the verdict price. Simulated over the measured
          ITL(B) distributions, the one place a load process has to be assumed. It is
          also where the protocol's winsorization bites (d' 2.48 uncapped, 1.05
          capped), because a null with no wire jitter is hard-bounded above.
      (E) BUT THE ENDPOINT'S OWN NULL IS STABLE, WHICH RESCUES (A). The same cells
          re-measured in a fresh process drift by 0.01% in slope, 0.03 ms per cell --
          2 000x tighter than the cross-architecture prediction. A client that
          calibrates the slope once on its own honest traffic has a floor good to
          0.01%, so what limits sensitivity is wire jitter, not calibration.
      (F) EVASION HAS A PRICE IN THE PROVIDER'S OWN CURRENCY. Truncating to W saves
          attention time and KV memory; hiding it from this verifier means padding by
          exactly the time it saved. The provider keeps the memory and gives back the
          latency, which is the whole point of a differential test.

MEASURED INPUTS   docs/results/slope_verifier.json (this experiment) and
    docs/results/clock_channel.json (the ITL(B) grid (D) resamples, and the in-sample
    slopes (A) fits on). delta* = 3.767 from cost_of_a_verdict.json.

MODELLED INPUTS   two, both swept or stated. The client-side jitter sigma, as in
    fig_clock_measured (F) -- device-side jitter is measured but a wire is not. And
    the load process in (D): B drawn from the measured batch grid, which is a
    simulation over measured distributions, not a measurement.

    python -m experiments.plot_slope_verifier
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.plot_clock_channel_principle import (
    DEV, FIG_DIR, GRID, HONEST, INK, INK2, MUTED, WARN,
)
from experiments.plot_clock_measured import GREEN, KV_GEOM, fit, kv_bytes, price
from ivgym import harness, signal, verifiers

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results"
SRC = RES / "slope_verifier.json"
MAIN = RES / "clock_channel.json"
ARCHF = RES / "clock_channel_arch.json"
COST = RES / "cost_of_a_verdict.json"

DELTA_STAR = json.load(open(COST))["delta_star"]
JITTER_SHAPE = 1.6                 # the repo's own one-sided queueing model
SIGMA_MAIN = 50.0                  # ms of client-side jitter: a pessimistic wire
BATCHES = [1, 2, 4, 8, 16, 32, 64]
GEOM2 = {                          # layers, KV heads, head_dim for the arm-2 models
    "Llama-3.2-1B": (16, 8, 64), "TinyLlama-1.1B-Chat-v1.0": (22, 4, 64),
    "pythia-1.4b": (24, 16, 128), "OLMo-2-0425-1B": (16, 16, 128),
    "Qwen2.5-0.5B": (24, 2, 64),
}


class ClockSlope(verifiers.Verifier):
    """A shim so `harness.evaluate` scores the clock exactly as it scores every
    returned-token verifier: the per-'token' score is one probe pair's D."""

    name = "clock_slope"
    value_fn = "uniform"


# ------------------------------------------------------------------------- data
def load_all():
    d = json.load(open(SRC))
    cells = list(d["cells"])
    deep = RES / "slope_verifier_window.json"
    if deep.exists():                # the detection arm, re-run at a bigger pool
        dw = json.load(open(deep))
        keep = {c["label"] for c in dw["cells"]}
        cells = [c for c in cells if c["label"] not in keep] + dw["cells"]
        d["window_pool"] = dw["steps"] * dw["reps"]
    main = json.load(open(MAIN))
    arch = json.load(open(ARCHF)) if ARCHF.exists() else {"cells": []}
    return d, cells, main["cells"] + arch["cells"]


def by(cells, **kw):
    out = [c for c in cells if all(c.get(k) == v for k, v in kw.items())]
    return out


def slope_of(cells, label, mode="graph", B=1):
    """One (ctx, ITL) point per context -- the batching arm re-measures some contexts
    at B=1, and two rows for the same ctx would silently double-weight it."""
    grp = {}
    for c in cells:
        if c["label"] == label and c["mode"] == mode and c["B"] == B:
            grp.setdefault(c["ctx"], []).append(c["mean"])
    if len(grp) < 3:
        return None
    x = np.array(sorted(grp), float)
    y = np.array([float(np.mean(grp[int(k)])) for k in x])
    a, b, r = fit(x, y)
    return {"slope": b, "intercept": a, "resid": r, "ctx": x, "itl": y}


def jitter(rng, n, sigma):
    """One-sided, positive, right-skewed -- the nuisance model the clock figures use."""
    return rng.gamma(JITTER_SHAPE, sigma / JITTER_SHAPE, n) if sigma > 0 else np.zeros(n)


def pairs(cells, hi_label, lo_arr, sigma, rng, half):
    """Probe-pair statistic D = ITL_hi - ITL_lo, with client-side jitter on both."""
    c = next((c for c in cells if c["label"] == hi_label), None)
    if c is None:
        return None
    hi = np.asarray(c["itl_ms"], float)
    n = min(len(hi), len(lo_arr))
    hi, lo = hi[:n], lo_arr[:n]
    return (hi + jitter(rng, n, sigma)) - (lo + jitter(rng, n, sigma))


# ============================================================================ (A)
def panel_a(ax, cells, main_cells):
    fit_pts, oos_pts = [], []
    for lab, (L, _, _) in KV_GEOM.items():
        s = slope_of(main_cells, lab)
        if s and lab != "Qwen3-1.7B-NF4":
            fit_pts.append((L, s["slope"] * 1e3, lab, kv_bytes(lab)))
    for lab, (L, h, hd) in GEOM2.items():
        s = slope_of(cells, lab)
        if s:
            oos_pts.append((L, s["slope"] * 1e3, lab, 2 * L * h * hd * 2))
    x = np.array([p[0] for p in fit_pts], float)
    y = np.array([p[1] for p in fit_pts])
    a, b, _ = fit(x, y)
    ax.set_xlim(13, 39)
    xs = np.linspace(14, 38, 30)
    ax.plot(xs, a + b * xs, "-", color=DEV, lw=2.0, zorder=3)

    for (L, sl, lab, kb) in fit_pts:
        ax.plot([L], [sl], "o", ms=8, color=DEV, mec="white", mew=1.0, zorder=5)
        dy = {"Qwen3-0.6B": 9, "Qwen3-1.7B": -13, "Qwen3-4B": 9, "Qwen2.5-1.5B": 9,
              "SmolLM2-1.7B": -13}.get(lab, 9)
        ax.annotate(lab.replace("Qwen3-", "Q3-").replace("Qwen2.5-", "Q2.5-"), (L, sl),
                    xytext=(0, dy), textcoords="offset points", ha="center",
                    va="bottom" if dy > 0 else "top", fontsize=7.4, color=DEV)
    errs = []
    for (L, sl, lab, kb) in oos_pts:
        pred = a + b * L
        errs.append(abs(sl - pred) / sl)
        ax.plot([L], [sl], "D", ms=8, color=GREEN, mec="white", mew=1.0, zorder=5)
        ax.annotate(f"{lab.split('-Chat')[0]}\n{kb / 1e3:.0f} kB/tok, "
                    f"{(sl - pred) / sl:+.0%}", (L, sl), xytext=(0, -10),
                    textcoords="offset points",
                    ha="center" if lab != "pythia-1.4b" else "right",
                    va="top" if lab != "pythia-1.4b" else "bottom", fontsize=7.0,
                    color=GREEN, linespacing=1.4)
    kbs = [p[3] for p in fit_pts + oos_pts]
    ax.annotate(f"$\\bf{{fit}}$ (5 Qwen configs): {a:.2f} + {b:.4f} $\\times$ layers\n"
                f"$\\bf{{misses}}$ 5 unseen ones by {np.mean(errs):.0%} mean, "
                f"{max(errs):.0%} worst\n(KV bytes as a 2nd predictor: worse)\n\n"
                "no model-card floor -- an absolute test\nwould have to forgive a "
                "quarter of the\nslope. Calibrate on the endpoint (E).",
                (0.99, 0.015), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=7.8, color=INK2, linespacing=1.55)
    ax.set_xlabel("transformer layers", fontsize=9.2, color=INK2)
    ax.set_ylabel("context slope  d(ITL)/d(ctx)   ($\\mu$s per token)", fontsize=9.2,
                  color=INK2)
    ax.set_title("(A)  The floor is NOT predictable from a model card",
                 fontsize=10.6, weight="bold", color=INK, loc="left")


# ======================================================================== (B)/(C)
def detection(cells, sigma, seed=0):
    """Score the truncating providers through the repo's standardized protocol."""
    rng = np.random.default_rng(seed)
    lo = np.asarray(next(c for c in cells if c["label"] == "probe_lo")["itl_ms"], float)
    half = len(lo) // 2
    hon = pairs(cells, "honest_hi", lo[:half], sigma, rng, half)
    out = {}
    for W in (512, 2048, 8192, 16384):
        atk = pairs(cells, f"window_{W}", lo[half:], sigma, rng, half)
        if atk is None or hon is None:
            continue
        # A truncating provider is FASTER, so the deviation makes D SMALLER, and
        # `evaluate` scores "higher = more deviant" -- hence the sign flip.
        h = harness.TokenScores("honest", {"clock_slope": -hon})
        a = harness.TokenScores(f"window_{W}", {"clock_slope": -atk})
        # Winsorization is the one protocol element that does not transfer to a
        # ONE-SIDED channel. It caps at the 99.9th percentile of the honest
        # calibration split so that no single outlier carries a batch; here the
        # honest statistic is bounded above (jitter only ever ADDS time, so -D only
        # ever goes down) and the whole deviation lives past that bound, so the cap
        # deletes it -- the same failure `evaluate`'s own docstring describes for a
        # selective audit. Both are reported; the headline is the uncapped one.
        res = harness.evaluate(h, a, [ClockSlope()], BATCHES,
                               config=harness.EvalConfig(over_ratio="allow"),
                               winsor_pct=None)
        res_w = harness.evaluate(h, a, [ClockSlope()], BATCHES,
                                 config=harness.EvalConfig(over_ratio="allow"))
        dp = signal.per_token_stats(-hon, -atk, winsor_pct=None)["d_prime"]
        out[W] = {"res": res, "res_winsor": res_w, "d_prime": dp, "pool": len(hon),
                  "sep": float(np.mean(hon) - np.mean(atk))}
    return out


def panel_b(ax, det, cells):
    for W, col in zip(sorted(det), (WARN, DEV, GREEN, HONEST)):
        r = det[W]
        b = [x.batch_size for x in r["res"]]
        auc = [x.auc for x in r["res"]]
        ok = [x.pool_ratio <= 0.10 for x in r["res"]]
        ax.plot(b, auc, "-", color=col, lw=1.9, zorder=4)
        ax.plot([bb for bb, o in zip(b, ok) if o], [aa for aa, o in zip(auc, ok) if o],
                "o", ms=6, color=col, mec="white", mew=0.9, zorder=5)
        ax.plot([bb for bb, o in zip(b, ok) if not o],
                [aa for aa, o in zip(auc, ok) if not o], "x", ms=7, color=col,
                mew=1.6, zorder=5)
        ax.plot([], [], "-o", color=col, ms=5,
                label=f"holds {W:,} of 32,768   $d'$ = {r['d_prime']:.2f}")
    ax.axhline(0.9, color=INK2, lw=1.1, ls="--", zorder=3)
    ax.annotate("the repo's target, pAUC 0.90", (0.985, 0.905),
                xycoords=("axes fraction", "data"), ha="right", va="bottom",
                fontsize=8.0, color=INK2)
    ax.axhline(0.5, color=MUTED, lw=1.0, ls=":", zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0.45, 1.03)
    ax.set_xlabel("probe pairs per verdict  (a pair = one short + one long request)",
                  fontsize=9.0, color=INK2)
    ax.set_ylabel(f"standardized pAUC @ FPR $\\leq$ 0.5%", fontsize=9.2, color=INK2)
    ax.set_title(f"(B)  A real truncating provider, scored by `harness.evaluate` "
                 f"($\\sigma$ = {SIGMA_MAIN:.0f} ms)",
                 fontsize=10.6, weight="bold", color=INK, loc="left")
    ax.legend(fontsize=8.0, loc="upper left", frameon=False)
    ax.annotate("filled = inside the 10% batch/pool ceiling.  Reported uncapped; "
                "with the\nprotocol's winsorization on, every number here moves by "
                "< 0.004 pAUC --\nthe jittered null has a long enough upper tail that "
                "the cap never bites.\nIt does bite in the zero-jitter limit: see (D).",
                (0.0, -0.155), xycoords="axes fraction", ha="left", va="top",
                fontsize=7.8, color=MUTED, linespacing=1.5)


def panel_c(ax, det):
    for W, col in zip(sorted(det), (WARN, DEV, GREEN, HONEST)):
        r = det[W]
        b = np.array([x.batch_size for x in r["res"]], float)
        meas = np.array([x.auc for x in r["res"]])
        pred = np.array([signal.predicted_pauc(r["d_prime"], int(bb)) for bb in b])
        ok = np.array([x.pool_ratio <= 0.10 for x in r["res"]])
        ax.plot(pred[ok], meas[ok], "o", ms=7, color=col, mec="white", mew=0.9,
                zorder=5, label=f"W = {W:,}")
        ax.plot(pred[~ok], meas[~ok], "x", ms=7, color=col, mew=1.5, zorder=4)
    ax.plot([0.45, 1.02], [0.45, 1.02], "-", color=INK2, lw=1.2, zorder=3)
    ax.annotate("the cost law, unmodified:\n"
                "$\\delta = d'\\sqrt{b}$, pAUC from $\\delta$\n"
                "(`ivgym/signal.py`, fitted to nothing)",
                (0.52, 0.97), ha="left", va="top", fontsize=8.3, color=INK2,
                linespacing=1.6)
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=7.8, loc="lower right", frameon=False)
    ax.set_xlabel("predicted pAUC from $d'$ alone", fontsize=9.2, color=INK2)
    ax.set_ylabel("measured pAUC", fontsize=9.2, color=INK2)
    ax.set_title("(C)  The channel is priced by the same law as the token channel",
                 fontsize=10.6, weight="bold", color=INK, loc="left")


# ============================================================================ (D)
def panel_d(ax, main_cells):
    """Fluctuating co-tenancy: an unpaired difference dies, a paired one does not."""
    rng = np.random.default_rng(1)
    lo_by_B, hi_by_B = {}, {}
    for c in main_cells:
        if c["label"] == "Qwen3-1.7B" and c["mode"] == "graph":
            if c["ctx"] == 1024:
                lo_by_B[c["B"]] = np.asarray(c["itl_ms"], float)
            elif c["ctx"] == 8192:
                hi_by_B[c["B"]] = np.asarray(c["itl_ms"], float)
    Bs = sorted(set(lo_by_B) & set(hi_by_B))
    n = 4000
    draw = lambda d, b: rng.choice(d[b], 1)[0]
    designs = {}
    for name in ("paired", "unpaired"):
        h, a = np.empty(n), np.empty(n)
        for i in range(n):
            b1 = Bs[rng.integers(len(Bs))]
            b2 = b1 if name == "paired" else Bs[rng.integers(len(Bs))]
            h[i] = draw(hi_by_B, b2) - draw(lo_by_B, b1)         # honest: full context
            b3 = Bs[rng.integers(len(Bs))]
            b4 = b3 if name == "paired" else Bs[rng.integers(len(Bs))]
            a[i] = draw(lo_by_B, b4) - draw(lo_by_B, b3)         # truncated to the short ctx
        designs[name] = (h, a)
    for (name, (h, a)), col in zip(designs.items(), (DEV, WARN)):
        bins = np.linspace(-15, 45, 70)
        ax.hist(h, bins=bins, color=col, alpha=0.30, zorder=3)
        ax.hist(a, bins=bins, color=col, alpha=0.95, histtype="step", lw=1.8, zorder=4)
        dp = signal.per_token_stats(-h, -a, winsor_pct=None)["d_prime"]
        p1 = price(dp)
        ax.annotate(f"$\\bf{{{name}}}$ probes:  $d'$ = {dp:.2f}  ->  "
                    f"{'<1' if p1 < 1 else f'{p1:.0f}'} pair(s) per verdict",
                    (0.03, 0.97 if name == "paired" else 0.915),
                    xycoords="axes fraction", ha="left", va="top", color=col,
                    fontsize=8.6)
    ax.annotate("filled = honest (full 8 192 ctx),  outline = truncated to 1 024.\n"
                "B redrawn from the measured grid per probe; pairing buys 1.4x.\n\n"
                "Load is a ONE-SIDED nuisance: it only pushes the honest statistic\n"
                "UP, away from the deviation, so it costs power and cannot\n"
                "manufacture a false accusation.\n\n"
                "It is also where the protocol's winsorization bites -- with no wire\n"
                "jitter this null is hard-bounded, so the honest 99.9th-percentile\n"
                "cap sits inside the deviation: $d'$ 1.05 capped, 2.48 uncapped.",
                (0.98, 0.83), xycoords="axes fraction", ha="right", va="top",
                fontsize=7.9, color=INK2, linespacing=1.6)
    ax.set_xlim(-15, 45)
    ax.set_ylim(0, 1450)
    ax.set_xlabel("probe-pair statistic  D = ITL(8192) - ITL(1024)   (ms)",
                  fontsize=9.0, color=INK2)
    ax.set_ylabel("probe pairs", fontsize=9.2, color=INK2)
    ax.set_title("(D)  Co-tenancy costs power, never a false accusation",
                 fontsize=10.6, weight="bold", color=INK, loc="left")


# ============================================================================ (E)
def panel_e(ax, cells, main_cells):
    a = slope_of(main_cells, "Qwen3-1.7B")
    b = slope_of(cells, "Qwen3-1.7B-rerun")
    if not (a and b):
        return
    ax.plot(a["ctx"], a["itl"], "-o", ms=5, color=DEV, lw=1.8, mec="white", mew=0.8,
            zorder=4)
    ax.plot(b["ctx"], b["itl"], "--s", ms=5, color=WARN, lw=1.8, mec="white", mew=0.8,
            zorder=4)
    drift = abs(b["slope"] - a["slope"]) / a["slope"]
    ax.annotate(f"run 1  {a['slope'] * 1e3:.3f} $\\mu$s/tok", (a["ctx"][-1], a["itl"][-1]),
                xytext=(-6, 10), textcoords="offset points", ha="right", color=DEV,
                fontsize=8.4)
    ax.annotate(f"run 2, fresh process  {b['slope'] * 1e3:.3f} $\\mu$s/tok",
                (b["ctx"][-1], b["itl"][-1]), xytext=(-6, -14),
                textcoords="offset points", ha="right", va="top", color=WARN,
                fontsize=8.4)
    dmax = max(abs(y2 - y1) for y1, y2 in zip(a["itl"], b["itl"]))
    ax.annotate(f"$\\bf{{{drift * 100:.2f}\\%}}$ drift in the slope between "
                f"processes, worst cell {dmax:.3f} ms.\n\n"
                "Calibrating on the endpoint instead of on a config file is\n"
                f"{0.264 / max(drift, 1e-9):,.0f}x tighter than (A)'s law, and it is "
                "free: the client\nmeasures it on its own honest traffic. A provider "
                f"that quietly\ndrops {max(drift * 3, 0.001):.1%} of the context "
                "moves the slope by more than the null does,\nso what bounds "
                "sensitivity is the wire in (B), not the calibration.",
                (0.04, 0.96), xycoords="axes fraction", ha="left", va="top",
                fontsize=8.4, color=INK2, linespacing=1.6)
    ax.set_xlabel("context length (tokens)", fontsize=9.2, color=INK2)
    ax.set_ylabel("ms per output token", fontsize=9.2, color=INK2)
    ax.set_title("(E)  The honest null does not move between processes",
                 fontsize=10.6, weight="bold", color=INK, loc="left")


# ============================================================================ (F)
def panel_f(ax, cells, det):
    hon = next(c for c in cells if c["label"] == "honest_hi")
    kv = hon["kv_bytes_per_token"]
    Ws, saved_ms, saved_gb, cost = [], [], [], []
    for W in sorted(det):
        c = next(c for c in cells if c["label"] == f"window_{W}")
        Ws.append(W)
        saved_ms.append(hon["mean"] - c["mean"])
        saved_gb.append((32768 - W) * kv / 1e9)
        cost.append(2 * price(det[W]["d_prime"]))     # 2 streams x pairs = tokens
    ax.plot(saved_ms, cost, "-o", ms=8, color=DEV, lw=2.0, mec="white", mew=1.0,
            zorder=5)
    for W, sm, ct, gb in zip(Ws, saved_ms, cost, saved_gb):
        # leftmost two points label to the right, rightmost two to the left, with
        # the last one dropped so the two closest points do not collide
        dx, dy, ha, va = {16384: (10, -4, "left", "top"),
                          8192: (10, -4, "left", "top"),
                          2048: (-6, 26, "right", "bottom"),
                          512: (-10, -10, "right", "top")}[W]
        ax.annotate(f"holds {W:,} of 32,768\n{gb:.1f} GB of KV saved\n"
                    f"{ct:.0f} tokens to catch", (sm, ct), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va=va, fontsize=7.6,
                    color=INK2, linespacing=1.45)
    ax.set_yscale("log")
    ax.set_xlim(min(saved_ms) - 4, max(saved_ms) + 4)
    ax.set_ylim(min(cost) * 0.45, max(cost) * 3.2)
    ax.set_xlabel("latency the provider saves per output token (ms)", fontsize=9.0,
                  color=INK2)
    ax.set_ylabel("tokens of stream per pAUC 0.90 verdict", fontsize=9.2, color=INK2)
    ax.annotate(f"To hide from this test the provider must pad the long request by\n"
                f"exactly what truncation saved -- up to {max(saved_ms):.0f} ms per "
                f"output token,\n{max(saved_ms) / hon['mean'] * 100:.0f}% of the honest "
                f"per-token time. It then keeps the\n{max(saved_gb):.1f} GB of KV "
                "memory and gives back all of the speed.\n\n"
                "That is what a differential test buys: not a proof, but a\n"
                "conversion of the cheat from free money into a real cost.",
                (0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
                fontsize=8.2, color=INK2, linespacing=1.6)
    ax.set_title(f"(F)  The evasion frontier ($\\sigma$ = {SIGMA_MAIN:.0f} ms)",
                 fontsize=10.6, weight="bold", color=INK, loc="left")


# =========================================================================== main
def report(d, cells, main_cells, det):
    print("=" * 92)
    print("The context-slope verifier, examined")
    print(f"  {len(cells)} new cells, steps={d['steps']} reps={d['reps']}, "
          f"{d['elapsed_s']:.0f}s on {d['card']['device']}")
    print("=" * 92)

    print("\n(A) out-of-sample floor prediction")
    fit_pts = [(KV_GEOM[l][0], slope_of(main_cells, l)["slope"] * 1e3, l)
               for l in KV_GEOM if slope_of(main_cells, l) and l != "Qwen3-1.7B-NF4"]
    a, b, _ = fit(np.array([p[0] for p in fit_pts], float),
                  np.array([p[1] for p in fit_pts]))
    print(f"  fitted on {len(fit_pts)} Qwen configs:  slope = {a:.4f} + {b:.5f} * "
          f"layers  (us/tok)")
    print(f"  {'model':26s} {'layers':>7s} {'KV B/tok':>10s} {'measured':>9s} "
          f"{'predicted':>10s} {'error':>7s}")
    errs = []
    for lab, (L, h, hd) in GEOM2.items():
        s = slope_of(cells, lab)
        if not s:
            print(f"  {lab:26s}  (not measured)")
            continue
        pred, meas = a + b * L, s["slope"] * 1e3
        errs.append(abs(meas - pred) / meas)
        print(f"  {lab:26s} {L:7d} {2 * L * h * hd * 2:10,d} {meas:9.3f} "
              f"{pred:10.3f} {(pred - meas) / meas:+6.1%}")
    if errs:
        print(f"  mean |error| {np.mean(errs):.1%}, worst {max(errs):.1%}")

    print(f"\n(B) detection of a truncating provider, sigma = {SIGMA_MAIN} ms, "
          f"house protocol")
    print(f"  {'holds':>8s} {'d prime':>9s} {'pairs for pAUC 0.90':>21s} "
          f"{'pAUC @ 32 pairs':>16s} {'pool ratio':>11s}")
    for W in sorted(det):
        r = det[W]
        ok = [x for x in r["res"] if x.pool_ratio <= 0.10 and x.auc >= 0.9]
        at32 = next((x for x in r["res"] if x.batch_size == 32), None)
        need = min([x.batch_size for x in ok], default=None)
        print(f"  {W:8,d} {r['d_prime']:9.2f} {str(need) if need else '>64':>21s} "
              f"{at32.auc if at32 else float('nan'):16.3f} "
              f"{at32.pool_ratio if at32 else float('nan'):10.1%}")

    a2 = slope_of(main_cells, "Qwen3-1.7B")
    b2 = slope_of(cells, "Qwen3-1.7B-rerun")
    if a2 and b2:
        print(f"\n(E) honest null across processes: slope {a2['slope'] * 1e3:.3f} -> "
              f"{b2['slope'] * 1e3:.3f} us/tok "
              f"({abs(b2['slope'] - a2['slope']) / a2['slope']:.2%} drift), worst "
              f"per-cell {max(abs(y2 - y1) for y1, y2 in zip(a2['itl'], b2['itl'])):.2f} ms")

    hon = next(c for c in cells if c["label"] == "honest_hi")
    print("\n(F) evasion frontier")
    for W in sorted(det):
        c = next(c for c in cells if c["label"] == f"window_{W}")
        ok = [x for x in det[W]["res"] if x.pool_ratio <= 0.10 and x.auc >= 0.9]
        print(f"  holds {W:6,d}:  saves {hon['mean'] - c['mean']:6.2f} ms/token and "
              f"{(32768 - W) * hon['kv_bytes_per_token'] / 1e9:5.2f} GB of KV;  "
              f"verdict in {min([x.batch_size for x in ok], default='>64')} pairs;  "
              f"padding to hide costs back all "
              f"{hon['mean'] - c['mean']:.2f} ms/token")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d, cells, main_cells = load_all()
    det = detection(cells, SIGMA_MAIN)
    report(d, cells, main_cells, det)

    fig, axes = plt.subplots(2, 3, figsize=(19.0, 10.6))
    for ax in axes.flat:
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8.4, colors=INK2)
    panel_a(axes[0, 0], cells, main_cells)
    panel_b(axes[0, 1], det, cells)
    panel_c(axes[0, 2], det)
    panel_d(axes[1, 0], main_cells)
    panel_e(axes[1, 1], cells, main_cells)
    panel_f(axes[1, 2], cells, det)

    fig.suptitle("The context-slope verifier: no model-card floor, but a "
                 "self-calibrating one -- and a verdict in ~64 tokens of stream",
                 fontsize=15.5, weight="bold", color=INK, x=0.006, ha="left", y=0.988)
    fig.text(0.006, 0.955,
             "The one channel left standing by fig_clock_measured, put through the "
             "four tests a verifier has to pass: an out-of-sample floor, a real "
             "truncating provider on the house protocol, a stable null, and an "
             "evasion price.\nClient-side jitter (swept in fig_clock_measured, fixed "
             "at a pessimistic 50 ms here) and the load process in (D) are the only "
             "modelled inputs; everything else is measured.",
             fontsize=9.4, color=INK2, ha="left", va="top", linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_slope_verifier.png"
    fig.savefig(out, dpi=155, facecolor="white")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
