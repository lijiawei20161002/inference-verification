"""Three more clocks and two better estimators: the result of exp_clock_algos_gpu.

`fig_clock_basic.png`'s premise is that a token cannot arrive before its bytes have
moved. `fig_clock_measured.png` tested that premise on the ONE observable the figures
use -- the inter-token gap of a batch-1 decode -- and found the worst possible place
to read it: an unknown 4.15 ms stack constant, a weight term that inverts under
batching, and an honest speculative server living inside the "impossible" region.

This figure keeps the premise and changes where it is read.

  docs/figures/fig_clock_algos.png
      (A) THE PREFILL CLOCK. Time to the FIRST token against prompt length, four
          serving configurations. Prefill is compute-bound and its work is
          2 x params x prompt plus attention, so unlike a decode step it is not
          sitting on a launch-latency floor -- and it is measured here to 0.3%
          relative sd, tighter than the decode channel manages.
      (B) IT GIVES BACK THE ROW THE DECODE CLOCK LOST. `0.6B served as 1.7B` is a
          1.20x gap in decode ITL at B=1 and 1.03x at B=64; in prefill at a 4k prompt
          it is 1.62x, and the probe costs ONE token instead of a stream.
      (C) AND IT SEES THE QUANTIZATION ROW, WITH THE OPPOSITE SIGN. Real NF4 weights
          are 1.80x SLOWER to prefill a short prompt, because dequantization is
          compute and prefill is compute-bound. A deviation does not have to be fast
          to be visible -- it has to be DIFFERENT, which is a strictly weaker
          requirement than the figure's "the saving is the signal".
      (D) THE INTRA-STREAM CLOCK, AND WHY IT DOES NOT WORK HERE. Your own KV cache
          grows one position per output token, so an honest ITL must rise inside the
          stream you already bought. Measured against a real sliding-window provider,
          the growth term is swamped by GPU clock drift in this stack; the panel
          reports the ratio that decides it, which is the number a production stack
          would have to beat.
      (E) READ IT WITH A LOW QUANTILE, NOT A MEAN. Wire jitter is one-sided and
          positive -- the figure's own model -- so the low quantiles of a probe block
          are cleaner than its mean. Same measured samples, same cost, swept over
          sigma: the estimator alone moves the verdict price.
      (F) MODULATE, THEN CORRELATE. Slow load drift is the one nuisance a paired
          probe does not remove, and a strictly alternating probe aliases with a
          period-2 scheduler. A pseudorandom context schedule correlated against the
          returned latencies survives both.

MEASURED INPUTS   docs/results/clock_algos.json (+ _ttft.json) for (A)-(D);
    slope_verifier_window.json for the probe samples in (E); clock_channel.json for
    the ITL(B) grid the load process in (F) resamples. delta* = 3.767 as everywhere.

MODELLED INPUTS   the client-side jitter in (E) (swept) and the load process in (F)
    (a drift and a period-2 interferer over measured distributions). Nothing else.

    python -m experiments.plot_clock_algos
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.plot_clock_channel_principle import (
    DEV, FIG_DIR, GRID, HONEST, INK, INK2, MUTED, WARN,
)
from experiments.plot_clock_measured import GREEN, price

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results"
JITTER_SHAPE = 1.6
SIGMAS = np.array([1.0, 5.0, 10.0, 25.0, 50.0, 100.0])
PROBE_N = 32                      # probe pairs per block in (E)


def load_all():
    d = json.load(open(RES / "clock_algos.json"))
    tt = RES / "clock_algos_ttft.json"
    if not d.get("ttft") and tt.exists():
        d["ttft"] = json.load(open(tt))["ttft"]
    if not d.get("intra") and tt.exists():
        d["intra"] = json.load(open(tt)).get("intra", [])
    win = json.load(open(RES / "slope_verifier_window.json"))["cells"]
    grid = json.load(open(RES / "clock_channel.json"))["cells"]
    return d, {c["label"]: c for c in win}, grid


def ttft(d, label):
    rows = sorted([r for r in d["ttft"] if r["label"] == label],
                  key=lambda r: r["ctx"])
    return (np.array([r["ctx"] for r in rows], float),
            np.array([r["min"] for r in rows]),          # min: one-sided nuisance
            np.array([r["mean"] for r in rows]),
            np.array([r["sd"] for r in rows]))


def jit(rng, shape, sigma):
    return rng.gamma(JITTER_SHAPE, sigma / JITTER_SHAPE, shape)


# ============================================================================ (A)
def panel_a(ax, d):
    for lab, col, ls in (("Qwen3-4B", GREEN, "-"), ("Qwen3-1.7B", DEV, "-"),
                         ("Qwen3-1.7B-NF4", WARN, "--"), ("Qwen3-0.6B", HONEST, "-")):
        x, y, ym, sd = ttft(d, lab)
        if not len(x):
            continue
        ax.plot(x, y, ls, marker="o", ms=4.6, color=col, lw=1.9, mec="white", mew=0.7,
                zorder=4)
        dy = {"Qwen3-4B": 8, "Qwen3-1.7B": 10, "Qwen3-1.7B-NF4": -14,
              "Qwen3-0.6B": -14}[lab]
        ax.annotate(f"{lab}   {y[-1]:,.0f} ms", (x[-1], y[-1]), xytext=(-6, dy),
                    textcoords="offset points", ha="right",
                    va="bottom" if dy > 0 else "top", color=col, fontsize=8.2,
                    zorder=6)
    rel = float(np.mean([(sd / ym)[x >= 4096]
                         for lab in ("Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B")
                         for x, _, ym, sd in [ttft(d, lab)]]))
    ax.annotate("Prefill is COMPUTE-bound: no launch-latency floor, and the work is\n"
                "the client's own prompt times the provider's own parameters.\n"
                f"Relative sd over 5 reps at prompts $\\geq$ 4k: {rel:.2%} -- "
                "tighter than the\ndecode channel, on an observable 100x larger.",
                (0.03, 0.97), xycoords="axes fraction", ha="left", va="top",
                color=INK2, fontsize=8.4, linespacing=1.6)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("prompt length (tokens)", fontsize=9.2, color=INK2)
    ax.set_ylabel("time to first token (ms)", fontsize=9.2, color=INK2)
    ax.set_title("(A)  The prefill clock: one probe, one token, no stream",
                 fontsize=10.6, weight="bold", color=INK, loc="left")


# ============================================================================ (B)
def channel_table(d, grid):
    """Deviation x channel, as the ratio each channel sees. Decode numbers come from
    the batching grid of fig_clock_measured; prefill from this experiment."""
    g = {(c["label"], c["B"], c["ctx"]): c for c in grid if c["mode"] == "graph"}

    def dec(lab_h, lab_d, B):
        h, a = g.get((lab_h, B, 1024)), g.get((lab_d, B, 1024))
        return (h["mean"] / a["mean"]) if h and a else np.nan

    def pre(lab_h, lab_d, ctx):
        xh, yh, _, _ = ttft(d, lab_h)
        xa, ya, _, _ = ttft(d, lab_d)
        ih = int(np.where(xh == ctx)[0][0])
        ia = int(np.where(xa == ctx)[0][0])
        return yh[ih] / ya[ia]

    return [
        ("0.6B served\nas 1.7B", dec("Qwen3-1.7B", "Qwen3-0.6B", 1),
         dec("Qwen3-1.7B", "Qwen3-0.6B", 64), pre("Qwen3-1.7B", "Qwen3-0.6B", 4096)),
        ("real NF4 weights\n(1.7B)", dec("Qwen3-1.7B", "Qwen3-1.7B-NF4", 1),
         dec("Qwen3-1.7B", "Qwen3-1.7B-NF4", 64),
         pre("Qwen3-1.7B", "Qwen3-1.7B-NF4", 256)),
    ]


def panel_b(ax, d, grid):
    rows = channel_table(d, grid)
    y = np.arange(len(rows))[::-1]
    h = 0.25
    for (name, d1, d64, pf), yy in zip(rows, y):
        for val, off, col, lab in ((d1, h, MUTED, "decode ITL, B=1"),
                                   (d64, 0.0, HONEST, "decode ITL, B=64"),
                                   (pf, -h, DEV, "prefill TTFT")):
            v = val if np.isfinite(val) else 1.0
            ax.barh(yy + off, v - 1.0, height=h * 0.92, left=1.0,
                    color=col, zorder=4 if col == DEV else 3)
            ax.annotate(f"{v:.2f}x", (v, yy + off),
                        xytext=(5 if v >= 1 else -5, 0), textcoords="offset points",
                        va="center", ha="left" if v >= 1 else "right",
                        fontsize=8.4, color=col,
                        weight="bold" if col == DEV else "normal")
    ax.axvline(1.0, color=INK2, lw=1.2, zorder=5)
    for lab, col, yy in (("decode ITL, B=1", MUTED, 0.36), ("decode ITL, B=64", HONEST, 0.30),
                         ("prefill TTFT", DEV, 0.24)):
        ax.annotate(lab, (0.985, yy), xycoords="axes fraction", ha="right",
                    va="top", color=col, fontsize=8.4, weight="bold")
    ax.annotate("A ratio of 1.00x is a channel that sees nothing.\n"
                "Prefill is the only column with a row it can read\n"
                "at production batch sizes -- and the NF4 row is\n"
                "read with the opposite SIGN, because dequantization\n"
                "is compute and prefill is compute-bound.",
                (0.0, -0.165), xycoords="axes fraction", ha="left", va="top",
                color=INK2, fontsize=8.3, linespacing=1.6)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.8)
    ax.set_xlim(0.4, 2.0)
    ax.set_xlabel("honest time / deviating time   (1.00x = invisible)", fontsize=9.0,
                  color=INK2)
    ax.set_title("(B)  Prefill reads the rows decode cannot", fontsize=10.6,
                 weight="bold", color=INK, loc="left")


# ============================================================================ (C)
def panel_c(ax, d):
    xh, yh, _, _ = ttft(d, "Qwen3-1.7B")
    xa, ya, _, _ = ttft(d, "Qwen3-1.7B-NF4")
    xs, ys, _, _ = ttft(d, "Qwen3-0.6B")
    n = min(len(xh), len(xa), len(xs))
    ax.plot(xh[:n], ya[:n] / yh[:n], "-o", ms=5, color=WARN, lw=2.0, mec="white",
            mew=0.8, zorder=5)
    ax.plot(xh[:n], ys[:n] / yh[:n], "-o", ms=5, color=HONEST, lw=2.0, mec="white",
            mew=0.8, zorder=5)
    ax.axhline(1.0, color=INK2, lw=1.2, ls="--", zorder=3)
    ax.annotate(f"NF4 weights: {ya[0] / yh[0]:.2f}x SLOWER at a 256-token prompt,\n"
                f"{ya[-1] / yh[-1]:.3f}x at 32k -- dequantization is a fixed compute\n"
                "tax per weight, so it shows where weights dominate the\n"
                "prefill and vanishes where attention does.",
                (0.985, 0.97), xycoords="axes fraction", ha="right", va="top",
                color=WARN, fontsize=8.2, linespacing=1.6)
    ax.annotate(f"0.6B served as 1.7B: {ys[2] / yh[2]:.2f}x faster at 4k.\n"
                "The separation peaks where the parameter term still\n"
                "matters and attention has not taken over.",
                (0.03, 0.03), xycoords="axes fraction", ha="left", va="bottom",
                color=HONEST, fontsize=8.2, linespacing=1.6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("prompt length (tokens)", fontsize=9.2, color=INK2)
    ax.set_ylabel("deviating TTFT / honest TTFT", fontsize=9.2, color=INK2)
    ax.set_title("(C)  Where in the prompt-length axis each deviation is loudest",
                 fontsize=10.6, weight="bold", color=INK, loc="left")


# ============================================================================ (D)
def panel_d(ax, d):
    if not d.get("intra"):
        ax.set_title("(D)  intra-stream arm not present", fontsize=10.6, loc="left")
        return
    recs = {r["label"]: r for r in d["intra"]}
    hon, win = recs.get("honest"), recs.get("window_1024")
    K = 64                                   # bin the trace so the trend is visible
    for r, col, lab in ((hon, DEV, "honest, cache grows"),
                        (win, WARN, "sliding window 1 024, cache does not")):
        if r is None:
            continue
        a = np.asarray(r["itl_ms"], float)
        m = a.mean(axis=0)
        x = np.arange(len(m))
        mb = m[: len(m) // K * K].reshape(-1, K).mean(axis=1)
        xb = x[: len(m) // K * K].reshape(-1, K).mean(axis=1)
        ax.plot(xb, mb, "-", color=col, lw=1.7, zorder=4)
        ax.annotate(f"{lab}\nfitted growth "
                    f"{r['slope_us_per_token']:+.2f} $\\mu$s/token",
                    (xb[-1], mb[-1]), xytext=(-6, 22 if col == DEV else -12),
                    textcoords="offset points", ha="right",
                    va="bottom" if col == DEV else "top", color=col, fontsize=8.0,
                    linespacing=1.5, zorder=6)
    if hon and win:
        spread = float(np.std(hon["slope_reps"] + win["slope_reps"]))
        kv = hon["kv_bytes_per_token"]
        ax.annotate("The physics is real: the cache grows one position per token, so an\n"
                    "honest stream must get slower and a windowed one must not.\n\n"
                    "But this stack's context term is 0.031 $\\mu$s/token (eager, "
                    f"measured in\nCLOCK_MEASURED) and the run-to-run spread of the "
                    f"fitted slope is\n{spread:.3f} $\\mu$s/token -- clock ramp, not "
                    "the KV cache. Signal below\nnoise: the two arms are "
                    "indistinguishable, and the panel says so.\n\n"
                    "A CUDA-graph stack's context term is 1.65 $\\mu$s/token, 30x "
                    f"this noise\nfloor -- but a padded static cache does not grow, "
                    "so the growth is\nonly readable on a server that attends valid "
                    "positions only (vLLM).\n\n"
                    "It is the only free probe in the channel, so this is the arm a\n"
                    "production (paged-attention) stack should be re-run on.",
                    (0.03, 0.97), xycoords="axes fraction", ha="left", va="top",
                    color=INK2, fontsize=8.2, linespacing=1.6)
    lo_, hi_ = ax.get_ylim()
    ax.set_ylim(lo_, lo_ + (hi_ - lo_) * 2.9)
    ax.set_xlabel("output token index within one generation", fontsize=9.2, color=INK2)
    ax.set_ylabel("inter-token latency (ms)", fontsize=9.2, color=INK2)
    ax.set_title("(D)  The intra-stream clock: free, and drift-limited here",
                 fontsize=10.6, weight="bold", color=INK, loc="left")


# ============================================================================ (E)
ESTIMATORS = {
    "mean": lambda v: float(np.mean(v)),
    "20% trimmed mean": lambda v: float(np.mean(np.sort(v)[int(.1 * len(v)):
                                                          len(v) - int(.1 * len(v))])),
    "median": lambda v: float(np.median(v)),
    "25th pct": lambda v: float(np.quantile(v, 0.25)),
    "10th pct": lambda v: float(np.quantile(v, 0.10)),
    "minimum": lambda v: float(np.min(v)),
}


def burst(rng, v, eps):
    """Client-visible arrival bursts: a fraction `eps` of gaps collapse to ~0 and the
    rest carry their time, so the total is conserved. This is what speculation and
    SSE frame coalescing do to a stream -- measured at 73% zero gaps in
    `fig_clock_measured` (E), and used here at a mild rate."""
    if eps <= 0:
        return v
    z = rng.random(len(v)) < eps
    out = v / (1.0 - eps)
    out[z] = 0.0
    return out


def estimator_prices(win, n=PROBE_N, n_block=600, seed=3, eps=0.0):
    """Tokens of stream per verdict for each estimator, as a function of sigma.

    One block = `n` probe pairs = 2n tokens of stream. The honest arm holds the full
    32 768 context, the deviating arm holds 8 192."""
    rng = np.random.default_rng(seed)
    lo = np.asarray(win["probe_lo"]["itl_ms"], float)
    hi_h = np.asarray(win["honest_hi"]["itl_ms"], float)
    hi_a = np.asarray(win["window_8192"]["itl_ms"], float)
    out = {k: [] for k in ESTIMATORS}
    for sigma in SIGMAS:
        for name, f in ESTIMATORS.items():
            stats = {"h": np.empty(n_block), "a": np.empty(n_block)}
            for b in range(n_block):
                l = burst(rng, rng.choice(lo, n), eps) + jit(rng, n, sigma)
                stats["h"][b] = f(burst(rng, rng.choice(hi_h, n), eps)
                                  + jit(rng, n, sigma)) - f(l)
                l2 = burst(rng, rng.choice(lo, n), eps) + jit(rng, n, sigma)
                stats["a"][b] = f(burst(rng, rng.choice(hi_a, n), eps)
                                  + jit(rng, n, sigma)) - f(l2)
            dp = (stats["h"].mean() - stats["a"].mean()) / (stats["h"].std() + 1e-12)
            out[name].append(2 * n * price(dp))       # tokens of stream per verdict
    return {k: np.array(v) for k, v in out.items()}


def panel_e(ax, win):
    res = estimator_prices(win)
    cols = {"mean": MUTED, "20% trimmed mean": HONEST, "median": INK2,
            "25th pct": GREEN, "10th pct": DEV, "minimum": WARN}
    for name, v in res.items():
        ax.plot(SIGMAS, v, "-o", ms=4.4, color=cols[name], lw=1.9, mec="white",
                mew=0.7, zorder=5 if name == "10th pct" else 4)
        ax.plot([], [], "-o", ms=4, color=cols[name], label=name)
    burst_res = estimator_prices(win, eps=0.15)
    i = len(SIGMAS) - 2
    gain = res["mean"][i] / res["10th pct"][i]
    ax.annotate("The wire's nuisance is one-sided and positive, so the LOW order\n"
                "statistics of a probe block beat its mean: at $\\sigma$ = "
                f"{SIGMAS[i]:.0f} ms the 10th\npercentile is {gain:.1f}x cheaper for "
                "the same verdict. An estimator\nchange, not more data.",
                (0.03, 0.87), xycoords="axes fraction", ha="left", va="top",
                color=INK2, fontsize=8.3, linespacing=1.6)
    ins = ax.inset_axes([0.60, 0.05, 0.38, 0.26])
    keys = ["mean", "25th pct", "10th pct", "minimum"]
    xk = np.arange(len(keys))
    ins.bar(xk - 0.2, [res[k][i] for k in keys], width=0.38, color=DEV, zorder=4)
    ins.bar(xk + 0.2, [burst_res[k][i] for k in keys], width=0.38, color=WARN,
            zorder=4)
    ins.set_xticks(xk)
    ins.set_xticklabels(["mean", "q25", "q10", "min"], fontsize=6.6)
    ins.set_yscale("log")
    ins.tick_params(labelsize=6.4, colors=INK2, pad=1)
    ins.set_title(f"at $\\sigma$ = {SIGMAS[i]:.0f} ms:  clean (blue) vs 15% burst "
                  "arrivals (orange)", fontsize=6.8, color=INK, weight="bold", pad=3)
    ins.set_ylabel("tokens", fontsize=6.6, color=INK2, labelpad=1)
    for sp in ("top", "right"):
        ins.spines[sp].set_visible(False)
    mn_c, mn_b = res["minimum"][i], burst_res["minimum"][i]
    q_c, q_b = res["25th pct"][i], burst_res["25th pct"][i]
    ax.annotate("The MINIMUM -- what fig_clock_basic (3) proposes -- wins on a clean\n"
                f"wire ({mn_c:,.0f} tokens) and dies on a real one "
                f"({mn_b:,.0f} under 15% burst\narrivals): one collapsed gap per "
                "block is enough. The 25th percentile\nis "
                f"{res['mean'][i] / q_c:.1f}x cheaper than the mean clean and "
                f"{burst_res['mean'][i] / q_b:.1f}x cheaper bursty. Ship the\n"
                "quantile, not the minimum and not the mean.",
                (0.03, 0.56), xycoords="axes fraction", ha="left", va="top",
                color=INK2, fontsize=8.0, linespacing=1.55)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=7.6, loc="upper left", frameon=False, ncol=3,
              bbox_to_anchor=(0.0, 1.02), columnspacing=1.0, handletextpad=0.4)
    ax.set_xlim(0.8, 130)
    ax.set_xlabel("client-side jitter $\\sigma$ (ms)", fontsize=9.2, color=INK2)
    ax.set_ylabel("tokens of stream per pAUC 0.90 verdict", fontsize=9.2, color=INK2)
    ax.set_title("(E)  Same samples, same cost: the estimator moves the price",
                 fontsize=10.6, weight="bold", color=INK, loc="left")


# ============================================================================ (F)
def lockin(grid, n_probe=64, n_trial=1200, seed=5):
    """Three probe schedules against two nuisances, over the measured ITL(B) grid.

    The nuisance is drawn fresh for every trial -- a drift with a random slope and
    sign, or a period-2 scheduler with a random phase. That is the point: a fixed
    schedule absorbs a KNOWN nuisance as a constant bias and looks fine, and it is
    the UNKNOWN realization that turns the bias into variance and eats the verdict."""
    rng = np.random.default_rng(seed)
    lo = {c["B"]: np.asarray(c["itl_ms"], float) for c in grid
          if c["label"] == "Qwen3-1.7B" and c["mode"] == "graph" and c["ctx"] == 1024}
    hi = {c["B"]: np.asarray(c["itl_ms"], float) for c in grid
          if c["label"] == "Qwen3-1.7B" and c["mode"] == "graph" and c["ctx"] == 8192}
    Bs = sorted(set(lo) & set(hi))
    K = len(Bs)

    def load(kind):
        if kind == "drift":                       # random slope and direction
            a, b = rng.integers(0, K), rng.integers(0, K)
            return [Bs[int(round(a + (b - a) * t / (n_probe - 1)))]
                    for t in range(n_probe)]
        ph = int(rng.integers(0, 2))              # period-2, random phase
        return [Bs[0] if (t + ph) % 2 == 0 else Bs[-1] for t in range(n_probe)]

    fixed = {"two blocks": np.r_[np.ones(n_probe // 2), np.zeros(n_probe // 2)],
             "alternating": np.array([t % 2 for t in range(n_probe)], float)}
    out = {}
    for kind in ("drift", "period-2 interferer"):
        for sname in ("two blocks", "alternating", "pseudorandom"):
            h, a = np.empty(n_trial), np.empty(n_trial)
            for t in range(n_trial):
                L = load("drift" if kind == "drift" else "p2")
                s_ = (rng.integers(0, 2, n_probe).astype(float)
                      if sname == "pseudorandom" else fixed[sname])
                w = s_ - s_.mean()
                den = float(w @ w) or 1.0
                # honest: a long probe really is long. truncating provider: every
                # request costs what the short one costs, whatever was asked for.
                yh = np.array([float(rng.choice((hi if s_[i] else lo)[L[i]]))
                               for i in range(n_probe)])
                ya = np.array([float(rng.choice(lo[L[i]])) for i in range(n_probe)])
                h[t] = float(w @ yh) / den
                a[t] = float(w @ ya) / den
            out[(kind, sname)] = (h, a, (h.mean() - a.mean()) / (h.std() + 1e-12))
    return out


def panel_f(ax, grid):
    res = lockin(grid)
    kinds = ["drift", "period-2 interferer"]
    names = ["two blocks", "alternating", "pseudorandom"]
    w = 0.26
    worst = {sn: min(res[(k, sn)][2] for k in kinds) for sn in names}
    for j, sname in enumerate(names):
        vals = [res[(k, sname)][2] for k in kinds]
        col = (MUTED, HONEST, DEV)[j]
        ax.bar(np.arange(len(kinds)) + (j - 1) * w, vals, width=w * 0.9, color=col,
               zorder=4, label=sname)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.2f}", (i + (j - 1) * w, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8.0,
                        color=col)
    ax.set_xticks(np.arange(len(kinds)))
    ax.set_xticklabels(["load DRIFTS during the probe",
                        "load ALTERNATES (period-2 scheduler)"], fontsize=8.8)
    ax.legend(fontsize=8.2, frameon=False, loc="upper left",
              bbox_to_anchor=(0.0, 0.66))
    ax.annotate("Same 64 probes, three schedules, and a nuisance whose\n"
                "realization is unknown to the client. Two blocks confound\n"
                "an unknown drift with the deviation; alternating fixes drift\n"
                "and then aliases onto a period-2 scheduler; a pseudorandom\n"
                "context schedule survives both.\n\n"
                "",
                (0.02, 0.97), xycoords="axes fraction", ha="left", va="top",
                color=INK2, fontsize=8.3, linespacing=1.6)

    ax.set_yscale("log")
    ax.set_ylim(0.7, 3000)
    ax.set_ylabel("$d'$ per 64-probe block  (log)", fontsize=9.2, color=INK2)
    ax.set_title(f"(F)  The schedule is a design variable  (worst case "
                 f"{worst['two blocks']:.1f} / {worst['alternating']:.1f} / "
                 f"{worst['pseudorandom']:.1f})",
                 fontsize=10.6, weight="bold", color=INK, loc="left")


# =========================================================================== main
def report(d, win, grid):
    print("=" * 92)
    print("Three more clocks, and how to read them")
    print("=" * 92)
    print("\n(A)/(B)/(C) the prefill clock -- TTFT (min of 5) in ms")
    labs = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-1.7B-NF4"]
    x0 = ttft(d, labs[0])[0]
    print(f"  {'prompt':>8s} " + " ".join(f"{l:>16s}" for l in labs))
    for i, ctx in enumerate(x0):
        row = " ".join(f"{ttft(d, l)[1][i]:16.1f}" for l in labs)
        print(f"  {int(ctx):8,d} {row}")
    print("\n  deviation x channel (honest time / deviating time; 1.00 = invisible)")
    for name, d1, d64, pf in channel_table(d, grid):
        print(f"  {name.replace(chr(10), ' '):26s} decode B=1 {d1:5.2f}x   "
              f"decode B=64 {d64:5.2f}x   prefill {pf:5.2f}x")

    print("\n  the same rows priced at a common wire jitter (gamma, one-sided)")
    print(f"  {'deviation':26s} {'channel':22s} {'delta (ms)':>11s} "
          f"{'d prime':>9s} {'probe pairs':>12s}")
    g = {(c["label"], c["B"], c["ctx"]): c for c in grid if c["mode"] == "graph"}
    for sg in (50.0, 500.0):
        sd_pair = 0.79 * sg * np.sqrt(2.0)     # gamma(1.6) sd = sigma/sqrt(1.6)
        rows = [
            ("0.6B served as 1.7B", f"decode ITL B=1, s={sg:.0f}ms",
             g[("Qwen3-1.7B", 1, 1024)]["mean"] - g[("Qwen3-0.6B", 1, 1024)]["mean"]),
            ("0.6B served as 1.7B", f"prefill 32k, s={sg:.0f}ms",
             float(ttft(d, "Qwen3-1.7B")[1][-1] - ttft(d, "Qwen3-0.6B")[1][-1])),
            ("real NF4 weights", f"prefill 256, s={sg:.0f}ms",
             float(ttft(d, "Qwen3-1.7B-NF4")[1][0] - ttft(d, "Qwen3-1.7B")[1][0])),
        ]
        for name, chan, delta in rows:
            dp = abs(delta) / sd_pair
            print(f"  {name:26s} {chan:22s} {delta:11.2f} {dp:9.3f} "
                  f"{price(dp):12,.0f}")

    if d.get("intra"):
        print("\n(D) intra-stream growth, eager stack")
        for r in d["intra"]:
            print(f"  {r['label']:12s} fitted {r['slope_us_per_token']:+7.3f} us/token"
                  f"   reps {[round(v, 2) for v in r['slope_reps']]}")
        allr = [v for r in d["intra"] for v in r["slope_reps"]]
        print(f"  run-to-run spread of the fitted slope: {np.std(allr):.3f} us/token "
              f"-- this stack's own "
              f"context term is 0.031, so the growth sits below its noise floor.")

    print("\n(E) estimator, tokens of stream per verdict (32-pair blocks)")
    res = estimator_prices(win)
    print(f"  {'sigma (ms)':>11s} " + " ".join(f"{k:>17s}" for k in ESTIMATORS))
    for i, s in enumerate(SIGMAS):
        print(f"  {s:11.0f} " + " ".join(f"{res[k][i]:17,.0f}" for k in ESTIMATORS))
    best = min(ESTIMATORS, key=lambda k: res[k][-2])
    print(f"  best at sigma={SIGMAS[-2]:.0f}: {best} "
          f"({res['mean'][-2] / res[best][-2]:.1f}x cheaper than the mean)")
    bres = estimator_prices(win, eps=0.15)
    print("  the same, with 15% of arrivals collapsed into bursts:")
    print(f"  {'sigma (ms)':>11s} " + " ".join(f"{k:>17s}" for k in ESTIMATORS))
    for i, sg in enumerate(SIGMAS):
        print(f"  {sg:11.0f} " + " ".join(f"{bres[k][i]:17,.0f}" for k in ESTIMATORS))
    b2 = min(ESTIMATORS, key=lambda k: bres[k][-2])
    print(f"  best under bursts at sigma={SIGMAS[-2]:.0f}: {b2}; the minimum goes "
          f"{res['minimum'][-2]:,.0f} -> {bres['minimum'][-2]:,.0f} tokens")

    print("\n(F) probe schedule, d' per 64-probe block")
    lk = lockin(grid)
    for kind in ("drift", "period-2 interferer"):
        row = "   ".join(f"{s}: {lk[(kind, s)][2]:5.2f}"
                         for s in ("two blocks", "alternating", "pseudorandom"))
        print(f"  {kind:22s} {row}")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d, win, grid = load_all()
    report(d, win, grid)

    fig, axes = plt.subplots(2, 3, figsize=(19.0, 10.6))
    for ax in axes.flat:
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8.4, colors=INK2)
    panel_a(axes[0, 0], d)
    panel_b(axes[0, 1], d, grid)
    panel_c(axes[0, 2], d)
    panel_d(axes[1, 0], d)
    panel_e(axes[1, 1], win)
    panel_f(axes[1, 2], grid)

    fig.suptitle("Same premise, better observables: the clock is loudest before the "
                 "first token, and cheapest read with a low quantile",
                 fontsize=15.5, weight="bold", color=INK, x=0.006, ha="left", y=0.988)
    fig.text(0.006, 0.955,
             "fig_clock_basic's claim is that a token cannot arrive before its bytes "
             "have moved. It reads that claim in a batch-1 decode gap -- the one part "
             "of serving that is neither compute-bound nor\nproportional to what the "
             "client asked for. Read the same claim in the PREFILL and it recovers "
             "the deviations the decode channel lost; read it with the right "
             "estimator and schedule and it gets cheaper again.",
             fontsize=9.4, color=INK2, ha="left", va="top", linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_clock_algos.png"
    fig.savefig(out, dpi=155, facecolor="white")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
