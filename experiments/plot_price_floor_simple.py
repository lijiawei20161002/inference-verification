"""The one-idea version of `fig_price_floor_principle.png`.

That figure makes three arguments at once. This one makes the argument that has
to land first, and answers the only question that decides whether the rest is
worth reading: does the conclusion generalize, or is it a fact about this run?

  docs/figures/fig_price_floor_simple.png
      (A) The mechanism, on one cell.  fp8 KV / token DiFR has a fixed effect,
          d' = 0.020.  At 10 sequences per arm the estimate's spread swamps it and
          27% of runs report "no budget buys a verdict"; at 80 it is 4%.  Same
          provider, same deviation -- what changed is the audit's budget.
      (B) Whether that generalizes.  Rescale every cell's effect by the noise its
          own estimate carries at its own pool size, and all 35 cells at 4 pool
          sizes -- 5 deviations x 7 verifiers, effects spanning three orders of
          magnitude -- fall on the parameter-free curve P = Phi(-d'/se).  The x
          axis holds no model, verifier or deviation identity.  The "infinite
          price" label is a function of one number, and that number is a
          signal-to-noise ratio, not a property of the provider.

Reads the same per-token scores as the full figure and reuses its cache for (A).

    python -m experiments.plot_price_floor_simple
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "docs" / "figures"
RES = ROOT / "docs" / "results"
COST = RES / "cost_of_a_verdict.json"
SCORES = RES / "cost_of_a_verdict_scores.npz"
FULL_CACHE = RES / "price_floor_principle.json"     # (A) comes from the full figure
CACHE = RES / "price_floor_simple.json"

# Same palette and the same meaning as plot_price_floor_principle.py.
DEV, HONEST, INF = "#2a78d6", "#52514e", "#d1622b"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8880", "#e3e2dd"

N_REP = 1500
POOLS = (10, 20, 40, 80)
FOCUS = ("kv_fp8", "token_difr")
SHOW_A = (10, 80)                       # the two rows (A) needs to make the point


# ------------------------------------------------------------------------ data
def _dprime_rows(h, a):
    """`signal.per_token_stats` d', vectorized over an axis of independent draws.

    h, a are (R, n_tok). Winsorize at the honest 99.9th percentile of each row,
    exactly as the scalar version does, then d' = (mean_a - mean_h) / sd_h.
    """
    cap = np.percentile(h, 99.9, axis=1, keepdims=True)
    h, a = np.minimum(h, cap), np.minimum(a, cap)
    return (a.mean(1) - h.mean(1)) / (h.std(1) + 1e-12)


def _draws(h_seq, a_seq, m, reps, seed):
    """`reps` runs of the experiment at m sequences per arm, resampled from the pool."""
    rng = np.random.default_rng(seed)
    n = h_seq.shape[0]
    out = np.empty(reps)
    for lo in range(0, reps, 200):                       # chunked: (200, m*256) floats
        hi = min(lo + 200, reps)
        i = rng.integers(0, n, (hi - lo, m))
        j = rng.integers(0, n, (hi - lo, m))
        out[lo:hi] = _dprime_rows(h_seq[i].reshape(hi - lo, -1),
                                  a_seq[j].reshape(hi - lo, -1))
    return out


def measure():
    meta = json.loads(COST.read_text())
    z = np.load(SCORES, allow_pickle=True)
    n_seq, t = int(z["n_prompts"]), int(z["tokens"])
    get = lambda k: np.asarray(z[k], float).reshape(n_seq, t)
    honest = {v: get(f"honest__{v}") for v in meta["verifiers"]}

    # Every cell at every pool size: the effect, the noise the estimate carries at
    # that pool, and how often the run comes back "no budget buys a verdict".
    pts = []
    for atk in meta["attacks"]:
        for v in meta["verifiers"]:
            h, a = honest[v], get(f"{atk}__{v}")
            d_full = float(_dprime_rows(h.reshape(1, -1), a.reshape(1, -1))[0])
            for m in POOLS:
                d = _draws(h, a, m, N_REP, seed=1000 + m)
                pts.append({"attack": atk, "verifier": v, "m": m,
                            "d_full": d_full, "se": float(d.std(ddof=1)),
                            "p_inf": float((d <= 0).mean())})
        print(f"  {atk} done", flush=True)
    out = {"points": pts, "meta": {k: meta[k] for k in
                                   ("M", "proxy", "n_prompts", "tokens",
                                    "delta_star", "verifiers", "attacks")}}
    CACHE.write_text(json.dumps(out))
    print(f"wrote {CACHE}")
    return out


def load():
    return json.loads(CACHE.read_text()) if CACHE.exists() else measure()


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.set_axisbelow(True)


# =========================================================================== (A)
def panel_a(axes, F):
    """One cell, two budgets. The effect is fixed; the verdict is not."""
    lo, hi = -0.085, 0.115
    bins = np.linspace(lo, hi, 84)
    mid = 0.5 * (bins[1:] + bins[:-1])
    d_full = F["d_full"]

    for k, (ax, m) in enumerate(zip(axes, SHOW_A)):
        _clean(ax)
        cell = np.array(F["cell"][str(m)]["draws"])
        p_inf = F["cell"][str(m)]["p_inf"]
        dc, _ = np.histogram(cell, bins=bins, density=True)
        dc = dc / dc.max()

        ax.axvline(d_full, color=DEV, lw=1.0, ls=(0, (2, 2.5)), alpha=0.8, zorder=2)
        ax.axvline(0.0, color=INK, lw=1.3, zorder=6)
        ax.fill_between(mid, 0, dc, color=DEV, alpha=0.28, lw=0, zorder=3)
        ax.plot(mid, dc, color=DEV, lw=1.7, zorder=4)
        ax.fill_between(mid, 0, dc, where=mid <= 0, color=INF, alpha=0.95, lw=0,
                        zorder=5)

        ax.set_ylim(0, 1.22)
        ax.set_xlim(lo, hi)
        ax.set_yticks([])
        ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
        ax.spines["left"].set_visible(False)
        ax.annotate(f"$\\bf{{{m}}}$ sequences per arm", (0.015, 0.98),
                    xycoords="axes fraction", ha="left", va="top", color=INK2,
                    fontsize=9.5)
        ax.annotate(f"$\\bf{{{p_inf * 100:.0f}\\%}}$ of runs report\n\"no budget buys a verdict\"",
                    (0.985, 0.74), xycoords="axes fraction", ha="right", va="top",
                    color=INF, fontsize=9, linespacing=1.35)
        if k == 0:
            ax.annotate(f"the effect: $d'$ = {d_full:.3f}", (d_full, 1.06),
                        xytext=(6, 0), textcoords="offset points", ha="left",
                        va="bottom", color=DEV, fontsize=9, weight="bold",
                        annotation_clip=False)
            ax.annotate("orange = the runs\npriced at $\\infty$", (0.02, 0.70),
                        xycoords="axes fraction", ha="left", va="top", color=INF,
                        fontsize=9, weight="bold", linespacing=1.35)
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("$\\hat{d'}$ measured by one run of the experiment",
                          color=INK2, fontsize=10)

    axes[0].set_title("A.  Same cell, same deviation, different budget",
                      color=INK, fontsize=11.5, weight="bold", loc="left", pad=40)
    axes[0].annotate("fp8 KV / token DiFR.  The experiment's price rule returns "
                     "\"$\\infty$\" exactly when $\\hat{d'} \\leq 0$.",
                     (0.0, 1.14), xycoords="axes fraction", ha="left", va="bottom",
                     color=INK2, fontsize=9, annotation_clip=False)


# =========================================================================== (B)
def panel_b(ax, P):
    """Everything, rescaled by its own noise. One curve, no fitted parameter."""
    import math
    phi = lambda x: 0.5 * math.erfc(x / math.sqrt(2.0))      # P(Z <= -x)

    zs = np.linspace(-3.2, 5.2, 400)
    ax.plot(zs, [phi(z) for z in zs], color=INK, lw=2.0, zorder=5,
            label="$\\Phi(-d'/\\mathrm{se})$  $-$ no fitted parameter")

    ax.axvspan(-3.2, 1.645, color=INF, alpha=0.07, lw=0, zorder=0)
    ax.axvline(1.645, color=INF, lw=1.1, ls=(0, (4, 2.5)), zorder=2)
    ax.annotate("inside the shaded band the run has\na $\\geq$ 5% chance of retiring a "
                "real\ndeviation as unpriceable", (-3.05, 0.58), ha="left", va="top",
                color=INF, fontsize=9, linespacing=1.5)

    focus, other = [], []
    for p in P:
        (focus if (p["attack"], p["verifier"]) == FOCUS else other).append(p)
    zx = lambda p: p["d_full"] / p["se"]

    ax.scatter([zx(p) for p in other], [p["p_inf"] for p in other], s=26,
               facecolor=DEV, edgecolor="white", lw=0.6, alpha=0.85, zorder=6,
               label=f"the other {len(other) // len(POOLS)} cells, 4 pool sizes each")
    focus.sort(key=zx)
    ax.plot([zx(p) for p in focus], [p["p_inf"] for p in focus], color=INF, lw=1.4,
            marker="o", ms=6.5, mec="white", mew=0.8, zorder=7,
            label="the cell in (A), 10 $\\rightarrow$ 80 sequences")

    ax.set_xlim(-3.2, 5.2)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("the effect in units of the noise its own estimate carries,  "
                  "$d' / \\mathrm{se}(\\hat{d'})$", color=INK2, fontsize=10)
    ax.set_ylabel("P(the run reports \"no budget buys a verdict\")", color=INK2,
                  fontsize=10)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_title("B.  Rescale by the noise and every cell is the same cell",
                 color=INK, fontsize=11.5, weight="bold", loc="left", pad=40)
    ax.annotate("5 deviations $\\times$ 7 verifiers $\\times$ 4 pool sizes.  Effects span "
                "$d'$ = $-$0.04 to 10; the $x$ axis knows none of that.",
                (0.0, 1.02), xycoords="axes fraction", ha="left", va="bottom",
                color=INK2, fontsize=9, annotation_clip=False)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9, labelspacing=0.6,
                    handlelength=1.7, borderaxespad=0.4)
    for t in leg.get_texts():
        t.set_color(INK2)


# ====================================================================== footer
def footer(fig, ratios):
    import matplotlib.pyplot as plt
    fig.add_artist(plt.Line2D([0.008, 0.992], [0.175, 0.175], color=MUTED, lw=0.9))
    cols = [
        ("What generalizes:  the conclusion.",
         "(B) is arithmetic, not a finding.  The price is $b^{*} = (\\delta^{*}/d')^{2}$ "
         "and the code returns $\\infty$ on\nthe $\\it{sign}$ of $\\hat{d'}$, so the label "
         "is $\\Phi(-d'/\\mathrm{se})$ for $\\it{any}$ model, verifier or deviation.  "
         "\"$\\infty$\" means the\npool ran out, not that the deviation is undetectable.  "
         "Report a price floor $b^{*} \\geq N$ instead."),
        ("What does not:  the numbers.",
         f"Which cells land left of the dashed line is a fact about this run "
         f"(Qwen3-1.7B audited by 0.6B,\n80$\\times$256 tokens), and so is the cost of "
         f"fixing it: resolving a price runs {ratios[0]:.1f}$\\times$ to "
         f"{ratios[1]:.1f}$\\times$ $b^{{*}}$ here,\nbut that ratio scales with the "
         f"verifier's own noise, which already varies 4$\\times$ across these seven."),
    ]
    for x, (head, body) in zip((0.010, 0.512), cols):
        fig.text(x, 0.150, head, color=INK, fontsize=11, weight="bold", ha="left",
                 va="top")
        fig.text(x, 0.113, body, color=INK, fontsize=9.5, ha="left", va="top",
                 linespacing=1.62)


# ========================================================================== main
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D = load()
    full = json.loads(FULL_CACHE.read_text())
    F = full["flip"][f"{FOCUS[0]}__{FOCUS[1]}"]
    r = [f["pool_over_bstar"] for f in full["floor"].values()]

    fig = plt.figure(figsize=(14.6, 7.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.92, 1.08], wspace=0.20,
                          left=0.052, right=0.985, top=0.760, bottom=0.245)
    sub = gs[0, 0].subgridspec(len(SHOW_A), 1, hspace=0.22)
    panel_a([fig.add_subplot(sub[i, 0]) for i in range(len(SHOW_A))], F)

    ax_b = fig.add_subplot(gs[0, 1])
    _clean(ax_b)
    panel_b(ax_b, D["points"])
    footer(fig, (min(r), max(r)))

    fig.suptitle("An infinite price is a coin flip on the sign of a noisy number",
                 color=INK, fontsize=15, weight="bold", x=0.008, ha="left", y=0.975)
    fig.text(0.008, 0.930,
             "The cost-of-a-verdict experiment retires 14 of 35 (deviation, verifier) "
             "cells at an infinite price.  Its rule, $b^{*} = (\\delta^{*}/d')^{2}$, returns "
             "\"no budget buys a verdict\" exactly when\nthe point estimate "
             "$\\hat{d'} \\leq 0$ $-$ a decision on the $\\it{sign}$ of a statistic whose "
             "standard error is 0.011 to 0.049 at this pool.\nBoth panels are re-read from "
             "that experiment's own per-token scores.  The fuller argument, and what to "
             "report instead:  fig_price_floor_principle.png.",
             color=INK2, fontsize=9, ha="left", va="top", linespacing=1.55)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_price_floor_simple.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
