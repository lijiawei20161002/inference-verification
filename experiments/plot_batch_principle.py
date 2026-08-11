"""Why does a larger audit batch work? The gap is a bias, the noise is not.

Companion to `fig_int8_wall.png` panel (A): that figure shows *that* the deployable
cross-stack channel goes 0.533 -> 0.995 between batch 245 and batch 8 000, this one
shows *why*, in the data's own units.

  docs/figures/fig_batch_principle.png
      (A) One token carries ~nothing. 98.96% of honest tokens and 98.37% of int8's
          are exactly-zero margins; the nonzero tails sit on top of each other.
          Per-token AUC 0.503.
      (B) Averaging n tokens does not touch the gap between the arms' means -- it
          shrinks the spread of the estimate as sigma/sqrt(n). Same x-axis in all
          four rows: the two population means are nailed in place while the
          sampling distributions collapse onto them.
      (C) The consequence. The audit's threshold is the honest 99.5th percentile
          (FPR <= 0.5%), which falls as mu_h + z*sigma/sqrt(n) and flattens onto
          the honest mean. int8's mean does not move. So the bar eventually drops
          below the evidence, at n ~ (z/d')^2.

Everything is bootstrapped from the same 3 072-token pool the sweep used, with the
same margin cap and the same FPR budget, so the numbers here are the ones behind
`fig_int8_wall.png`.

    .venv/bin/python -m experiments.plot_batch_principle
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
SWEEP = ROOT / "docs" / "results" / "specdec_difr_sweep.jsonl"

# Same palette slots as plot_int8_wall.py, and the same meaning: the blue is the
# deployable cross-stack channel. The honest arm is the null, so it wears ink, not a
# hue. Every mark is also direct-labeled, so identity is never carried by color alone.
DEV, HONEST = "#2a78d6", "#52514e"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8880", "#e3e2dd"

PAIR = "q2.5-1.5b"
BASE = dict(K=4, T=1.0, top_k=0, top_p=1.0, pbatch=1)
MARGIN_CAP = 30.0       # the sweep's `_cap`
FPR = 0.005             # the sweep's FPR budget
Z = 2.5758              # the 99.5th percentile of a standard normal
SWEEP_BATCH = 245
ROWS = (61, 245, 1000, 4000)    # batch sizes shown in (B)
N_BOOT = 40000


# ------------------------------------------------------------------------ data
def load():
    cells = {}
    for line in open(SWEEP):
        r = json.loads(line)
        if (r["pair"] == PAIR and r["mode"] == "coupled"
                and all(r[k] == v for k, v in BASE.items())):
            cells[r["prov"]] = r
    missing = [t for t in ("clean", "int8") if t not in cells]
    if missing:
        sys.exit(f"missing provider cells {missing} in {SWEEP}")
    marg = lambda t: np.minimum(np.asarray(cells[t]["marg"], float), MARGIN_CAP)
    return marg("clean"), marg("int8")


def batch_means(x, b, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    return x[rng.integers(0, len(x), size=(n, b))].mean(1)


def threshold_and_power(h, a, b):
    """The audit's decision at batch `b`: the honest 99.5th percentile of the batch
    mean, and the fraction of int8 batches that clear it."""
    thr = float(np.quantile(batch_means(h, b), 1 - FPR))
    return thr, float((batch_means(a, b) > thr).mean())


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.set_axisbelow(True)


# =========================================================================== (A)
def panel_a(ax, h, a):
    """One token, both arms, every token drawn. The story is the pile at zero."""
    rng = np.random.default_rng(1)
    for y, x, color, name in ((1.0, h, HONEST, "honest provider"),
                              (0.0, a, DEV, "int8 RTN-g128 weights")):
        nz = x[x > 0]
        ax.scatter(nz, y + rng.uniform(-0.11, 0.11, len(nz)), s=26, color=color,
                   alpha=0.75, edgecolor="white", linewidth=0.6, zorder=3)
        # The zeros cannot be drawn as 3 000 dots on one x-value; draw the pile as a
        # bar whose length is the count, and label it.
        ax.plot([0, 0], [y - 0.17, y + 0.17], color=color, lw=6, solid_capstyle="butt",
                zorder=4)
        ax.annotate(name, (-0.004, y + 0.40), ha="left", va="center", color=color,
                    fontsize=10, weight="bold")
        ax.annotate(f"{(x == 0).sum():,} of {len(x):,} tokens agree exactly "
                    f"(margin 0)   |   {len(nz)} disagree ({len(nz) / len(x):.2%})",
                    (-0.004, y + 0.24), ha="left", va="center", color=color,
                    fontsize=8.5)

    ax.annotate("Per-token AUC = 0.503.  One token is a coin that lands\n"
                "'deviant' 50.3% of the time -- no statistic fixes that.\n"
                "But 1.63% $\\neq$ 1.04%, and a rate difference is all\n"
                "aggregation needs: it is a $\\bf{bias}$, not noise.",
                (-0.004, -0.66), ha="left", va="center", color=INK, fontsize=9.5,
                linespacing=1.6)

    ax.set_yticks([])
    ax.set_ylim(-1.05, 1.72)
    ax.set_xlim(-0.006, 0.155)
    ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel("per-token margin (nats), capped at 30", color=INK2, fontsize=10)
    ax.set_title("A.  One token carries ~nothing", color=INK, fontsize=11,
                 weight="bold", loc="left", pad=8)


# =========================================================================== (B)
def panel_b(axes, h, a):
    """The mechanism: the gap between the means is fixed, the spread is sigma/sqrt(n)."""
    lo, hi = 0.0, 0.0034
    bins = np.linspace(lo, hi, 220)
    mid = 0.5 * (bins[1:] + bins[:-1])
    mu_h, mu_a, sd_h = h.mean(), a.mean(), h.std(ddof=1)

    for k, (ax, b) in enumerate(zip(axes, ROWS)):
        _clean(ax)
        H, A = batch_means(h, b), batch_means(a, b)
        thr = float(np.quantile(H, 1 - FPR))
        power = float((A > thr).mean())
        dh, _ = np.histogram(H, bins=bins, density=True)
        da, _ = np.histogram(A, bins=bins, density=True)
        top = max(dh.max(), da.max())

        # The two population means, in the same place in every row. This is the point
        # of the panel: these do not move, the humps around them do.
        for m, c in ((mu_h, HONEST), (mu_a, DEV)):
            ax.axvline(m, color=c, lw=1.0, ls=(0, (2, 2.5)), alpha=0.7, zorder=2)

        ax.fill_between(mid, 0, dh / top, color=HONEST, alpha=0.30, lw=0, zorder=3)
        ax.plot(mid, dh / top, color=HONEST, lw=1.6, zorder=4)
        ax.fill_between(mid, 0, da / top, color=DEV, alpha=0.30, lw=0, zorder=3)
        ax.plot(mid, da / top, color=DEV, lw=1.6, zorder=4)
        # The audit's decision: everything right of the honest 99.5th percentile.
        ax.fill_between(mid, 0, da / top, where=mid > thr, color=DEV, alpha=0.95,
                        lw=0, zorder=5)
        ax.axvline(thr, color=INK, lw=1.2, ls=(0, (4, 2.5)), zorder=6)

        ax.set_ylim(0, 1.30)
        ax.set_xlim(lo, hi)
        ax.set_yticks([])
        ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
        ax.spines["left"].set_visible(False)
        xr = 0.985 if thr / hi < 0.6 else thr / hi - 0.02
        ax.annotate(f"$\\bf{{n = {b:,}}}$   sd $= \\sigma/\\sqrt{{n}}$ = "
                    f"{sd_h / np.sqrt(b):.1e}",
                    (xr, 0.90), xycoords="axes fraction", ha="right", va="top",
                    color=INK2, fontsize=9)
        ax.annotate(f"{power:.0%} of int8 batches\nclear the threshold",
                    (xr, 0.66), xycoords="axes fraction", ha="right", va="top",
                    color=DEV, fontsize=8.5, weight="bold", linespacing=1.35)
        if k == 0:
            # The two population means, labelled once. They are the only thing in this
            # panel that is identical in all four rows.
            ax.annotate("honest", (mu_h, 1.10), xytext=(-3, 0), ha="right", va="bottom",
                        textcoords="offset points", color=HONEST, fontsize=9,
                        weight="bold", annotation_clip=False)
            ax.annotate("int8", (mu_a, 1.10), xytext=(4, 0), ha="left", va="bottom",
                        textcoords="offset points", color=DEV, fontsize=9,
                        weight="bold", annotation_clip=False)
            ax.annotate("", xy=(mu_a, 1.42), xytext=(mu_h, 1.42),
                        annotation_clip=False,
                        arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.2,
                                        shrinkA=0, shrinkB=0, mutation_scale=9))
            ax.annotate(f"the gap: {mu_a - mu_h:.1e} nats/token, the same in every row",
                        (mu_a, 1.42), xytext=(8, 0), textcoords="offset points",
                        ha="left", va="center", color=INK, fontsize=9, weight="bold",
                        annotation_clip=False)
            ax.annotate("threshold at FPR $\\leq$ 0.5% -- the honest 99.5th pct;\n"
                        "it walks left as n grows",
                        (thr, 0.17), xytext=(-9, 0), xycoords=("data", "axes fraction"),
                        textcoords="offset points", ha="right", va="center", color=INK,
                        fontsize=8.5, linespacing=1.35)
        if k == len(ROWS) - 1:
            ax.set_xticks([0, 0.001, 0.002, 0.003], ["0", "1", "2", "3"])
            ax.set_xlabel("mean margin over the batch  ($\\times 10^{-3}$ nats/token)",
                          color=INK2, fontsize=10)
        else:
            ax.set_xticks([0, 0.001, 0.002, 0.003], ["", "", "", ""])

    axes[0].set_title("B.  Averaging shrinks the noise, not the gap", color=INK,
                      fontsize=11, weight="bold", loc="left", pad=34)


# =========================================================================== (C)
def panel_c(ax, h, a):
    """The payoff: a falling bar and a fixed height."""
    import matplotlib
    mu_h, mu_a, sd_h = h.mean(), a.mean(), h.std(ddof=1)
    bs = np.unique(np.round(np.geomspace(50, 20000, 34)).astype(int))
    thr = np.array([np.quantile(batch_means(h, int(b), 20000), 1 - FPR) for b in bs])

    ax.fill_between(bs, mu_h, thr, color=HONEST, alpha=0.14, lw=0, zorder=2)
    ax.plot(bs, thr, color=INK, lw=2.2, zorder=5,
            label="what the audit must clear: honest 99.5th pct")
    ax.plot(bs, mu_h + Z * sd_h / np.sqrt(bs), color=INK, lw=1.2, ls=(0, (3, 3)),
            zorder=4, label="$\\mu_{honest} + 2.58\\,\\sigma/\\sqrt{n}$ (Gaussian)")
    ax.axhline(mu_a, color=DEV, lw=2.2, zorder=5, label="int8's mean margin (fixed)")
    ax.axhline(mu_h, color=HONEST, lw=1.4, ls=(0, (1, 2)), zorder=3,
               label="honest mean margin (the floor the bar falls onto)")

    # Where the falling bar crosses the fixed evidence: 50% power, and the number the
    # cost law predicts.
    i = int(np.argmax(thr <= mu_a))
    ax.plot([bs[i]], [mu_a], marker="o", ms=9, color=DEV, mec="white", mew=1.5,
            zorder=6)
    ax.annotate(f"n $\\approx$ {bs[i]:,}: the bar drops below the\nevidence -- half of "
                f"int8 audits convict",
                (bs[i], mu_a), xytext=(64, 5.3e-4), textcoords="data",
                ha="left", va="bottom", color=DEV, fontsize=9, weight="bold",
                linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=DEV, lw=1.0, shrinkA=8,
                                shrinkB=6))

    for b, off, va, txt in ((SWEEP_BATCH, (-9, 7), "bottom", "batch {b:,}\n{p:.0%} power"),
                            (8000, (-9, -6), "top", "batch {b:,}, {p:.0%} power")):
        t, p = threshold_and_power(h, a, b)
        ax.plot([b], [t], marker="o", ms=6.5, color=INK, mec="white", mew=1.2,
                zorder=6)
        ax.annotate(txt.format(b=b, p=p), (b, t), xytext=off,
                    textcoords="offset points", ha="right", va=va, color=INK2,
                    fontsize=8.5, linespacing=1.35)

    ax.annotate("The noise is sampling error -- it vanishes.  The gap is a "
                "$\\bf{bias}$ --\nit does not.  So detection is only a question of "
                "price:\n"
                "        $n^{*} \\approx (z_{\\alpha}+z_{\\beta})^{2}/d'^{2}$,"
                "   $d' = 0.073$\n"
                "Halve the effect, pay $4\\times$ the tokens.  The same-stack channel\n"
                "raises $d'$ to 0.90 instead -- the wall figure's $\\sim$10$\\times$ "
                "saving is\na bigger gap, not a smaller noise.",
                (0.012, 0.020), xycoords="axes fraction", ha="left", va="bottom",
                color=INK, fontsize=9, linespacing=1.6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1.0e-4, 6e-3)
    ax.set_yticks([4.5e-4, 8.4e-4, 2e-3, 4e-3],
                  ["4.5e-4\n(honest)", "8.4e-4\n(int8)", "2e-3", "4e-3"])
    ax.set_xlim(50, 20000)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("audit batch size n (tokens aggregated per decision)", color=INK2,
                  fontsize=10)
    ax.set_ylabel("mean margin (nats/token)", color=INK2, fontsize=10)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_title("C.  The bar falls onto the floor; the gap is left over",
                 color=INK, fontsize=11, weight="bold", loc="left", pad=8)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelspacing=0.7,
                    handlelength=1.8, borderaxespad=0.6)
    for t in leg.get_texts():
        t.set_color(INK2)


# ========================================================================== main
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, a = load()
    d_prime = (a.mean() - h.mean()) / h.std(ddof=1)

    fig = plt.figure(figsize=(16.6, 6.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.0, 1.12], wspace=0.24,
                          left=0.040, right=0.995, top=0.785, bottom=0.105)
    ax_a = fig.add_subplot(gs[0, 0])
    _clean(ax_a)
    panel_a(ax_a, h, a)

    sub = gs[0, 1].subgridspec(len(ROWS), 1, hspace=0.30)
    panel_b([fig.add_subplot(sub[i, 0]) for i in range(len(ROWS))], h, a)

    ax_c = fig.add_subplot(gs[0, 2])
    _clean(ax_c)
    panel_c(ax_c, h, a)

    fig.suptitle("Why a bigger audit batch works: the noise averages away, the bias "
                 "does not", color=INK, fontsize=13.5, weight="bold", x=0.008,
                 ha="left", y=0.985)
    fig.text(0.008, 0.945,
             f"Why fig_int8_wall's cross-stack curve climbs 0.533 $\\rightarrow$ 0.995 "
             f"between batch 245 and batch 8 000, in the data's own units.  Same pool: "
             f"int8 RTN-g128 weights vs an honest provider, {PAIR} coupled\n"
             f"speculation, {len(h):,} tokens per arm, cross-stack replay (the "
             f"deployable channel), per-token $d'$ = {d_prime:.3f}.  Sampling "
             f"distributions are {N_BOOT:,} bootstrap batches from that pool, so n "
             f"past {len(h):,} assumes the observed margin tail is representative.",
             color=INK2, fontsize=8.5, ha="left", va="top", linespacing=1.5)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_batch_principle.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
