"""Figures for the report, from the experiment JSONs. Light mode, print-safe."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

RES = REPO / "docs/results"
# `docs/results/prefix_cost.json` holds whichever attack ran last; each attack's
# run is archived here so the figures always name the attack they plot.
BASE = REPO / "runs/baseline_json"
OUT = HERE / "figs"
OUT.mkdir(parents=True, exist_ok=True)

# Validated categorical slots (light mode); see the design-system palette.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8a85"
GRID = "#dedddA"

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.frameon": False, "legend.fontsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def load(name, root=RES):
    p = root / name
    return json.loads(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------- figure 1
def fig_ratio():
    hl, ps = load("headline_ratio.json"), load("pool_scaling.json")
    if hl is None and ps is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))

    ax = axes[0]
    if hl:
        xs, ys, labs = [], [], []
        for a, row in hl["cells"].items():
            for d, cell in row.items():
                xs.append(cell["readme"]["auc"])
                ys.append(cell["ratioed"]["auc"])
                labs.append(f"{a}/{d}")
        xs, ys = np.array(xs), np.array(ys)
        ax.plot([0.35, 1.02], [0.35, 1.02], color=MUTED, lw=1.0, ls="--", zorder=1)
        ax.scatter(xs, ys, s=34, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
        n_inf = int(np.sum(xs > ys))
        ax.text(0.97, 0.06, f"{n_inf} of {len(xs)} cells fall\nbelow the diagonal",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                color=INK2)
        ax.text(0.04, 0.96, "inflated by the\nsmall pool", transform=ax.transAxes,
                ha="left", va="top", fontsize=8, color=MUTED, style="italic")
        ax.set_xlabel("AUC measured at a 78% batch/pool ratio\n(the published configuration)")
        ax.set_ylabel("AUC at an 8.9% ratio\n(same per-token scores)")
        ax.set_title("A.  Every cell of the headline grid, twice", loc="left")
        ax.set_xlim(0.35, 1.02)
        ax.set_ylim(0.35, 1.02)
        ax.set_aspect("equal")

    ax = axes[1]
    if ps:
        rows = ps["rows"]
        b = np.array([r["batch"] for r in rows], float)
        auc = np.array([r["auc"] for r in rows])
        sd = np.array([r["sd"] for r in rows])
        pred = np.array([r["predicted"] for r in rows])
        inside = np.array([r["in_ceiling"] for r in rows])
        ceil_b = ps["ratio_ceiling"] * ps["null_split"]
        ax.axvspan(ceil_b, b.max() * 1.35, color=GRID, alpha=0.7, zorder=0, lw=0)
        ax.text(ceil_b * 1.10, 0.73, "over the 10% ratio\nceiling: not evidence",
                fontsize=7.5, color=MUTED, va="center", style="italic")
        order = np.argsort(b)
        ax.plot(b[order], pred[order], color=ORANGE, lw=1.6, ls="--", zorder=2,
                label=r"predicted from $d'\sqrt{b}$")
        ax.errorbar(b[inside], auc[inside], yerr=sd[inside], fmt="o", ms=6,
                    color=BLUE, ecolor=BLUE, elinewidth=1.2, capsize=2.5,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=4,
                    label="measured (valid ratio)")
        ax.errorbar(b[~inside], auc[~inside], yerr=sd[~inside], fmt="o", ms=6,
                    mfc="white", color=BLUE, ecolor=BLUE, elinewidth=1.2,
                    capsize=2.5, zorder=4, label="measured (over ceiling)")
        ax.axhline(0.90, color=MUTED, lw=0.9, ls=":", zorder=1)
        ax.text(b.min() * 0.95, 0.912, "AUC 0.90", fontsize=7.5, color=MUTED)
        ax.set_xscale("log")
        ax.set_xlabel("batch size $b$ (tokens), pool fixed at "
                      f"{ps['eval_tokens']:,}")
        ax.set_ylabel("AUC @ FPR $\\leq$ 0.5%")
        ax.set_title("B.  The effect size predicts the batch", loc="left")
        ax.set_ylim(0.42, 1.02)
        ax.legend(loc="lower right", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(OUT / "fig_ratio.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_ratio.pdf")


# ---------------------------------------------------------------- figure 2
def fig_infodirected():
    d = load("info_directed.json")
    if d is None or d.get("n_eval", 0) < 32:
        print("info_directed.json not at full scale yet; skipping")
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    budgets = np.array(d["budgets"]) * 100
    head = d["headline_attack"]
    auc = d["auc"][head]

    # Panel A: allocation held FIXED at the derived `info` ranking, so the only
    # thing varying is the aggregation rule -- which is the claim being tested.
    ax = axes[0]
    sdv = d["auc_sd"][head]
    names = [("mean", "plain batch mean", BLUE),
             ("matched", "matched filter (derived weights)", ORANGE),
             ("oracle_mf", "matched filter (oracle weights)", AQUA)]
    for agg, lab, col in names:
        y = np.array(auc[f"{agg}/info"])
        e = np.array(sdv[f"{agg}/info"])
        ax.plot(budgets, y, "o-", color=col, lw=1.7, ms=5,
                markeredgecolor="white", markeredgewidth=0.7, label=lab, zorder=3)
        ax.fill_between(budgets, y - e, y + e, color=col, alpha=0.13, lw=0, zorder=1)
    ax.set_xlabel("recompute budget (% of tokens audited)")
    ax.set_ylabel("AUC @ FPR $\\leq$ 0.5%")
    ax.set_title("A.  The derived aggregator loses to a plain mean", loc="left")
    ax.legend(loc="lower right", fontsize=7.5)

    # Panel B: the diagnosis -- which fitted piece is wrong.
    ax = axes[1]
    dg = d["diagnostics"][head]
    keys = [("spearman_v_hat_vs_oracle_v", "variance\n$\\hat{v}$ vs $v$"),
            ("spearman_delta_hat_vs_oracle_delta", "signal\n$\\hat{\\Delta}$ vs $\\Delta$"),
            ("spearman_info_vs_oracle", "information\n$\\hat{I}$ vs $I$")]
    vals = [dg[k] for k, _ in keys]
    cols = [BLUE if v > 0 else ORANGE for v in vals]
    ypos = np.arange(len(vals))[::-1]
    ax.barh(ypos, vals, height=0.5, color=cols, edgecolor="white", linewidth=0.8)
    for y, v in zip(ypos, vals):
        ax.text(v + (0.04 if v > 0 else -0.04), y, f"{v:+.2f}", va="center",
                ha="left" if v > 0 else "right", fontsize=8, color=INK)
    ax.set_yticks(ypos)
    ax.set_yticklabels([lab for _, lab in keys], fontsize=8)
    ax.axvline(0, color=INK2, lw=0.9)
    ax.set_xlim(-1.15, 1.15)
    ax.set_xlabel("Spearman correlation with the labelled-pair oracle")
    ax.set_title("B.  Why: the noise is learnable, the signal is not", loc="left")
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    fig.savefig(OUT / "fig_infodirected.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_infodirected.pdf")


# ---------------------------------------------------------------- figure 3
def fig_cost():
    """Nominal vs realized audit cost, and the wall clock, for the headline attack."""
    d = load("prefix_cost_quant2bit.json", BASE)
    if d is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    topk, pref = d["curves"]["topk"], d["curves"]["prefix"]

    def series(rows, k):
        return np.asarray(rows[k], float)

    ax = axes[0]
    ax.plot(series(topk, "budget") * 100, series(topk, "prefill_ratio") * 100,
            "o-", color=ORANGE, lw=1.7, ms=5, markeredgecolor="white",
            markeredgewidth=0.7, label="global top-$k$ tokens")
    ax.plot(series(pref, "budget") * 100, series(pref, "prefill_ratio") * 100,
            "o-", color=BLUE, lw=1.7, ms=5, markeredgecolor="white",
            markeredgewidth=0.7, label="prefix schedule", zorder=3)
    ax.plot([0, 100], [0, 100], color="#3d3d3a", ls=(0, (5, 4)), lw=1.1,
            label="what you asked for", zorder=5)
    ax.set_xlabel("nominal budget (% of tokens)")
    ax.set_ylabel("realized cost (% of a full audit's prefill)")
    ax.set_title("A.  A 5% token budget is not a 5% cost", loc="left")
    ax.legend(loc="lower right", fontsize=7.5)

    ax = axes[1]
    ax.plot(series(topk, "budget") * 100, series(topk, "seconds"), "o-",
            color=ORANGE, lw=1.7, ms=5, markeredgecolor="white",
            markeredgewidth=0.7, label="global top-$k$ tokens")
    ax.plot(series(pref, "budget") * 100, series(pref, "seconds"), "o-",
            color=BLUE, lw=1.7, ms=5, markeredgecolor="white",
            markeredgewidth=0.7, label="prefix schedule")
    full = series(topk, "seconds")[-1]
    ax.axhline(full, color=MUTED, ls=":", lw=0.9)
    ax.text(52, full * 1.03, "auditing every token", fontsize=7.5, color=MUTED)
    ax.set_xlabel("nominal budget (% of tokens)")
    ax.set_ylabel("measured seconds")
    ax.set_title("B.  The stopwatch agrees", loc="left")
    ax.legend(loc="center right", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(OUT / "fig_cost.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_cost.pdf")


# ---------------------------------------------------------------- figure 4
PREFIX_RUNS = [
    ("quant_2bit", "prefix_cost_quant2bit.json", BLUE),
    ("kv_fp8", "prefix_cost_kvfp8.json", ORANGE),
    ("bug_k32", "prefix_cost_bugk32.json", AQUA),
]


def fig_prefix_attacks():
    """Does the cost result depend on the attack? (A) no. (B) the honest Pareto."""
    runs = [(a, load(f, BASE), c) for a, f, c in PREFIX_RUNS]
    runs = [(a, d, c) for a, d, c in runs if d is not None]
    if len(runs) < 2:
        print("fig_prefix_attacks: fewer than 2 prefix runs; skipping")
        return
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))

    ax = axes[0]
    ax.plot([0, 100], [0, 100], color="#3d3d3a", ls=(0, (5, 4)), lw=1.1, zorder=5)
    for a, d, col in runs:
        tk, pf = d["curves"]["topk"], d["curves"]["prefix"]
        ax.plot(np.asarray(tk["budget"], float) * 100,
                np.asarray(tk["prefill_ratio"], float) * 100, "o-", color=col,
                lw=1.6, ms=4.5, markeredgecolor="white", markeredgewidth=0.7,
                label=a)
        ax.plot(np.asarray(pf["budget"], float) * 100,
                np.asarray(pf["prefill_ratio"], float) * 100, "s--", color=col,
                lw=1.2, ms=3.5, alpha=0.85, zorder=4)
    ax.set_xlabel("nominal budget (% of tokens)")
    ax.set_ylabel("realized cost (% of a full audit's prefill)")
    ax.set_title("A.  Cost geometry does not depend on the attack", loc="left")
    handles = [Line2D([], [], color=c, marker="o", ls="-", ms=4.5, lw=1.6,
                      label=f"top-$k$, {a}") for a, _, c in runs]
    handles.append(Line2D([], [], color=MUTED, marker="s", ls="--", ms=3.5, lw=1.2,
                          label="prefix (all three)"))
    ax.legend(handles=handles, loc="lower right", fontsize=7)
    ax.text(3, 62, "top-$k$ pays almost\neverything, always", fontsize=7.5,
            color=MUTED, style="italic")

    # Panel B: detection against REALIZED cost -- the only honest Pareto plane.
    ax = axes[1]
    for a, d, col in runs:
        tk, pf = d["curves"]["topk"], d["curves"]["prefix"]
        ax.plot(np.asarray(tk["prefill_ratio"], float) * 100, tk["auc"], "o-",
                color=col, lw=1.4, ms=4, alpha=0.5, markeredgecolor="white",
                markeredgewidth=0.7)
        ax.plot(np.asarray(pf["prefill_ratio"], float) * 100, pf["auc"], "s--",
                color=col, lw=1.7, ms=4.5, markeredgecolor="white",
                markeredgewidth=0.7, zorder=3)
    ax.axhline(0.5, color=MUTED, lw=0.9, ls=":", zorder=1)
    ax.text(3, 0.487, "chance", fontsize=7.5, color=MUTED)
    ax.set_xlabel("realized cost (% of a full audit's prefill)")
    ax.set_ylabel("AUC @ FPR $\\leq$ 0.5%")
    ax.set_title("B.  Detection against what it actually costs", loc="left")
    handles = [Line2D([], [], color=c, ls="-", lw=1.6, label=a)
               for a, _, c in runs]
    handles += [Line2D([], [], color=MUTED, marker="o", ls="-", ms=4, lw=1.4,
                       alpha=0.6, label="top-$k$"),
                Line2D([], [], color=MUTED, marker="s", ls="--", ms=4.5, lw=1.7,
                       label="prefix")]
    ax.legend(handles=handles, loc="upper left", fontsize=6.5, ncol=2,
              columnspacing=1.0, handlelength=1.8)
    ax.set_ylim(0.47, 0.93)

    fig.tight_layout()
    fig.savefig(OUT / "fig_prefix_attacks.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_prefix_attacks.pdf")


if __name__ == "__main__":
    fig_ratio()
    fig_infodirected()
    fig_cost()
    fig_prefix_attacks()
