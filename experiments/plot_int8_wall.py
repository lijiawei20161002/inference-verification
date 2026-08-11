"""Figure for the int8 wall: it is the CHANNEL, not the statistic, and not a wall.

Renders the cached JSON `exp_int8_gbt_aggregation` writes -- pure numpy + matplotlib,
no GPU, like every other figure here:

  docs/figures/fig_int8_wall.png
      (A) pAUC vs audit batch size for the two verifier replay channels. The
          deployable cross-stack channel convicts int8 too, it just needs ~16x the
          tokens; matching the verifier's dtype buys the same decision far cheaper.
      (B) Every aggregator at the sweep's batch 245, ranked. Nine ways of reading the
          cross-stack per-token features -- including gradient-boosted trees over both
          replays -- all land within 0.504-0.537, while changing the CHANNEL jumps to
          0.875. The x-axis is the same pAUC as (A), so the panels are comparable.

    python -m experiments.plot_int8_wall
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
RES_DIR = ROOT / "docs" / "results"
SRC = RES_DIR / "int8_gbt_aggregation_int8.json"

# Slots 1-2 of the validated reference palette (light mode), in the documented fixed
# order, used for the one thing this figure is about: WHICH CHANNEL a number came from.
# The same two hues mean the same two channels in both panels, and every mark is also
# direct-labeled, so identity is never carried by color alone.
XSTACK, MATCHED = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8880", "#e3e2dd"

CONVICT = 0.99          # "pAUC this high decides the audit" -- the reference line in (A)
CHANCE = 0.5

# Which rows of the JSON to show in (B), in the order they should be read, with the
# channel each one uses. Names are shortened for the tick labels.
PANEL_B = [
    ("XGBoost (margin only)", "XGBoost [difr]", "x"),
    ("XGBoost (both replays + position + role)", "XGBoost [xstack]", "x"),
    ("mean mismatch (match rate)", "mean mismatch (match rate)", "x"),
    ("both cross-stack replays mismatch", "both cross-stack replays mismatch", "x"),
    ("min margin over 2 replays", "min margin over 2 replays", "x"),
    ("mean sqrt(margin)", "mean sqrt(margin)", "x"),
    ("margin $\\times$ 1[2nd replay mismatch]", "margin x 1[2nd replay mismatch]", "x"),
    ("mean margin$^2$", "mean margin^2", "x"),
    ("mean margin  ($\\bf{token\\_difr}$, current)", "mean margin (token_difr)", "x"),
    ("XGBoost (+ same-stack replay)", "XGBoost [matched]", "m"),
    ("same-stack margin", "* same-stack margin", "m"),
    ("same-stack margin$^2$", "* same-stack margin^2", "m"),
]


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def plt_patch(color, label):
    from matplotlib.patches import Patch
    return Patch(facecolor=color, edgecolor="none", label=label)


def _crossing(batches, paucs, level):
    """Batch size at which a curve first reaches `level`, log-interpolated. Returns
    None if it never gets there, rather than extrapolating past the measured range."""
    b, p = np.asarray(batches, float), np.asarray(paucs, float)
    hit = np.nonzero(p >= level)[0]
    if not len(hit) or hit[0] == 0:
        return float(b[hit[0]]) if len(hit) else None
    i = hit[0]
    f = (level - p[i - 1]) / (p[i] - p[i - 1])
    return float(np.exp(np.log(b[i - 1]) + f * (np.log(b[i]) - np.log(b[i - 1]))))


# =========================================================================== (A)
def panel_a(ax, d):
    sc = d["scaling"]
    # Both reference lines label at the LEFT edge: the right half of the 0.99 line is
    # where the finding (the horizontal gap between the curves) gets annotated.
    ax.axhline(CONVICT, color=MUTED, lw=1.4, ls=(0, (4, 3)), zorder=1)
    ax.annotate(f"pAUC {CONVICT} -- the audit decides", (0.012, CONVICT),
                xycoords=("axes fraction", "data"), xytext=(0, 5), ha="left",
                va="bottom", textcoords="offset points", color=INK2, fontsize=8.5)
    ax.axhline(CHANCE, color=GRID, lw=1.4, zorder=1)
    ax.annotate("chance", (0.012, CHANCE), xycoords=("axes fraction", "data"),
                xytext=(0, -4), ha="left", va="top", textcoords="offset points",
                color=MUTED, fontsize=8.5)

    cross = {}
    for key, color, label in (("cross-stack margin", XSTACK,
                               "cross-stack replay\n(bf16 verifier, fp16 provider)"
                               "\n$\\bf{deployable\\ today}$"),
                              ("same-stack margin", MATCHED,
                               "same-stack replay\n(verifier matches the provider's dtype)"
                               "\n$\\it{requires\\ numeric\\ attestation}$")):
        b = [r["batch"] for r in sc[key]]
        p = [r["pauc"] for r in sc[key]]
        ax.plot(b, p, color=color, lw=2.0, marker="o", ms=6.0, zorder=3,
                markeredgecolor="white", markeredgewidth=1.2, label=label)
        cross[key] = _crossing(b, p, CONVICT)

    # The finding is the horizontal distance between the two curves, so annotate THAT
    # rather than either curve's height: same decision, 16x the tokens.
    xa, xb = cross["same-stack margin"], cross["cross-stack margin"]
    if xa and xb:
        ax.annotate("", xy=(xb, CONVICT), xytext=(xa, CONVICT),
                    arrowprops=dict(arrowstyle="<|-|>", color=INK2, lw=1.3,
                                    shrinkA=2, shrinkB=2, mutation_scale=11))
        ax.annotate(f"same decision, {xb/xa:.0f}$\\times$ the tokens",
                    (np.exp((np.log(xa) + np.log(xb)) / 2), CONVICT), xytext=(0, -14),
                    ha="center", va="top", textcoords="offset points", color=INK,
                    fontsize=9, weight="bold")

    # Where the sweep actually measured, and what it concluded there. Text goes LEFT of
    # the rule, into the only empty corner of this panel.
    ax.axvline(d["batch"], color=MUTED, lw=1.0, ls=(0, (2, 3)), zorder=1)
    at245 = next(r["pauc"] for r in sc["cross-stack margin"] if r["batch"] == d["batch"])
    ax.annotate(f"batch {d['batch']}: where\nthe sweep read\n'int8 is at chance'\n"
                f"(pAUC {at245:.3f})",
                (d["batch"], 0.615), xytext=(-7, 0), ha="right", va="center",
                textcoords="offset points", color=INK2, fontsize=8.5, linespacing=1.4)

    b = [r["batch"] for r in sc["cross-stack margin"]]
    p = [r["pauc"] for r in sc["cross-stack margin"]]
    i = int(np.argmin(np.abs(np.asarray(p) - 0.995)))
    ax.annotate(f"the deployable channel\nconvicts too: {p[i]:.3f}\n@ batch {b[i]:,}"
                f" ($\\approx${b[i] * 10 // 1000}k\ntokens of pool)",
                (b[i], p[i]), xytext=(b[-1], 0.80), textcoords="data",
                ha="right", va="center", color=XSTACK, fontsize=9, weight="bold",
                linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=XSTACK, lw=1.0,
                                shrinkA=4, shrinkB=5))

    ax.set_xscale("log")
    ax.set_xlabel("audit batch size (tokens aggregated per decision)", color=INK2,
                  fontsize=10)
    ax.set_ylabel("pAUC @ FPR $\\leq$ 0.5%", color=INK2, fontsize=10)
    ax.set_ylim(0.45, 1.035)
    ax.set_title("A.  int8 is not undetectable -- it is expensive", color=INK,
                 fontsize=11, weight="bold", loc="left", pad=8)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=8.5,
                    labelspacing=1.0, handlelength=1.6, borderaxespad=0.9)
    for t in leg.get_texts():
        t.set_color(INK2)


# =========================================================================== (B)
def panel_b(ax, d):
    by = {r["name"]: r for r in d["rows"]}
    missing = [k for _, k, _ in PANEL_B if k not in by]
    if missing:
        sys.exit(f"{SRC} is missing rows {missing} -- rerun the experiment")
    labels = [lab for lab, _, _ in PANEL_B]
    vals = np.array([by[k]["pauc"] for _, k, _ in PANEL_B])
    cols = [XSTACK if ch == "x" else MATCHED for _, _, ch in PANEL_B]
    # A one-row gap between the channel groups, so the grouping is carried by position
    # as well as by hue (and the legend names both, so never by hue alone).
    n_x = sum(1 for _, _, ch in PANEL_B if ch == "x")
    y = np.array([i if i < n_x else i + 0.9 for i in range(len(vals))], float)

    ax.axvline(CHANCE, color=MUTED, lw=1.4, zorder=1)
    # Bars start at chance, not at zero: 0.5 is this metric's origin, and a bar drawn
    # from 0 would spend 94% of its length on the part that carries no information.
    ax.barh(y, vals - CHANCE, left=CHANCE, height=0.62, color=cols, zorder=3)
    for yi, v, c in zip(y, vals, cols):
        ax.annotate(f"{v:.3f}", (v, yi), xytext=(5, 0), va="center",
                    textcoords="offset points", color=c, fontsize=9, weight="bold")

    # The finding, placed in the empty right half that the nine short bars leave.
    span = vals[:n_x].max() - vals[:n_x].min()
    ax.annotate(f"{n_x} ways to read the same channel --\nmeans, convex transforms, "
                f"replay agreement,\ngradient-boosted trees -- span {span:.3f} of pAUC.\n"
                f"Changing the CHANNEL adds "
                f"{vals[cols.index(MATCHED) + 1:].max() - vals[:n_x].max():+.3f}.",
                (0.60, (n_x - 1) / 2), va="center", color=INK, fontsize=9.5,
                linespacing=1.5)

    handles = [plt_patch(XSTACK, "cross-stack replay only ($\\bf{deployable}$)"),
               plt_patch(MATCHED, "adds the same-stack replay "
                                  "($\\it{needs\\ numeric\\ attestation}$)")]
    leg = ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5,
                    labelspacing=0.9, handlelength=1.1, borderaxespad=0.9)
    for t in leg.get_texts():
        t.set_color(INK2)

    ax.set_yticks(y, labels, color=INK2, fontsize=9)
    ax.set_xlim(CHANCE, 0.985)
    ax.set_ylim(-0.8, y[-1] + 0.6)
    ax.set_xlabel(f"pAUC @ FPR $\\leq$ 0.5%, at the sweep's batch {d['batch']}",
                  color=INK2, fontsize=10)
    ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)
    ax.set_title("B.  No aggregator rescues it -- the statistic is not the constraint",
                 color=INK, fontsize=11, weight="bold", loc="left", pad=8)


# ========================================================================== main
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not SRC.exists():
        sys.exit(f"missing {SRC} -- run experiments.exp_int8_gbt_aggregation first")
    d = json.loads(SRC.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.4),
                             gridspec_kw=dict(width_ratios=[1.0, 1.22], wspace=0.42))
    for ax in axes:
        _clean(ax)
    panel_a(axes[0], d)
    panel_b(axes[1], d)

    fig.suptitle("The int8 detection wall is a channel and a budget, not a statistic",
                 color=INK, fontsize=13, weight="bold", x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.925,
             f"int8 RTN-g128 weights vs an honest provider, {d['pair']} coupled "
             f"speculation, {d['n_tokens']:,} tokens per arm.  Trees are grouped 4-fold "
             f"CV by prompt.  Curves in (A) are bootstrapped, so batches past "
             f"{d['n_tokens']:,} assume the observed margin tail is representative.",
             color=INK2, fontsize=8.5, ha="left", va="top")
    fig.subplots_adjust(left=0.045, right=0.995, top=0.855, bottom=0.085)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_int8_wall.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
