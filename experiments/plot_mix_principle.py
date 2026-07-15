"""One explanatory figure for the proxy-tie-triage 'mix' algorithm:
    (A) the principle  -- cheap proxy q scores every token, keep the most-tied,
        recompute the trusted M only there;
    (B) the inspiration -- WHY it works: proxy tie-ness ranks the tokens a
        forward-pass corruption (quant) actually moves (Spearman ~= 0.73), so the
        most-tied decile carries ~orders-of-magnitude more corruption than the
        least-tied one;
    (C) the payoff      -- detection AUC vs recompute ratio: triaging by proxy
        tie-ness reaches full-recompute detection at a small fraction of M-calls.

All three panels are computed from the same real-Qwen3 cache used by
exp_tie_triage_pareto.py (M=Qwen3-1.7B true p*, proxy=Qwen3-0.6B, deterministic
int-n weight-only fake-quant).  Re-plot only, no GPU:

    python3 -m experiments.plot_mix_principle
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ivgym.metrics import roc_auc  # noqa: E402

NPZ = ROOT / "experiments" / "difr_data" / "tie_triage_pareto.npz"
FIG = ROOT / "docs" / "figures" / "fig_mix_principle.png"

# validated dataviz palette (light mode)
BLUE   = "#2a78d6"   # slot 1 -- the triaged / winning method + sequential hue
GREEN  = "#008300"   # slot 2
ORANGE = "#eb6834"   # slot 6 -- random baseline (distinct from blue)
RED    = "#e34948"   # slot 8 -- corruption
INK    = "#0b0b0b"
INK2   = "#52514e"
MUTE   = "#8a8984"
SURF   = "#fcfcfb"
GRID   = "#e6e5e1"

BATCH, N_BATCH, BOOT = 48, 300, 10
BITS_PAYOFF = 6  # representative middle bit-width for panel C


def load():
    d = dict(np.load(NPZ))
    return d


# --------------------------------------------------------------------------- C
def auc_curve(tv_hon, tv_att, tieness, rhos, mode):
    Mn = len(tv_hon)
    mean = np.zeros(len(rhos)); std = np.zeros(len(rhos))
    for i, rho in enumerate(rhos):
        k = max(1, int(round(rho * BATCH)))
        seed_aucs = []
        for s in range(BOOT):
            r = np.random.default_rng(1000 + s)
            h = np.empty(N_BATCH); a = np.empty(N_BATCH)
            for b in range(N_BATCH):
                idx = r.choice(Mn, size=BATCH, replace=False)
                if mode == "triage":
                    sel = idx[np.argsort(-tieness[idx])[:k]]
                else:
                    sel = r.choice(idx, size=k, replace=False)
                h[b] = tv_hon[sel].mean(); a[b] = tv_att[sel].mean()
            seed_aucs.append(roc_auc(h, a))
        mean[i] = np.mean(seed_aucs); std[i] = np.std(seed_aucs)
    return mean, std


def cost_to_reach(rhos, m, target=0.95):
    for i in range(len(rhos)):
        if m[i] >= target:
            if i == 0:
                return rhos[0]
            x0, x1 = rhos[i - 1], rhos[i]; y0, y1 = m[i - 1], m[i]
            return float(x0 + (target - y0) * (x1 - x0) / max(y1 - y0, 1e-9))
    return None


# ----------------------------------------------------------------- panel A art
def box(ax, x, y, w, h, text, fc, ec, tc=INK, fs=9.6, lw=1.4, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.02",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=4, weight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color=INK2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=13, lw=1.7, color=color, zorder=2,
                 shrinkA=1, shrinkB=1))


def draw_principle(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("A.  The principle — audit only where the cheap proxy is unsure",
                 fontsize=11.5, weight="bold", color=INK, loc="left", pad=8)

    # token strip: proxy scores every token; a few are "near-ties"
    tied = {2, 5, 6, 9}  # positions the proxy flags as near-tie
    n = 12
    x0, w, gap = 0.045, 0.058, 0.0125
    ytok = 0.80
    ax.text(x0, ytok + 0.135, "prompt  →  provider completion  (every token scored once by cheap proxy $q$)",
            fontsize=9.3, color=INK2, ha="left")
    for i in range(n):
        x = x0 + i * (w + gap)
        istie = i in tied
        box(ax, x, ytok, w, 0.085,
            "tie" if istie else "·",
            fc="#fdecef" if istie else "#f4f4f2",
            ec=RED if istie else GRID,
            tc=RED if istie else MUTE,
            fs=8.2 if istie else 11, lw=1.6 if istie else 1.0,
            bold=istie)
    ax.annotate("top-$k$ most-tied\npositions kept",
                xy=(x0 + 5.5 * (w + gap), ytok - 0.012),
                xytext=(0.62, ytok - 0.12), fontsize=8.6, color=RED, ha="left",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))

    # pipeline of steps
    yb, hb = 0.30, 0.20
    xs = [0.02, 0.275, 0.53, 0.785]
    ws = 0.195
    steps = [
        ("1.  Run cheap\nproxy $q$", "one forward\npass, no $M$", "#eef4fc", BLUE),
        ("2.  Tie-ness\nper token", "top-2 logit\nmargin of $q$", "#eef4fc", BLUE),
        ("3.  Select\ntop-$k$ tied", "rank by\ntie-ness", "#fdecef", RED),
        ("4.  Recompute\n$M$ there only", "trusted anchor,\nrare & costly", "#eaf6ea", GREEN),
    ]
    for i, (t, sub, fc, ec) in enumerate(steps):
        box(ax, xs[i], yb, ws, hb, t, fc=fc, ec=ec, tc=INK, fs=9.6, lw=1.6, bold=True)
        ax.text(xs[i] + ws / 2, yb - 0.085, sub, ha="center", va="center",
                fontsize=7.9, color=INK2, linespacing=1.2)
        if i < 3:
            arrow(ax, xs[i] + ws + 0.004, yb + hb / 2, xs[i + 1] - 0.004, yb + hb / 2)
    # cost caption
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.10,
                 boxstyle="round,pad=0.004,rounding_size=0.02",
                 fc="#f7f7f5", ec=GRID, lw=1.0, zorder=1))
    ax.text(0.5, 0.07,
            "cheap steps 1–2 touch every token;  the expensive $M$ recompute (step 4) fires on only a "
            "small fraction  →  full-recompute detection at a fraction of the $M$-calls",
            ha="center", va="center", fontsize=8.7, color=INK, style="italic")


# ----------------------------------------------------------------- panel B art
def draw_mechanism(ax, d):
    tie = d["tieness"]
    tvf = d[f"tvf_{BITS_PAYOFF}"]
    order = np.argsort(-tie)  # most-tied first
    dec = np.array_split(order, 10)
    means = np.array([tvf[idx].mean() for idx in dec])  # decile 1 = most tied
    xs = np.arange(10)
    # sequential blue ramp, darkest = most tied (decile 1)
    ramp = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf",
            "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#9ec5f4"]
    ax.bar(xs, means, width=0.74, color=ramp, zorder=3, edgecolor=SURF, linewidth=0.8)
    hon = d["tv_hon"].mean()
    ax.axhline(hon, ls="--", lw=1.2, color=INK2, zorder=4)
    ax.text(9.4, hon + 0.006, f"honest baseline (TV≈{hon:.3f})",
            ha="right", va="bottom", fontsize=8.2, color=INK2)

    # spearman (numpy rank corr)
    ra = np.argsort(np.argsort(tie)); rb = np.argsort(np.argsort(tvf))
    rho = np.corrcoef(ra, rb)[0, 1]
    ratio = means[0] / max(means[-1], 1e-9)
    ax.set_title("B.  The inspiration — proxy ties rank the tokens quant corrupts",
                 fontsize=11.5, weight="bold", color=INK, loc="left", pad=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(["most\ntied", "2", "3", "4", "5", "6", "7", "8", "9", "least\ntied"],
                       fontsize=8.2, color=INK2)
    ax.set_xlabel("provider tokens, bucketed by cheap-proxy tie-ness (deciles)",
                  fontsize=9.3, color=INK2)
    ax.set_ylabel(f"quant corruption  TV($M_{{quant}}$, $M$)   [{BITS_PAYOFF}-bit]",
                  fontsize=9.3, color=INK2)
    ax.text(0.5, means[0] + 0.006, f"{ratio:.0f}×\nmore corrupted", ha="center",
            va="bottom", fontsize=8.6, color="#0d366b", weight="bold")
    ax.annotate(f"Spearman(tie-ness, corruption) ≈ {rho:.2f}",
                xy=(0.02, 0.965), xycoords="axes fraction", fontsize=9.2,
                color=INK, ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f5", ec=GRID, lw=1.0))
    ax.set_ylim(0, means[0] * 1.24)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=3)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------- panel C art
def draw_payoff(ax, d):
    tv_hon = d["tv_hon"]; tie = d["tieness"]; tvf = d[f"tvf_{BITS_PAYOFF}"]
    rhos = np.unique(np.round(np.geomspace(1.0 / BATCH, 1.0, 22) * BATCH) / BATCH)
    m_tri, s_tri = auc_curve(tv_hon, tvf, tie, rhos, "triage")
    m_rnd, s_rnd = auc_curve(tv_hon, tvf, tie, rhos, "random")
    TARGET = 0.95
    c_tri, c_rnd = cost_to_reach(rhos, m_tri), cost_to_reach(rhos, m_rnd)
    factor = c_rnd / c_tri if (c_tri and c_rnd) else float("nan")

    ax.axhline(1.0, ls=":", color=MUTE, lw=1.2, zorder=1)
    ax.text(1.0, 1.003, "full recompute (AUC 1.0, 100% of $M$-calls)",
            ha="right", va="bottom", fontsize=7.8, color=MUTE)
    ax.axhline(TARGET, ls="--", color=MUTE, lw=0.8, zorder=1)
    ax.fill_between(rhos, m_tri - s_tri, m_tri + s_tri, color=BLUE, alpha=0.15, zorder=2)
    ax.fill_between(rhos, m_rnd - s_rnd, m_rnd + s_rnd, color=ORANGE, alpha=0.15, zorder=2)
    ax.plot(rhos, m_rnd, "--s", color=ORANGE, ms=3.4, lw=1.9, zorder=3,
            label="random subsample")
    ax.plot(rhos, m_tri, "-o", color=BLUE, ms=4.2, lw=2.3, zorder=4,
            label="triaged by proxy ties")

    if c_tri and c_rnd:
        ax.axvspan(c_tri, c_rnd, color="#d9d9d5", alpha=0.45, zorder=0)
        ax.annotate(f"{factor:.1f}× fewer\nrecomputes\nto AUC {TARGET}",
                    xy=(np.sqrt(c_tri * c_rnd), TARGET), xytext=(0.13, 0.66),
                    fontsize=9.0, color=INK, ha="left", weight="bold",
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0))
    ax.set_xscale("log")
    ax.set_xlim(1.0 / BATCH, 1.0); ax.set_ylim(0.55, 1.02)
    ax.set_title(f"C.  The payoff — same detection, far fewer $M$-calls  ({BITS_PAYOFF}-bit quant)",
                 fontsize=11.5, weight="bold", color=INK, loc="left", pad=8)
    ax.set_xlabel("recompute ratio  (fraction of tokens re-run on trusted $M$)",
                  fontsize=9.3, color=INK2)
    ax.set_ylabel("detection AUC  (honest vs quant)", fontsize=9.3, color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=3, which="both")
    ax.grid(color=GRID, lw=0.8, which="major", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95, edgecolor=GRID)


def main():
    d = load()
    fig = plt.figure(figsize=(15.2, 7.4), facecolor=SURF)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.02, 1.0], hspace=0.62, wspace=0.20,
                          left=0.055, right=0.985, top=0.775, bottom=0.145)
    axA = fig.add_subplot(gs[0, :]); axA.set_facecolor(SURF)
    axB = fig.add_subplot(gs[1, 0]); axB.set_facecolor(SURF)
    axC = fig.add_subplot(gs[1, 1]); axC.set_facecolor(SURF)

    draw_principle(axA)
    draw_mechanism(axB, d)
    draw_payoff(axC, d)

    fig.suptitle("Proxy-tie-triaged selective recompute — the 'mix' verification algorithm",
                 fontsize=15, weight="bold", color=INK, x=0.055, ha="left", y=0.965)
    fig.text(0.055, 0.915,
             "A forward-pass corruption (quantization / fp8) flips the sampled token only at near-tie positions.\n"
             "The client's own small proxy $q$ is an uncorrupted estimate of where the trusted model $M$ has those\n"
             "ties — so proxy tie-ness tells the verifier which few tokens are worth auditing on $M$.",
             fontsize=10, color=INK2, ha="left", va="top", linespacing=1.4)
    fig.text(0.5, 0.018,
             "real Qwen3:  $M$=Qwen3-1.7B (true $p^*$),  proxy $q$=Qwen3-0.6B,  deterministic int-$n$ "
             "weight-only quant  ·  exp_tie_triage_pareto cache  ·  AUC band = ±1 std / 10 seeds",
             fontsize=8, color=MUTE, ha="center", va="bottom", style="italic")

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150, facecolor=SURF)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
