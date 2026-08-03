"""Figures for the speculative-provider results (docs/figures/fig_spec_*.png).

Plots the three measurements written up in docs/SPECULATIVE_PROVIDERS.md. Reads
only committed artifacts -- no GPU, no model, no regeneration:

  docs/results/spec_decode_difr.json     -> fig_spec_provider_fpr.png
  docs/results/spec_aware_verifier.json  -> fig_spec_aware_verifier.png
  docs/results/spec_batch_numerics.json  -> fig_spec_batch_numerics.png

Figure 1 -- the false positive. An honest speculative server is flagged by
`token_difr` at 94.5%, the mechanism is a break in seed synchronisation rather
than in the served distribution, and tolerating speculation costs ~0.47 AUC on
exactly the two output-preserving deviations.

Figure 2 -- the spec-aware ladder. Each replay verifier reproduces only the
server whose decoding algorithm it assumes (read the diagonal), the cheap
seed-free rung is blind, and the sound rung costs 11.8x a prefill.

Figure 3 -- the floor no disclosure can fix. prefill / decode / speculative-verify
are the same number in different float arithmetic; the speculative path flips the
sampled token 3.2x more often than the batch-composition noise a verifier already
tolerates -- but only 0.97x an ordinary sequential decode.

Run:  python -m experiments.plot_spec_providers
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results"
OUT = ROOT / "docs" / "figures"

# ---- validated, colorblind-safe palette (dataviz skill, light surface) -------
SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8880", "#e7e6e2"
# categorical slots in fixed order, never cycled
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"

HONEST_ARMS = {"honest", "honest_null", "honest_spec", "honest_spec_seeded"}


def style(ax, grid_axis="both"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def title(ax, text):
    ax.set_title(text, color=INK, fontsize=10.5, loc="left", fontweight="bold")


def footnote(ax, text, y=-0.16, color=INK2, size=8.2):
    """Supporting prose, below the axis -- never over the marks."""
    ax.text(0.0, y, text, transform=ax.transAxes, ha="left", va="top",
            color=color, fontsize=size, linespacing=1.45)


def ylab(name):
    """Row label: arm name, with honest arms marked so a mark is never read as a hit."""
    return f"{name}   (honest)" if name in HONEST_ARMS else name


# =============================================================================
# Figure 1 -- the false positive (exp_spec_decode_difr_gpu)
# =============================================================================
def fig_fpr(difr):
    tpr, cert, alibi = difr["tpr"], difr["certificate"], difr["alibi"]

    detectors = [("token_difr", BLUE), ("cross_entropy", ORANGE),
                 ("token_toploc", AQUA), ("activation_difr", YELLOW)]
    arms = ["honest_null", "honest_spec", "honest_spec_seeded",
            "spec_lenient", "spec_topk", "seed_43", "quant_2bit"]

    fig = plt.figure(figsize=(16.8, 5.4), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.1, 1.0], wspace=0.32)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))

    # ---- Panel A: flag rate at a 0.5%-FPR threshold -------------------------
    # A dot plot rather than bars: the values span three decades, and bar length
    # on a log axis misencodes magnitude. One sub-row per detector, so coincident
    # values (several detectors at 100%) never stack on one mark.
    style(axA, grid_axis="x")
    FLOOR = 0.012  # percent; where an exact 0.0% is drawn, as an open marker
    dodge = [0.33, 0.11, -0.11, -0.33]
    rows = np.arange(len(arms))[::-1]
    for y, arm in zip(rows, arms):
        if arm in HONEST_ARMS:
            axA.axhspan(y - 0.5, y + 0.5, color=INK2, alpha=0.07, zorder=0)
        for d in dodge:
            axA.plot([FLOOR, 130], [y + d, y + d], color=GRID, lw=0.7, zorder=1)
    for (det, col), d in zip(detectors, dodge):
        for y, arm in zip(rows, arms):
            v = tpr[arm][det][0] * 100
            zero = v <= 0
            axA.plot([FLOOR if zero else v], [y + d], marker="o", ms=7.5,
                     color=SURFACE if zero else col, markeredgecolor=col,
                     markeredgewidth=1.7, zorder=3)
    axA.axvline(0.5, color=INK2, lw=1, ls=(0, (4, 4)), alpha=0.8, zorder=2)
    axA.text(0.5, len(arms) - 0.52, "  0.5% nominal FPR", color=INK2, fontsize=8.5,
             ha="left", va="center")
    axA.set_xscale("log")
    axA.set_xlim(FLOOR * 0.8, 300)
    axA.set_ylim(-0.62, len(arms) - 0.38)
    axA.set_xticks([0.1, 1, 10, 100])
    axA.set_xticklabels(["0.1%", "1%", "10%", "100%"])
    axA.set_yticks(rows)
    axA.set_yticklabels([ylab(a) for a in arms], fontsize=9)
    axA.set_xlabel("flag rate at a threshold calibrated to 0.5% FPR on honest traffic",
                   color=INK2, fontsize=10)
    title(axA, "A  An honest speculative server is flagged at "
               f"{tpr['honest_spec']['token_difr'][0]:.1%}")
    axA.legend(handles=[Line2D([], [], marker="o", ls="none", ms=7.5, color=c,
                              markeredgecolor=c, label=d) for d, c in detectors],
               frameon=False, fontsize=8.5, labelcolor=INK, ncol=4,
               loc="upper left", bbox_to_anchor=(-0.02, -0.10),
               handletextpad=0.25, columnspacing=1.4)

    # direct labels on the two honest speculative arms -- the finding
    for arm in ("honest_spec", "honest_spec_seeded"):
        y = rows[arms.index(arm)] + dodge[0]
        v = tpr[arm]["token_difr"][0] * 100
        axA.annotate(f"{v:.1f}%", xy=(v, y), xytext=(9, 4), textcoords="offset points",
                     color=INK, fontsize=9.5, fontweight="bold")
    footnote(axA,
             "Grey rows are honest servers, so those marks are false positives.\n"
             "Open marker = exactly 0%. honest_null is the same honest config on a\n"
             "disjoint prompt range — the protocol's own calibration check, and\n"
             f"cross_entropy fails it ({tpr['honest_null']['cross_entropy'][0]:.1%} ≫ 0.5%): "
             f"its threshold does not transfer across\nprompt ranges. Read the other three "
             f"columns as false-positive rates.", y=-0.19)

    # ---- Panel B: mechanism -- seed-free vs seeded --------------------------
    style(axB)
    order = ["honest", "honest_null", "honest_spec", "honest_spec_seeded",
             "seed_43", "spec_lenient", "spec_topk", "quant_2bit"]
    # label offsets, hand-placed to keep the crowded lower cluster legible
    off = {"honest": (9, 6), "honest_null": (9, -13), "quant_2bit": (10, 3),
           "honest_spec": (-9, 8), "seed_43": (0, -16), "spec_lenient": (12, 7),
           "honest_spec_seeded": (12, -11), "spec_topk": (-10, 7)}
    for arm in order:
        c = cert[arm]
        honest = arm in HONEST_ARMS
        col = BLUE if honest else ORANGE
        axB.errorbar(c["nll"], c["agree"] * 100, xerr=c["nll_se"], fmt="none",
                     ecolor=col, elinewidth=1.2, capsize=2.5, alpha=0.85, zorder=2)
        axB.plot([c["nll"]], [c["agree"] * 100], marker="o" if honest else "s",
                 ms=9, color=col, markeredgecolor=SURFACE, markeredgewidth=1.5,
                 zorder=3, label=("honest server" if arm == "honest" else
                                  "deviation" if arm == "seed_43" else None))
        dx, dy = off[arm]
        axB.annotate(arm, xy=(c["nll"], c["agree"] * 100), textcoords="offset points",
                     xytext=(dx, dy), color=INK if honest else INK2, fontsize=8.5,
                     ha="left" if dx > 0 else "right")
    axB.set_xlabel("mean NLL of the claimed tokens under M   (seed-free)",
                   color=INK2, fontsize=10)
    axB.set_ylabel("seed replay agreement   (seeded)", color=INK2, fontsize=10)
    axB.set_xlim(0.66, 1.31)
    axB.set_ylim(56, 104)
    axB.set_yticks([60, 70, 80, 90, 100])
    axB.set_yticklabels(["60%", "70%", "80%", "90%", "100%"])
    title(axB, "B  A break in seed sync, not in the distribution")
    axB.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right")
    hs, s43 = cert["honest_spec"], cert["seed_43"]
    axB.annotate("honest speculation and the wrong-seed attack\n"
                 f"land in the same place: {hs['agree']:.1%} vs {s43['agree']:.1%}\n"
                 "replay agreement, ordinary NLL either way",
                 xy=(0.5 * (hs["nll"] + s43["nll"]), 0.5 * (hs["agree"] + s43["agree"]) * 100),
                 xytext=(0.875, 80.5), color=INK2, fontsize=8.2, va="top", ha="left",
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0,
                                 connectionstyle="arc3,rad=0.28"))
    footnote(axB,
             "The seed-free axis says the honest speculative\n"
             "server's tokens are ordinary draws from M. The\n"
             "seeded axis says the replay cannot reproduce\n"
             "them — and that is what token_difr reports.", y=-0.13)

    # ---- Panel C: what tolerating speculation costs -------------------------
    style(axC, grid_axis="x")
    attacks = sorted(alibi, key=lambda a: -alibi[a]["cost"])
    rowsC = np.arange(len(attacks))[::-1]
    for y, a in zip(rowsC, attacks):
        full, free = alibi[a]["full"], alibi[a]["seed_free"]
        if abs(full - free) < 2e-3:
            # both reach the same AUC: one mark, ringed in the other series' hue,
            # rather than two marks stacked invisibly
            axC.plot([full], [y], marker="o", ms=11, color=BLUE,
                     markeredgecolor=ORANGE, markeredgewidth=2.6, zorder=3)
            axC.annotate("no cost", xy=(full, y), xytext=(-14, 0),
                         textcoords="offset points", ha="right", va="center",
                         color=INK2, fontsize=8.5)
            continue
        axC.plot([free, full], [y, y], color=GRID, lw=3.5, solid_capstyle="round",
                 zorder=1)
        axC.plot([full], [y], marker="o", ms=10, color=BLUE, markeredgecolor=SURFACE,
                 markeredgewidth=1.5, zorder=3)
        axC.plot([free], [y], marker="o", ms=10, color=ORANGE, markeredgecolor=SURFACE,
                 markeredgewidth=1.5, zorder=3)
        axC.annotate(f"−{alibi[a]['cost']:.2f} AUC", xy=(0.5 * (free + full), y),
                     xytext=(0, 12), textcoords="offset points", ha="center",
                     color=INK, fontsize=9.5, fontweight="bold")
    axC.axvline(0.5, color=INK2, lw=1, ls=(0, (4, 4)), alpha=0.8)
    axC.text(0.5, rowsC[0] + 0.44, " chance", color=INK2, fontsize=8.5, ha="left")
    axC.set_xlim(0.42, 1.07)
    axC.set_ylim(-0.6, len(attacks) - 0.25)
    axC.set_yticks(rowsC)
    axC.set_yticklabels(attacks, fontsize=9)
    axC.set_xlabel("best detection AUC  (pAUC @ FPR ≤ 0.5%)", color=INK2, fontsize=10)
    title(axC, "C  Output-preserving cheats cost half an AUC")
    axC.legend(handles=[Line2D([], [], marker="o", ls="none", ms=9, color=BLUE,
                               markeredgecolor=BLUE, label="best of all 4 detectors"),
                        Line2D([], [], marker="o", ls="none", ms=9, color=ORANGE,
                               markeredgecolor=ORANGE, label="best without token_difr")],
               frameon=False, fontsize=8.5, labelcolor=INK, ncol=2,
               loc="upper left", bbox_to_anchor=(-0.02, -0.10),
               handletextpad=0.25, columnspacing=1.4)
    footnote(axC,
             "If a provider is allowed to speculate, token_difr must be\n"
             "dropped. The two deviations that preserve the served\n"
             "distribution are exactly the two that needed it; forward-pass\n"
             "cheats are untouched, because activation_difr reads activations\n"
             "before a token is drawn. Ringed mark = both reach the same AUC.",
             y=-0.19)

    fig.suptitle("An honest speculative-decoding provider is indistinguishable from a "
                 "cheating one to a seeded-replay verifier",
                 color=INK, fontsize=12.5, fontweight="bold", x=0.5, y=1.00)
    p = OUT / "fig_spec_provider_fpr.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {p}")


# =============================================================================
# Figure 2 -- the spec-aware ladder (exp_spec_aware_verifier_gpu)
# =============================================================================
def fig_aware(aware, difr):
    servers = ["honest", "honest_spec_seeded", "honest_spec", "seed_43", "spec_lenient"]
    # V0's disagreement is 1 - the seed-replay agreement of the same rerun, taken from
    # the certificate block of the companion run (matches this experiment's own printed
    # V0 column to 0.1%; see docs/results/logs/spec_aware_verifier_n48_t128.log).
    v0 = {s: 1.0 - difr["certificate"][s]["agree"] for s in servers}
    v2 = aware["replay_disagreement"]
    assumed = {"V0": "honest", "V2": "honest_spec_seeded"}  # the diagonal
    # row labels (panel A, horizontal) and tick labels (panel B, vertical)
    wide = {"honest": "honest, sequential\n(honest)",
            "honest_spec_seeded": "honest_spec_seeded\n(honest)",
            "honest_spec": "honest_spec, own RNG\n(honest)",
            "seed_43": "seed_43\n(cheat)", "spec_lenient": "spec_lenient\n(cheat)"}
    tall = {"honest_spec": "honest_spec\nown RNG\n(honest)",
            "honest_spec_seeded": "honest_spec_\nseeded\n(honest)",
            "seed_43": "seed_43\n(cheat)", "spec_lenient": "spec_lenient\n(cheat)"}

    fig = plt.figure(figsize=(16.8, 5.4), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.05, 1.0], wspace=0.42)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))

    # ---- Panel A: read the diagonal ----------------------------------------
    style(axA, grid_axis="x")
    rows = np.arange(len(servers))[::-1]
    h = 0.34
    for i, (name, vals, col, key) in enumerate(
            [("V0  Gumbel rerun", v0, BLUE, "V0"), ("V2  spec replay", v2, ORANGE, "V2")]):
        ys = rows + (0.5 - i) * h
        bars = axA.barh(ys, [vals[s] * 100 for s in servers], height=h * 0.90,
                        color=col, label=name, zorder=2)
        for s, b in zip(servers, bars):
            matched = s == assumed[key]
            if matched:
                b.set_edgecolor(INK)
                b.set_linewidth(1.6)
            axA.annotate(f"{vals[s]:.1%}", xy=(b.get_width(), b.get_y() + b.get_height() / 2),
                         xytext=(5, 0), textcoords="offset points", va="center",
                         color=INK if matched else INK2, fontsize=9,
                         fontweight="bold" if matched else "normal")
    axA.set_yticks(rows)
    axA.set_yticklabels([wide[s] for s in servers], fontsize=8.6)
    axA.set_xlabel("share of the server's tokens the verifier CANNOT reproduce",
                   color=INK2, fontsize=10)
    axA.set_xticks([0, 10, 20, 30, 40, 50])
    axA.set_xticklabels(["0%", "10%", "20%", "30%", "40%", "50%"])
    axA.set_xlim(0, 56)
    axA.set_ylim(-0.6, len(servers) - 0.4)
    title(axA, "A  Each verifier reproduces only the algorithm it assumes")
    # explicit handles: the diagonal's outline is an annotation, not part of a series
    axA.legend(handles=[Line2D([], [], marker="s", ls="none", ms=9, color=BLUE,
                               label="V0  Gumbel rerun"),
                        Line2D([], [], marker="s", ls="none", ms=9, color=ORANGE,
                               label="V2  spec replay")],
               frameon=False, fontsize=8.5, labelcolor=INK, ncol=2, loc="upper left",
               bbox_to_anchor=(-0.02, -0.11), handletextpad=0.4, columnspacing=1.6)
    footnote(axA,
             "Outlined bars are the diagonal — the server whose decoding algorithm the\n"
             "verifier assumes. Off the diagonal these shares are total losses at the batch\n"
             "level: V2 flags the honest sequential server at 100%, and the honest\n"
             "speculative server that draws its own randomness at 100% too.", y=-0.21)

    # ---- Panel B: the cheap seed-free rung sees nothing --------------------
    style(axB, grid_axis="y")
    arms = ["honest_spec", "honest_spec_seeded", "seed_43", "spec_lenient"]
    auc = aware["auc"]
    w = 0.34
    for i, (score, col, name) in enumerate([("token_difr", BLUE, "V0  token_difr"),
                                            ("spec_accept", VIOLET, "V1  spec_accept")]):
        xs = np.arange(len(arms)) + (i - 0.5) * w
        hs = [auc[a][score][0] for a in arms]
        es = [auc[a][score][1] for a in arms]
        axB.bar(xs, hs, width=w * 0.90, color=col, label=name, zorder=2)
        axB.errorbar(xs, hs, yerr=es, fmt="none", ecolor=INK2, elinewidth=1.1,
                     capsize=2.5, zorder=3)
        for x, v in zip(xs, hs):
            axB.annotate(f"{v:.3f}", xy=(x, v), xytext=(0, 7), textcoords="offset points",
                         ha="center", color=INK2, fontsize=8.2)
    axB.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 4)), alpha=0.8, zorder=1)
    axB.text(-0.55, 0.474, "chance", color=INK2, fontsize=8.5, ha="left")
    axB.set_xticks(np.arange(len(arms)))
    axB.set_xticklabels([tall[a] for a in arms], fontsize=8.2)
    axB.set_xlim(-0.6, len(arms) - 0.4)
    axB.set_ylim(0.42, 1.09)
    axB.set_ylabel("detection AUC vs each verifier's own honest null",
                   color=INK2, fontsize=10)
    title(axB, "B  V1 is seed-free — and blind")
    axB.legend(frameon=False, fontsize=8.5, labelcolor=INK, ncol=2, loc="upper left",
               bbox_to_anchor=(-0.02, -0.14), handletextpad=0.4, columnspacing=1.6)
    pr = aware["accept_pass_rate"]
    footnote(axB,
             f"V1 reruns the attested draft and asks a seed-free question,\n"
             f"so it cannot false-positive on honest speculation — and it\n"
             f"sees nothing. As a binary test (V1b) it passes "
             f"{pr['seed_43']:.1%} of the\nwrong-seed attack's tokens against "
             f"{pr['honest']:.1%} of honest ones: a\ndistributional check, which is "
             f"exactly what an output-\npreserving deviation preserves.", y=-0.25)

    # ---- Panel C: the price of the sound rung ------------------------------
    style(axC, grid_axis="x")
    cost = aware["cost_seconds"]
    rungs = [("V0  token_difr\none prefill over\n[prompt + claimed]", cost["prefill_difr"], BLUE),
             ("V2  spec_replay\nreplay the\nspeculative loop", cost["spec_replay"], ORANGE)]
    rowsC = np.arange(len(rungs))[::-1]
    for y, (name, sec, col) in zip(rowsC, rungs):
        axC.barh([y], [sec * 1000], height=0.40, color=col, zorder=2)
        axC.annotate(f"{sec * 1000:,.1f} ms", xy=(sec * 1000, y), xytext=(7, 0),
                     textcoords="offset points", va="center", color=INK,
                     fontsize=10.5, fontweight="bold")
    axC.set_yticks(rowsC)
    axC.set_yticklabels([r[0] for r in rungs], fontsize=8.8)
    axC.set_xlim(0, cost["spec_replay"] * 1000 * 1.30)
    axC.set_ylim(-0.55, len(rungs) - 0.45)
    axC.set_xlabel("audit cost per audited sequence, GPU-synchronised (ms)",
                   color=INK2, fontsize=10)
    title(axC, f"C  The sound rung costs {cost['ratio']:.1f}× a prefill")
    axC.annotate(f"{cost['ratio']:.1f}×", xy=(cost["spec_replay"] * 1000 * 0.48, 0.52),
                 color=INK, fontsize=17, fontweight="bold", ha="center", va="center")
    footnote(axC,
             "V2 is not a prefill: it replays ~32 sequential rounds, each a\n"
             "4-step draft rollout plus a batched target pass, because the\n"
             "round structure is only knowable by walking it. The sound\n"
             "spec-aware audit costs more than the generation it checks —\n"
             "and the economic case for recomputation is that auditing is\n"
             "cheaper than generating.", y=-0.15)

    fig.suptitle("Teaching the verifier to speculate: sound only against the algorithm it "
                 f"assumes, noisier than what it replaces, and {cost['ratio']:.1f}× the cost",
                 color=INK, fontsize=12.5, fontweight="bold", x=0.5, y=1.00)
    p = OUT / "fig_spec_aware_verifier.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {p}")


# =============================================================================
# Figure 3 -- the numerical floor (exp_spec_batch_numerics_gpu)
# =============================================================================
def fig_numerics(num):
    st = num["stats"]
    paths = ["prefill_repeat", "batch2", "batch8", "decode", "verify_g4", "verify_g8"]
    role = {"prefill_repeat": ("determinism control", INK2),
            "batch2": ("benign yardstick", AQUA), "batch8": ("benign yardstick", AQUA),
            "decode": ("ordinary sequential server", YELLOW),
            "verify_g4": ("speculative server", BLUE),
            "verify_g8": ("speculative server", BLUE)}
    note = {"prefill_repeat": "the same call twice — bit-identical",
            "batch2": "batch of 2 identical rows", "batch8": "batch of 8 identical rows",
            "decode": "one token at a time vs a KV cache",
            "verify_g4": "γ = 4 drafted tokens at once",
            "verify_g8": "γ = 8 drafted tokens at once"}

    lo = min(st["batch2"]["token_flip_rate"], st["batch8"]["token_flip_rate"])
    yard = max(st["batch2"]["token_flip_rate"], st["batch8"]["token_flip_rate"])
    spec = max(st["verify_g4"]["token_flip_rate"], st["verify_g8"]["token_flip_rate"])
    dec = st["decode"]["token_flip_rate"]

    fig = plt.figure(figsize=(14.4, 5.2), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.46)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    rows = np.arange(len(paths))[::-1]

    # ---- Panel A: sampled-token flip rate ----------------------------------
    style(axA, grid_axis="x")
    axA.axvspan(lo * 100, yard * 100, color=AQUA, alpha=0.16, zorder=0)
    axA.annotate("benign band", xy=(yard * 100, len(paths) - 0.52), xytext=(4, 0),
                 textcoords="offset points", color=INK2, fontsize=8.2,
                 va="center", fontweight="bold")
    for y, p_ in zip(rows, paths):
        v = st[p_]["token_flip_rate"] * 100
        if v == 0:  # a bar of length zero, marked so it does not read as missing data
            axA.plot([0], [y], marker="|", ms=13, mew=2.2, color=role[p_][1], zorder=2)
        axA.barh([y], [v], height=0.42, color=role[p_][1], zorder=2)
        axA.annotate(f"{v:.2f}%", xy=(v, y), xytext=(6, 0), textcoords="offset points",
                     va="center", color=INK, fontsize=9.5,
                     fontweight="bold" if p_.startswith("verify") else "normal")
        axA.annotate(note[p_], xy=(0, y), xytext=(4, -15), textcoords="offset points",
                     va="center", color=MUTED, fontsize=7.8)
    axA.set_yticks(rows)
    axA.set_yticklabels(paths, fontsize=9.5)
    axA.set_xlim(0, spec * 100 * 1.34)
    axA.set_ylim(-0.68, len(paths) - 0.32)
    axA.set_xlabel("sampled-token flip rate vs the verifier's prefill  "
                   f"({num['n_positions']} positions, no simulated noise)",
                   color=INK2, fontsize=10)
    title(axA, "A  The speculative path flips tokens like a sequential one")
    axA.legend(handles=[Line2D([], [], marker="s", ls="none", ms=9, color=c, label=r)
                        for r, c in dict(role.values()).items()],
               frameon=False, fontsize=8.5, labelcolor=INK, ncol=4, loc="upper left",
               bbox_to_anchor=(-0.02, -0.11), handletextpad=0.3, columnspacing=1.2)
    footnote(axA,
             f"Against the benign yardstick the speculative verify pass flips the sampled "
             f"token\n{spec / yard:.1f}× as often as the batch-composition nondeterminism a "
             f"verifier already tolerates.\nAgainst the right control it is only "
             f"{spec / dec:.2f}× an ordinary sequential decode: speculation\nadds no "
             f"numerical penalty of its own. The floor is the prefill-versus-incremental\n"
             f"mismatch every serving stack has, and every prefill-based verifier inherits.",
             y=-0.20)

    # ---- Panel B: the underlying logit divergence --------------------------
    style(axB, grid_axis="x")
    for y, p_ in zip(rows, paths):
        v = st[p_]["mean_abs_logit_diff"]
        axB.barh([y], [v], height=0.42, color=role[p_][1], zorder=2)
        axB.annotate("0 (bit-identical)" if v == 0 else f"{v:.3f}", xy=(v, y),
                     xytext=(6, 0), textcoords="offset points", va="center",
                     color=INK, fontsize=9)
        axB.annotate(f"max {st[p_]['max_abs_logit_diff']:.2f}   ·   "
                     f"argmax flips {st[p_]['argmax_flip_rate']:.2%}",
                     xy=(0, y), xytext=(4, -15), textcoords="offset points",
                     va="center", color=MUTED, fontsize=7.8)
    axB.set_yticks(rows)
    axB.set_yticklabels(paths, fontsize=9.5)
    axB.set_xlim(0, max(st[p_]["mean_abs_logit_diff"] for p_ in paths) * 1.60)
    axB.set_ylim(-0.68, len(paths) - 0.32)
    axB.set_xlabel("mean │Δ logit│ vs the verifier's prefill", color=INK2, fontsize=10)
    title(axB, "B  Same arithmetic, different reduction order")
    footnote(axB,
             "Colours as in A. The control is load-bearing: the same call twice is\n"
             "bit-identical, so every other row is attributable to the shape of the forward\n"
             "pass rather than to run-to-run jitter (asserted in tests/test_claims.py).\n"
             "All three paths are the same number mathematically — they are different\n"
             "reduction orders, dispatched to different kernels.", y=-0.20)

    fig.suptitle("The floor no disclosure can fix: reading M's logits by prefill, decode or "
                 "speculative verify gives the same number in different floats  "
                 f"({num['model'].split('/')[-1]}, bf16, one H100)",
                 color=INK, fontsize=12, fontweight="bold", x=0.5, y=1.00)
    p = OUT / "fig_spec_batch_numerics.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {p}")


def main():
    difr = json.loads((RES / "spec_decode_difr.json").read_text())
    aware = json.loads((RES / "spec_aware_verifier.json").read_text())
    num = json.loads((RES / "spec_batch_numerics.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)

    fig_fpr(difr)
    fig_aware(aware, difr)
    fig_numerics(num)

    # ---- report the numbers behind the figures ------------------------------
    print(f"\nsetup: target={difr['model']} draft={difr['draft']} gamma={difr['gamma']} "
          f"prompts={difr['n_prompts']} tokens={difr['n_tokens']} seeds={difr['n_seeds']} "
          f"batch={difr['batch']} pool_ratio={difr['pool_ratio']:.1%}")
    print("\nfig 1  flag rate @0.5% FPR (token_difr):")
    for a in ("honest_null", "honest_spec", "honest_spec_seeded", "seed_43", "spec_lenient"):
        print(f"    {a:>19} {difr['tpr'][a]['token_difr'][0]:>7.1%}"
              f"   {'(honest -> false positive)' if a in HONEST_ARMS else '(cheat)'}")
    print("  alibi cost (best AUC lost when token_difr must be dropped):")
    for a, v in sorted(difr["alibi"].items(), key=lambda kv: -kv[1]["cost"]):
        print(f"    {a:>19} {v['full']:.3f} -> {v['seed_free']:.3f}   cost {v['cost']:.3f}")
    print("\nfig 2  replay disagreement (V0 Gumbel rerun / V2 spec replay):")
    for s in ("honest", "honest_spec_seeded", "honest_spec", "seed_43", "spec_lenient"):
        print(f"    {s:>19} {1 - difr['certificate'][s]['agree']:>7.1%}"
              f" {aware['replay_disagreement'][s]:>7.1%}")
    c = aware["cost_seconds"]
    print(f"  cost: V0 {c['prefill_difr']*1000:.1f} ms, V2 {c['spec_replay']*1000:.1f} ms "
          f"({c['ratio']:.1f}x)")
    print("\nfig 3  sampled-token flip rate vs prefill:")
    for p_ in ("prefill_repeat", "batch2", "batch8", "decode", "verify_g4", "verify_g8"):
        print(f"    {p_:>15} {num['stats'][p_]['token_flip_rate']:>7.2%}"
              f"   mean |dlogit| {num['stats'][p_]['mean_abs_logit_diff']:.4f}")


if __name__ == "__main__":
    main()
