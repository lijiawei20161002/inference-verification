"""Figures for the learned triage head and the audit-cost experiments.

Renders the cached JSON the two GPU experiments write, the same way every other
experiment here ships a figure -- pure numpy + matplotlib, no GPU, so the figures
can be iterated without re-running the model:

  docs/figures/fig_confidence_head_pareto.png
      Detection AUC vs recompute budget for each triage value signal: the four
      hand-crafted ones, the deployable learned head, and the oracle-labeled head.

  docs/figures/fig_confidence_head_diagnostics.png
      (A) reliability before/after Sequential Temperature Scaling, with ECE;
      (B) the head's coefficient per proxy feature; (C) AUC at a fixed budget
      across deviations -- does the honest-only surrogate label transfer?

  docs/figures/fig_prefix_cost.png
      (A) nominal token budget vs realized prefill cost -- the accounting gap;
      (B) AUC vs REAL cost for top-k and the prefix scheduler; (C) the same
      against measured prefill seconds.

    python -m experiments.plot_triage
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
RES_DIR = ROOT / "docs" / "results"

# Categorical slots 1-6 of the validated reference palette (light mode), assigned
# in the documented fixed order and never cycled. Every line is ALSO direct-
# labeled, which is both the >=2-series identity rule and the relief rule for the
# slots that sit under 3:1 contrast on a light surface (yellow, magenta).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8880", "#e3e2dd"
LABEL = {"topk": "top-k over tokens (current)", "prefix": "prefix-scheduled (DSpark port)"}


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def _line(ax, x, y, color, label):
    ax.plot(x, y, color=color, lw=2.0, marker="o", ms=5.5, zorder=3,
            markeredgecolor="white", markeredgewidth=1.2, label=label)


# ======================================================================== head
def fig_pareto(p, path):
    import matplotlib.pyplot as plt
    names, budgets, atk = p["value_fns"], p["budgets"], p["headline_attack"]
    curves = p["pareto"][atk]
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    _clean(ax)
    full = curves["uniform"][-1]
    ax.axhline(full, color=MUTED, lw=1.4, ls=(0, (4, 3)), zorder=1)
    ax.annotate(f"full recompute (budget 100%)  AUC {full:.3f}", (0.015, full),
                xytext=(0, 6), textcoords="offset points", color=INK2, fontsize=9)
    for name, c in zip(names, SERIES):
        y = curves[name]
        _line(ax, budgets, y, c, name)
        ax.annotate(name, (budgets[-1], y[-1]), xytext=(8, 0), va="center",
                    textcoords="offset points", color=c, fontsize=9.5, weight="bold")
    ax.set_xlabel("recompute budget (fraction of tokens audited)", color=INK2, fontsize=10)
    ax.set_ylabel("detection AUC @ FPR $\\leq$ 0.5%", color=INK2, fontsize=10)
    ax.set_title("Learned triage head vs hand-crafted value signals", color=INK,
                 fontsize=12, weight="bold", loc="left")
    ax.annotate(f"attack={atk}   M={p['M'].split('/')[-1]}   proxy={p['proxy'].split('/')[-1]}"
                f"   {p['n_eval']} prompts x {p['tokens']} tokens   batch {p['batch']}"
                f"\nhead trained on {p['n_train']} DISJOINT honest prompts "
                f"(probe sigma {p['probe_sigma']}, {p['n_probe']} probes); "
                f"'oracle' = same features, labeled with {p['oracle_attack']} pairs",
                (0, 1.015), xycoords="axes fraction", color=INK2, fontsize=8.5, va="bottom")
    ax.set_xlim(0, 1.26)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right", ncol=2)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_diagnostics(p, path):
    import matplotlib.pyplot as plt
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16.8, 4.9))
    for a in (axA, axB, axC):
        _clean(a)

    rel = p["reliability"]
    axA.plot([0, 1], [0, 1], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    axA.annotate("perfectly calibrated", (0.40, 0.44), color=INK2, fontsize=8.5,
                 rotation=36, rotation_mode="anchor")
    for key, c, lab in (("raw", SERIES[0], f"raw  ECE {rel['ece_raw']:.4f}"),
                        ("sts", SERIES[1], f"+STS  ECE {rel['ece_sts']:.4f}")):
        _line(axA, rel[key]["conf"], rel[key]["acc"], c, lab)
    axA.set_xlabel("predicted P(high-value position)", color=INK2, fontsize=10)
    axA.set_ylabel("observed fraction high-value", color=INK2, fontsize=10)
    axA.set_title(f"A  Calibration on held-out honest  (STS T={rel['temperature']:.3f})",
                  color=INK, fontsize=11, weight="bold", loc="left")
    axA.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")

    w = p["weights"]
    names = list(w)
    vals = np.array([w[k] for k in names])
    order = np.argsort(np.abs(vals))
    y = np.arange(len(order))
    axB.barh(y, vals[order], color=SERIES[0], height=0.62, zorder=3)
    axB.axvline(0, color=MUTED, lw=1.0)
    axB.set_yticks(y, [names[i] for i in order], fontsize=9)
    axB.grid(axis="y", visible=False)
    for yi, v in zip(y, vals[order]):
        axB.annotate(f"{v:+.2f}", (v, yi), xytext=(6 if v >= 0 else -6, 0),
                     textcoords="offset points", va="center",
                     ha="left" if v >= 0 else "right", color=INK2, fontsize=8.5)
    axB.set_xlabel("logistic coefficient (standardized features)", color=INK2, fontsize=10)
    axB.set_title("B  Which cheap signals the head uses", color=INK, fontsize=11,
                  weight="bold", loc="left")
    pad = 0.42 * max(np.abs(vals).max(), 1e-6)
    axB.set_xlim(vals.min() - pad, vals.max() + pad)

    tr, atks = p["transfer"], p["eval_attacks"]
    rows = [("tie_margin", SERIES[2]), ("learned", SERIES[0]), ("oracle", SERIES[1])]
    xs = np.arange(len(atks))
    wb = 0.26
    for i, (lab, c) in enumerate(rows):
        vals = [tr[a][lab] for a in atks]
        axC.bar(xs + (i - 1) * wb, vals, width=wb - 0.03, color=c, zorder=3, label=lab)
        for x, v in zip(xs + (i - 1) * wb, vals):
            axC.annotate(f"{v:.2f}", (x, v), xytext=(0, 3), textcoords="offset points",
                         ha="center", color=INK2, fontsize=8)
    for i, a in enumerate(atks):
        axC.plot([i - 1.6 * wb, i + 1.6 * wb], [tr[a]["full"]] * 2, color=MUTED,
                 lw=1.6, ls=(0, (3, 2)), zorder=4)
    axC.annotate("dashes = full recompute", (0.98, 0.03), xycoords="axes fraction",
                 ha="right", color=INK2, fontsize=8.5)
    axC.axhline(0.5, color=MUTED, lw=1.0, zorder=1)
    axC.set_xticks(xs, atks, fontsize=9)
    axC.grid(axis="x", visible=False)
    axC.set_ylabel(f"AUC @ FPR $\\leq$ 0.5%  (budget {p['transfer_budget']:.0%})",
                   color=INK2, fontsize=10)
    axC.set_title("C  Does the label transfer across deviations?", color=INK,
                  fontsize=11, weight="bold", loc="left")
    axC.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ======================================================================== cost
def fig_cost(p, path):
    import matplotlib.pyplot as plt
    d, budgets, scheds = p["curves"], p["budgets"], p["schedulers"]
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16.8, 4.9))
    for a in (axA, axB, axC):
        _clean(a)

    # The prefix curve IS the identity line (it schedules against the cost budget
    # directly), so drawing a separate diagonal would just double it. Shade the gap
    # between the two instead -- that area is the overcharge.
    axA.fill_between(budgets, d["prefix"]["prefill_ratio"], d["topk"]["prefill_ratio"],
                     color=SERIES[0], alpha=0.10, lw=0, zorder=2)
    for s, c in zip(scheds, SERIES):
        _line(axA, budgets, d[s]["prefill_ratio"], c, LABEL[s])
    i10 = budgets.index(0.10) if 0.10 in budgets else 1
    axA.annotate(f"a \"{budgets[i10]:.0%}\" budget really costs\n"
                 f"{d['topk']['prefill_ratio'][i10]:.0%} of a full audit",
                 (budgets[i10], d["topk"]["prefill_ratio"][i10]), xytext=(16, -34),
                 textcoords="offset points", color=SERIES[0], fontsize=9,
                 arrowprops=dict(arrowstyle="-", color=SERIES[0], lw=1.0))
    axA.annotate("spends exactly what\nit is budgeted", (0.62, 0.50), color=SERIES[1],
                 fontsize=8.5, ha="left", va="top")
    axA.set_xlabel("nominal budget", color=INK2, fontsize=10)
    axA.set_ylabel("realized prefill cost / full-audit cost", color=INK2, fontsize=10)
    axA.set_title("A  The accounting gap", color=INK, fontsize=11, weight="bold", loc="left")
    axA.set_ylim(0, 1.06)
    axA.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")

    full = max(d["topk"]["auc"][-1], d["prefix"]["auc"][-1])
    axB.axhline(full, color=MUTED, lw=1.3, ls=(0, (4, 3)), zorder=1)
    axB.annotate(f"full recompute  AUC {full:.3f}", (0.02, full), xytext=(0, 6),
                 textcoords="offset points", color=INK2, fontsize=8.5)
    for s, c in zip(scheds, SERIES):
        _line(axB, d[s]["prefill_ratio"], d[s]["auc"], c, LABEL[s])
    axB.set_xlabel("realized prefill cost / full-audit cost", color=INK2, fontsize=10)
    axB.set_ylabel("detection AUC @ FPR $\\leq$ 0.5%", color=INK2, fontsize=10)
    axB.set_title("B  The honest Pareto", color=INK, fontsize=11, weight="bold", loc="left")
    axB.set_xlim(0, 1.06)
    axB.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")
    axB.annotate(f"AUC over {p['n']}x{p['tokens']} tokens is the noisy axis here;\n"
                 f"A and C are exact geometry / measured time",
                 (0.02, 0.03), xycoords="axes fraction", color=MUTED, fontsize=8, va="bottom")

    for s, c in zip(scheds, SERIES):
        _line(axC, d[s]["prefill_ratio"], d[s]["seconds"], c, LABEL[s])
    full_sec = d["topk"]["seconds"][-1]
    axC.axhline(full_sec, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    lo = min(d["topk"]["seconds"])
    if d["topk"]["seconds"][0] > full_sec:
        axC.annotate(f"top-k at a \"{budgets[0]:.0%}\" budget takes "
                     f"{d['topk']['seconds'][0]:.2f}s --\nMORE than auditing every token "
                     f"({full_sec:.2f}s)",
                     xy=(d["topk"]["prefill_ratio"][0], d["topk"]["seconds"][0]),
                     xytext=(0.03, 0.80), textcoords="axes fraction", ha="left",
                     va="top", color=SERIES[0], fontsize=8.5,
                     arrowprops=dict(arrowstyle="-", color=SERIES[0], lw=1.0,
                                     connectionstyle="arc3,rad=-0.15"))
    axC.set_xlabel("realized prefill cost / full-audit cost", color=INK2, fontsize=10)
    axC.set_ylabel("measured reference-prefill seconds", color=INK2, fontsize=10)
    axC.set_title("C  Cost model vs wall clock", color=INK, fontsize=11,
                  weight="bold", loc="left")
    axC.set_xlim(0, 1.06)
    axC.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")

    fig.suptitle("What a selective audit costs: token fraction vs prefill reality",
                 color=INK, fontsize=12.5, weight="bold", x=0.005, ha="left", y=1.06)
    fig.text(0.005, 1.005, f"attack={p['attack']}   value={p['value_fn']}   "
             f"M={p['M'].split('/')[-1]}   {p['n']} prompts x {p['tokens']} tokens   "
             f"full-audit cost {p['full_prefill_cost']} reference-forward tokens   "
             f"lazy_reference backend (nothing prefilled until the verifier pays)",
             color=INK2, fontsize=8.5, ha="left", va="bottom")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    head = RES_DIR / "confidence_head.json"
    cost = RES_DIR / "prefix_cost.json"
    wrote = []
    if head.exists():
        p = json.loads(head.read_text())
        fig_pareto(p, FIG_DIR / "fig_confidence_head_pareto.png")
        fig_diagnostics(p, FIG_DIR / "fig_confidence_head_diagnostics.png")
        wrote += ["fig_confidence_head_pareto.png", "fig_confidence_head_diagnostics.png"]
    else:
        print(f"skip: {head} missing (run exp_confidence_head_gpu)")
    if cost.exists():
        fig_cost(json.loads(cost.read_text()), FIG_DIR / "fig_prefix_cost.png")
        wrote.append("fig_prefix_cost.png")
    else:
        print(f"skip: {cost} missing (run exp_prefix_cost_gpu)")
    for w in wrote:
        print(f"wrote {FIG_DIR/w}")


if __name__ == "__main__":
    main(sys.argv[1:])
