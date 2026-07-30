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


def _line(ax, x, y, color, label, sd=None):
    """One direct-labeled series. `sd` draws a +-1 sd band: for the AUC axes that
    band is the difference between a finding and noise, so it is not decoration."""
    if sd is not None:
        y, sd = np.asarray(y, float), np.asarray(sd, float)
        ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.13, lw=0, zorder=2)
    ax.plot(x, y, color=color, lw=2.0, marker="o", ms=5.5, zorder=3,
            markeredgecolor="white", markeredgewidth=1.2, label=label)


# ======================================================================== head
def fig_pareto(p, path):
    import matplotlib.pyplot as plt
    names, budgets, atk = p["value_fns"], p["budgets"], p["headline_attack"]
    curves = p["pareto"][atk]
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    _clean(ax)
    full = curves["uniform"][-1]
    ax.axhline(full, color=MUTED, lw=1.4, ls=(0, (4, 3)), zorder=1)
    ax.annotate(f"full recompute (budget 100%)  AUC {full:.3f}", (0.42, full),
                xytext=(0, 5), ha="left", va="bottom", textcoords="offset points",
                color=INK2, fontsize=9)
    sds = p.get("pareto_sd", {}).get(atk, {})
    # Six series that all CONVERGE at budget 1.0 (every signal audits everything
    # there), so a direct label at the right edge is six labels on one point.
    # Error bars rather than filled bands for the same reason: six overlapping
    # +-1 sd bands hide the curves they are supposed to qualify. The legend
    # carries identity; the two annotated extremes carry the finding.
    for name, c in zip(names, SERIES):
        y = curves[name]
        ax.errorbar(budgets, y, yerr=sds.get(name), color=c, lw=2.0, marker="o",
                    ms=5.5, capsize=2.5, elinewidth=1.0, zorder=3,
                    markeredgecolor="white", markeredgewidth=1.2, label=name)
    if sds:
        ax.annotate(f"error bars = $\\pm$1 sd over {p.get('n_seed', 1)} protocol "
                    f"seeds; signals separated by less than that are not separated",
                    (0.0, 0.02), xycoords="axes fraction", color=MUTED, fontsize=8.5)
        # Name the best and the control at the budget where they are furthest
        # apart -- the whole question this figure asks is whether allocation beats
        # spending the same tokens at random.
        best = max(names, key=lambda n: max(curves[n][:-1]))
        i = int(np.argmax(curves[best][:-1]))
        ax.annotate(f"{best} {curves[best][i]:.3f} $\\pm$ {sds[best][i]:.3f}\n"
                    f"at a {budgets[i]:.0%} budget -- above full recompute",
                    (budgets[i], curves[best][i]), xytext=(10, 8),
                    textcoords="offset points", color=SERIES[names.index(best)],
                    fontsize=9, weight="bold", linespacing=1.35)
        if "uniform" in names:
            j = int(np.argmin(curves["uniform"][:-1]))
            ax.annotate(f"uniform {curves['uniform'][j]:.3f} -- the equal-cost\n"
                        f"random control at the same {budgets[j]:.0%} budget",
                        (budgets[j], curves["uniform"][j]), xytext=(30, 7),
                        textcoords="offset points", va="bottom",
                        color=SERIES[names.index("uniform")], fontsize=9,
                        linespacing=1.35)
    ax.set_xlabel("recompute budget (fraction of tokens audited)", color=INK2, fontsize=10)
    ax.set_ylabel("detection AUC @ FPR $\\leq$ 0.5%", color=INK2, fontsize=10)
    ax.set_title("Learned triage head vs hand-crafted value signals", color=INK,
                 fontsize=12, weight="bold", loc="left", pad=34)
    ax.annotate(f"attack={atk}   M={p['M'].split('/')[-1]}   proxy={p['proxy'].split('/')[-1]}"
                f"   {p['n_eval']} prompts x {p['tokens']} tokens   batch {p['batch']}"
                f" ({p.get('batch_frac_of_null', 0):.1%} of the honest null split)"
                f"\nhead trained on {p['n_train']} DISJOINT honest prompts "
                f"(probe sigma {p['probe_sigma']}, {p['n_probe']} probes); "
                f"'oracle' = same features, labeled with {p['oracle_attack']} pairs",
                (0, 1.015), xycoords="axes fraction", color=INK2, fontsize=8.5, va="bottom")
    ax.set_xlim(0, 1.06)
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
    scope = ("held-out honest split" if rel.get("held_out")
             else "STS's OWN fit split (IN-SAMPLE)")
    axA.set_title(f"A  Calibration on a {scope}", color=INK, fontsize=11,
                  weight="bold", loc="left")
    axA.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    ins = p.get("reliability_insample")
    note = f"STS temperature T = {rel['temperature']:.3f}"
    if ins:
        note += (f"\nin-sample, for contrast: ECE {ins['ece_raw']:.4f} "
                 f"$\\rightarrow$ {ins['ece_sts']:.4f}")
    axA.annotate(note, (0.98, 0.10), xycoords="axes fraction", ha="right",
                 color=MUTED, fontsize=8.5, linespacing=1.4)

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
    sd = p.get("transfer_sd", {})
    # `entropy` is here because it is the incumbent default value signal and the
    # one the head has to beat -- a transfer panel without it would compare the
    # head only against signals it already beats. Colours are looked up from the
    # figure's own value-fn order, so a hue means the same signal in both figures.
    order = list(p["value_fns"])
    rows = [n for n in ("entropy", "tie_margin", "learned", "oracle") if n in tr[atks[0]]]
    xs = np.arange(len(atks))
    wb = 0.78 / len(rows)
    off = lambda i: (i - (len(rows) - 1) / 2) * wb
    for i, lab in enumerate(rows):
        c = SERIES[order.index(lab) % len(SERIES)] if lab in order else MUTED
        vals = [tr[a][lab] for a in atks]
        errs = [sd.get(a, {}).get(lab, 0.0) for a in atks] if sd else None
        axC.bar(xs + off(i), vals, width=wb - 0.03, color=c, zorder=3, label=lab,
                yerr=errs, capsize=2.5, ecolor=INK2,
                error_kw=dict(elinewidth=1.0, zorder=5))
        for x, v, e in zip(xs + off(i), vals, errs or [0.0] * len(vals)):
            axC.annotate(f"{v:.2f}", (x, v + e), xytext=(0, 3),
                         textcoords="offset points", ha="center", color=INK2,
                         fontsize=8)
    for i, a in enumerate(atks):
        axC.plot([i - 0.44, i + 0.44], [tr[a]["full"]] * 2, color=MUTED,
                 lw=1.6, ls=(0, (3, 2)), zorder=4)
    axC.set_ylim(0, 1.24)
    axC.annotate("dashes = full recompute   error bars = $\\pm$1 sd over "
                 f"{p.get('n_seed', 1)} seeds", (0.02, 0.99),
                 xycoords="axes fraction", ha="left", va="top", color=INK2,
                 fontsize=8.5)
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
    # With more than one value signal on record, panel B carries the step-5 cross:
    # the SHAPE of the audit (scheduler) against WHICH tokens it picks (value_fn).
    by_val = p.get("curves_by_value")
    vfs = p.get("value_fns", [p.get("value_fn", "tie_margin")])
    if by_val and len(vfs) > 1:
        i = 0
        for v in vfs:
            for s in scheds:
                c = SERIES[i % len(SERIES)]
                dash = (0, (3, 2)) if s == "topk" else None
                _line(axB, by_val[v][s]["prefill_ratio"], by_val[v][s]["auc"], c,
                      f"{s} / {v}", sd=by_val[v][s].get("auc_sd"))
                if dash:
                    axB.lines[-1].set_linestyle(dash)
                i += 1
    else:
        for s, c in zip(scheds, SERIES):
            _line(axB, d[s]["prefill_ratio"], d[s]["auc"], c, LABEL[s],
                  sd=d[s].get("auc_sd"))
    axB.set_xlabel("realized prefill cost / full-audit cost", color=INK2, fontsize=10)
    axB.set_ylabel("detection AUC @ FPR $\\leq$ 0.5%", color=INK2, fontsize=10)
    axB.set_title("B  The honest Pareto", color=INK, fontsize=11, weight="bold", loc="left")
    axB.set_xlim(0, 1.06)
    axB.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")
    note = (f"bands = $\\pm$1 sd over {p.get('n_seed')} protocol seeds"
            if d.get("auc_sd") else
            f"AUC over {p['n']}x{p['tokens']} tokens is the noisy axis here")
    axB.annotate(f"{note};\nA and C are exact geometry / measured time",
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
