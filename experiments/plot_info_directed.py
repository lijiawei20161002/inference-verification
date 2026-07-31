"""Figure for information-directed verification (`exp_info_directed_gpu`).

Renders the cached JSON that experiment writes -- pure numpy + matplotlib, no GPU,
so the figure can be iterated without re-running the model:

  docs/figures/fig_info_directed.png
      (A) ALLOCATION: detection AUC vs recompute budget for every value signal,
          under the matched-filter statistic the derivation is about -- does
          ranking by information `I = Delta^2/v` beat ranking by sensitivity
          `Delta` (the negative result of docs/TRIAGE_AND_AUDIT_COST.md Part 1)
          and beat the hand-crafted signals already in the library? The ORACLE
          cell (labeled `(Delta, v)`, not deployable) is the ceiling.
      (B) AGGREGATION: the same audit sets scored by the weighted, centered
          statistic instead of the batch mean, as a difference.
      (C) THEORY: `signal.pauc_of_capture` predicted from the information capture
          alone, against the measured curve, deployably and at the oracle. A
          prediction, not a fit.
      (D) The capture curves themselves -- what fraction of the pool's REAL
          information each cheap ranking keeps -- plus how each Tier-0 ranking
          correlates with the oracle per-token information.
      (E) The claims of `ivgym/infogain.py`, each as a difference paired by
          protocol seed with its standard error, at the budget where the effect is
          largest. This is the panel that says which claims survived.

Everything annotated here is computed from the payload (which signal wins, by how
much, at which budget, with what t) rather than written in, so the figure states
whatever the run actually found.

    python -m experiments.plot_info_directed [attack_name]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
RES_DIR = ROOT / "docs" / "results"

# Categorical slots 1-6 of the validated reference palette (light mode), assigned in
# the documented fixed order and never cycled: one hue per DEPLOYABLE value signal,
# the same hue in every panel, so a colour means "this allocation rule" throughout.
# The oracle arms are deliberately NOT given a categorical slot -- they are a
# ceiling, not a competitor, and are drawn in ink like every other reference line
# in this repo's figures.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8880", "#e3e2dd"
ORACLE = "oracle_info"
ORACLE_CELL = "oracle_mf/oracle_info"

# Readable row labels for panel E. Keys are the comparison names the experiment
# writes; anything unknown falls back to its raw key, so adding a comparison to the
# experiment does not silently drop it from the figure.
CLAIM = {
    "P1a_info_vs_sensitivity": "P1a  rank by $I$ vs by $\\Delta$  (matched agg.)",
    "P1a_info_vs_sensitivity_mean_agg": "P1a  rank by $I$ vs by $\\Delta$  (mean agg.)",
    "P1a_info_vs_best_handcrafted": "P1a  rank by $I$ vs the best hand-crafted signal",
    "P1b_matched_vs_mean_at_info": "P1b  matched filter vs batch mean  ($I$ audit)",
    "P1b_matched_vs_mean_at_uniform": "P1b  matched filter vs batch mean  (random audit)",
    "oracle_alloc_vs_info": "ceiling  oracle $I$ vs estimated $\\hat{I}$  (allocation)",
    "oracle_agg_vs_mean": "ceiling  oracle weights vs batch mean  (aggregation)",
    "oracle_both_vs_deployable": "ceiling  oracle both vs deployable both",
    "oracle_both_vs_full_mean": "ceiling  oracle both vs a full audit, batch mean",
}


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def _deployable(p):
    return [v for v in p["value_fns"] if v != ORACLE]


def _colors(p):
    c = {v: SERIES[i % len(SERIES)] for i, v in enumerate(_deployable(p))}
    c[ORACLE] = INK
    return c


def _cell(p, atk, agg, vf):
    """`(auc, sd)` arrays for one (aggregation, allocation) cell."""
    key = f"{agg}/{vf}"
    return (np.asarray(p["auc"][atk][key], float),
            np.asarray(p["auc_sd"][atk].get(key, np.zeros(len(p["budgets"]))), float))


def _sem(p, atk, agg, vf):
    """Standard error of the mean AUC over protocol seeds -- the right bar for
    comparing two cells' means, as opposed to the sd of a single seed's draw."""
    return _cell(p, atk, agg, vf)[1] / max(np.sqrt(p.get("n_seed", 1)), 1.0)


# ------------------------------------------------------- A: which tokens to audit
def panel_alloc(ax, p, atk, agg="matched"):
    budgets, col = p["budgets"], _colors(p)
    full = _cell(p, atk, "mean", "uniform")[0][-1]
    ax.axhline(full, color=MUTED, lw=1.3, ls=(0, (4, 3)), zorder=1)
    ax.annotate(f"full recompute, batch mean  AUC {full:.3f}", (0.02, full),
                xytext=(0, 5), textcoords="offset points", color=INK2, fontsize=8.5)
    # Error bars, not filled bands: overlapping +-1 sem bands hide the curves they
    # are there to qualify (same reasoning as plot_triage.fig_pareto).
    for vf in _deployable(p):
        y, _ = _cell(p, atk, agg, vf)
        ax.errorbar(budgets, y, yerr=_sem(p, atk, agg, vf), color=col[vf], lw=2.0,
                    marker="o", ms=5.0, capsize=2.5, elinewidth=1.0, zorder=3,
                    markeredgecolor="white", markeredgewidth=1.1, label=vf)
    if ORACLE in p["value_fns"]:
        y, _ = _cell(p, atk, "oracle_mf", ORACLE)
        ax.errorbar(budgets, y, yerr=_sem(p, atk, "oracle_mf", ORACLE), color=INK,
                    lw=2.0, ls=(0, (5, 2)), marker="D", ms=4.5, capsize=2.5,
                    elinewidth=1.0, zorder=4, markeredgecolor="white",
                    markeredgewidth=1.1,
                    label="ORACLE $(\\Delta, v)$ -- ceiling, not deployable")
    # The finding, computed from the payload: `info` (rank by information) against
    # `sensitivity` (rank by Delta, the earlier head's target), paired by seed.
    cmp = p.get("comparisons", {}).get(atk, {}).get("P1a_info_vs_sensitivity")
    if cmp:
        k = int(np.argmax([abs(r["diff"]) for r in cmp[:-1]]))   # 1.0 = same audit
        r = cmp[k]
        yi = _cell(p, atk, agg, "info")[0][k]
        ys = _cell(p, atk, agg, "sensitivity")[0][k]
        verb = "above" if r["diff"] > 0 else "below"
        ax.annotate(f"info {yi:.3f} is {abs(r['diff']):.3f} {verb} sensitivity "
                    f"{ys:.3f}\nat a {r['budget']:.0%} budget "
                    f"(paired sem {r['sem']:.3f}, t {r['t']:+.1f})",
                    (r["budget"], max(yi, ys)), xytext=(12, 12),
                    textcoords="offset points", color=col["info"], fontsize=9,
                    weight="bold", linespacing=1.35)
    ax.set_xlabel("recompute budget (fraction of tokens audited)", color=INK2, fontsize=10)
    ax.set_ylabel("detection AUC @ FPR $\\leq$ 0.5%", color=INK2, fontsize=10)
    ax.set_title(f"A  Allocation: what to rank by ({agg} aggregation)", color=INK,
                 fontsize=11, weight="bold", loc="left")
    ax.set_xlim(0, 1.06)
    ax.annotate(f"error bars = $\\pm$1 sem over {p.get('n_seed', 1)} protocol seeds",
                (0.0, 0.015), xycoords="axes fraction", color=MUTED, fontsize=8)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right", ncol=2)


# --------------------------------------------------- B: how to combine the audit
def panel_agg(ax, p, atk):
    budgets, col = p["budgets"], _colors(p)
    ax.axhline(0.0, color=INK2, lw=1.2, zorder=2)
    best = (None, 0.0, 0)
    for vf in _deployable(p):
        ym = _cell(p, atk, "mean", vf)[0]
        yx = _cell(p, atk, "matched", vf)[0]
        d = yx - ym
        # Paired by seed where the payload carries the per-seed values, which is the
        # error bar that belongs on a difference of two cells sharing protocol draws.
        err = _paired_err(p, atk, f"matched/{vf}", f"mean/{vf}")
        ax.errorbar(budgets, d, yerr=err, color=col[vf], lw=2.0, marker="o", ms=5.0,
                    capsize=2.5, elinewidth=1.0, zorder=3, markeredgecolor="white",
                    markeredgewidth=1.1, label=vf)
        k = int(np.argmax(np.abs(d)))
        if abs(d[k]) > abs(best[1]):
            best = (vf, float(d[k]), k)
    if ORACLE in p["value_fns"]:
        d = _cell(p, atk, "oracle_mf", "uniform")[0] - _cell(p, atk, "mean", "uniform")[0]
        ax.errorbar(budgets, d, yerr=_paired_err(p, atk, "oracle_mf/uniform",
                                                 "mean/uniform"),
                    color=INK, lw=2.0, ls=(0, (5, 2)), marker="D", ms=4.5,
                    capsize=2.5, elinewidth=1.0, zorder=4, markeredgecolor="white",
                    markeredgewidth=1.1,
                    label="ORACLE weights, same (uniform) audit")
    if best[0] is not None:
        vf, d, k = best
        ax.annotate(f"largest deployable effect: {vf} {d:+.3f} AUC\n"
                    f"at a {budgets[k]:.0%} budget",
                    (budgets[k], d), xytext=(10, 8 if d > 0 else -26),
                    textcoords="offset points", color=col[vf], fontsize=9,
                    weight="bold", linespacing=1.35)
    ax.set_xlabel("recompute budget", color=INK2, fontsize=10)
    ax.set_ylabel("AUC(weighted, centered) $-$ AUC(batch mean)", color=INK2, fontsize=10)
    ax.set_title("B  Aggregation: same tokens, weighted and centered", color=INK,
                 fontsize=11, weight="bold", loc="left")
    ax.set_xlim(0, 1.06)
    ax.annotate("above 0 = the matched filter wins at an unchanged\n"
                "budget and an unchanged audit set",
                (0.02, 0.97), xycoords="axes fraction", va="top", color=MUTED,
                fontsize=8.5, linespacing=1.4)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right", ncol=2)


def _paired_err(p, atk, lhs, rhs):
    """+-1 sem of the seed-paired difference `lhs - rhs`, or None if the per-seed
    values were not stored (older payloads)."""
    seeds = p.get("auc_seeds", {}).get(atk)
    if not seeds or lhs not in seeds or rhs not in seeds:
        return None
    a, b = np.asarray(seeds[lhs], float), np.asarray(seeds[rhs], float)
    n = a.shape[1]
    return (a - b).std(axis=1, ddof=1) / np.sqrt(n) if n > 1 else None


# --------------------------------------------------------- C: theory vs measured
def panel_theory(ax, p, atk):
    budgets, th = p["budgets"], p["theory"][atk]
    pairs = [("predicted", "measured", SERIES[5], "matched / info (deployable)", None),
             ("oracle_predicted", "oracle_measured", INK, "oracle (ceiling)",
              (0, (5, 2)))]
    for pk, mk, c, lab, dash in pairs:
        if pk not in th:
            continue
        pred, meas = np.asarray(th[pk], float), np.asarray(th[mk], float)
        ax.plot(budgets, pred, color=c, lw=1.6, ls=(0, (2, 2)), marker="s", ms=4.0,
                zorder=3, markeredgecolor="white", markeredgewidth=1.0,
                label=f"predicted -- {lab}")
        ax.plot(budgets, meas, color=c, lw=2.0, ls=dash, marker="o", ms=5.0, zorder=4,
                markeredgecolor="white", markeredgewidth=1.1,
                label=f"measured -- {lab}")
        ax.fill_between(budgets, pred, meas, color=c, alpha=0.09, lw=0, zorder=2)
    ax.set_xlabel("recompute budget", color=INK2, fontsize=10)
    ax.set_ylabel("detection AUC @ FPR $\\leq$ 0.5%", color=INK2, fontsize=10)
    ax.set_title("C  Theory: the curve predicted before any recompute", color=INK,
                 fontsize=11, weight="bold", loc="left")
    ax.set_xlim(0, 1.06)
    note = (f"per-token d' {th['d_prime_mean']:.4f} (mean) $\\rightarrow$ "
            f"{th['d_prime_matched']:.4f} (matched)")
    if "d_prime_oracle_matched" in th:
        # %g: the oracle weights are fit on the pool they are scored on, so this
        # number is an in-sample bound and can run orders of magnitude high.
        note += f" $\\rightarrow$ {th['d_prime_oracle_matched']:.3g} (oracle)"
    note += f"\nMAE {th['mae']:.3f} deployable"
    if "oracle_mae" in th:
        note += f", {th['oracle_mae']:.3f} at the oracle"
    note += (f"\npredicted batch for AUC 0.90: "
             f"{th['predicted_full_batch_for_090']} tokens")
    ax.annotate(note, (0.98, 0.03), xycoords="axes fraction", ha="right", va="bottom",
                color=MUTED, fontsize=8.5, linespacing=1.4)
    ax.legend(frameon=False, fontsize=8.0, labelcolor=INK2, loc="upper left")


# ------------------------------------------- D: capture, and is the estimate right
def panel_capture(ax, p, atk):
    budgets, col = p["budgets"], _colors(p)
    cap = p.get("capture_oracle", {}).get(atk)
    which = "ORACLE" if cap else "Tier-0"
    cap = cap or p["capture"]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate("a random audit keeps its budget's share", (0.30, 0.245), color=INK2,
                fontsize=8.5, rotation=31, rotation_mode="anchor")
    for vf in p["value_fns"]:
        if vf not in cap:
            continue
        ax.plot(budgets, cap[vf], color=col[vf], lw=2.0,
                ls=(0, (5, 2)) if vf == ORACLE else None,
                marker="D" if vf == ORACLE else "o", ms=4.5 if vf == ORACLE else 5.0,
                zorder=4 if vf == ORACLE else 3, markeredgecolor="white",
                markeredgewidth=1.1, label=vf)
    ax.set_xlabel("recompute budget", color=INK2, fontsize=10)
    ax.set_ylabel(f"fraction of {which} information $\\sum I(t)$ kept",
                  color=INK2, fontsize=10)
    ax.set_title(f"D  What a budget keeps of the {which} information", color=INK,
                 fontsize=11, weight="bold", loc="left")
    ax.set_xlim(0, 1.06)
    ax.set_ylim(0, 1.06)
    dg = p["diagnostics"][atk]
    rows = [("info", "spearman_info_vs_oracle"),
            ("sensitivity", "spearman_sensitivity_vs_oracle"),
            ("entropy", "spearman_entropy_vs_oracle"),
            ("tie_margin", "spearman_tie_vs_oracle"),
            ("Delta_hat", "spearman_delta_hat_vs_oracle_delta"),
            ("v_hat", "spearman_v_hat_vs_oracle_v")]
    txt = "\n".join(f"{n:<12s}{dg[k]:+.3f}" for n, k in rows if k in dg)
    ax.annotate("Spearman vs the oracle quantity\n" + txt, (0.985, 0.03),
                xycoords="axes fraction", ha="right", va="bottom", color=INK2,
                fontsize=8.5, family="monospace", linespacing=1.5)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left", ncol=2)


# ------------------------------------------------------------- E: did it survive?
def panel_claims(ax, p, atk):
    """One row per claim: the seed-paired difference at the budget where it is
    largest, with its sem. Green = the claim's prediction held there, orange =
    the reverse happened, grey = not separated from zero."""
    cmps = p.get("comparisons", {}).get(atk)
    if not cmps:
        ax.set_axis_off()
        return
    labels, diffs, errs, colors, notes = [], [], [], [], []
    for name, out in cmps.items():
        k = int(np.argmax([abs(r["diff"]) for r in out]))
        r = out[k]
        sig = abs(r["t"]) >= 2.0
        labels.append(f"{CLAIM.get(name, name)}\nat a {r['budget']:.0%} budget"
                      + (f", vs {r['vs'].split('/')[-1]}" if "vs" in r else ""))
        diffs.append(r["diff"])
        errs.append(r["sem"])
        colors.append(SERIES[2] if (sig and r["diff"] > 0) else
                      SERIES[1] if (sig and r["diff"] < 0) else MUTED)
        notes.append(f"t {r['t']:+.1f}" if np.isfinite(r["t"]) else "t inf")
    y = np.arange(len(labels))[::-1]
    ax.errorbar(diffs, y, xerr=errs, fmt="o", ms=6.5, capsize=3.0, elinewidth=1.2,
                lw=0, zorder=3, markeredgecolor="white", markeredgewidth=1.2,
                ecolor=INK2)
    for yi, d, c in zip(y, diffs, colors):
        ax.plot([d], [yi], marker="o", ms=6.5, color=c, zorder=4,
                markeredgecolor="white", markeredgewidth=1.2)
    for yi, d, e, n in zip(y, diffs, errs, notes):
        ax.annotate(n, (d + e, yi), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=8)
    ax.axvline(0.0, color=INK2, lw=1.2, zorder=2)
    ax.set_yticks(y, labels, fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("seed-paired difference in AUC @ FPR $\\leq$ 0.5%  "
                  "($\\pm$1 sem)", color=INK2, fontsize=10)
    ax.set_title("E  Which claims survived (green = held, orange = reversed, "
                 "grey = null)", color=INK, fontsize=11, weight="bold", loc="left")
    pad = 0.35 * max(max(np.abs(diffs)), 1e-6)
    ax.set_xlim(min(diffs) - pad - max(errs), max(diffs) + pad + max(errs))


# ------------------------------------------------------------------------- figure
def fig_info_directed(p, path, atk=None):
    import matplotlib.pyplot as plt
    atk = atk or p["headline_attack"]
    # Two columns, not three: every panel carries a seven-entry legend, and at three
    # to a row the legends do not fit inside their own axes.
    fig = plt.figure(figsize=(15.4, 16.4))
    gs = fig.add_gridspec(3, 2, hspace=0.30, wspace=0.22, top=0.945,
                          height_ratios=[1.0, 1.0, 0.82])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[2, :])]
    for a in axes:
        _clean(a)
    panel_alloc(axes[0], p, atk)
    panel_agg(axes[1], p, atk)
    panel_theory(axes[2], p, atk)
    panel_capture(axes[3], p, atk)
    panel_claims(axes[4], p, atk)
    fig.suptitle("Information-directed verification: rank by "
                 "$I=\\Delta^2/v$, aggregate by matched filter",
                 color=INK, fontsize=13, weight="bold", x=0.005, ha="left", y=0.995)
    fig.text(0.005, 0.978,
             f"attack={atk}   M={p['M'].split('/')[-1]}   "
             f"proxy={p['proxy'].split('/')[-1]}   {p['n_eval']} prompts x "
             f"{p['tokens']} tokens = {p['eval_tokens']} tokens   batch {p['batch']} "
             f"({p['batch_frac_of_null']:.1%} of the honest null split)   "
             f"{p.get('n_seed', 1)} protocol seeds\n"
             f"both fits: {p['n_train']} DISJOINT honest prompts, "
             f"{p['n_probe']} probes/side at sigma {p['probe_sigma']} "
             f"(benign {p['benign_sigma']:.4f}); same nine Tier-0 features, "
             f"differing only in target -- Delta (sensitivity) vs Delta^2/v (info).  "
             f"R2(delta)={p['fit_report']['r2_delta']:.3f}  "
             f"R2(log v)={p['fit_report']['r2_log_v']:.3f}  "
             f"oracle arms read labeled honest/attack pairs in "
             f"{p.get('oracle_bins', 40)} bins",
             color=INK2, fontsize=8.5, ha="left", va="top", linespacing=1.5)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    src = RES_DIR / "info_directed.json"
    if not src.exists():
        print(f"skip: {src} missing (run exp_info_directed_gpu)")
        return
    p = json.loads(src.read_text())
    out = FIG_DIR / "fig_info_directed.png"
    fig_info_directed(p, out, argv[0] if argv else None)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
