"""Figures for the selective-vs-full recompute robustness sweep.

Renders `docs/results/selective_robustness.json` (from
exp_selective_robustness_gpu) the same way every other experiment here ships a
figure. Pure numpy + matplotlib on the cached JSON -- no GPU:

  docs/figures/fig_selective_robustness_pareto.png
      Representative AUC-vs-recompute-ratio curves: information-directed triage
      (best value fn) vs equal-cost random subsample, with the full-recompute
      line and the target-AUC crossing, on forward-pass cells across families.

  docs/figures/fig_selective_robustness_summary.png
      Four panels: (A) triage-vs-random saving heatmap (model × attack);
      (B) saving by attack type (forward-pass vs sampling-only); (C) selective @
      fixed budget vs full recompute per attack; (D) value-fn robustness.

    .venv/bin/python -m experiments.plot_selective_robustness [path.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "docs" / "results" / "selective_robustness.json"
FIG_DIR = ROOT / "docs" / "figures"
HEADLINE = "token_difr"
FIXED_BUDGET = 0.25
FAM_COLORS = {"qwen3": "#1f77b4", "llama3.2": "#d1902f", "smollm2": "#2ca02c",
              "pythia": "#9467bd"}
C_TRI, C_RND, C_FULL = "#2ca02c", "#d1902f", "#1f77b4"


def family_of(tag): return tag.rsplit("-", 1)[0]
def interp(rhos, curve, x): return float(np.interp(x, rhos, curve))
def gmean(xs):
    xs = [x for x in xs if x]
    return float(np.exp(np.mean(np.log(xs)))) if xs else np.nan


def _load(path):
    payload = json.loads(Path(path).read_text())
    ok = [m for m in payload["models"] if "error" not in m]
    if not ok:
        raise SystemExit(f"no successful models in {path}")
    return payload, ok


def _cell(m, a): return m["cells"][a][HEADLINE]


# --------------------------------------------------------------------------- fig 1
def fig_pareto(payload, ok):
    import matplotlib.pyplot as plt
    cfg = payload["config"]
    rhos = np.array(cfg["rhos"])
    target = cfg["target"]
    fwd = cfg["forward_pass"]
    # one representative forward-pass cell per family (prefer quant_4bit, else the
    # forward-pass attack with the largest triage saving on that family's largest model)
    panels = []
    for fam in sorted({family_of(m["tag"]) for m in ok}):
        ms = sorted([m for m in ok if family_of(m["tag"]) == fam], key=lambda m: -m["params"])
        m = ms[0]
        # the forward-pass cell where triage helps most (largest relative saving)
        a = max(fwd, key=lambda x: (_cell(m, x).get("saving_rel") or 0))
        panels.append((m, a))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (m, a) in zip(axes, panels):
        c = _cell(m, a)
        vf = c["best_value_fn"]
        tri = np.array(c["triaged"][vf])
        rmean, rstd = np.array(c["random_mean"]), np.array(c["random_std"])
        rel = c.get("rel_target", target)
        ax.axhline(c["full_auc"], ls=":", color=C_FULL, lw=1.4,
                   label=f"full recompute (AUC {c['full_auc']:.2f})")
        ax.axhline(rel, ls="--", color="#888", lw=0.8)
        ax.fill_between(rhos, rmean - rstd, rmean + rstd, color=C_RND, alpha=0.18)
        ax.plot(rhos, tri, "-o", color=C_TRI, ms=4, lw=2.1, label=f"triage ({vf})")
        ax.plot(rhos, rmean, "--s", color=C_RND, ms=3.2, lw=1.8, label="random subsample")
        sav = c.get("saving_rel")
        if sav:
            ax.text(0.03, 0.06, f"{sav:.1f}× fewer recomputes\nto {int(0.95*100)}% of full AUC",
                    transform=ax.transAxes, fontsize=9, color="#222",
                    bbox=dict(boxstyle="round", fc="white", ec="#bbb", alpha=0.9))
        ax.set_xscale("log")
        ax.set_title(f"{m['tag']} · {a}", fontsize=10)
        ax.set_xlabel("recompute ratio", fontsize=9.5)
        ax.set_xlim(rhos.min(), 1.0)
        ax.set_ylim(0.45, 1.02)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=7.8, loc="lower right", framealpha=0.92)
    axes[0].set_ylabel("detection AUC (honest vs attack)", fontsize=10)
    fig.suptitle("Information-directed selective recompute vs equal-cost random, across families\n"
                 f"`{HEADLINE}` on a representative forward-pass cell per family "
                 f"({len(ok)} models, H100). Band = ±1 std over random-selection seeds.",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = FIG_DIR / "fig_selective_robustness_pareto.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    plt.close(fig)


# --------------------------------------------------------------------------- fig 2
def _panel_heat(ax, ok, attacks):
    tags = [m["tag"] for m in sorted(ok, key=lambda m: (family_of(m["tag"]), m["params"]))]
    order = sorted(ok, key=lambda m: (family_of(m["tag"]), m["params"]))
    M = np.array([[(_cell(m, a).get("saving_rel") or np.nan) for a in attacks] for m in order])
    Mc = np.clip(M, 0.5, 4.0)
    im = ax.imshow(Mc, aspect="auto", cmap="RdYlGn", vmin=0.8, vmax=3.0)
    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels(attacks, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels(tags, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            txt = "n/a" if np.isnan(v) else f"{v:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="#111" if (np.isnan(v) or 0.9 < v < 3) else "white")
    ax.set_title("Triage vs random — saving factor (×) to reach target AUC", fontsize=10)
    return im


def _panel_attacktype(ax, ok, cfg):
    groups = [("forward-pass", cfg["forward_pass"]), ("sampling-only", cfg["sampling_only"])]
    labels, savings, colors = [], [], []
    for label, grp in groups:
        s = gmean([_cell(m, a).get("saving_rel") for m in ok for a in grp])
        labels.append(label); savings.append(s)
        colors.append("#2ca02c" if label == "forward-pass" else "#9467bd")
    x = np.arange(len(labels))
    bars = ax.bar(x, savings, color=colors, alpha=0.85, width=0.55)
    ax.axhline(1.0, ls=":", color="#888", lw=1, label="no gain (=random)")
    for r, v in zip(bars, savings):
        if not np.isnan(v):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.03, f"{v:.2f}×", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("gmean saving factor", fontsize=9)
    ax.set_title("Where triage pays: attack type", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)


def _panel_selvsfull(ax, ok, cfg):
    attacks = cfg["attacks"]
    rhos = cfg["rhos"]
    full = [np.mean([_cell(m, a)["full_auc"] for m in ok]) for a in attacks]
    tri = [np.mean([interp(rhos, _cell(m, a)["triaged"][_cell(m, a)["best_value_fn"]],
                           FIXED_BUDGET) for m in ok]) for a in attacks]
    x = np.arange(len(attacks)); w = 0.38
    ax.bar(x - w / 2, full, w, color=C_FULL, alpha=0.85, label="full (100%)")
    ax.bar(x + w / 2, tri, w, color=C_TRI, alpha=0.85,
           label=f"triage @ {int(FIXED_BUDGET*100)}%")
    ax.axhline(0.5, ls=":", color="#888", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(attacks, rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0.4, 1.02); ax.set_ylabel("mean AUC", fontsize=9)
    ax.set_title(f"Selective @ {int(FIXED_BUDGET*100)}% vs full recompute", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)


def _panel_valuefn(ax, ok, cfg):
    vfs = cfg["value_fns"]
    fwd = cfg["forward_pass"]
    means = []
    for vf in vfs:
        ratios = [_cell(m, a).get("cost_rel", {}).get(f"triage:{vf}") for m in ok for a in fwd]
        ratios = [r for r in ratios if r]
        means.append(np.mean(ratios) if ratios else np.nan)
    x = np.arange(len(vfs))
    bars = ax.bar(x, means, color="#8c564b", alpha=0.85, width=0.55)
    for r, v in zip(bars, means):
        if not np.isnan(v):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.005, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(vfs, fontsize=9)
    ax.set_ylabel("mean recompute ratio to target", fontsize=9)
    ax.set_title("Value-fn robustness (lower = cheaper, FWD attacks)", fontsize=10)
    ax.grid(axis="y", alpha=0.25)


def fig_summary(payload, ok):
    import matplotlib.pyplot as plt
    cfg = payload["config"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    im = _panel_heat(axes[0, 0], ok, cfg["attacks"])
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04).set_label("saving ×", fontsize=9)
    _panel_attacktype(axes[0, 1], ok, cfg)
    _panel_selvsfull(axes[1, 0], ok, cfg)
    _panel_valuefn(axes[1, 1], ok, cfg)
    fig.suptitle("Selective vs full recompute — robustness synthesis "
                 f"({len(ok)} real models × {len(cfg['attacks'])} attacks, H100)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG_DIR / "fig_selective_robustness_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    payload, ok = _load(path)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"source: {path}  ({len(ok)} models ok)")
    fig_pareto(payload, ok)
    fig_summary(payload, ok)


if __name__ == "__main__":
    main()
