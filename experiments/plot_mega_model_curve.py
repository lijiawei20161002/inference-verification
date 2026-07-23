"""Sibling of `plot_mega_cost_curve` with the x-axis = the CLAIMED MODEL itself
(size ladder, categorical) instead of verifier FLOPs.

Motivation: on the FLOPs axis the cheap proxy is drawn at the *proxy model's*
cost, so several claimed models that share one proxy pile up at the same x --
collapsing the proxy "curve" into a single marker (or, when their AUCs differ,
a spurious vertical segment, e.g. Llama-3.x). Here every point -- recompute AND
proxy -- is plotted at the *claimed* model's slot on the x-axis, so the proxy
becomes a proper left-to-right curve, one point per claimed model.

  * one panel PER FAMILY (qwen, llama, smollm2, pythia, gpt2). x = claimed model
    size ladder (categorical, small->large):
      - **full recompute** (`token_difr`): solid family-coloured line.
      - **cheap same-family proxy** (`surface_stat`): grey dashed line, placed at
        the CLAIMED model's slot (absent for the smallest size, which has no
        smaller sibling to use as a proxy).
      - light band = spread of detection AUC across the six canonical attacks.
  * a sixth SUMMARY panel overlaying all families' recompute curves against a
    normalised size rank (small / mid / large), so families with different
    absolute sizes stay comparable.

Detection AUC is the repo headline: standardized partial AUC @ FPR<=0.5%.

    python -m experiments.plot_mega_model_curve
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results" / "cost_curve"
FIG_DIR = ROOT / "docs" / "figures"

FAM_ORDER = ["qwen", "llama", "smollm2", "pythia", "gpt2"]
FAM_TITLE = {"qwen": "Qwen3", "llama": "Llama-3.x", "smollm2": "SmolLM2",
             "pythia": "Pythia", "gpt2": "GPT-2"}
FAM_COLORS = {"qwen": "#1f77b4", "llama": "#d1902f", "smollm2": "#2ca02c",
              "pythia": "#9467bd", "gpt2": "#d62728"}


def load() -> dict[str, list[dict]]:
    """All per-model results, grouped by family and sorted small->large."""
    by_fam: dict[str, list[dict]] = {}
    for p in sorted(RESULTS.glob("*.json")):
        d = json.loads(p.read_text())
        by_fam.setdefault(d["family"], []).append(d)
    for fam in by_fam:
        by_fam[fam].sort(key=lambda d: d["params"])
    return by_fam


def _param_label(params: float) -> str:
    return f"{params/1e9:.2f}B" if params >= 1e9 else f"{params/1e6:.0f}M"


def _flops_label(flops: float) -> str:
    """Compact mantissa-exponent FLOPs tag, e.g. 9.2e11."""
    exp = int(np.floor(np.log10(flops)))
    return f"{flops/10**exp:.1f}e{exp}"


def attack_aucs(d: dict, verifier: str) -> list[float]:
    """Per-attack headline AUCs for one verifier (skips attacks lacking it)."""
    out = []
    for a in d["config"]["core_attacks"]:
        cell = d["attacks"].get(a, {}).get(verifier)
        if cell is not None:
            out.append(cell["auc"])
    return out


def _panel(ax, fam: str, models: list[dict]):
    color = FAM_COLORS.get(fam, "#333")
    xs = np.arange(len(models))                       # one slot per claimed model
    labels = [_param_label(d["params"]) for d in models]

    # -- full recompute curve (one point per size) --
    ry, rlo, rhi = [], [], []
    for d in models:
        a = attack_aucs(d, "token_difr")
        ry.append(np.mean(a)); rlo.append(np.min(a)); rhi.append(np.max(a))
    ax.fill_between(xs, rlo, rhi, color=color, alpha=0.13, lw=0)
    ax.plot(xs, ry, "-o", color=color, ms=8, lw=2.2, zorder=3,
            label="full recompute (token_difr)")
    for x, y, d in zip(xs, ry, models):          # recompute FLOPs above each point
        ax.annotate(_flops_label(d["recompute_flops"]), (x, y),
                    textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=7.5, color=color, fontweight="bold")

    # -- cheap same-family proxy, plotted at the CLAIMED model's slot --
    px, py, pflops = [], [], []
    for x, d in zip(xs, models):
        if not d.get("proxy"):
            continue
        a = attack_aucs(d, "surface_stat")
        if a:
            px.append(x); py.append(np.mean(a)); pflops.append(d["proxy"]["flops"])
    if px:
        ax.plot(px, py, "--x", color="#666", ms=9, lw=1.6, zorder=2,
                label="cheap proxy (surface_stat)")
        for x, y, fl in zip(px, py, pflops):     # proxy FLOPs below each point
            ax.annotate(_flops_label(fl), (x, y), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=7, color="#666")

    ax.axhline(0.5, ls=":", color="0.5", lw=1.1)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.45, 1.03)
    ax.set_xlim(-0.35, len(models) - 0.65)
    ax.set_title(f"{FAM_TITLE.get(fam, fam)}  ({len(models)} sizes)",
                 fontsize=11, color=color, fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)


def _summary(ax, by_fam: dict[str, list[dict]]):
    rank_labels = ["small", "mid", "large"]
    for fam in FAM_ORDER:
        models = by_fam.get(fam)
        if not models:
            continue
        ys = [np.mean(attack_aucs(d, "token_difr")) for d in models]
        xs = np.arange(len(models))
        ax.plot(xs, ys, "-o", color=FAM_COLORS[fam], ms=6, lw=2,
                label=FAM_TITLE.get(fam, fam))
    ax.axhline(0.5, ls=":", color="0.5", lw=1.1)
    ax.set_xticks(range(len(rank_labels)))
    ax.set_xticklabels(rank_labels)
    ax.set_ylim(0.45, 1.03)
    ax.set_title("all families: recompute detection vs size rank", fontsize=11,
                 fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)


def build():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_fam = load()
    n_models = sum(len(v) for v in by_fam.values())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    axes = axes.ravel()

    for ax, fam in zip(axes[:5], FAM_ORDER):
        models = by_fam.get(fam)
        if models:
            _panel(ax, fam, models)
        else:
            ax.set_title(f"{FAM_TITLE.get(fam, fam)}  (no data yet)", fontsize=11)
            ax.axis("off")
    _summary(axes[5], by_fam)

    for ax in axes:
        ax.set_xlabel("claimed model  (size ladder, small → large) · labels = verifier FLOPs/seq",
                      fontsize=8.5)
        ax.set_ylabel("detection AUC  (partial AUC @ FPR≤0.5%)", fontsize=9)

    fig.suptitle(
        "Performance of inference verification, across model families and sizes\n"
        f"{n_models} models × {len(by_fam)} families × 6 canonical attacks (H100). "
        "x-axis = the claimed model itself (size ladder). Each panel: full recompute "
        "(solid) vs a cheap same-family proxy (dashed), both plotted at the claimed "
        "model's slot. Band = spread across attacks; line = mean.",
        fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_mega_model_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out, f"({n_models} models across {len(by_fam)} families)")
    plt.close(fig)


if __name__ == "__main__":
    build()
