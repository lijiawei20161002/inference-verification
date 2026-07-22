"""The mega figure: verification performance vs cost, ONE curve per model, laid
out systematically across five model families and their size ladders.

Reads every `docs/results/cost_curve/<tag>.json` written by the sweep
(`run_cost_curve_sweep` -> `exp_cost_curve_gpu`) and renders a 2x3 grid:

  * one panel PER FAMILY (qwen, llama, smollm2, pythia, gpt2). Within a panel,
    the SAME curve is drawn for every model size:
      - **full recompute** (`token_difr`): a solid family-coloured line, one
        marker per size, at (recompute FLOPs/seq, detection AUC). As the claimed
        model grows the marker moves RIGHT (more verify cost) and UP (recompute
        catches more) -- the performance/cost tradeoff, made explicit.
      - **cheap same-family proxy** (`surface_stat`): a grey dashed line at
        (proxy FLOPs/seq, detection AUC) -- cheap but near chance, the baseline
        the recompute has to beat.
      - a light band = spread of detection AUC across the six canonical attacks
        (the line itself is their mean).
  * a sixth SUMMARY panel overlaying all five families' recompute curves, so the
    families can be compared head to head on the same cost axis.

Detection AUC is the repo headline: standardized partial AUC @ FPR<=0.5%.
Cost (x, log) is verifier FLOPs/sequence (2*N_non_embed*T) -- deterministic and
hardware-independent, so the curves are comparable across families and sizes.

    python -m experiments.plot_mega_cost_curve
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


def attack_aucs(d: dict, verifier: str) -> list[float]:
    """Per-attack headline AUCs for one verifier (skips attacks lacking it)."""
    out = []
    for a in d["config"]["core_attacks"]:
        cell = d["attacks"].get(a, {}).get(verifier)
        if cell is not None:
            out.append(cell["auc"])
    return out


def recompute_xy(d: dict):
    aucs = attack_aucs(d, "token_difr")
    return d["recompute_flops"], np.mean(aucs), np.min(aucs), np.max(aucs)


def proxy_xy(d: dict):
    if not d.get("proxy"):
        return None
    aucs = attack_aucs(d, "surface_stat")
    if not aucs:
        return None
    return d["proxy"]["flops"], np.mean(aucs), np.min(aucs), np.max(aucs)


def _panel(ax, fam: str, models: list[dict]):
    color = FAM_COLORS.get(fam, "#333")

    # -- full recompute curve (one point per size) --
    rx, ry, rlo, rhi, rlab = [], [], [], [], []
    for d in models:
        x, mean, lo, hi = recompute_xy(d)
        rx.append(x); ry.append(mean); rlo.append(lo); rhi.append(hi)
        rlab.append(_param_label(d["params"]))
    order = np.argsort(rx)
    rx = np.array(rx)[order]; ry = np.array(ry)[order]
    rlo = np.array(rlo)[order]; rhi = np.array(rhi)[order]
    rlab = [rlab[i] for i in order]

    ax.fill_between(rx, rlo, rhi, color=color, alpha=0.13, lw=0)
    ax.plot(rx, ry, "-o", color=color, ms=8, lw=2.2, zorder=3,
            label="full recompute (token_difr)")
    for x, y, lab in zip(rx, ry, rlab):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color=color, fontweight="bold")

    # -- cheap same-family proxy points --
    px, py = [], []
    for d in models:
        pk = proxy_xy(d)
        if pk:
            px.append(pk[0]); py.append(pk[1])
    if px:
        po = np.argsort(px)
        px = np.array(px)[po]; py = np.array(py)[po]
        ax.plot(px, py, "--x", color="#666", ms=8, lw=1.6, zorder=2,
                label="cheap proxy (surface_stat)")

    ax.axhline(0.5, ls=":", color="0.5", lw=1.1)
    ax.set_xscale("log")
    ax.set_ylim(0.45, 1.03)
    ax.set_title(f"{FAM_TITLE.get(fam, fam)}  ({len(models)} sizes)",
                 fontsize=11, color=color, fontweight="bold")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)


def _summary(ax, by_fam: dict[str, list[dict]]):
    for fam in FAM_ORDER:
        models = by_fam.get(fam)
        if not models:
            continue
        pts = sorted(recompute_xy(d)[:2] for d in models)
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-o", color=FAM_COLORS[fam], ms=6, lw=2,
                label=FAM_TITLE.get(fam, fam))
    ax.axhline(0.5, ls=":", color="0.5", lw=1.1)
    ax.set_xscale("log")
    ax.set_ylim(0.45, 1.03)
    ax.set_title("all families: recompute detection vs cost", fontsize=11,
                 fontweight="bold")
    ax.grid(alpha=0.25, which="both")
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
        ax.set_xlabel("verifier cost  —  FLOPs / sequence  (2·N·T, log)", fontsize=9)
        ax.set_ylabel("detection AUC  (partial AUC @ FPR≤0.5%)", fontsize=9)

    fig.suptitle(
        "Performance vs cost of inference verification, across model families and sizes\n"
        f"{n_models} models × {len(by_fam)} families × 6 canonical attacks (H100). "
        "Each curve: full recompute of the claimed model (solid, cost grows with "
        "model size) vs a cheap same-family proxy (dashed). "
        "Band = spread across attacks; line = mean.",
        fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_mega_cost_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out, f"({n_models} models across {len(by_fam)} families)")
    plt.close(fig)


if __name__ == "__main__":
    build()
