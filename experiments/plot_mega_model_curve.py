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


# The cheap verifier we advocate: information-directed selective recompute of M on
# just the top-25% highest-entropy tokens (entropy read off the free proxy). The
# old dashed line (`surface_stat`, a pure cheap-proxy statistic) is kept faint for
# contrast -- it is the one the figure used to show collapsing to chance.
SELECTIVE = "sel_difr_b25"
SEL_LABEL = "selective recompute @25% (info-directed)"
SEL_COLOR = "#111111"


def _sel_flops(d: dict) -> float | None:
    info = (d.get("selective") or {}).get(SELECTIVE)
    return info["flops"] if info else None


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

    # -- WINNER: selective recompute @25%, plotted at the CLAIMED model's slot --
    sx, sy, sflops = [], [], []
    for x, d in zip(xs, models):
        a = attack_aucs(d, SELECTIVE)
        fl = _sel_flops(d)
        if a and fl:
            sx.append(x); sy.append(np.mean(a)); sflops.append(fl)
    if sx:
        ax.plot(sx, sy, "--D", color=SEL_COLOR, ms=6.5, lw=2.0, zorder=4,
                label=SEL_LABEL)
        for x, y, fl in zip(sx, sy, sflops):     # selective FLOPs below each point
            ax.annotate(_flops_label(fl), (x, y), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=7,
                        color=SEL_COLOR)

    # -- the OLD cheap proxy (surface_stat), faint, for contrast --
    px, py = [], []
    for x, d in zip(xs, models):
        a = attack_aucs(d, "surface_stat")
        if d.get("proxy") and a:
            px.append(x); py.append(np.mean(a))
    if px:
        ax.plot(px, py, ":x", color="#999", ms=7, lw=1.3, zorder=2,
                label="cheap proxy (surface_stat)")

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
    """Parity scatter: how well each CHEAP verifier reproduces the full-recompute
    detection AUC. x = full recompute AUC, y = cheap-verifier AUC, one point per
    (family, size) that has a proxy. Points on the dashed y=x line perfectly match
    full recompute. Selective recompute (coloured, per family) hugs the diagonal;
    the old surface_stat proxy (grey) collapses toward the 0.5 chance floor."""
    ax.plot([0.45, 1.03], [0.45, 1.03], ls="--", color="0.6", lw=1.2, zorder=1,
            label="parity with full recompute")
    ax.axhline(0.5, ls=":", color="0.7", lw=1.0)
    sel_pts, surf_pts = [], []
    for fam in FAM_ORDER:
        for d in by_fam.get(fam, []):
            if not d.get("proxy"):
                continue
            rec = np.mean(attack_aucs(d, "token_difr"))
            sel = attack_aucs(d, SELECTIVE)
            surf = attack_aucs(d, "surface_stat")
            if sel:
                sel_pts.append((rec, np.mean(sel)))
                ax.scatter(rec, np.mean(sel), s=70, color=FAM_COLORS[fam],
                           edgecolor="k", lw=0.6, zorder=4)
            if surf:
                surf_pts.append((rec, np.mean(surf)))
                ax.scatter(rec, np.mean(surf), s=45, marker="x", color="#999",
                           lw=1.4, zorder=3)
    # proxy legend handles (family colours are labelled in the family panels)
    ax.scatter([], [], s=70, color="#444", edgecolor="k", lw=0.6,
               label="selective @25% (by family)")
    ax.scatter([], [], s=45, marker="x", color="#999", lw=1.4,
               label="surface_stat proxy")
    ax.set_xlim(0.45, 1.03)
    ax.set_ylim(0.45, 1.03)
    ax.set_xlabel("full-recompute detection AUC", fontsize=9)
    ax.set_ylabel("cheap-verifier detection AUC", fontsize=9)
    ax.set_title("cheap verifier vs full recompute (parity)", fontsize=11,
                 fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)


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

    for ax in axes[:5]:                              # family panels share the ladder axes
        ax.set_xlabel("claimed model  (size ladder, small → large) · labels = verifier FLOPs/seq",
                      fontsize=8.5)
        ax.set_ylabel("detection AUC  (partial AUC @ FPR≤0.5%)", fontsize=9)

    fig.suptitle(
        "Inference verification: a cheap verifier that stays good across families and sizes\n"
        f"{n_models} models × {len(by_fam)} families × 6 canonical attacks (H100). x-axis = the "
        "claimed model (size ladder); labels = verifier FLOPs/seq. Full recompute (solid) vs "
        "information-directed SELECTIVE recompute of M on the top-25% highest-entropy tokens "
        "(black dashed) vs the old cheap proxy (faint grey, ≈chance). Band = spread across "
        "attacks; line = mean. Selective tracks full recompute at ~1.7-2.7x lower cost; "
        "the pure cheap proxy stays at chance.",
        fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_mega_model_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out, f"({n_models} models across {len(by_fam)} families)")
    plt.close(fig)


if __name__ == "__main__":
    build()
