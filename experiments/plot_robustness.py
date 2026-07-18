"""Turn the robustness sweep into figures (not just the markdown tables).

`exp_robustness_gpu.py` runs the unified verification algorithm against every
attack on a matrix of real model families/sizes and writes RAW AUCs to
`docs/results/robustness_sweep.json`; `analyze_robustness.py` renders orientation-
correct markdown tables. This script renders the *same* data as figures, so the
robustness story is visual like every other experiment in the repo:

  docs/figures/fig_robustness_heatmap.png
      Two model x attack heatmaps: `token_difr` full-recompute AUC (does the
      one recompute detector catch every attack on every family/size?) and the
      best-verifier detectability per cell (does the unified registry, taking the
      strongest verifier per attack, close the gaps token_difr leaves?).

  docs/figures/fig_robustness_summary.png
      Four synthesis panels: per-verifier detectability across all cells
      (mean + worst), attack difficulty ranking, the size trend within each
      family, and information-directed selective recompute vs full on the hard
      forward-pass attacks.

Pure numpy + matplotlib on the cached JSON -- no GPU, no model reload:

    .venv/bin/python -m experiments.plot_robustness              # default json
    .venv/bin/python -m experiments.plot_robustness path.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ATTACKS = ["quant_4bit", "kv_fp8", "temp_1.1", "seed_43", "bug_k2", "bug_k32",
           "adv_quant_temp"]
TIER1 = {"token_difr", "cross_entropy", "activation_difr", "token_toploc"}
HARD = ["quant_4bit", "kv_fp8", "bug_k2", "adv_quant_temp"]
ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "docs" / "results" / "robustness_sweep.json"
FIG_DIR = ROOT / "docs" / "figures"

# family -> stable colour, shared across panels
FAM_COLORS = {
    "qwen3": "#1f77b4", "llama3.2": "#d1902f", "smollm2": "#2ca02c",
    "pythia": "#9467bd",
}


def detect(auc: float, verifier: str) -> float:
    """Orientation-correct detectability: raw AUC for Tier-1 recompute verifiers,
    max(auc, 1-auc) for Tier-0 black-box detectors (a reversed signal still
    separates) -- matches analyze_robustness.detect and the DiFR convention."""
    if verifier in TIER1:
        return auc
    return max(auc, 1.0 - auc)


def family_of(tag: str) -> str:
    return tag.rsplit("-", 1)[0]


def _load(path: Path):
    data = json.loads(path.read_text())
    ok = [r for r in data if "error" not in r]
    if not ok:
        raise SystemExit(f"no successful models in {path}")
    return data, ok, ok[0]["verifiers"]


def _heat(ax, M, row_labels, col_labels, title, cmap, vmin, vmax):
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8.5)
    ax.set_title(title, fontsize=10.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v < (vmin + vmax) / 2 else "#111")
    return im


# --------------------------------------------------------------------------- fig 1
def fig_heatmap(ok, verifiers):
    import matplotlib.pyplot as plt

    tags = [r["tag"] for r in ok]
    # token_difr AUC per cell
    Mtd = np.array([[r["full"].get(a, {}).get("token_difr", np.nan) for a in ATTACKS]
                    for r in ok])
    # best-verifier detectability per cell (orientation-correct over the registry)
    Mbest = np.array([[max((detect(r["full"][a][d], d) for d in verifiers
                            if d in r["full"].get(a, {})), default=np.nan)
                       for a in ATTACKS] for r in ok])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    im0 = _heat(axes[0], Mtd, tags, ATTACKS,
                f"`token_difr` full-recompute AUC   (min {np.nanmin(Mtd):.2f})",
                "RdYlGn", 0.5, 1.0)
    im1 = _heat(axes[1], Mbest, tags, ATTACKS,
                f"best verifier per cell — detectability   (min {np.nanmin(Mbest):.2f})",
                "RdYlGn", 0.5, 1.0)
    for im, ax in ((im0, axes[0]), (im1, axes[1])):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
            "detection AUC", fontsize=9)
    axes[0].set_ylabel("reference model  (family × size)", fontsize=10)
    fig.suptitle(
        "Robustness of the unified verification algorithm across families, sizes, attacks\n"
        f"{len(ok)} real models × {len(ATTACKS)} attacks on an H100.  Left: does one recompute "
        "detector (token_difr) generalise?  Right: does the registry's best-per-attack close its gaps?",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG_DIR / "fig_robustness_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    plt.close(fig)


# --------------------------------------------------------------------------- fig 2
def _panel_verifiers(ax, ok, verifiers):
    means, mins, names = [], [], []
    for d in verifiers:
        cells = [detect(r["full"][a][d], d) for r in ok for a in ATTACKS
                 if d in r["full"].get(a, {})]
        if cells:
            names.append(d)
            means.append(float(np.mean(cells)))
            mins.append(float(np.min(cells)))
    order = np.argsort(means)
    names = [names[i] for i in order]
    means = [means[i] for i in order]
    mins = [mins[i] for i in order]
    y = np.arange(len(names))
    colors = ["#1f77b4" if n in TIER1 else "#d1902f" for n in names]
    ax.barh(y, means, color=colors, alpha=0.85, label="mean")
    ax.plot(mins, y, "o", color="#c0392b", ms=5, label="worst cell")
    ax.axvline(0.5, ls=":", color="#888", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n} {'(T1)' if n in TIER1 else '(T0)'}" for n in names],
                       fontsize=8.5)
    ax.set_xlim(0.4, 1.02)
    ax.set_xlabel("detectability across all model × attack cells", fontsize=9)
    ax.set_title("Per-verifier robustness (blue=Tier-1 recompute, orange=Tier-0)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25)


def _panel_attacks(ax, ok):
    rows = []
    for a in ATTACKS:
        vals = [r["full"][a]["token_difr"] for r in ok
                if "token_difr" in r["full"].get(a, {})]
        if vals:
            rows.append((a, float(np.mean(vals)), float(np.min(vals)),
                         float(np.max(vals))))
    rows.sort(key=lambda x: x[1])
    names = [r[0] for r in rows]
    mean = np.array([r[1] for r in rows])
    lo = mean - np.array([r[2] for r in rows])
    hi = np.array([r[3] for r in rows]) - mean
    y = np.arange(len(names))
    ax.barh(y, mean, xerr=[lo, hi], color="#2ca02c", alpha=0.85,
            error_kw=dict(ecolor="#555", lw=1, capsize=3))
    ax.axvline(0.5, ls=":", color="#888", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlim(0.4, 1.02)
    ax.set_xlabel("`token_difr` AUC  (mean over models; bars=min…max)", fontsize=9)
    ax.set_title("Attack difficulty (lower = harder to catch)", fontsize=10)
    ax.grid(axis="x", alpha=0.25)


def _panel_size(ax, ok, verifiers):
    for fam in sorted({family_of(r["tag"]) for r in ok}):
        pts = []
        for r in ok:
            if family_of(r["tag"]) != fam:
                continue
            td = [r["full"][a]["token_difr"] for a in ATTACKS
                  if "token_difr" in r["full"].get(a, {})]
            pts.append((r["params"] / 1e9, float(np.mean(td))))
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        c = FAM_COLORS.get(fam, "#333")
        ax.plot(xs, ys, "-o", color=c, ms=6, lw=1.8, label=fam)
    ax.set_xscale("log")
    ax.axhline(0.5, ls=":", color="#888", lw=1)
    ax.set_xlabel("reference model size (B params, log)", fontsize=9)
    ax.set_ylabel("mean `token_difr` AUC over attacks", fontsize=9)
    ax.set_title("Size trend within family (does a bigger reference help?)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25, which="both")


def _panel_selective(ax, ok):
    budgets = sorted({float(b) for r in ok for b in r.get("selective", {})})
    labels = ["full\n(100%)"]
    means = [float(np.mean([r["full"][a]["token_difr"] for r in ok for a in HARD
                            if "token_difr" in r["full"].get(a, {})]))]
    for b in budgets:
        sv, rr = [], []
        for r in ok:
            sel = r.get("selective", {})
            cell = sel.get(str(b), sel.get(b, {}))
            rr.append(r.get("realized_recompute_ratio", {}).get(
                str(b), r.get("realized_recompute_ratio", {}).get(b, b)))
            for a in HARD:
                if a in cell and "token_difr" in cell[a]:
                    sv.append(cell[a]["token_difr"])
        if sv:
            labels.append(f"selective\n({np.mean(rr)*100:.0f}%)")
            means.append(float(np.mean(sv)))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, color=["#1f77b4"] + ["#2ca02c"] * (len(labels) - 1),
                  alpha=0.85, width=0.6)
    for rect, v in zip(bars, means):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", fontsize=8.5)
    ax.axhline(0.5, ls=":", color="#888", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("mean `token_difr` AUC", fontsize=9)
    ax.set_title(f"Selective recompute vs full on hard attacks {HARD}", fontsize=9.5)
    ax.grid(axis="y", alpha=0.25)


def fig_summary(ok, verifiers):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
    _panel_verifiers(axes[0, 0], ok, verifiers)
    _panel_attacks(axes[0, 1], ok)
    _panel_size(axes[1, 0], ok, verifiers)
    _panel_selective(axes[1, 1], ok)
    fig.suptitle(
        "Unified verification algorithm — robustness synthesis "
        f"({len(ok)} real models × {len(ATTACKS)} attacks, H100)",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG_DIR / "fig_robustness_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    data, ok, verifiers = _load(path)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"source: {path}  ({len(ok)}/{len(data)} models ok)")
    fig_heatmap(ok, verifiers)
    fig_summary(ok, verifiers)


if __name__ == "__main__":
    main()
