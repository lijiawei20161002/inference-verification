"""One big figure spanning every experiment in the repo, grouped by model family.

Every other `plot_*.py` here renders a single experiment. This one is the
mega-view: it re-parses each experiment's already-written result artifact
(the cached JSON/`.txt` logs under `docs/results/`) and lays every result out
as a grid of family-grouped panels, so the whole robustness + efficiency +
detectability story reads in one image instead of ~15 separate PNGs.

Rows = model family (qwen3, llama3.2, smollm2, pythia). Every family gets the
two sweeps that were actually run across all four families:
  (A) `token_difr` full-recompute AUC by attack (`robustness_sweep.json`)
  (B) selective-recompute saving, forward-pass vs sampling-only attacks
      (`selective_robustness.json`)
qwen3 and llama3.2 additionally get the cheap-proxy distance-grid heatmap
that was run with THAT family's flagship as the reference model
(`exp_proxy_distance_grid[_llama].txt`); qwen3 alone gets the extra
qwen-only diagnostics that don't have a per-family analogue yet: cross-family
proxy acceptance (`exp_cross_family_accept_*.txt`), tie-triage recompute
saving by quant bit-width (`exp_tie_triage_margin_*.txt`), and the
speculative-verifier cost/detection tradeoff (`exp_spec_verifier_cost.txt`).
The asymmetry is real, not a layout accident: most single-config diagnostics
in this repo were only ever run on Qwen3.

    .venv/bin/python -m experiments.plot_mega_summary
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"
FIG_DIR = ROOT / "docs" / "figures"

ATTACKS = ["quant_4bit", "kv_fp8", "temp_1.1", "seed_43", "bug_k2", "bug_k32",
           "adv_quant_temp"]
FAM_COLORS = {"qwen3": "#1f77b4", "llama3.2": "#d1902f", "smollm2": "#2ca02c",
              "pythia": "#9467bd"}
FAMILIES = ["qwen3", "llama3.2", "smollm2", "pythia"]
HEADLINE = "token_difr"


def family_of(tag: str) -> str:
    return tag.rsplit("-", 1)[0]


def gmean(xs):
    xs = [x for x in xs if x]
    return float(np.exp(np.mean(np.log(xs)))) if xs else np.nan


# --------------------------------------------------------------------------- loaders
def load_robustness():
    data = json.loads((RESULTS / "robustness_sweep.json").read_text())
    return [r for r in data if "error" not in r]


def load_selective():
    payload = json.loads((RESULTS / "selective_robustness.json").read_text())
    ok = [m for m in payload["models"] if "error" not in m]
    return payload["config"], ok


def parse_proxy_distance_grid(path: Path):
    """First `CHEAP-PROXY DETECTION AUC` table in an `exp_proxy_distance_grid*.txt`
    log -> (ref_name, row_labels, col_labels, matrix)."""
    text = path.read_text()
    ref = re.search(r"claimed M = ([^\s;]+)", text)
    ref_name = ref.group(1).split("/")[-1] if ref else "?"
    lines = text.splitlines()
    header_i = next(i for i, l in enumerate(lines) if "distance group |" in l)
    cols = [c.strip() for c in lines[header_i].split("|", 1)[1].split()]
    # column headers are space-separated but may be multi-word; re-split robustly
    cols = re.split(r"\s{2,}", lines[header_i].split("|", 1)[1].strip())
    rows, mat = [], []
    for l in lines[header_i + 2:]:
        if not l.strip() or l.startswith("-") or "|" not in l:
            break
        label, vals = l.split("|", 1)
        # label = "<model name>  <distance-group description>" (2+ spaces between);
        # keep just the model name for the tick, the group is already implied by
        # the row's position (rows are pre-sorted near -> far from the reference).
        name = re.split(r"\s{2,}", label.strip())[0]
        rows.append(name)
        mat.append([float(v) for v in vals.split()])
    return ref_name, rows, cols, np.array(mat)


def parse_cross_family_accept(path: Path):
    pat = re.compile(
        r"^\s*(?P<name>.+?)\s*\[\s*(?P<cat>[^\]]*?)\s*\]:\s*[\d.]+B\s+"
        r"top1=[\d.]+\s+accept=(?P<accept>[\d.]+)\s+KL=[\d.]+", re.M)
    return [(m["name"], m["cat"].strip(), float(m["accept"]))
            for m in pat.finditer(path.read_text())]


def parse_tie_triage(path: Path):
    pat = re.compile(r"^\s*(\d+)\s+([\d.]+)%\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x\s*$", re.M)
    return [(int(b), float(f), float(t), float(r), float(s))
            for b, f, t, r, s in pat.findall(path.read_text())]


def parse_spec_cost(path: Path):
    text = path.read_text()
    pat = re.compile(
        r"^\s*(\(honest null\)|[a-z_0-9]+)\s*\|\s*([\d.]+) \(TPR\s*[\d.]+\)\s*\|\s*"
        r"[\d.]+ \(TPR\s*[\d.]+\)\s*\|\s*([\d.]+) \(TPR\s*[\d.]+\)", re.M)
    rows = [(c, float(sa), float(td)) for c, sa, td in pat.findall(text)]
    saving = re.search(
        r"cost saving \(recompute / proxy\):\s*([\d.]+)x wall-clock\s*\|\s*([\d.]+)x FLOPs",
        text)
    return rows, (float(saving.group(1)), float(saving.group(2))) if saving else (None, None)


# --------------------------------------------------------------------------- panels
def _heat(ax, M, row_labels, col_labels, title, vmin=0.5, vmax=1.0, cmap="RdYlGn"):
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7.5)
    ax.set_title(title, fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.3,
                    color="white" if v < (vmin + vmax) / 2 else "#111")
    return im


def panel_robustness_heatmap(ax, robustness_ok, family):
    rows = sorted([r for r in robustness_ok if family_of(r["tag"]) == family],
                  key=lambda r: r["params"])
    M = np.array([[r["full"].get(a, {}).get("token_difr", np.nan) for a in ATTACKS]
                  for r in rows])
    _heat(ax, M, [r["tag"] for r in rows], ATTACKS,
          f"{family}: token_difr AUC by attack")


def panel_selective_saving(ax, cfg, selective_ok, family):
    ms = [m for m in selective_ok if family_of(m["tag"]) == family]
    groups = [("forward-pass", cfg["forward_pass"], "#2ca02c"),
              ("sampling-only", cfg["sampling_only"], "#9467bd")]
    labels, vals, colors = [], [], []
    for label, attacks, c in groups:
        s = gmean([m["cells"][a][HEADLINE].get("saving_rel") for m in ms for a in attacks
                   if a in m["cells"]])
        labels.append(label); vals.append(s); colors.append(c)
    x = np.arange(len(labels))
    plot_vals = [0.0 if np.isnan(v) else v for v in vals]
    bars = ax.bar(x, plot_vals, color=colors, alpha=0.85, width=0.55)
    ax.axhline(1.0, ls=":", color="#888", lw=1)
    ymax = max([v for v in plot_vals if v] or [1.0])
    for r, v in zip(bars, vals):
        if np.isnan(v):
            ax.text(r.get_x() + r.get_width() / 2, ymax * 0.04, "n/a\n(no saving)",
                    ha="center", va="bottom", fontsize=7.5, color="#888")
        else:
            ax.text(r.get_x() + r.get_width() / 2, v + ymax * 0.02, f"{v:.2f}×",
                    ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("gmean saving ×", fontsize=8)
    ax.set_title(f"{family}: selective-recompute saving ({len(ms)} models)", fontsize=9)
    ax.grid(axis="y", alpha=0.25)


def panel_proxy_distance(ax, path: Path, family):
    ref_name, rows, cols, M = parse_proxy_distance_grid(path)
    _heat(ax, M, rows, cols, f"{family}: cheap-proxy AUC (ref {ref_name})",
          vmin=0.3, vmax=1.0, cmap="RdYlGn")


def panel_cross_family_accept(ax, path: Path):
    entries = parse_cross_family_accept(path)
    cat_colors = {"same family": "#1f77b4", "cross gen": "#d1902f",
                  "cross domain": "#2ca02c", "cross post": "#9467bd"}
    names = [e[0] for e in entries]
    accepts = [e[2] for e in entries]
    colors = [cat_colors.get(e[1], "#888") for e in entries]
    x = np.arange(len(names))
    ax.bar(x, accepts, color=colors, alpha=0.88, width=0.6)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=6.8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accept rate = 1−TV", fontsize=8)
    ax.set_title("qwen3: cross-family proxy acceptance", fontsize=9)
    handles = [plt_patch(c, l) for l, c in cat_colors.items()]
    ax.legend(handles=handles, fontsize=6.5, loc="upper right")
    ax.grid(axis="y", alpha=0.25)


def plt_patch(color, label):
    import matplotlib.patches as mpatches
    return mpatches.Patch(color=color, label=label)


def panel_tie_triage(ax, path: Path):
    rows = parse_tie_triage(path)
    bits = [r[0] for r in rows]
    saving = [r[4] for r in rows]
    x = np.arange(len(bits))
    bars = ax.bar(x, saving, color="#1f77b4", alpha=0.85, width=0.55)
    for r, v, (b, f, *_ ) in zip(bars, saving, rows):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.05, f"{v:.1f}×\n({f:.0f}% flip)",
                ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([f"{b}-bit" for b in bits], fontsize=8)
    ax.set_ylabel("recompute saving × (to AUC≥0.95)", fontsize=7.8)
    ax.set_title("qwen3: tie-triage saving by quant bit-width", fontsize=9)
    ax.grid(axis="y", alpha=0.25)


def panel_spec_cost(ax, path: Path):
    rows, (wallclock, flops) = parse_spec_cost(path)
    names = [r[0] for r in rows]
    spec = [r[1] for r in rows]
    td = [r[2] for r in rows]
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w / 2, spec, w, color="#d1902f", alpha=0.85, label="spec_accept (cheap)")
    ax.bar(x + w / 2, td, w, color="#1f77b4", alpha=0.85, label="token_difr (recompute)")
    ax.axhline(0.5, ls=":", color="#888", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("detection AUC", fontsize=8)
    title = "qwen3: spec-proxy verifier vs recompute"
    if wallclock:
        title += f"\n(proxy is {wallclock:.1f}× faster, {flops:.1f}× fewer FLOPs)"
    ax.set_title(title, fontsize=8.5)
    ax.legend(fontsize=6.8, loc="upper left")
    ax.grid(axis="y", alpha=0.25)


# --------------------------------------------------------------------------- figure
def build():
    import matplotlib.pyplot as plt

    robustness_ok = load_robustness()
    cfg, selective_ok = load_selective()

    fig = plt.figure(figsize=(19, 17))
    gs = fig.add_gridspec(5, 6, hspace=0.75, wspace=0.55)

    # -- qwen3 (rows 0-1): heatmap, selective saving, proxy-distance / cross-family
    #    accept, tie-triage, spec-verifier cost
    panel_robustness_heatmap(fig.add_subplot(gs[0, 0:2]), robustness_ok, "qwen3")
    panel_selective_saving(fig.add_subplot(gs[0, 2:4]), cfg, selective_ok, "qwen3")
    panel_proxy_distance(fig.add_subplot(gs[0, 4:6]), RESULTS / "exp_proxy_distance_grid.txt",
                          "qwen3")
    panel_cross_family_accept(fig.add_subplot(gs[1, 0:2]),
                               RESULTS / "exp_cross_family_accept_qwen3-4b.txt")
    panel_tie_triage(fig.add_subplot(gs[1, 2:4]), RESULTS / "exp_tie_triage_margin_qwen3-1.7b.txt")
    panel_spec_cost(fig.add_subplot(gs[1, 4:6]), RESULTS / "exp_spec_verifier_cost.txt")

    # -- llama3.2 (row 2): heatmap, selective saving, proxy-distance
    panel_robustness_heatmap(fig.add_subplot(gs[2, 0:2]), robustness_ok, "llama3.2")
    panel_selective_saving(fig.add_subplot(gs[2, 2:4]), cfg, selective_ok, "llama3.2")
    panel_proxy_distance(fig.add_subplot(gs[2, 4:6]),
                          RESULTS / "exp_proxy_distance_grid_llama.txt", "llama3.2")

    # -- smollm2 (row 3) / pythia (row 4): heatmap + selective saving only
    panel_robustness_heatmap(fig.add_subplot(gs[3, 0:3]), robustness_ok, "smollm2")
    panel_selective_saving(fig.add_subplot(gs[3, 3:6]), cfg, selective_ok, "smollm2")
    panel_robustness_heatmap(fig.add_subplot(gs[4, 0:3]), robustness_ok, "pythia")
    panel_selective_saving(fig.add_subplot(gs[4, 3:6]), cfg, selective_ok, "pythia")

    # family-group labels down the left margin, colour-matched to FAM_COLORS
    row_y = {"qwen3": 0.895, "llama3.2": 0.60, "smollm2": 0.395, "pythia": 0.205}
    for fam, y in row_y.items():
        fig.text(0.005, y, fam.upper(), rotation=90, va="center", ha="left",
                  fontsize=13, fontweight="bold", color=FAM_COLORS[fam])

    fig.suptitle(
        "Inference-verification: every experiment, grouped by model family\n"
        f"{len({r['tag'] for r in robustness_ok})} real models across "
        f"{len(FAMILIES)} families × {len(ATTACKS)} attacks, H100. "
        "qwen3 carries the extra single-family diagnostics (proxy distance, "
        "cross-family acceptance, tie-triage, spec-verifier cost) because "
        "that's the only family they were run on.",
        fontsize=12.5)
    fig.tight_layout(rect=(0.018, 0, 1, 0.955))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_mega_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    build()


if __name__ == "__main__":
    main()
