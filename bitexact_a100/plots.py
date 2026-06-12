"""Render the experiment results (results/results.json) to figures in results/."""
import json, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def fig_e1_e4(r, path):
    """Summary bar chart: distinct outputs over repeated identical runs.
    1 distinct = deterministic; >1 = nondeterministic."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels, vals, colors = [], [], []
    for k, v in r["E1_within_condition"].items():
        labels.append(f"GEMM\n{k}"); vals.append(v["distinct"])
        colors.append("#2ca02c")
    for k, v in r["E4_genuine_nondeterminism"].items():
        labels.append(f"{k}\n(atomicAdd)"); vals.append(v["distinct"])
        colors.append("#d62728")
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(1, ls="--", c="gray", lw=1)
    ax.set_ylabel("distinct bit-patterns over 10 identical runs")
    ax.set_title("Within-condition determinism: GEMMs (green) vs genuine\n"
                 "non-determinism from float atomicAdd (red)")
    for b, val in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.15, str(val),
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 11)
    ax.text(0.02, 0.95, "1 = bit-exact reproducible", transform=ax.transAxes,
            fontsize=9, color="gray", va="top")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_e2(r, path):
    """Same floats summed three ways -> different bits."""
    e = r["E2_non_associativity"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    order = ["tree", "L->R", "R->L"]
    vals = [e["values"][k] for k in order]
    bars = ax.bar(order, vals, color=["#1f77b4", "#ff7f0e", "#9467bd"],
                  edgecolor="black", linewidth=0.6)
    ax.set_ylabel("computed sum")
    ax.set_title("Non-associativity of float reduction (root cause)\n"
                 f"identical $2^{{20}}$ values summed 3 ways -> "
                 f"{e['distinct_bit_patterns']} distinct bit patterns")
    span = max(vals) - min(vals)
    ax.set_ylim(min(vals) - 0.12 * span, max(vals) + 0.22 * span)
    for k, b, v in zip(order, bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v / 2,
                f"{v:.3f}\n{e['hashes'][k]}",
                ha="center", va="center", fontsize=10, color="white", fontweight="bold")
    ax.axhline(0, c="black", lw=0.8)
    ax.text(0.5, 0.97, "same numbers, same hardware — only the summation ORDER differs",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color="gray")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_e3(r, path):
    """Heatmaps: which (batch, K) shapes change the bits vs B=1, and L2 magnitude."""
    g = r["E3_noninvariance_grid"]
    Bs, Ks = g["batch_sizes"], g["K_values"]
    diff = np.array(g["diff_from_b1"])      # rows=K, cols=B
    l2 = np.array(g["l2"])
    nonrepro = np.array(g["nonreproducible"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # Left: categorical map — same bits as B=1 (green) vs differs (orange).
    cmap = ListedColormap(["#2ca02c", "#ff7f0e"])
    ax = axes[0]
    ax.imshow(diff, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(Bs))); ax.set_xticklabels(Bs)
    ax.set_yticks(range(len(Ks))); ax.set_yticklabels(Ks)
    ax.set_xlabel("batch size B"); ax.set_ylabel("contraction length K")
    ax.set_title("Bits of element-0 output vs B=1\n"
                 "green = identical (equivalence class), orange = differs")
    for i in range(len(Ks)):
        for j in range(len(Bs)):
            mark = "x" if nonrepro[i, j] else ("=" if diff[i, j] == 0 else "Δ")
            ax.text(j, i, mark, ha="center", va="center",
                    color="white", fontsize=11, fontweight="bold")

    # Right: continuous L2 magnitude of the divergence.
    ax = axes[1]
    im = ax.imshow(l2, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(Bs))); ax.set_xticklabels(Bs)
    ax.set_yticks(range(len(Ks))); ax.set_yticklabels(Ks)
    ax.set_xlabel("batch size B"); ax.set_ylabel("contraction length K")
    ax.set_title("L2 distance of element-0 output vs B=1\n(deterministic, but shape-dependent)")
    for i in range(len(Ks)):
        for j in range(len(Bs)):
            ax.text(j, i, f"{l2[i, j]:.1e}", ha="center", va="center",
                    color="white", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="L2")
    fig.suptitle("Non-invariance: same inputs, different shape => different bits "
                 "(every cell individually reproducible)", fontsize=12, y=1.02)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_llm(path, llm_path):
    """Real-LLM batch-size non-invariance: L2 of element-0 hidden state vs B,
    colored by whether the bit-fingerprint matches B=1."""
    import os
    if not os.path.exists(llm_path):
        return False
    with open(llm_path) as f:
        d = json.load(f)
    rows = d["batch_sweep"]
    Bs = [r["B"] for r in rows]
    l2 = [r["l2_vs_b1"] for r in rows]
    same = [r["same_as_b1"] for r in rows]
    colors = ["#2ca02c" if s else "#ff7f0e" for s in same]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bars = ax.bar([str(b) for b in Bs], l2, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xlabel("batch size B  (element-0 tokens held FIXED)")
    ax.set_ylabel("L2 of last-layer hidden state vs B=1")
    ax.set_title(f"Real LLM ({d['model']}, {d['dtype']}): batch-size non-invariance\n"
                 "green = bit-identical to B=1, orange = different bits "
                 "(all individually reproducible)")
    for b, r in zip(bars, rows):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                ("=" if r["same_as_b1"] else "Δ"),
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    wc = d["within_condition_distinct"]
    ax.text(0.02, 0.95, f"within-condition: {wc} distinct fingerprint over 5 runs "
            f"({'bit-exact' if wc == 1 else 'diverged'})",
            transform=ax.transAxes, va="top", fontsize=9, color="gray")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.json")
    ap.add_argument("--llm", default="results/llm_results.json")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    with open(args.results) as f:
        r = json.load(f)
    fig_e1_e4(r, f"{args.outdir}/fig1_determinism_vs_atomics.png")
    fig_e2(r, f"{args.outdir}/fig2_non_associativity.png")
    fig_e3(r, f"{args.outdir}/fig3_noninvariance_grid.png")
    if fig_llm(f"{args.outdir}/fig4_llm_noninvariance.png", args.llm):
        print("wrote LLM figure")
    print(f"wrote figures to {args.outdir}/")


if __name__ == "__main__":
    main()
