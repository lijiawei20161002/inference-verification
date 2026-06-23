"""Render the two attack results as labelled figures.

Reuses the exact sweeps from `exp_safe_set` and `exp_seed_free`, so the plots
and the printed tables always agree. Writes PNGs to `docs/figures/`.

Run:  python -m experiments.figures
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import exp_safe_set, exp_seed_free

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"

# One consistent style per defense across both figures.
STYLE = {
    "token_difr":      dict(color="#d62728", marker="o", label="Token-DiFR (seed-synced)"),
    "cross_entropy":   dict(color="#1f77b4", marker="s", label="cross-entropy"),
    "activation_difr": dict(color="#2ca02c", marker="^", label="activation-DiFR (spoofed)"),
    "topk_overlap":    dict(color="#9467bd", marker="D", label="rank / TOPLOC"),
}
CHANCE = dict(ls=":", color="0.5", lw=1.2)


def _finish(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, framealpha=0.9)


def fig_safe_set(path: Path):
    """Seed-synced regime: where the SAFE-set substitution stays hidden, and the
    one-sidedness of Token-DiFR (detection sharpens with batch)."""
    epsilons = (0.01, 0.03, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
    batches = (50, 100, 300, 1000, 3000)
    rows = exp_safe_set.run_sweep(epsilons=epsilons, batch_sizes=batches, n_prompts=120)
    devs = [r.deviation_rate * 100 for r in rows]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    # (a) AUC vs how many tokens are changed, at the operating batch (1000).
    for d in ("token_difr", "cross_entropy", "activation_difr"):
        st = dict(STYLE[d]); st["label"] = st["label"].replace(" (spoofed)", "")
        axA.plot(devs, [r.results[(d, 1000)].auc for r in rows], **st, ms=5)
    axA.axhline(0.5, **CHANCE, label="chance (AUC = 0.5)")
    axA.set_ylim(0.0, 1.02)
    _finish(axA, "(a) Seed-synced verifier: what each defense tolerates",
            "deviation rate  (% of tokens changed from the honest sample)",
            "detection AUC   (batch = 1000 tokens)")

    # (b) Token-DiFR AUC vs batch size for a few deviation rates: one-sidedness.
    shown = (0.01, 0.05, 0.2, 0.5, 1.0)
    cmap = plt.cm.viridis
    sel = [r for r in rows if r.epsilon in shown]
    for i, r in enumerate(sel):
        c = cmap(i / max(len(sel) - 1, 1))
        axB.plot(batches, [r.results[("token_difr", b)].auc for b in batches],
                 marker="o", ms=5, color=c,
                 label=f"{r.deviation_rate*100:4.1f}% dev  (ε={r.epsilon})")
    axB.axhline(0.5, **CHANCE, label="chance")
    axB.set_xscale("log")
    axB.set_ylim(0.4, 1.02)
    _finish(axB, "(b) Token-DiFR is one-sided: detection sharpens with batch",
            "batch size  (tokens averaged into S)  [log]",
            "Token-DiFR detection AUC")

    fig.suptitle("SAFE-set substitution — an indistinguishable token-choice deviation "
                 "(no compute saving)", fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_seed_free(path: Path):
    """Seed-free regime: quant + temp-retune + fingerprint spoof vs every defense,
    and the rank-detector frontier on hideable compute saving."""
    # Same pool size as exp_seed_free so the shaded frontier matches its table.
    quants = (0.1, 0.2, 0.3, 0.4, 0.7, 1.0, 1.4)
    _, rows = exp_seed_free.run_sweep(quants=quants)
    qs = [r.quant for r in rows]
    hidden = [r.quant for r in rows if r.hidden]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    panels = [(axA, "auc", "detection AUC", 0.5, "chance (AUC = 0.5)", (0.0, 1.05)),
              (axB, "tpr_at_1pct", "TPR @ FPR = 1%   (catch rate)", 0.01,
               "chance (= 1% budget)", (-0.03, 1.05))]

    for ax, attr, ylabel, chance, chance_lbl, ylim in panels:
        if hidden:                      # shade the indistinguishable region
            ax.axvspan(min(qs), max(hidden), color="#2ca02c", alpha=0.07,
                       label="indistinguishable to all\nseed-free defenses")
        for d in ("token_difr", "cross_entropy", "activation_difr", "topk_overlap"):
            ax.plot(qs, [getattr(r.spoof[d], attr) for r in rows], **STYLE[d], ms=5)
        ax.plot(qs, [getattr(r.honest_act, attr) for r in rows],
                color="#2ca02c", marker="x", ls="--", ms=6,
                label="activation-DiFR (honest report)")
        ax.axhline(chance, **CHANCE, label=chance_lbl)
        ax.set_ylim(*ylim)
        _finish(ax, f"({'a' if attr == 'auc' else 'b'}) " + ("threshold-free (AUC)"
                if attr == "auc" else "operational (catch rate @ 1% FPR)"),
                "quantization strength q   (→ more compute saved)", ylabel)

    fig.suptitle("Seed-free verifier: quantize + temperature-retune + spoof the "
                 "fingerprint — indistinguishable while genuinely cheating", fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f1 = OUT / "fig1_safe_set.png"
    f2 = OUT / "fig2_seed_free.png"
    print("rendering fig1 (seed-synced / SAFE-set) ...")
    fig_safe_set(f1)
    print("rendering fig2 (seed-free / compute cheat) ...")
    fig_seed_free(f2)
    print(f"wrote:\n  {f1}\n  {f2}")


if __name__ == "__main__":
    main()
