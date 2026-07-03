"""Figures for the acceptance-rate fingerprint (docs/figures/fig_accept_rate_*.png).

Illustrates the one no-recompute lever the speculative-decoding trace adds over
generic black-box statistics: the realized acceptance rate = 1 - TV(p-hat, q),
anchored on the trusted cheap draft q.

Three panels:
  A. Mechanism -- per-trace acceptance-rate distributions shift left as the target
     is quantized more (TV(p-hat,q) grows), separating from the honest reference.
  B. Detection AUC vs quantization strength -- accept-rate fingerprint tracks (and
     slightly leads) the generic entropy fingerprint; both climb with sigma; the
     recompute spot-check is flat at 1.0.
  C. Why it survives the temperature-retune evasion -- in (entropy, accept-rate)
     space, adv_quant_temp is tuned back INTO the honest entropy band (an entropy
     detector goes blind) but stays OUTSIDE the honest accept-rate band (the
     draft-anchored detector still separates it).

Run:  python -m experiments.plot_accept_rate_fingerprint
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ivgym import spec_decode as sd
from ivgym.core import SamplingSpec
from ivgym.metrics import roc_auc

# ---- validated, colorblind-safe palette (dataviz skill, light surface) -------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e7e6e2"
BLUE, AQUA, YELLOW, ORANGE, VIOLET = "#2a78d6", "#1baf7a", "#eda100", "#eb6834", "#4a3aa7"
SEQ_BLUE = ["#9ec5f4", "#5598e7", "#256abf", "#0d366b"]  # sequential ramp for sigma

N = 120
N_POS = 200
VOCAB = 64
AGREE = 0.8


def build(cheat, seed0, n=N):
    spec = SamplingSpec()
    out = []
    for t in range(n):
        rng = np.random.default_rng(1_000_003 * seed0 + t)
        pos = sd.synthetic_positions(rng, N_POS, vocab=VOCAB, agreement=AGREE)
        out.append(sd.generate_trace(rng, cheat, pos, spec))
    return out


def accepts(traces):
    return np.array([sd.trace_features(t)["accept_rate"] for t in traces])


def entropies(traces):
    return np.array([sd.trace_features(t)["mean_entropy"] for t in traces])


def feat_auc(fp, feature, honest_null, cheat):
    h = np.array([fp.score(t)[feature] for t in honest_null])
    c = np.array([fp.score(t)[feature] for t in cheat])
    return roc_auc(h, c)


def spot_auc(honest_null, cheat):
    spot = sd.get_check("target_spotcheck")
    h = np.array([spot.score(t, sd.make_oracle(t)) for t in honest_null])
    c = np.array([spot.score(t, sd.make_oracle(t)) for t in cheat])
    return roc_auc(h, c)


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def main():
    print("building traces ...", flush=True)
    honest_fit = build(sd.get_cheat("honest"), 1)
    honest_null = build(sd.get_cheat("honest"), 2)
    fp = sd.ReferenceFingerprint().fit(honest_fit)
    h_acc, h_ent = accepts(honest_null), entropies(honest_null)
    h_acc_m, h_acc_s = h_acc.mean(), h_acc.std()
    h_ent_m, h_ent_s = h_ent.mean(), h_ent.std()

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.5, 4.6), facecolor=SURFACE)

    # ---- Panel A: acceptance-rate distributions shift with quant strength ----
    style(axA)
    sig_show = [0.25, 0.5, 1.0]
    bins = np.linspace(0.45, 0.9, 40)
    axA.hist(h_acc, bins=bins, color=INK2, alpha=0.28, label="honest")
    axA.axvline(h_acc_m, color=INK, lw=2, label="honest mean")
    for sig, col in zip(sig_show, SEQ_BLUE[1:]):
        tr = build(sd.QuantTarget(name=f"q{sig}", sigma=sig), 10 + int(sig * 10))
        axA.hist(accepts(tr), bins=bins, color=col, alpha=0.55,
                 label=f"quant σ={sig}")
    axA.set_xlabel("acceptance rate  = 1 − TV(p̂, q)", color=INK2, fontsize=10)
    axA.set_ylabel("traces", color=INK2, fontsize=10)
    axA.set_title("A  Quantization lowers the acceptance rate",
                  color=INK, fontsize=11, loc="left", fontweight="bold")
    axA.legend(frameon=False, fontsize=8.5, labelcolor=INK)

    # ---- Panel B: detection AUC vs quant strength ---------------------------
    style(axB)
    sigmas = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    ent_auc, acc_auc, spot = [], [], []
    for sig in sigmas:
        tr = build(sd.QuantTarget(name=f"q{sig}", sigma=sig), 400 + int(sig * 100))
        ent_auc.append(feat_auc(fp, "mean_entropy", honest_null, tr))
        acc_auc.append(feat_auc(fp, "accept_rate", honest_null, tr))
        spot.append(spot_auc(honest_null, tr))
    axB.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 4)), alpha=0.7)
    axB.text(sigmas[0], 0.515, "chance", color=INK2, fontsize=8)
    series = [("target_spotcheck (recompute)", spot, BLUE, "o"),
              ("accept-rate fingerprint", acc_auc, AQUA, "s"),
              ("entropy fingerprint", ent_auc, YELLOW, "^")]
    for name, ys, col, mk in series:
        axB.plot(sigmas, ys, color=col, lw=2, marker=mk, ms=7, label=name)
    axB.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    axB.set_xlabel("quantization strength  σ (logit-error scale)", color=INK2, fontsize=10)
    axB.set_ylabel("detection AUC", color=INK2, fontsize=10)
    axB.set_ylim(0.4, 1.04)
    axB.set_xlim(sigmas[0] - 0.03, sigmas[-1] + 0.03)
    axB.set_title("B  Accept-rate fingerprint leads the entropy one",
                  color=INK, fontsize=11, loc="left", fontweight="bold")

    # ---- Panel C: why accept-rate survives temperature retune ---------------
    style(axC)
    # tune adv temperature to match honest entropy (sigma = 1.0)
    best_T, best_gap = 1.0, 1e9
    for T in np.linspace(0.6, 2.0, 29):
        tr = build(sd.AdvQuantTemp(name="adv", sigma=1.0, temp=float(T)), 90, n=40)
        gap = abs(entropies(tr).mean() - h_ent_m)
        if gap < best_gap:
            best_gap, best_T = gap, float(T)
    plain = build(sd.QuantTarget(name="q", sigma=1.0), 5)
    adv = build(sd.AdvQuantTemp(name="adv", sigma=1.0, temp=best_T), 6)

    # honest reference bands (mean ± 2σ) on each axis
    axC.axvspan(h_ent_m - 2 * h_ent_s, h_ent_m + 2 * h_ent_s, color=INK2, alpha=0.10)
    axC.axhspan(h_acc_m - 2 * h_acc_s, h_acc_m + 2 * h_acc_s, color=INK2, alpha=0.10)
    for name, tr, col in [("honest", honest_null, BLUE),
                          ("quant (plain)", plain, ORANGE),
                          (f"adv_quant_temp (T={best_T:.2f})", adv, VIOLET)]:
        axC.scatter(entropies(tr), accepts(tr), s=26, color=col, alpha=0.65,
                    edgecolor=SURFACE, linewidth=0.5, label=name)
    axC.set_xlabel("mean target entropy  (entropy fingerprint axis)", color=INK2, fontsize=10)
    axC.set_ylabel("acceptance rate  (accept-rate fingerprint axis)", color=INK2, fontsize=10)
    axC.set_title("C  Temp-retune hides in entropy, not in accept-rate",
                  color=INK, fontsize=11, loc="left", fontweight="bold")
    axC.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower left")
    # annotate the shaded honest bands
    axC.text(0.98, 0.97, "grey bands = honest ± 2σ", transform=axC.transAxes,
             ha="right", va="top", color=INK2, fontsize=8)

    fig.suptitle(
        "Acceptance-rate fingerprint: detecting a quantized target without "
        "per-request recompute, via the trusted draft",
        color=INK, fontsize=12.5, fontweight="bold", x=0.5, y=1.02)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    combined = out / "fig_accept_rate_fingerprint.png"
    fig.savefig(combined, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {combined}")

    # standalone Panel C (the key mechanism figure)
    figC, ax = plt.subplots(figsize=(6.2, 5.2), facecolor=SURFACE)
    style(ax)
    ax.axvspan(h_ent_m - 2 * h_ent_s, h_ent_m + 2 * h_ent_s, color=INK2, alpha=0.10)
    ax.axhspan(h_acc_m - 2 * h_acc_s, h_acc_m + 2 * h_acc_s, color=INK2, alpha=0.10)
    for name, tr, col in [("honest", honest_null, BLUE),
                          ("quant (plain)", plain, ORANGE),
                          (f"adv_quant_temp (T={best_T:.2f})", adv, VIOLET)]:
        ax.scatter(entropies(tr), accepts(tr), s=30, color=col, alpha=0.65,
                   edgecolor=SURFACE, linewidth=0.5, label=name)
    ax.set_xlabel("mean target entropy  (entropy-fingerprint axis)", color=INK2, fontsize=10)
    ax.set_ylabel("acceptance rate = 1 − TV(p̂, q)  (accept-rate axis)", color=INK2, fontsize=10)
    ax.set_title("adv_quant_temp hides in the honest entropy band\n"
                 "but not in the honest acceptance-rate band",
                 color=INK, fontsize=11, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    ax.text(0.98, 0.02, "grey bands = honest ± 2σ", transform=ax.transAxes,
            ha="right", va="bottom", color=INK2, fontsize=8)
    figC.tight_layout()
    mech = out / "fig_accept_rate_mechanism.png"
    figC.savefig(mech, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {mech}")

    # report the numbers behind the figures
    print("\nnumbers:")
    print(f"  honest accept={h_acc_m:.3f}±{h_acc_s:.3f}  entropy={h_ent_m:.3f}±{h_ent_s:.3f}")
    print(f"  Panel B sigmas={sigmas}")
    print(f"    entropy_fp AUC = {[round(x,3) for x in ent_auc]}")
    print(f"    accept_fp  AUC = {[round(x,3) for x in acc_auc]}")
    print(f"    spotcheck  AUC = {[round(x,3) for x in spot]}")
    print(f"  adv T={best_T:.3f}: entropy_fp={feat_auc(fp,'mean_entropy',honest_null,adv):.3f} "
          f"accept_fp={feat_auc(fp,'accept_rate',honest_null,adv):.3f} "
          f"spotcheck={spot_auc(honest_null,adv):.3f}")


if __name__ == "__main__":
    main()
