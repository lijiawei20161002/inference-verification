"""Figure for the baseline-headroom diagnostic.

Renders `docs/results/baseline_headroom.json` (written by
`exp_baseline_headroom_gpu.py`) into `docs/figures/fig_baseline_headroom.png`.
Pure numpy + matplotlib, no GPU, so the figure can be iterated without re-running
the 80-minute sweep.

Three panels, one per candidate explanation the sweep separates:

  A  AUC vs the batch/pool RATIO at fixed batch -- the accounting artifact. Same
     model, same attack, same per-token scores throughout: only the size of the
     token pool the batches are resampled from changes. The repo's own historical
     configurations are marked on the x axis, which is the point of the panel.
  B  AUC vs BATCH SIZE with the ratio pinned at 10% -- legitimate power, plus the
     Gaussian prediction `d' * sqrt(b)` from the measured per-token effect size.
  C  Per-token d' vs attack strength -- the ladder, and where a rung has to sit
     for a properly-ratioed 8k-token pool to reach AUC 0.90 at all.

    python -m experiments.plot_headroom
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import signal

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
RES_DIR = ROOT / "docs" / "results"

# Slots 1-3 of the same validated reference palette `plot_triage.py` uses, taken
# in the same documented fixed order so a colour means the same thing across the
# two figures of this write-up. Identity is never colour-alone: every series is
# direct-labeled AND, where two series share a hue by design (panel A, where hue
# encodes batch size), the second is dashed and the legend states the encoding.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8880", "#e3e2dd"

# The batch-level separation a Gaussian pair needs for standardized
# pAUC@FPR<=0.5% = 0.90 -- solved, not a rounded literal, by the same library
# function `exp_baseline_headroom_gpu.batch_for_target` predicts from, so panel B's
# dashed curve and the experiment's printed prediction cannot drift apart. (3.7673;
# earlier versions of both hard-coded the rounded 3.767 separately.)
TARGET_DELTA = signal.delta_for_pauc(0.90, 0.005)

# Configurations this repo actually published detection numbers at, so panel A can
# say where they sit on the ratio axis rather than leaving it to the reader.
HISTORICAL = [
    (0.694, "exp_gpu.py  12x48, b200"),
    (0.781, "README  20x128, b1000"),
    (0.049, "triage run  64x128, b200"),
]


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def _line(ax, x, y, color, label, sd=None, dashed=False):
    """One direct-labeled series; `sd` draws the +-1 sd band over protocol seeds.

    On an AUC axis that band is the difference between a finding and noise, so it
    is load-bearing rather than decoration -- panel A's whole claim is that the
    0.98 at a 69% ratio sits many sd away from the 0.53 at 2%."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if sd is not None:
        sd = np.asarray(sd, float)
        ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.13, lw=0, zorder=2)
    ax.plot(x, y, color=color, lw=2.0, marker="o", ms=6.0, zorder=3,
            ls=(0, (4, 2)) if dashed else "-",
            markeredgecolor="white", markeredgewidth=1.2, label=label)


def _short(model: str) -> str:
    return model.split("/")[-1]


# ------------------------------------------------------------------ A: the ratio
def panel_ratio(ax, p, models):
    for mi, model in enumerate(models):
        sweep = p["models"][model]["ratio_sweep"]
        for b in p["ratio_batch"]:
            rows = sweep.get(str(b), [])
            if not rows:
                continue
            rows = sorted(rows, key=lambda r: r["ratio"])
            _line(ax, [r["ratio"] for r in rows], [r["auc"] for r in rows],
                  SERIES[p["ratio_batch"].index(b)],
                  f"batch {b}   {_short(model)}",
                  sd=[r["auc_sd"] for r in rows], dashed=mi > 0)
    ax.set_xscale("log")
    ax.set_ylim(0.33, 1.06)
    ax.axhline(0.5, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.axvspan(0.10, 1.06, color=SERIES[1], alpha=0.06, lw=0, zorder=0)
    ax.annotate("chance (0.5)", (0.26, 0.5), xytext=(0, 5), ha="left",
                textcoords="offset points", color=INK2, fontsize=8.5)
    ax.annotate("shaded: above the $\\leq$10% batch/pool ceiling\n"
                "`EvalConfig` documents. There the batches\n"
                "overlap, honest variance collapses, and the\n"
                "AUC scores the two POOLS, not a fresh draw.",
                (0.02, 0.98), xycoords="axes fraction", va="top", ha="left",
                color=INK2, fontsize=8.5, linespacing=1.35)
    # The repo's own published configurations, as vertical rules with rotated
    # labels: they are the reason this panel exists, and rotating keeps three
    # labels (two of them nearly coincident) from colliding.
    # The 69% and 78% rules are nearly coincident on a log axis, so their labels
    # go on opposite sides of their own rule.
    for i, (x, lab) in enumerate(HISTORICAL):
        ax.axvline(x, color=INK2, lw=0.9, ls=(0, (1, 2)), zorder=1)
        right = i == 1
        ax.annotate(lab, (x, 0.02), xycoords=("data", "axes fraction"),
                    xytext=(4 if right else -3, 0), textcoords="offset points",
                    rotation=90, ha="left" if right else "right", va="bottom",
                    color=INK2, fontsize=7.5)
    ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    ax.set_xticklabels(["2%", "5%", "10%", "20%", "50%", "100%"])
    ax.xaxis.set_minor_locator(__import__("matplotlib").ticker.NullLocator())
    ax.set_xlabel("batch size / honest eval-split tokens   (the ratio)", color=INK2,
                  fontsize=10)
    ax.set_ylabel("detection AUC @ FPR $\\leq$ 0.5%", color=INK2, fontsize=10)
    ax.set_title("A  Same scores, same attack -- only the POOL grows",
                 color=INK, fontsize=11, weight="bold", loc="left")
    leg = ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left",
                    bbox_to_anchor=(0.02, 0.70),
                    title="hue = batch, dash = model", title_fontsize=8.5)
    leg.get_title().set_color(INK2)


# ------------------------------------------------------------- B: honest power
def panel_batch(ax, p, models):
    for mi, model in enumerate(models):
        rows = [r for r in p["models"][model]["batch_sweep"] if "auc_fixed" in r]
        _line(ax, [r["batch"] for r in rows], [r["auc_fixed"] for r in rows],
              SERIES[mi], _short(model), sd=[r["auc_fixed_sd"] for r in rows])
        ax.annotate(_short(model), (rows[-1]["batch"], rows[-1]["auc_fixed"]),
                    xytext=(8, 0), va="center", textcoords="offset points",
                    color=SERIES[mi], fontsize=9.5, weight="bold")

    # The Gaussian prediction from the measured per-token d', on the same axes:
    # a batch of b independent tokens separates by d'*sqrt(b), and pAUC@0.5% is a
    # monotone function of that separation. It is what says no batch this pool can
    # afford reaches 0.90 at this attack strength.
    head = models[0]
    d = p["models"][head]["per_token"]["d_prime"]
    b_need = p["models"][head]["batch_for_target"]
    b_grid = np.geomspace(40, max(5000, 1.6 * b_need), 80)
    ax.plot(b_grid, [signal.pauc_of_delta(d * np.sqrt(b)) for b in b_grid], color=MUTED,
            lw=1.6, ls=(0, (5, 3)), zorder=2)
    ax.plot([b_need], [p["target_auc"]], marker="o", ms=6.0, color=MUTED,
            markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    ax.annotate(f"batch {b_need:,} -- and a\n$\\geq${20*b_need:,}-token pool\n"
                f"to afford it honestly",
                (b_need, p["target_auc"]), xytext=(-8, -6), textcoords="offset points",
                ha="right", va="top", color=INK2, fontsize=8.5, linespacing=1.35)
    ax.annotate(f"Gaussian prediction from the measured per-token\n"
                f"$d'$ = {d:.3f} ({_short(head)}): a batch of $b$ independent\n"
                f"tokens separates by $d'\\sqrt{{b}}$",
                (0.98, 0.03), xycoords="axes fraction", xytext=(0, 0),
                textcoords="offset points", ha="right", va="bottom", color=INK2,
                fontsize=8.5, linespacing=1.35)
    ax.axhline(p["target_auc"], color=SERIES[1], lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate(f"target AUC {p['target_auc']:.2f}", (44, p["target_auc"]),
                xytext=(0, 5), textcoords="offset points", color=SERIES[1],
                fontsize=8.5)
    ax.set_xscale("log")
    ax.set_ylim(0.40, 1.06)
    ax.set_xticks([50, 100, 200, 400, 800, 1600, 3200])
    ax.set_xticklabels(["50", "100", "200", "400", "800", "1600", "3200"])
    ax.xaxis.set_minor_locator(__import__("matplotlib").ticker.NullLocator())
    ax.set_xlabel("batch size (pool grown to hold the ratio at 10%)", color=INK2,
                  fontsize=10)
    ax.set_ylabel("detection AUC @ FPR $\\leq$ 0.5%", color=INK2, fontsize=10)
    ax.set_title("B  Legitimate power, ratio held fixed", color=INK, fontsize=11,
                 weight="bold", loc="left")


# ------------------------------------------------------------ C: attack strength
def panel_ladder(ax, p, models):
    head = models[0]
    # The ladder plus the default rung, which is the main run's own measurement of
    # the same quantity at sigma = 0.18 -- so the middle point is not re-measured.
    rungs = [(float(s), v["per_token"]["d_prime"])
             for s, v in p["sigma_ladder"].items() if v["model"] == head]
    rungs.append((0.18, p["models"][head]["per_token"]["d_prime"]))
    rungs.sort()
    xs, ys = [r[0] for r in rungs], [r[1] for r in rungs]
    _line(ax, xs, ys, SERIES[2], "per-token $d'$")
    ax.set_xscale("log")
    ax.set_yscale("log")

    # A pool of `pool` tokens can afford batch = 10% of its honest half; the d'
    # that reaches the target there is the horizontal line. Where the ladder
    # crosses it is the cheapest honest source of headroom.
    pool = 8192
    b_afford = 0.10 * 0.5 * pool
    d_need = TARGET_DELTA / np.sqrt(b_afford)
    ax.axhline(d_need, color=SERIES[1], lw=1.3, ls=(0, (4, 3)), zorder=1)
    ax.annotate(f"$d'$ = {d_need:.3f}: what AUC {p['target_auc']:.2f} needs at\n"
                f"batch {b_afford:.0f}, the biggest a properly-ratioed\n"
                f"{pool:,}-token pool can afford",
                (0.02, 0.97), xycoords="axes fraction", va="top", ha="left",
                color=SERIES[1], fontsize=8.5, linespacing=1.35)
    # Each rung labeled with what it costs to detect: the batch its d' needs.
    # Placement alternates side so the three labels cannot collide, and the
    # left-most sits ABOVE its point so it does not run into the y axis.
    place = {0: (8, 10, "left", "bottom"), 1: (10, -2, "left", "top"),
             2: (-8, -6, "right", "top")}
    for i, (x, y) in enumerate(rungs):
        b_need = int(np.ceil((TARGET_DELTA / y) ** 2)) if y > 0 else -1
        name = {0.18: " = quant_4bit", 0.36: " = quant_2bit"}.get(x, "")
        dx, dy, ha, va = place[i]
        ax.annotate(f"$\\sigma$={x:g}{name}\n$d'$={y:.3f}\nbatch for 0.90: {b_need:,}",
                    (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, color=INK2, fontsize=8, linespacing=1.35)
    ax.set_xlim(min(xs) * 0.82, max(xs) * 1.28)
    ax.set_ylim(min(ys) * 0.40, max(ys) * 3.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.xaxis.set_minor_locator(__import__("matplotlib").ticker.NullLocator())
    ax.set_xlabel("quantization $\\sigma$ (logit-deviation magnitude)", color=INK2,
                  fontsize=10)
    ax.set_ylabel("per-token effect size $d'$", color=INK2, fontsize=10)
    ax.set_title("C  Where headroom can come from instead",
                 color=INK, fontsize=11, weight="bold", loc="left")


# ------------------------------------------------------------------------- main
def fig_headroom(p, path):
    import matplotlib.pyplot as plt
    models = list(p["models"])
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16.4, 5.2))
    for ax in (axA, axB, axC):
        _clean(ax)
    panel_ratio(axA, p, models)
    panel_batch(axB, p, models)
    panel_ladder(axC, p, models)

    fig.suptitle("Why full-recompute `token_difr` scored 0.570: the batch/pool "
                 "ratio, not the detector",
                 color=INK, fontsize=12.5, weight="bold", x=0.005, ha="left", y=1.07)
    fig.text(0.005, 1.005,
             f"attack={p['attack']}   pool = {p['n_prompts']} prompts x "
             f"{p['tokens']} tokens = {p['n_prompts']*p['tokens']:,} tokens per "
             f"configuration   every AUC is a mean $\\pm$ sd over {p['n_seed']} "
             f"independent protocol seeds (bands)   "
             f"standardized EvalConfig: pAUC @ FPR $\\leq$ 0.5%, n_batches=2000",
             color=INK2, fontsize=8.5, ha="left", va="bottom")
    fig.tight_layout(w_pad=3.0)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    src = RES_DIR / "baseline_headroom.json"
    if not src.exists():
        print(f"skip: {src} missing (run exp_baseline_headroom_gpu)")
        return
    out = FIG_DIR / "fig_baseline_headroom.png"
    fig_headroom(json.loads(src.read_text()), out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
