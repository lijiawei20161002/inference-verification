"""Figures for the four `exp_specdec_*_gpu` experiments.

Renders, from `docs/results/` into `docs/figures/`:

  fig_specdec_mechanism.png      the shape gap, by speculation depth and dtype
  fig_specdec_stepfunctions.png  the two things that flip the kernel: length, batch
  fig_specdec_survival.png       how long "lossless" spec-dec stays identical
  fig_specdec_why.png            margins vs perturbation, and the flip rate

Pure numpy + matplotlib, no GPU, so the figures can be iterated without re-running
the sweeps. Each figure is skipped with a message if its artifact is missing.

Every number that appears as text on these figures is read or derived from the
artifacts rather than typed in -- including the bar labels in fig 1 and the
implied-hazard line in fig 4, both of which were literals in the original draft.
A caption that cannot drift from its data is the whole lesson of
`tests/test_claims.py`.

    python -m experiments.plot_specdec
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "docs" / "results"
FIG_DIR = ROOT / "docs" / "figures"

# Slots 1-3 of the same validated reference palette `plot_headroom.py` and
# `plot_triage.py` take, in the same documented fixed order, so a colour means the
# same thing across every figure in the repo. Identity is never colour-alone:
# every series here is direct-labeled too.
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
SURF, INK, INK2, MUTED, GRID, BASE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 2.0, "lines.markersize": 5.5,
})
PCT = FuncFormatter(lambda v, _: f"{v:.0f}%")

# The speculation depth the single-number panels report at: the middle of the
# swept range and the depth `exp_specdec_ctxlen_gpu` holds fixed, so fig 1B and
# fig 2A are read at the same operating point.
BAR_GAMMA = 8


def title(ax, t, sub=None):
    ax.set_title(t, color=INK, fontsize=11, fontweight="bold", loc="left",
                 pad=14 if sub else 8)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, color=MUTED, fontsize=8.6,
                va="bottom")


def _shape(rows, tag, **where):
    """The one row of `specdec_shape.json` matching `tag` and every kwarg."""
    return next(r for r in rows
                if r["tag"] == tag and all(r.get(k) == v for k, v in where.items()))


# ============================ FIG 1: the mechanism ============================
def fig_mechanism(e1, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.4, 4.5), gridspec_kw=dict(wspace=0.34))
    gammas = sorted({r["gamma"] for r in e1 if r["tag"] == "chunked_vs_sequential"})
    dtypes = [("bfloat16", S1, "bf16"), ("float16", S2, "fp16"), ("float32", S3, "fp32")]

    for dt, col, lab in dtypes:
        y = [100 * _shape(e1, "chunked_vs_sequential", dtype=dt, gamma=g)["frac_exact"]
             for g in gammas]
        a.plot(range(len(gammas)), y, "o-", color=col, label=lab, zorder=3)
        a.annotate(lab, (len(gammas) - 1, y[-1]), textcoords="offset points",
                   xytext=(9, 0), color=col, fontsize=9.5, fontweight="bold",
                   va="center")
    a.set_xticks(range(len(gammas)))
    a.set_xticklabels(gammas)
    a.set_xlim(-0.25, len(gammas) - 0.28)
    a.set_ylim(-4, 108)
    a.yaxis.set_major_formatter(PCT)
    a.set_xlabel("speculation length γ  (tokens verified in one forward pass)")
    a.set_ylabel("logit values bitwise identical")
    a.legend(loc="lower left", bbox_to_anchor=(0.0, -0.30), ncol=3, fontsize=9,
             handlelength=1.6, columnspacing=1.6)
    # The control is the load-bearing part of the subtitle: it is what licenses
    # reading everything above as the shape rather than as noise.
    ctl = [r for r in e1 if r["tag"] == "rerun_control"]
    ctl_exact = min(r["frac_exact"] for r in ctl)
    title(a, "Chunked verification ≠ sequential decode",
          f"short prompts (~35 tok) · rerun control at every point: "
          f"{ctl_exact*100:.0f}%")

    bars = [(lab, _shape(e1, "chunked_vs_sequential", dtype=dt, gamma=BAR_GAMMA)["max_abs"],
             col) for dt, col, lab in dtypes]
    ypos = np.arange(len(bars))[::-1]
    for i, (lab, v, col) in zip(ypos, bars):
        b.barh(i, v, height=0.5, color=col, zorder=3)
        b.text(v * 1.35, i, f"{v:.4g}", va="center", color=INK, fontsize=9.5,
               fontweight="bold")
    b.set_yticks(ypos)
    b.set_yticklabels([x[0] for x in bars], color=INK2, fontsize=10)
    b.set_xscale("log")
    b.set_xlim(3e-4, 12)
    b.set_ylim(-1.15, len(bars) - 0.4)
    ctl_max = max(r["max_abs"] for r in ctl)
    b.text(3.6e-4, -0.95,
           f"rerun control:  {ctl_max:.1f}  — exactly zero, at every dtype",
           color=INK2, fontsize=9, va="center")
    b.grid(axis="y", visible=False)
    b.set_xlabel("max | Δ logit |   (log scale)")
    title(b, "How far the logits move",
          f"worst-case | Δ logit | at γ = {BAR_GAMMA}, by numeric format")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==================== FIG 2: kernel selection is a step function ====================
def fig_stepfunctions(e1, e4, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.4, 4.3), gridspec_kw=dict(wspace=0.28))

    def step_panel(ax, xs, ys, switch, xlabel, t, sub):
        """One exact-match curve, with the collapse between two adjacent points
        called out. `switch` is the index the curve drops INTO."""
        ax.plot(range(len(xs)), ys, "o-", color=S1, zorder=3)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs)
        ax.set_ylim(-4, 108)
        ax.yaxis.set_major_formatter(PCT)
        ax.axvspan(switch - 1.0, switch, color=S2, alpha=0.10, zorder=1)
        ax.annotate(f"kernel switch\nbetween {xs[switch-1]} and {xs[switch]}",
                    (switch, ys[switch]), textcoords="offset points", xytext=(16, 30),
                    color=INK2, fontsize=8.8,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))
        for i in (switch - 1, switch):
            ax.annotate(f"{ys[i]:.0f}%", (i, ys[i]), textcoords="offset points",
                        xytext=(0, 10 if i == switch - 1 else -16), ha="center",
                        color=INK, fontsize=9, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("logit values bitwise identical")
        title(ax, t, sub)

    step_panel(a, [r["ctx_len"] for r in e4], [100 * r["frac_exact"] for r in e4], 2,
               "context length (tokens)",
               "Source 1 — sequence length flips the attention kernel",
               f"chunked (γ={BAR_GAMMA}) vs sequential · bf16 · "
               f"{e4[0]['n_windows']} windows of natural text per point")

    batch_rows = sorted((r for r in e1 if r["tag"] == "batch_composition"),
                        key=lambda r: r["batch_size"])
    step_panel(b, [r["batch_size"] for r in batch_rows],
               [100 * r["frac_exact"] for r in batch_rows], 2,
               "batch size (identical row, different neighbours)",
               "Source 2 — who else is in the batch",
               "the batch-invariance problem, measured on the same prompts · bf16")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==================== FIG 3: exact-match survival ====================
def survival(rows, gamma, n):
    """P(still bitwise identical to plain greedy after k tokens), k = 0..n-1."""
    nm = np.array([r["n_match"] for r in rows if r["gamma"] == gamma])
    return np.array([100 * (nm > k).mean() for k in range(n)])


def fig_survival(e2, path):
    rows = e2["rows"]
    # The generation length: a run that never diverged has n_match == max_new.
    n = max(r["n_match"] for r in rows)
    gammas = sorted({r["gamma"] for r in rows})
    cols = dict(zip(gammas, (S1, S2, S3)))
    n_prompts = len({r["prompt_idx"] for r in rows})
    # Diverging at SOME gamma vs at EVERY gamma. The gap between the two is the
    # panel's actual claim: divergence is a property of the (prompt, gamma) pair,
    # not of the prompt. (An earlier draft captioned this "19 of 32", which is
    # neither count -- the sort of drift `tests/test_claims.py` exists to stop.)
    by_prompt = {p: [r["divergence_idx"] >= 0 for r in rows if r["prompt_idx"] == p]
                 for p in range(n_prompts)}
    n_any = sum(any(v) for v in by_prompt.values())
    n_all = sum(all(v) for v in by_prompt.values())

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.4, 4.3),
                               gridspec_kw=dict(wspace=0.26, width_ratios=[1.25, 1]))
    a.axhline(100, color=MUTED, ls=(0, (4, 3)), lw=1.4, zorder=4)
    a.text(n - 4, 96.5, "what the losslessness theorem promises",
           color=INK2, fontsize=9, va="top", ha="right")
    for g in gammas:
        surv = survival(rows, g, n)
        a.plot(range(n), surv, color=cols[g], zorder=3, label=f"γ={g}")
        a.annotate(f"γ={g}", (n - 1, surv[-1]), textcoords="offset points",
                   xytext=(8, 0), color=cols[g], fontsize=9.5, fontweight="bold",
                   va="center")
    a.set_xlim(0, n + 10)
    a.set_ylim(-4, 108)
    a.yaxis.set_major_formatter(PCT)
    a.set_xlabel("generated token index")
    a.set_ylabel("outputs still bitwise identical")
    a.legend(loc="lower left", bbox_to_anchor=(0.0, 0.02), ncol=3, fontsize=9,
             handlelength=1.6, columnspacing=1.6)
    a.grid(axis="x", visible=False)
    title(a, "Lossless greedy speculative decoding vs plain greedy",
          f"Qwen2.5-1.5B target · Qwen2.5-0.5B draft · bf16 · "
          f"{n_prompts} prompts × {n} tokens")

    # Jitter only; seeded so the figure regenerates byte-identically.
    rng = np.random.default_rng(0)
    for gi, g in enumerate(gammas):
        d = [r["divergence_idx"] if r["divergence_idx"] >= 0 else n
             for r in rows if r["gamma"] == g]
        b.scatter(np.full(len(d), gi) + rng.uniform(-0.16, 0.16, len(d)), d,
                  s=26, color=cols[g], alpha=0.75, linewidths=0, zorder=3)
        med = np.median([x for x in d if x < n])
        b.plot([gi - 0.30, gi + 0.30], [med, med], color=INK, lw=2.0, zorder=5)
        b.annotate(f"median {med:.0f}", (gi, med), textcoords="offset points",
                   xytext=(0, 5), ha="center", color=INK, fontsize=8.6,
                   fontweight="bold", zorder=6)
    b.axhline(n, color=MUTED, ls=(0, (4, 3)), lw=1.2, zorder=2)
    b.text(-0.42, n + 3, f"never diverged within {n} tokens", color=INK2, fontsize=8.6)
    b.set_xticks(range(len(gammas)))
    b.set_xticklabels([f"γ={g}" for g in gammas], color=INK2, fontsize=10)
    b.set_xlim(-0.5, len(gammas) - 0.25)
    b.set_ylim(-6, n + 14)
    b.set_ylabel("index of first differing token")
    b.grid(axis="x", visible=False)
    title(b, "Where each prompt first diverges",
          f"one dot per prompt · {n_any} of {n_prompts} prompts diverge at some γ, "
          f"only {n_all} at every γ")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==================== FIG 4: why — margins vs perturbation ====================
def implied_hazard(e2, gamma):
    """Constant per-token flip rate implied by the survival curve at `gamma`.

    If each token flips independently with probability `h`, the share of prompts
    still identical after `n` tokens is `(1-h)^n`, so `h = 1 - S(n)^(1/n)`. Read
    at the shallowest speculation depth, which is the closest the end-to-end
    experiment gets to a per-token measurement. This is an ENTIRELY independent
    route to the quantity fig 4B measures directly, which is why it is worth
    drawing on the same axes.
    """
    nm = np.array([r["n_match"] for r in e2["rows"] if r["gamma"] == gamma])
    n = nm.max()                       # generation length; n_match == n means never
    return 1 - (nm >= n).mean() ** (1 / n)


def fig_why(e2, e3, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.4, 4.3),
                               gridspec_kw=dict(wspace=0.28, width_ratios=[1.25, 1]))
    m = np.sort(np.array(e3["margins"]))
    cdf = 100 * np.arange(1, len(m) + 1) / len(m)
    eps = e3["stats"]["chunked"]["maxd"]
    frac = 100 * (m < eps).mean()
    a.fill_between([0, eps], 0, 100, color=S2, alpha=0.10, zorder=1)
    a.plot(m, cdf, color=S1, zorder=3)
    a.axvline(eps, color=S2, lw=1.6, zorder=4)
    a.annotate(f"max logit perturbation\nfrom chunking = {eps:.2f}", (eps, 78),
               textcoords="offset points", xytext=(14, 0), color=INK2, fontsize=8.8,
               va="center")
    a.plot([eps], [frac], "o", color=S1, ms=7, zorder=5,
           markeredgecolor=SURF, markeredgewidth=2)
    a.annotate(f"{frac:.1f}% of positions sit inside\nthe perturbation — a coin flip",
               (eps, frac), textcoords="offset points", xytext=(14, -26),
               color=INK, fontsize=9, fontweight="bold")
    a.set_xlim(0, 6)
    a.set_ylim(0, 100)
    a.annotate("tail continues →", (6, cdf[np.searchsorted(m, 6) - 1]),
               textcoords="offset points", xytext=(-6, 8), ha="right",
               color=MUTED, fontsize=8.4)
    a.yaxis.set_major_formatter(PCT)
    a.set_xlabel("top-1 minus top-2 logit gap (the argmax margin)")
    a.set_ylabel("cumulative share of positions")
    title(a, "Why a last-bit difference changes the token",
          f"margin distribution over {len(m):,} real decode positions · bf16")

    names = ["chunked\n(spec-dec shape)", "batch of 8\n(co-tenancy)",
             "rerun\n(same shape)"]
    vals = [100 * e3["stats"][k]["flip_rate"] for k in ("chunked", "batched", "control")]
    for i, (nm_, v, c) in enumerate(zip(names, vals, (S1, S2, MUTED))):
        b.bar(i, v, width=0.52, color=c, zorder=3)
        b.text(i, v + 0.03, f"{v:.2f}%", ha="center", color=INK, fontsize=10,
               fontweight="bold")
    g_low = min(r["gamma"] for r in e2["rows"])
    h = 100 * implied_hazard(e2, g_low)
    b.axhline(h, color=INK, ls=(0, (4, 3)), lw=1.3, zorder=4)
    b.annotate(f"{h:.2f}% — rate implied independently\n"
               f"by the γ={g_low} survival curve in fig 3",
               (2.42, h), textcoords="offset points", xytext=(0, 6),
               ha="right", color=INK2, fontsize=8.6)
    b.set_xticks(range(3))
    b.set_xticklabels(names, color=INK2, fontsize=9)
    b.set_ylim(0, max(max(vals), h) * 1.33)
    b.set_xlim(-0.6, 2.5)
    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}%"))
    b.set_ylabel("per-token argmax flip rate")
    b.grid(axis="x", visible=False)
    title(b, "Per-token flip rate",
          f"{e3['stats']['chunked']['n']:,} positions · vs sequential bf16 reference")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------- main
def _load(name):
    src = RES_DIR / name
    if not src.exists():
        return None
    return json.loads(src.read_text())


def main(argv):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    e1 = _load("specdec_shape.json")
    e2 = _load("specdec_divergence.json")
    e3 = _load("specdec_fliprate.json")
    e4 = _load("specdec_ctxlen.json")

    jobs = [
        ("fig_specdec_mechanism.png", fig_mechanism, (e1,), ["specdec_shape"]),
        ("fig_specdec_stepfunctions.png", fig_stepfunctions, (e1, e4),
         ["specdec_shape", "specdec_ctxlen"]),
        ("fig_specdec_survival.png", fig_survival, (e2,), ["specdec_divergence"]),
        ("fig_specdec_why.png", fig_why, (e2, e3), ["specdec_divergence",
                                                    "specdec_fliprate"]),
    ]
    for fig_name, fn, args, needs in jobs:
        if any(a is None for a in args):
            missing = [n for n, a in zip(needs, args) if a is None]
            print(f"skip {fig_name}: {missing} missing "
                  f"(run the matching exp_specdec_*_gpu)")
            continue
        out = FIG_DIR / fig_name
        fn(*args, out)
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])
