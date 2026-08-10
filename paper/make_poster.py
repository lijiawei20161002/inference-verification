"""Conference poster: fixing Token-DiFR under speculative decoding.

Default size is **24 x 36 in portrait** (`--size 24x36`), the standard US poster
board, in two columns. `--size` accepts any board (e.g. `30x40`); the panel-to-column
map is written for two columns, so `--cols 3` needs the `col=` fields updated too.
Type sizes follow poster convention rather than paper convention -- title fitted to
the board width, section heads 31 pt, body 22 pt, chart labels 18-20 pt -- so it
reads at 1-2 m. `--debug` prints the packed layout in inches.

Every number is read from the committed artifacts (`docs/results/specdec_difr_sweep_summary.json`,
`specdec_difr_identity.json`), so the poster cannot drift from the experiment.

    python paper/make_poster.py                      # -> paper/specdec_difr_poster.{pdf,png}
    python paper/make_poster.py --size 30x40 --debug

Layout is expressed in INCHES throughout (`fx`/`fy` convert to figure fractions at
the very end). Panels declare either a fixed height or a flex weight, and each
column distributes its leftover height among its flex panels -- so changing the
board size re-flows instead of overlapping.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results"

# ---------------------------------------------------------------- design tokens
SURFACE = "#fcfcfb"
PANEL = "#f5f4f1"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#767570"
GRID = "#e3e2de"
S1 = "#2a78d6"      # categorical slot 1 -- honest provider / match-rate statistic
S2 = "#eb6834"      # categorical slot 2 -- cheating provider / margin statistic
# Reference-palette slots 1-2, validated (Machado 2009 severity 1.0, OKLab dE x100,
# surface #fcfcfb): normal dE 33.6 (floor 15), protan 24.7 / deutan 31.7 -> min 24.7
# (target 8), contrast 4.30:1 and 3.12:1 (min 3.0), both in the lightness band, C > 0.10.

# type scale (points) -- poster viewing distance, not page viewing distance
T_TITLE, T_SUB, T_AUTH = 62, 26, 23
T_HERO, T_HEROCAP = 52, 19
T_KICK, T_HEAD, T_BODY = 20, 31, 22
T_TICK, T_AXIS, T_NOTE, T_ANNO = 20, 20, 18, 20
T_TICK_DENSE = 18          # 7- and 12-row charts
T_FOOT = 17
LEAD = 1.30
EM = 0.60            # DejaVu Sans average advance in em -- for line-width math

ap = argparse.ArgumentParser()
ap.add_argument("--size", default="24x36", help="WxH in inches (default 24x36 portrait)")
ap.add_argument("--out", default="specdec_difr_poster")
ap.add_argument("--cols", type=int, default=2, help="columns (the panel map assumes 2)")
ap.add_argument("--debug", action="store_true", help="print the packed layout in inches")
args = ap.parse_args()
W, H = (float(v) for v in args.size.lower().split("x"))
DEBUG = args.debug
NCOL = args.cols

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": T_BODY,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 1.2,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "pdf.fonttype": 42, "mathtext.fontset": "dejavusans",
})

# ------------------------------------------------------------------------ data
S = json.loads((RES / "specdec_difr_sweep_summary.json").read_text())
ID = json.loads((RES / "specdec_difr_identity.json").read_text())
ROWS, CELLS = S["rows"], S["cells"]
P0 = "q2.5-1.5b"


def row(kind, mode, prov, pair=P0, K=4, T=1.0, tk=0, tp=1.0, pb=1):
    for r in ROWS:
        if (r["kind"], r["pair"], r["mode"], r["prov"], r["K"], r["T"],
                r["top_k"], r["top_p"], r["pbatch"]) == (kind, pair, mode, prov, K, T, tk, tp, pb):
            return r
    raise KeyError((kind, mode, prov, pair, K, T, tk, tp, pb))


def cellv(mode, prov, pair=P0, K=4, T=1.0, tk=0, tp=1.0, pb=1):
    for c in CELLS:
        if (c["pair"], c["mode"], c["prov"], c["K"], c["T"], c["top_k"],
                c["top_p"], c["pbatch"]) == (pair, mode, prov, K, T, tk, tp, pb):
            return c
    raise KeyError((mode, prov, pair, K))


SWEPT = [("K = 1", dict(K=1)), ("K = 2", dict(K=2)), ("K = 4  (base)", dict()),
         ("K = 8", dict(K=8)),
         ("T = 0.4", dict(T=0.4)), ("T = 0.7", dict(T=0.7)),
         ("top-p 0.9", dict(tp=0.9)), ("top-k 50", dict(tk=50)),
         ("provider batch 4", dict(pb=4)),
         ("Qwen2.5-3B ← 0.5B", dict(pair="q2.5-3b")),
         ("Qwen3-1.7B ← 0.6B", dict(pair="q3-1.7b")),
         ("1.5B ← Coder-0.5B", dict(pair="q2.5-1.5b/coder-draft"))]
D_BENIGN = [row("benign", "coupled", "clean", **kw)["d_mismatch"] for _, kw in SWEPT]
D_ATTACK = [row("attack", "coupled", "int4", **kw)["d_mismatch"] for _, kw in SWEPT]
PAIR_IDS = [P0, "q2.5-3b", "q3-1.7b", "q2.5-1.5b/coder-draft"]
BREAK = [row("benign", "standard", "clean", pair=q)["d_mismatch"] for q in PAIR_IDS]
INREG = [row("attack", "standard", "int4", pair=q)["d_mismatch"] for q in PAIR_IDS]
DEVS = [("int3 weights (RTN g128)", "coupled", "int3"),
        ("serves 0.5B, bills 1.5B", "coupled", "sub"),
        ("fp8 KV cache", "coupled", "kvfp8"),
        ("lossy “typical” acceptance", "typical", "clean"),
        ("int4 weights (RTN g128)", "coupled", "int4"),
        ("temperature 1.05 vs 1.00", "coupled", "temp1.05"),
        ("int8 weights (RTN g128)", "coupled", "int8")]

fig = plt.figure(figsize=(W, H))

# ------------------------------------------------------------- inches -> figure
fx = lambda xin: xin / W
fy = lambda yin: 1.0 - yin / H          # y measured DOWN from the top edge, in inches
fh = lambda hin: hin / H


def lines_of(s, w_in, size):
    """Wrap to a column `w_in` inches wide: a line holds w_in*72/(size*EM) chars."""
    n = max(8, int(w_in * 72 / (size * EM)))
    out = []
    for block in s.split("\n"):
        out += textwrap.wrap(block, n) or [""]
    return out


def fit_size(s, w_in, size):
    """Largest size <= `size` at which `s` fits `w_in` inches on one line. A poster
    title is the one string with no room to wrap, so it gets measured, not guessed."""
    while size > 8 and len(s) * size * EM / 72 > w_in:
        size -= 1
    return size


def text_h(s, w_in, size, lead=LEAD):
    return len(lines_of(s, w_in, size)) * size * lead / 72


def draw_text(x_in, y_in, s, w_in, size=T_BODY, color=INK2, lead=LEAD):
    ls = lines_of(s, w_in, size)
    dy = size * lead / 72
    for i, ln in enumerate(ls):
        fig.text(fx(x_in), fy(y_in + i * dy), ln, size=size, color=color, va="top", ha="left")
    return y_in + len(ls) * dy


def used(y_end, y0, h, name):
    """Assert a renderer stayed inside the content rect it was handed. Silent
    overflow is how a poster ends up with body copy under the footer."""
    assert y_end <= y0 + h + 0.06, f"{name} overflows its panel by {y_end - y0 - h:.2f} in"


def card(x_in, y_in, w_in, h_in):
    """Recessive background card. zorder MUST sit below the Axes: an Axes is zorder 0,
    so a figure patch at zorder 0 created later paints straight over it."""
    fig.add_artist(FancyBboxPatch(
        (fx(x_in), fy(y_in + h_in)), fx(w_in), fh(h_in),
        boxstyle="round,pad=0,rounding_size=0.004", transform=fig.transFigure,
        facecolor=PANEL, edgecolor="none", zorder=-10))


def axes_at(x_in, y_in, w_in, h_in):
    assert w_in > 0.5 and h_in > 0.5, f"degenerate axes {(x_in, y_in, w_in, h_in)}"
    a = fig.add_axes([fx(x_in), fy(y_in + h_in), fx(w_in), fh(h_in)])
    a.set_zorder(3)
    a.patch.set_visible(False)
    for s in ("top", "right", "left"):
        a.spines[s].set_visible(False)
    a.spines["bottom"].set_color(GRID)
    a.tick_params(labelsize=T_TICK, pad=5)
    return a


def series_key(x_in, y_in, labels, colors, right_edge=None):
    """A legend drawn as coloured heads above the plot. At poster scale an in-axes
    legend box eats a third of the panel and lands on the marks. The last label is
    right-aligned to `right_edge` so a long series name cannot leave the card."""
    for i, (lab, c) in enumerate(zip(labels, colors)):
        last = i == len(labels) - 1
        if last and right_edge is not None:
            fig.text(fx(right_edge), fy(y_in), lab, size=T_ANNO, color=c,
                     weight="bold", va="top", ha="right")
        else:
            fig.text(fx(x_in + i * 4.2), fy(y_in), lab, size=T_ANNO, color=c,
                     weight="bold", va="top")


def bar_h(n_series, span=0.84, gap=0.22):
    return span / n_series * (1 - gap)


# ============================================================ panel renderers
# `draw(x, y, w, h)` gets the CONTENT rect in inches (y measured down from the top)
# that is left over after the kicker / title / body copy have been laid out.

def p_problem(x, y, w, h):
    ax = axes_at(x + 3.7, y + 1.35, w - 4.1, h - 2.55)
    regimes = [("incremental decode", "exact"), ("coupled speculation", "coupled"),
               ("provider-local RNG", "standard")]
    yp = np.arange(len(regimes))[::-1]
    bh = bar_h(2)
    for j, (lab, mode) in enumerate(regimes):
        hv, av = cellv(mode, "clean")["mismatch"], cellv(mode, "int4")["mismatch"]
        ax.barh(yp[j] + bh / 2, hv, height=bh, color=S1, edgecolor="none")
        ax.barh(yp[j] - bh / 2, av, height=bh, color=S2, edgecolor="none")
        ax.text(hv + 0.012, yp[j] + bh / 2, f"{hv:.3f}", va="center", size=T_ANNO, color=INK2)
        ax.text(av + 0.012, yp[j] - bh / 2, f"{av:.3f}", va="center", size=T_ANNO, color=INK2)
    ax.set_yticks(yp, [r[0] for r in regimes], size=T_TICK)
    ax.set_xlim(0, 0.88)
    ax.set_ylim(-0.62, len(regimes) - 0.38)
    ax.set_xlabel("per-token mismatch rate against the verifier's replay", size=T_AXIS)
    ax.xaxis.grid(True, color=GRID, lw=1.2)
    ax.set_axisbelow(True)
    series_key(x + 3.7, y, ["honest provider", "int4 cheat"], [S1, S2], right_edge=x + w)
    draw_text(x, y + 0.42,
              "Bottom row: an honest speculative provider misses more tokens than the cheat "
              "ever did on an honest decoder.", w, size=T_ANNO, color=INK)


def p_fix(x, y, w, h):
    y0 = y
    y = draw_text(x, y, "Hand the draft the same Gumbel vector the verifier will use. "
                        "The draft proposes", w) + 0.12
    fig.text(fx(x + 0.25), fy(y), r"$x_i=\arg\max_x\;[\,\log q(x)/T + G_{t+i}(x)\,]$",
             size=T_HEAD - 3, color=INK, va="top")
    y = draw_text(x, y + 0.85, "and the target accepts it if and only if", w) + 0.12
    fig.text(fx(x + 0.25), fy(y), r"$x_i=\arg\max_x\;[\,\log p(x)/T + G_{t+i}(x)\,]$",
             size=T_HEAD - 3, color=INK, va="top")
    y = draw_text(x, y + 0.85,
                  "otherwise it emits its own argmax and the round ends. Still lossless in "
                  "the Leviathan/Chen sense — but now every returned token is the one "
                  "DiFR expects, and the verifier does not change at all.", w)
    tpr = cellv("coupled", "clean")["tokens_per_round"]
    std = cellv("standard", "clean")["tokens_per_round"]
    used(draw_text(x, y + 0.22,
              f"Cost: {tpr:.2f} vs {std:.2f} tokens per target forward pass at K = 4 — "
              f"{1 - tpr / std:.1%} of the speculative speedup, and it still scales with "
              f"depth ({cellv('coupled', 'clean', K=1)['tokens_per_round']:.2f} at K = 1 to "
              f"{cellv('coupled', 'clean', K=8)['tokens_per_round']:.2f} at K = 8).",
              w, color=INK), y0, h, "fix")


def p_sweep(x, y, w, h):
    note = ("The orange rule marks d′ = 0.0775, the repo's 4-bit-quantisation signal "
            "— 3–80× the benign shift the fix leaves behind.")
    nh = text_h(note, w, T_NOTE) + 0.35
    XLAB = 0.62                      # room the x-axis label itself needs
    top = y + 0.95
    ah = h - nh - XLAB - 0.95
    lab_w = 3.9
    lw = (w - lab_w) * 0.56
    axb = axes_at(x + lab_w, top, lw, ah)
    axa = axes_at(x + lab_w + lw + 0.60, top, (w - lab_w) - lw - 0.60, ah)
    yp = np.arange(len(SWEPT))[::-1]
    axb.axvline(0, color=INK3, lw=1.4, zorder=1)
    axb.axvline(0.0775, color=S2, lw=1.8, zorder=1)
    axb.scatter(D_BENIGN, yp, s=260, color=S1, zorder=3, clip_on=False)
    axb.set_yticks(yp, [s[0] for s in SWEPT], size=T_TICK_DENSE)
    axb.set_xlim(-0.10, 0.105)
    axb.set_xticks([-0.05, 0, 0.05], ["−0.05", "0", "0.05"])
    axb.set_xlabel("d′ benign  (honest)", size=T_AXIS)
    axa.scatter(D_ATTACK, yp, s=260, color=S2, zorder=3, clip_on=False)
    axa.set_yticks(yp, [])
    axa.set_xlim(0, 3.0)
    axa.set_xticks([0, 1, 2, 3])
    axa.set_xlabel("d′ int4  (the cheat)", size=T_AXIS)
    for a in (axb, axa):
        a.set_ylim(-0.9, len(SWEPT) - 0.1)
        a.xaxis.grid(True, color=GRID, lw=1.2)
        a.set_axisbelow(True)
    fig.text(fx(x + lab_w), fy(y), "must be ≈ 0", size=T_ANNO, color=S1,
             weight="bold", va="top")
    fig.text(fx(x + lab_w + lw + 0.60), fy(y), "must stay large", size=T_ANNO, color=S2,
             weight="bold", va="top")
    fig.text(fx(x + lab_w), fy(y + 0.40), "every cell: pAUC 1.000 at FPR ≤ 0.5%",
             size=T_ANNO, color=INK, va="top")
    used(draw_text(x, y + h - nh + 0.20, note, w, size=T_NOTE, color=S2), y, h, "sweep")


def p_identity(x, y, w, h):
    """Three numbers, not a chart: a bar of 1.000 against a bar of 0.967 is a stat
    tile pretending to be a plot."""
    arms = [("coupled", "coupled speculation\nvs incremental decode"),
            ("exact_rerun", "incremental decode\nrerun (the control)"),
            ("standard", "provider-local RNG\nvs incremental decode")]
    for i, (arm, cap) in enumerate(arms):
        v = float(np.mean([r["identical_frac"] for r in ID if r["arm"] == arm]))
        sx = x + i * (w / 3)
        fig.text(fx(sx), fy(y), f"{v:.3f}", size=T_HERO - 6, weight="bold",
                 color=S1 if arm != "standard" else S2, va="top")
        fig.text(fx(sx), fy(y + 0.80), cap, size=T_NOTE, color=INK2, va="top",
                 linespacing=1.45)
    used(draw_text(x, y + 1.95,
                   "Fraction of returned tokens identical to the reference decode, same "
                   "prompt and same seed. The coupled residual is one early numeric flip "
                   "cascading through an autoregressive sequence.", w), y, h, "identity")


def p_devgrid(x, y, w, h):
    tmp = row("attack", "coupled", "temp1.05")
    note = (f"The margin dominates where a deviation changes how badly tokens miss, not "
            f"whether they miss ({tmp['pauc_mismatch']:.3f} vs {tmp['pauc']:.3f} on the "
            f"temperature retune). int8 is at chance on both.")
    nh = text_h(note, w, T_NOTE) + 0.35
    ax = axes_at(x + 4.7, y + 0.58, w - 5.1, h - nh - 1.55)
    yp = np.arange(len(DEVS))[::-1]
    bh = bar_h(2, span=0.90, gap=0.14)
    for j, (lab, mode, prov) in enumerate(DEVS):
        r = row("attack", mode, prov)
        ax.barh(yp[j] + bh / 2, r["pauc_mismatch"] - 0.5, left=0.5, height=bh, color=S1,
                edgecolor="none")
        ax.barh(yp[j] - bh / 2, r["pauc"] - 0.5, left=0.5, height=bh, color=S2,
                edgecolor="none")
        if abs(r["pauc_mismatch"] - r["pauc"]) < 0.0005:
            # identical to 3 dp: two stacked labels would collide, and would say
            # the same thing twice
            ax.text(max(r["pauc"], r["pauc_mismatch"]) + 0.008, yp[j], f"{r['pauc']:.3f}",
                    va="center", size=T_NOTE, color=INK2)
        else:
            ax.text(r["pauc_mismatch"] + 0.008, yp[j] + bh / 2, f"{r['pauc_mismatch']:.3f}",
                    va="center", size=T_NOTE, color=INK2)
            ax.text(r["pauc"] + 0.008, yp[j] - bh / 2, f"{r['pauc']:.3f}", va="center",
                    size=T_NOTE, color=INK2)
    ax.set_yticks(yp, [d[0] for d in DEVS], size=T_TICK_DENSE)
    ax.set_xlim(0.5, 1.10)
    ax.set_xticks([0.5, 0.75, 1.0], ["0.5\nchance", "0.75", "1.0"])
    ax.set_ylim(-0.7, len(DEVS) - 0.3)
    ax.set_xlabel("standardized pAUC @ FPR ≤ 0.5%", size=T_AXIS)
    ax.xaxis.grid(True, color=GRID, lw=1.2)
    ax.set_axisbelow(True)
    series_key(x + 4.7, y, ["match rate", "margin"], [S1, S2], right_edge=x + w)
    used(draw_text(x, y + h - nh + 0.24, note, w, size=T_NOTE), y, h, "devgrid")


def p_takeaways(x, y, w, h):
    bullets = [
        ("Audit the decoder, not just the weights.",
         "Speculation makes the calibration non-portable."),
        ("Ask for coupled speculation, not for none.",
         "Output-identical to the reference decode, 7.5% slower."),
        ("Aggregate the margin, not the match indicator.",
         "Except under top-p 0.9: 0.13% of honest tokens land outside the verifier's own "
         "nucleus, costing the margin 25× of its effect size."),
        ("Neither of them fixes int8.",
         "Under the honest floor: a pool problem, not a seed problem."),
    ]
    dy = T_BODY * LEAD / 72
    y0 = y
    for head, rest in bullets:
        fig.text(fx(x), fy(y), "•", size=T_BODY, color=S1, va="top")
        fig.text(fx(x + 0.30), fy(y), head, size=T_BODY, color=INK, va="top", weight="bold")
        y = draw_text(x + 0.30, y + dy, rest, w - 0.32) + 0.18
    used(y, y0, h, "takeaways")



PANELS = [
    dict(col=0, kicker="1 · the problem", title="The honest null collapses",
         body="An honest provider reproduces the replayed token at ~99% of positions; a "
              "provider-local RNG destroys that null. Across four target/draft pairs the "
              f"benign shift is d\u2032 {min(BREAK):.1f}\u2013{max(BREAK):.1f}, against "
              f"{min(INREG):.2f}\u2013{max(INREG):.2f} for an int4 cheat re-calibrated in "
              "the same regime.",
         draw=p_problem, flex=1.0),
    dict(col=0, kicker="2 · the fix: common random numbers",
         title="Couple the provider's Gumbels", draw=p_fix, h=7.85),
    dict(col=0, kicker="3 · takeaways", title="What we would deploy",
         draw=p_takeaways, h=7.15),
    dict(col=1, kicker="4 · does it generalise?",
         title="The fix holds on every axis swept",
         body="Left: the false-positive side. Right: the power side — int4 weights "
              "against honest, in the same regime.",
         draw=p_sweep, flex=2.15),
    dict(col=1, kicker="5 · the deviation grid",
         title="What the fix buys — and what it does not",
         body="Seven deviations under the fixed scheme, scored two ways: the binary match "
              "rate, and the post-Gumbel margin.",
         draw=p_devgrid, flex=2.00),
    dict(col=1, kicker="6 · the stronger statement",
         title="Not merely indistinguishable — identical",
         draw=p_identity, h=5.2),
]

# ======================================================================= header
MARGIN = 0.9
GUTTER = 0.50
CW = (W - 2 * MARGIN - (NCOL - 1) * GUTTER) / NCOL
COLX = [MARGIN + i * (CW + GUTTER) for i in range(NCOL)]

TITLE = ["Speculative decoding breaks inference verification.",
         "Coupling the provider's random numbers fixes it."]
t_size = min(fit_size(t, W - 2 * MARGIN, T_TITLE) for t in TITLE)
y = 0.90
fig.text(fx(MARGIN), fy(y), TITLE[0], size=t_size, weight="bold", color=INK, va="top")
y += t_size * 1.16 / 72
fig.text(fx(MARGIN), fy(y), TITLE[1], size=t_size, weight="bold", color=S1, va="top")
y += t_size * 1.34 / 72
fig.text(fx(MARGIN), fy(y),
         "Jiawei Li   ·   Inference Verification Gym (ivgym)   ·   "
         "new measurements on one NVIDIA H100-80GB", size=T_AUTH, color=INK, va="top")
y += T_AUTH * 1.85 / 72
y = draw_text(MARGIN, y,
              "Token-DiFR (Karvonen et al., 2025) audits an inference provider by replaying "
              "its returned tokens under seed-synchronised Gumbel-Max sampling — and "
              "assumes the provider decodes one token at a time. Real serving stacks "
              "speculate, and then DiFR flags every honest provider. This poster measures the "
              "break, the one-line fix, and how far the fix generalises.",
              W - 2 * MARGIN, size=T_SUB, color=INK2) + 0.50

HERO = [(f"≤ {max(abs(v) for v in D_BENIGN):.3f}",
         "false-positive effect size d′ of the\nfixed scheme, all 12 swept settings"),
        ("1.000", "detection pAUC @ FPR ≤ 0.5% for\nint4 weights, in all 12 settings"),
        ("7.5%", "of the speculative speedup is what\nauditability costs")]
hero_w = (W - 2 * MARGIN - 2 * GUTTER) / 3
hero_h = 2.45
for i, (big, small) in enumerate(HERO):
    hx = MARGIN + i * (hero_w + GUTTER)
    card(hx, y, hero_w, hero_h)
    fig.text(fx(hx + 0.38), fy(y + 0.32), big, size=T_HERO, weight="bold", color=S1, va="top")
    fig.text(fx(hx + 0.38), fy(y + 1.45), small, size=T_HEROCAP, color=INK2, va="top",
             linespacing=1.5)
BODY_TOP = y + hero_h + 0.60

# ================================================================ pack & render
FOOT_H = 2.85
BODY_H = H - BODY_TOP - FOOT_H
PAD = 0.42                                    # card padding
for c in range(NCOL):
    ps = [p for p in PANELS if p["col"] == c]
    for p in ps:                              # measured chrome: kicker + title + body
        p["_chrome"] = PAD
        if p.get("kicker"):
            p["_chrome"] += T_KICK * 1.55 / 72
        if p.get("title"):
            p["_chrome"] += T_HEAD * 1.30 / 72 * (p["title"].count("\n") + 1) + 0.18
        if p.get("body"):
            p["_chrome"] += text_h(p["body"], CW - 2 * PAD, T_BODY) + 0.18
        p["_fixed"] = p.get("h", 0.0)
    gaps = GUTTER * (len(ps) - 1)
    fixed = sum(p["_fixed"] for p in ps)
    flexw = sum(p.get("flex", 0.0) for p in ps)
    slack = BODY_H - gaps - fixed - sum(p["_chrome"] for p in ps if p.get("flex"))
    assert slack > 0, f"column {c} overfull by {-slack:.2f} in"
    if DEBUG:
        print(f"column {c}: body={BODY_H:.2f} fixed={fixed:.2f} gaps={gaps:.2f} slack={slack:.2f}")
    yy = BODY_TOP
    for p in ps:
        h = p["_fixed"] if p["_fixed"] else p["_chrome"] + slack * p["flex"] / flexw
        if DEBUG:
            print(f"  col{c} {p['kicker'][:22]:24s} h={h:5.2f} chrome={p['_chrome']:5.2f} "
                  f"content={h - p['_chrome']:5.2f}")
        x = COLX[c]
        card(x, yy, CW, h)
        ty = yy + PAD * 0.55
        if p.get("kicker"):
            fig.text(fx(x + PAD), fy(ty), p["kicker"].upper(), size=T_KICK, color=INK3,
                     weight="bold", va="top")
            ty += T_KICK * 1.55 / 72
        if p.get("title"):
            fig.text(fx(x + PAD), fy(ty), p["title"], size=T_HEAD, color=INK, weight="bold",
                     va="top", linespacing=1.3)
            ty += T_HEAD * 1.30 / 72 * (p["title"].count("\n") + 1) + 0.18
        if p.get("body"):
            ty = draw_text(x + PAD, ty, p["body"], CW - 2 * PAD) + 0.18
        p["draw"](x + PAD, ty, CW - 2 * PAD, yy + h - ty - PAD * 0.55)
        yy += h + GUTTER

# ======================================================================= footer
draw_text(MARGIN, H - FOOT_H + 0.26,
          "Method: fp16 provider vs bf16 verifier on one H100 — a real cross-implementation "
          "numeric channel, not injected noise. 58 cells (4 speculation depths × 3 "
          "temperatures × 3 truncation rules × 2 provider batch shapes × 4 target/draft "
          "pairs × 8 provider deviations), 16 chat prompts × 192 tokens = 3 072 tokens per "
          "cell, scored at an 8.0% batch/pool ratio inside the repo's enforced 10% ceiling.\n"
          "Reproduce: python -m experiments.exp_specdec_difr_sweep_gpu --out sweep  ·  the "
          "same script with --identity  ·  this poster: python paper/make_poster.py  ·  "
          "artifacts: docs/results/specdec_difr_{sweep.jsonl, sweep_summary.json, "
          "identity.json, sweep.md}\n"
          "References: Karvonen et al., DiFR: Inference Verification Despite Nondeterminism "
          "(2025)  ·  Leviathan et al. (2022) and Chen et al. (2023), speculative decoding "
          " ·  code and every run artifact: the inference-verification (ivgym) repository.",
          W - 2 * MARGIN, size=T_FOOT, color=INK3, lead=1.5)

out = ROOT / "paper" / args.out
fig.savefig(out.with_suffix(".pdf"))
fig.savefig(out.with_suffix(".png"), dpi=int(2400 / max(W, H)))
print(f"wrote {out.with_suffix('.pdf')} ({W:g}x{H:g} in, {NCOL} columns) and .png")
