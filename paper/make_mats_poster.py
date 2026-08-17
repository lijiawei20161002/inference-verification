"""MATS 10.0 symposium poster: auditing a computation you cannot see.

The board is framed from the governance setting -- a treaty clause, a promise to
a regulator and an API bill all reduce to one unverifiable sentence, "model M ran
under spec phi" -- and then measures the two things an outsider actually holds:
the tokens that came back (column 2) and their arrival times (column 3). Column 1
draws that surface; the two channel columns lead with the principle and keep only
the results that price it.

Built on the MATS **top-banner landscape 36x24** template
(`MATS_poster_top_banner_landscape36x24.pptx`, symposium template folder). The
template's geometry, type scale and palette are reproduced here rather than
approximated -- column origins 1.25 / 12.72 / 24.18 in at width 10.57, a 4.00 in
maroon banner, Libre Baskerville for display and Carlito (metric-compatible with
the template's Calibri) for body.

Every number on the board is read from a committed artifact in `docs/results/`
(named in the caption). Nothing is typed in twice: the byte ratios, the headline
grid, the verdict prices, the pooling law's inputs and residual, and every
latency in the clock column all come out of JSON at render time, so a rerun that
moves a number moves the poster. Figure 3 is the one drawn curve -- the pooling
law's normal approximation with this experiment's measured mean, spread and d'
substituted in, because the raw bootstrap of a score that is exactly zero on 98%
of tokens is a comb of discreteness artefacts rather than a picture of the
principle.

Column 3 goes one step further and re-scores the clock rather than illustrating
it: `experiments.plot_slope_verifier.detection` is imported and run here, so
Figure 6B's pAUC curves come out of the same `harness.evaluate` -- same
standardized pAUC @ FPR <= 0.5%, same honest calibration split, same pool
ceiling -- that the returned-token verifiers are scored with in column 2. That
costs a few seconds of render time and buys the guarantee that the two channels
on this board are measured against each other on one protocol.

Column origins and section heights are calibrated to the 36x24 landscape board,
so `--size` is for proofing at a smaller scale rather than for reflowing onto
the portrait template. Each column reports its own slack at the end of a run and
says so loudly if it would cross the footer rule.

    python paper/make_mats_poster.py                  # -> paper/mats_poster.png + .pdf
    python paper/make_mats_poster.py --dpi 300        # print resolution
"""
from __future__ import annotations

import argparse
import json
import sys
from math import erf
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results"
ASSETS = Path(__file__).resolve().parent / "assets"
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--size", default="36x24")
ap.add_argument("--out", default="mats_poster")
ap.add_argument("--dpi", type=float, default=150)
args = ap.parse_args()
W, H = (float(v) for v in args.size.lower().split("x"))

# ------------------------------------------------------------------ template
# Sampled straight out of the template's slide XML.
MAROON = "#800020"
CREAM = "#FFF7DF"
INK = "#141414"
GREY = "#5C5C5C"
SURFACE = "#F7F5F2"      # the template's figure-panel fill
RULE_C = "#C9C2BA"
WHITE = "#FFFFFF"

# Two accents for the two channels. Both are checked against the maroon/cream
# ground for >= 4.5:1 on white and are distinguishable under deuteranopia; the
# channel is ALSO carried by position (column 2 vs column 3) and by label, so
# colour is never the only encoding.
C_TOK = "#0B5D9E"        # the returned-token channel
C_CLK = "#B4531B"        # the clock channel
C_DEAD = "#9A948C"       # a cell with no price / a channel that sees nothing

SERIF = "Libre Baskerville"
SANS = "Carlito"

T_TITLE, T_SUB, T_AUTH, T_AFFIL = 46, 23, 22, 17
T_SEC, T_BODY, T_TAB, T_CAP, T_SMALL = 26, 18, 16, 14, 13
T_LEAD = 1.30

plt.rcParams.update({
    "font.family": SANS, "font.size": T_BODY, "mathtext.fontset": "dejavusans",
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GREY,
    "xtick.color": GREY, "ytick.color": GREY,
    "axes.linewidth": 1.0, "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "xtick.major.size": 4, "ytick.major.size": 4,
    "figure.facecolor": WHITE, "axes.facecolor": WHITE,
    "pdf.fonttype": 42, "savefig.facecolor": WHITE,
})

# ---------------------------------------------------------------------- data
HR = json.loads((RES / "headline_ratio.json").read_text())
CV = json.loads((RES / "cost_of_a_verdict.json").read_text())
PS = json.loads((RES / "pool_scaling.json").read_text())
CC = json.loads((RES / "clock_channel.json").read_text())
KVQ = json.loads((RES / "clock_channel_kvq.json").read_text())
SC = np.load(RES / "cost_of_a_verdict_scores.npz")

# The clock column is scored, not drawn, so it needs one thing the other two do
# not: the protocol itself. `detection` is imported from the experiment that
# published the slope verifier rather than reimplemented, so Figure 6B's pAUC
# curves are the run's own curves -- same `harness.evaluate`, same standardized
# pAUC @ FPR <= 0.5%, same honest calibration split as the token channel -- and
# the board cannot drift from the artifact. Everything else in the column is a
# measured latency read straight out of the committed timing grids.
from experiments.plot_clock_measured import price as clock_price   # (delta*/d')^2
from experiments.plot_slope_verifier import SIGMA_MAIN, detection, load_all

NICE_A = {"quant_4bit": "4-bit weights", "kv_fp8": "fp8 KV cache",
          "temp_1.1": "wrong temperature", "seed_43": "wrong seed",
          "bug_k2": "top-$k$ bug (k=2)", "bug_k32": "top-$k$ bug (k=32)"}
NICE_V = {"token_difr": "token\nDiFR", "cross_entropy": "cross\nentropy",
          "activation_difr": "activation\nDiFR", "token_toploc": "token\ntop-loc",
          "accept_rate": "accept\nrate", "surface_stat": "surface\nstat",
          "surface_tokens": "surface\ntokens"}

# ------------------------------------------------------------------- canvas
fig = plt.figure(figsize=(W, H))
# Text is positioned word by word so that `*emphasis*` can change colour mid
# line. Glyph widths come back quantised to whole device pixels, so at the
# default 100 dpi every word loses up to 0.01 in and a long line closes up on
# itself. Measure at 400 dpi; --dpi still sets the output resolution.
fig.set_dpi(400)
REND = fig.canvas.get_renderer()

fx = lambda xin: xin / W
fy = lambda yin: 1.0 - yin / H
fw = lambda win: win / W
fh = lambda hin: hin / H


def text_w(s, size, weight="normal", style="normal", family=SANS) -> float:
    """Width of `s` in INCHES, measured with the real font metrics."""
    t = fig.text(0, 0, s, size=size, weight=weight, style=style, family=family)
    bb = t.get_window_extent(renderer=REND)
    t.remove()
    return bb.width / fig.dpi



def _runs(s):
    """Split `s` into words; each word is a list of (text, emphasised) fragments.

    Three things have to survive the split. Words break on WHITESPACE only, so
    the `:` after `*relative*:` stays welded to the word rather than drifting
    off as its own token. `$...$` mathtext is atomic even when it contains
    spaces or a `*` (as `$(\\delta^*/d')^2$` does), so it is consumed whole
    before `*` is read as markup. And emphasis on a poster has to be legible
    from two metres, where italic at 16 pt is not -- so `*word*` is set in the
    accent colour at the same weight instead.
    """
    words, word, buf, emph, i = [], [], "", False, 0

    def end_frag():
        nonlocal buf
        if buf:
            word.append((buf, emph))
        buf = ""

    def end_word():
        nonlocal word
        end_frag()
        if word:
            words.append(word)
        word = []

    while i < len(s):
        c = s[i]
        if c == "$":                       # atomic mathtext span
            j = s.find("$", i + 1)
            j = len(s) - 1 if j < 0 else j
            buf += s[i:j + 1]
            i = j + 1
        elif c == "*":
            end_frag()
            emph = not emph
            i += 1
        elif c.isspace():
            end_word()
            i += 1
        else:
            buf += c
            i += 1
    end_word()
    return words


def para(x, y, s, w, size=T_BODY, color=INK, lead=T_LEAD, weight="normal",
         style="normal", family=SANS, indent0=0.0, emph_color=MAROON):
    """Wrap and draw, honouring `*emphasis*` and hard newlines."""
    dy = size * lead / 72
    space = (text_w("n n", size, weight, style, family)
             - text_w("nn", size, weight, style, family))

    # A mathtext span's bounding box grows upward with whatever is in it (a
    # radical, a superscript), so top-aligning it drops it below the line it
    # belongs to. Align those on the BASELINE instead, one text-ascent below
    # the top-aligned line.
    probe = fig.text(0, 0, "Hxp", size=size, family=family, va="baseline")
    asc = probe.get_window_extent(renderer=REND).y1 / fig.dpi
    probe.remove()

    def frag_w(word):
        return sum(text_w(t, size, weight, style, family) for t, _ in word)

    def draw(line, yy, indent):
        xx = x + indent
        for word in line:
            for text, em in word:
                math = "$" in text
                fig.text(fx(xx), fy(yy + asc) if math else fy(yy), text,
                         size=size, color=(emph_color if em else color),
                         weight=weight, style=style, family=family, ha="left",
                         va="baseline" if math else "top")
                xx += text_w(text, size, weight, style, family)
            xx += space

    indent = indent0
    for block in s.split("\n"):
        cur, cx = [], 0.0
        for word in _runs(block):
            ww = frag_w(word)
            if cur and cx + ww > w - indent:
                draw(cur, y, indent)
                y += dy
                cur, cx, indent = [], 0.0, 0.0
            cur.append(word)
            cx += ww + space
        if cur:
            draw(cur, y, indent)
            y += dy
            indent = 0.0
    return y


def rule(x, y, w, lw=1.4, color=RULE_C):
    fig.add_artist(Line2D([fx(x), fx(x + w)], [fy(y), fy(y)], lw=lw, color=color,
                          transform=fig.transFigure, zorder=5))


def section(x, y, w, title, color=MAROON):
    """The template's section head: maroon Libre Baskerville, hairline under."""
    size = T_SEC
    while text_w(title, size, "bold", family=SERIF) > w and size > 20:
        size -= 1
    fig.text(fx(x), fy(y), title, size=size, color=color, weight="bold",
             family=SERIF, va="top")
    y += size * 1.06 / 72
    rule(x, y, w, 1.6, RULE_C)
    return y + 0.17


def lead_line(x, y, s, w, size=T_BODY):
    """A maroon lead-in sentence -- the template uses these to open a section."""
    return para(x, y, s, w, size=size, color=MAROON, weight="bold")


def numbered(x, y, w, items, size=T_BODY):
    """The template's numbered CONTRIBUTIONS list: maroon numeral, hanging text."""
    for i, (head, body) in enumerate(items, 1):
        fig.text(fx(x), fy(y), f"{i}.", size=size, color=MAROON, weight="bold",
                 va="top")
        yy = para(x + 0.60, y, head, w - 0.60, size=size, weight="bold", color=INK)
        yy = para(x + 0.60, yy + 0.02, body, w - 0.60, size=size, color=GREY)
        y = yy + 0.20
    return y


def caption(x, y, w, label, text, size=T_CAP):
    pre = f"{label}  "
    ind = text_w(pre, size, "bold") + 0.06
    fig.text(fx(x), fy(y), pre, size=size, color=MAROON, weight="bold", va="top")
    return para(x, y, text, w, size=size, color=GREY, lead=1.26, indent0=ind,
                emph_color=INK)


def panel(x, y, w, h, fc=SURFACE):
    """The template's figure well."""
    fig.add_artist(Rectangle((fx(x), fy(y + h)), fw(w), fh(h), facecolor=fc,
                             edgecolor="none", transform=fig.transFigure, zorder=0))


def axes_at(x, y, w, h, frame=False):
    a = fig.add_axes([fx(x), fy(y + h), fw(w), fh(h)])
    a.set_facecolor("none")
    if not frame:
        for s in a.spines.values():
            s.set_visible(False)
        a.set_xticks([]), a.set_yticks([])
    else:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    a.tick_params(labelsize=T_SMALL, pad=2, colors=GREY)
    return a


def table(x, y, w, cols, rows, widths, size=T_TAB, head_bg=MAROON,
          aligns=None, zebra=True, colors=None, rowlead=1.62):
    """The template's table: maroon header band, hairline row rules, no verticals.

    `colors` is an optional [row][col] -> colour override, used to carry "this
    cell has no price" without a second legend.
    """
    aligns = aligns or (["left"] + ["right"] * (len(cols) - 1))
    tot = sum(widths)
    xs, acc = [], 0.0
    for ww in widths:
        xs.append(x + acc / tot * w)
        acc += ww
    cw = [ww / tot * w for ww in widths]
    hh = size * rowlead / 72

    fig.add_artist(Rectangle((fx(x), fy(y + hh)), fw(w), fh(hh), facecolor=head_bg,
                             edgecolor="none", transform=fig.transFigure, zorder=1))
    for j, c in enumerate(cols):
        ha = aligns[j]
        cx = xs[j] + (0.10 if ha == "left" else cw[j] - 0.10)
        fig.text(fx(cx), fy(y + hh / 2), c, size=size - 1, color=WHITE,
                 weight="bold", va="center", ha=ha, zorder=3)
    y += hh
    for i, r in enumerate(rows):
        if zebra and i % 2 == 1:
            fig.add_artist(Rectangle((fx(x), fy(y + hh)), fw(w), fh(hh),
                                     facecolor=SURFACE, edgecolor="none",
                                     transform=fig.transFigure, zorder=0))
        for j, cell in enumerate(r):
            ha = aligns[j]
            cx = xs[j] + (0.10 if ha == "left" else cw[j] - 0.10)
            col = INK if j == 0 else GREY
            wt = "bold" if j == 0 else "normal"
            if colors and colors[i][j]:
                col, wt = colors[i][j], "bold"
            fig.text(fx(cx), fy(y + hh / 2), cell, size=size, color=col,
                     weight=wt, va="center", ha=ha, zorder=3)
        y += hh
        rule(x, y, w, 0.8, RULE_C)
    return y


FOOT_Y = H - 0.55             # baseline of the footer band
FOOT_TOP = FOOT_Y - 1.50      # the rule above it; no column may cross this


def col_end(n, y):
    slack = FOOT_TOP - 0.20 - y
    print(f"  column {n} ends at {y:6.2f} in  ({slack:+5.2f} in slack)"
          + ("   <-- OVERRUN" if slack < 0 else ""))
    return slack


# ==========================================================================
# BANNER  (template: 36 x 4.00 in maroon, logo at 1.25/1.52, text block at 8.55)
# ==========================================================================
fig.add_artist(Rectangle((0, fy(4.00)), 1.0, fh(4.00), facecolor=MAROON,
                         edgecolor="none", transform=fig.transFigure, zorder=0))

logo = plt.imread(ASSETS / "mats_wordmark_white.png")
la = fig.add_axes([fx(1.25), fy(1.52 + 0.95), fw(5.77), fh(0.95)], zorder=2)
la.imshow(logo)
la.axis("off")
la.set_facecolor("none")

TX, TW = 8.55, 26.20
TITLE = "How do you audit a computation you cannot see?"
ts = T_TITLE
while text_w(TITLE, ts, "bold", family=SERIF) > TW and ts > 30:
    ts -= 1
fig.text(fx(TX), fy(0.52), TITLE, size=ts, color=WHITE, weight="bold",
         family=SERIF, va="top")
fig.text(fx(TX), fy(1.50), "A treaty clause, a promise to a regulator and an API "
         "bill are the same claim. An outsider has two ways to test it: the "
         "bytes that came back, and the clock.",
         size=T_SUB, color=CREAM, va="top")
fig.text(fx(TX), fy(2.22), "Jiawei Li¹      ·      mentored by Gabriel Kulp² "
         "and Roy Rinberg²", size=T_AUTH, color=WHITE, va="top")
fig.text(fx(TX), fy(2.82), "¹MATS 10.0 Scholar      ²MATS 10.0 Mentor      "
         "Code + every run artifact: github.com/lijiawei20161002/inference-verification",
         size=T_AFFIL, color=CREAM, va="top")

# column geometry, straight from the template
CX = [1.25, 12.72, 24.18]
CW = 10.57
Y0 = 4.62

# ==========================================================================
# COLUMN 1 -- the governed promise, the boundary, and the two observables
# ==========================================================================
x, y = CX[0], Y0


def box(a, cx, cy, w, h, label, sub=None, fc=WHITE, ec=GREY, lc=INK, lw=1.6,
        fs=15, subfs=12.5, subc=GREY, rad=1.2, ls="-", zo=2):
    a.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                               boxstyle=f"round,pad=0,rounding_size={rad}",
                               facecolor=fc, edgecolor=ec, linewidth=lw,
                               linestyle=ls, zorder=zo))
    dy = 0 if sub is None else h * 0.26
    a.text(cx, cy + dy, label, ha="center", va="center", fontsize=fs,
           color=lc, weight="bold", zorder=zo + 1)
    if sub:
        a.text(cx, cy - h * 0.15, sub, ha="center", va="center",
               fontsize=subfs, color=subc, zorder=zo + 1, linespacing=1.35)


def arrow(a, x0, y0, x1, y1, c=INK, lw=2.2, mut=16, ls="-"):
    a.annotate("", xy=(x1, y1), xytext=(x0, y0),
               arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                               mutation_scale=mut, linestyle=ls,
                               shrinkA=0, shrinkB=0), zorder=4)


y = section(x, y, CW, "THE AUDIT PROBLEM")
y = lead_line(x, y, "Two governments agree to stop training something. Who "
              "checks — and with what?", CW)
y = para(x, y + 0.08,
         "Every compute-governance instrument ends in a promise about work done "
         "inside a building the auditor cannot enter. Nothing in the stack "
         "between a GPU and the outside world emits a statement about what the "
         "silicon actually did, so the auditor is left with whatever crosses "
         "the boundary.", CW, color=GREY)

# ---- Figure 1: three promises, one sentence, one boundary, two wires
F1_H = 5.52
y += 0.16
panel(x, y, CW, F1_H)
ax = axes_at(x, y, CW, F1_H)
ax.set_xlim(0, 100), ax.set_ylim(0, 100)

ax.text(50, 99.5, "Three promises. One sentence. Two ways to test it.",
        ha="center", va="top", fontsize=16.5, weight="bold", color=INK)

for cx, head, sub in [
        (17.6, "A TREATY", "U.S. and China agree:\nneither trains above $X$"),
        (50.0, "A COMMITMENT", "a lab tells its regulator:\nthis model is frozen"),
        (82.4, "A BILL", "a provider charges you\nfor model $M$")]:
    box(ax, cx, 87.5, 31, 13.5, head, sub, fc=WHITE, ec=MAROON, lc=MAROON,
        fs=13.5, subfs=11.5, lw=1.5)
for x0, x1 in ((17.6, 41.0), (50.0, 50.0), (82.4, 59.0)):
    arrow(ax, x0, 80.6, x1, 75.0, c=MAROON, lw=1.8, mut=13)

box(ax, 50, 70.5, 74, 8.4, "“model $M$ ran under spec $\\varphi$ — and nothing "
    "else did”", "the sentence all three need, and no part of the stack produces",
    fc=CREAM, ec=MAROON, lc=MAROON, subc=MAROON, fs=15.5, subfs=11.5, lw=1.8)

# the boundary of what an auditor can see
ax.add_artist(Line2D([53.5, 53.5], [11.0, 61.0], color=INK, lw=3.0,
                     ls=(0, (5, 3)), zorder=3))
ax.text(53.5, 61.8, "the boundary", ha="center", va="bottom", fontsize=11.5,
        color=INK, weight="bold")

ax.add_patch(Rectangle((2.5, 12.0), 46.5, 47.0, facecolor="#ECEAE7",
                       edgecolor="none", zorder=0))
ax.text(25.75, 56.5, "INSIDE  ·  what nobody outside sees", ha="center",
        va="top", fontsize=12.5, color=GREY, weight="bold")
for cy, lab in [(47.0, "which weights were loaded"),
                (37.8, "at what precision, in what kernel"),
                (28.6, "how much context was really attended"),
                (19.4, "whether any of it was a training step")]:
    box(ax, 25.75, cy, 42, 7.2, lab, fc=WHITE, ec=C_DEAD, lc=GREY, fs=12,
        lw=1.2, ls=(0, (3, 2.4)), rad=0.8)

ax.text(78.5, 56.5, "OUTSIDE  ·  the auditor's entire budget", ha="center",
        va="top", fontsize=12.5, color=GREY, weight="bold")
for cy, col, head, tag in [(47.0, C_TOK, "CHANNEL 1 — WHAT IT SAID", "bytes out"),
                           (25.5, C_CLK, "CHANNEL 2 — WHEN IT SAID IT",
                            "the clock")]:
    arrow(ax, 49.5, cy, 57.2, cy, c=col, lw=2.6, mut=17)
    ax.text(53.4, cy + 1.4, tag, ha="center", va="bottom", fontsize=11,
            color=col, weight="bold")
    box(ax, 79.0, cy, 42, 6.6, head, fc=col, ec=col, lc=WHITE, fs=13, rad=0.8)

# what each channel literally is: nine tokens, and the nine gaps between them
for i in range(9):
    ax.add_patch(Rectangle((59.5 + i * 4.4, 37.4), 3.3, 4.6, facecolor="#DCE8F3",
                           edgecolor=C_TOK, lw=1.2, zorder=2))
ax.text(79.0, 34.8, "the tokens themselves — recompute and score them",
        ha="center", va="top", fontsize=11, color=GREY)

TICKS = [0.0, 4.0, 8.6, 12.4, 17.6, 21.4]
ax.add_artist(Line2D([59.0, 85.0], [15.0, 15.0], color=GREY, lw=1.2, zorder=2))
for t in TICKS:
    ax.add_artist(Line2D([59.5 + t, 59.5 + t], [15.0, 19.6], color=C_CLK, lw=2.2,
                         zorder=3))
ax.annotate("", xy=(59.5 + TICKS[5], 17.3), xytext=(59.5 + TICKS[4], 17.3),
            arrowprops=dict(arrowstyle="<|-|>", color=C_CLK, lw=1.1,
                            mutation_scale=8), zorder=4)
ax.text(84.0, 17.3, "$\\Delta t$", ha="left", va="center", fontsize=12,
        color=C_CLK, weight="bold")
ax.text(79.0, 12.8, "only the gaps between them — a stream already paid for",
        ha="center", va="top", fontsize=11, color=GREY)

ax.add_patch(Rectangle((2.5, 1.0), 96, 8.0, facecolor=WHITE, edgecolor=MAROON,
                       lw=1.4, zorder=1))
ax.text(50, 5.0, "Two observables, and no cooperation from the other side. This "
        "board measures what each one can prove,\nand what a verdict costs in "
        "each.", ha="center", va="center", fontsize=12.5, color=MAROON,
        weight="bold", zorder=3, linespacing=1.45)

y += F1_H
y = caption(x, y + 0.10, CW, "Figure 1",
            "The auditable surface of an inference provider. The same shape "
            "recurs one level up — a no-training clause is a promise about "
            "which computation ran on a fleet — so the honest place to start is "
            "the case where the auditor holds a spec and receives a stream, and "
            "every claim can be measured on real hardware.")

y = section(x, y + 0.26, CW, "WHY THE PROMISE IS WORTH BREAKING")
y = para(x, y, "A 4-bit copy, a smaller sibling or a truncated context is pure "
         "margin, and invisible in the tokens the client pays for. But the "
         "deviations worth doing are exactly the ones that *read fewer bytes* — "
         "which is why they leave a second trace.", CW, size=T_TAB, color=GREY)

y += 0.18


def _cell(label, ctx, mode="graph", B=1):
    return next(c for c in CC["cells"] if c["label"] == label and c["ctx"] == ctx
                and c["mode"] == mode and c["B"] == B)


BR = lambda a, b: _cell(*a)["bytes_read"] / _cell(*b)["bytes_read"]
y = table(x, y, CW,
          ["what the provider does instead", "bytes it saves", "worth it?"],
          [["Serve Qwen3-0.6B as Qwen3-1.7B",
            f"{BR(('Qwen3-1.7B', 256), ('Qwen3-0.6B', 256)):.2f}×", "capability"],
           ["4-bit (NF4) weights instead of bf16",
            f"{BR(('Qwen3-1.7B', 256), ('Qwen3-1.7B-NF4', 256)):.2f}×", "capacity"],
           ["Attend half the contracted context",
            f"{BR(('Qwen3-1.7B', 32768), ('Qwen3-1.7B', 16384)):.2f}×", "capacity"],
           ["Wrong seed / wrong temperature", "1.00×", "no"]],
          [5.6, 2.5, 2.5],
          aligns=["left", "right", "right"])
y = caption(x, y + 0.08, CW, "Table 1",
            "Weight + KV bytes per decode step, from the timing grid's own byte "
            "counts (clock_channel.json, H100 PCIe, B=1). The last row is the "
            "control: a deviation that saves no bytes leaves no clock signature "
            "at all — and the two channels split exactly there.")

# ---- Figure 2: the two channels are blind in opposite directions
y = section(x, y + 0.28, CW, "THE TWO CHANNELS ARE BLIND IN OPPOSITE DIRECTIONS")
F2A_H = 2.90
y += 0.06
panel(x, y, CW, F2A_H)
ax = axes_at(x, y, CW, F2A_H)
ax.set_xlim(0, 100), ax.set_ylim(0, 100)
ax.add_artist(Line2D([50, 50], [4, 96], color=RULE_C, lw=1.4, ls=(0, (4, 4))))

for cx, col, head, what, sees, blind in [
    (25.0, C_TOK, "CHANNEL 1 — WHAT IT SAID",
     "Recompute $M$ on the claimed tokens and read the\n"
     "per-token margin — or skip $M$ and score against a\n"
     "cheap trusted proxy $q$.",
     "the sampler, the seed, the exact distribution;\n"
     "wholesale substitution at AUC 0.998 with no $M$",
     "an fp8 KV cache — it moves the served\n"
     "distribution less than $M$'s own rerun noise"),
    (75.0, C_CLK, "CHANNEL 2 — WHEN IT SAID IT",
     "Read the arrival times of a stream already bought.\n"
     "Zero audit FLOPs, zero cooperation from the\n"
     "provider, zero added latency.",
     "which positions were attended, and which\n"
     "parameters were read at prefill",
     "anything reading the same bytes: wrong seed,\n"
     "wrong temperature, a top-$k$ bug"),
]:
    box(ax, cx, 91, 46, 11, head, fc=col, ec=col, lc=WHITE, fs=13.5, rad=0.8)
    left = cx - 23.0
    ax.text(left, 79, what, ha="left", va="top", fontsize=11.5, color=INK,
            linespacing=1.5)
    ax.text(left, 50, "sees", ha="left", va="top", fontsize=11.5, color=col,
            weight="bold")
    ax.text(left + 7.5, 50, sees, ha="left", va="top", fontsize=11.5, color=INK,
            linespacing=1.5)
    ax.text(left, 26, "blind", ha="left", va="top", fontsize=11.5, color=C_DEAD,
            weight="bold")
    ax.text(left + 8.2, 26, blind, ha="left", va="top", fontsize=11.5,
            color=GREY, linespacing=1.5)

y += F2A_H
y = caption(x, y + 0.10, CW, "Figure 2",
            "Complementary by mechanism, not by accident: a deviation either "
            "changes the distribution the tokens were drawn from, or the bytes "
            "the GPU had to move, or both. One protocol scores both channels — "
            "standardized pAUC @ FPR ≤ 0.5%, held-out honest calibration, "
            "batch/pool ratio ≤ 10% enforced in code.")

col_end(1, y)

# ==========================================================================
# COLUMN 2 -- the token verifier
# ==========================================================================
x, y = CX[1], Y0

y = section(x, y, CW, "CHANNEL 1  ·  READ WHAT CAME BACK", color=C_TOK)
y = lead_line(x, y, "One token proves almost nothing. A verdict is bought in "
              "bulk — and the price is fixed before you start.", CW)
y = para(x, y + 0.08,
         "Re-run $M$ on the tokens the provider claims it produced and compare "
         "the per-token sampling margin. The honest and the cheating stream "
         "overlap almost completely: the audit rests on one number, the "
         "per-token separation $d'$.", CW, color=GREY)

# ---- Figure 3: the batching principle, on this repo's own scores
F2_H = 4.35
y += 0.18
panel(x, y, CW, F2_H)

honest = SC["honest__token_difr"]
attack = SC["quant_4bit__token_difr"]
hi = np.percentile(honest, 99.9)                 # the protocol's own winsor point
hw = np.clip(honest, None, hi)
aw = np.clip(attack, None, hi)
dprime = (aw.mean() - hw.mean()) / hw.std()
GAP = aw.mean() - hw.mean()
ZERO = (hw == 0).mean()          # tokens whose margin carries nothing at all
# The law's own accuracy, recomputed from the pool-scaling run rather than typed
# in: mean |measured pAUC - pAUC predicted from d' alone| inside the ratio ceiling.
PSR = np.mean([abs(r["auc"] - r["predicted"]) for r in PS["rows"]
               if r["in_ceiling"]])
PSN = sum(1 for r in PS["rows"] if r["in_ceiling"])

fig.text(fx(x + CW / 2), fy(y + 0.18),
         "Averaging shrinks the noise. It never shrinks the gap.",
         size=16.5, weight="bold", color=INK, va="top", ha="center")
lx = x + 0.34
for c, lab in ((GREY, "honest provider"), (C_TOK, "provider serving 4-bit weights")):
    fig.add_artist(Rectangle((fx(lx), fy(y + 0.60)), fw(0.16), fh(0.11),
                             facecolor=c, alpha=0.45, edgecolor=c, lw=1.4,
                             transform=fig.transFigure, zorder=3))
    fig.text(fx(lx + 0.22), fy(y + 0.545), lab, size=12, color=c,
             weight="bold", va="center")
    lx += 0.30 + text_w(lab, 12, "bold")

# The three rows are the pooling law itself, with this experiment's measured
# per-token mean and spread substituted in: the sampling distribution of the
# audit statistic at pool size b. Drawing the raw bootstrap instead puts a comb
# of discreteness artefacts on the board -- 98% of per-token margins are exactly
# zero -- which is a fact about the score, not about the principle on show. The
# law's own accuracy against measured pAUC is the residual quoted in the caption.
BATCHES = [100, 600, 2494]
CALLOUT = 0.55
PA_H = (F2_H - 1.42 - CALLOUT) / 3
Z = 2.5758                                     # one-sided FPR of 0.5%
SE0_H, SE0_A = hw.std() / np.sqrt(BATCHES[0]), aw.std() / np.sqrt(BATCHES[0])
SPAN = (hw.mean() - 3.7 * SE0_H, aw.mean() + 3.7 * SE0_A)
SPAN = (SPAN[0], SPAN[1] + 0.30 * (SPAN[1] - SPAN[0]))     # room for the labels
grid = np.linspace(*SPAN, 700)
sf = lambda z: 0.5 * (1.0 - erf(z / np.sqrt(2.0)))

for k, n in enumerate(BATCHES):
    axk = axes_at(x + 0.34, y + 0.74 + k * PA_H, CW - 0.72, PA_H * 0.86,
                  frame=True)
    se_h, se_a = hw.std() / np.sqrt(n), aw.std() / np.sqrt(n)
    for mu, se, c in ((hw.mean(), se_h, GREY), (aw.mean(), se_a, C_TOK)):
        d = np.exp(-0.5 * ((grid - mu) / se) ** 2)
        axk.fill_between(grid, 0, d, color=c, alpha=0.30, lw=0, zorder=2)
        axk.plot(grid, d, color=c, lw=2.0, zorder=3)
        axk.plot([mu, mu], [0, 1.26], color=c, lw=1.2, ls=(0, (1, 2.5)), zorder=1)
    thr = hw.mean() + Z * se_h
    axk.plot([thr, thr], [0, 1.06], color=INK, lw=1.6, ls=(0, (3, 2)), zorder=4)
    power = sf((thr - aw.mean()) / se_a)
    axk.set_xlim(*SPAN), axk.set_ylim(0, 1.46)
    axk.set_yticks([]), axk.spines["left"].set_visible(False)
    axk.tick_params(labelsize=11.5, pad=1, length=3)
    if k < 2:
        axk.set_xticklabels([])
    axk.text(0.995, 0.99, f"pool $b$ = {n:,} tokens", transform=axk.transAxes,
             fontsize=13.5, color=INK, weight="bold", ha="right", va="top")
    axk.text(0.995, 0.60, f"{power*100:.0f}% of cheating providers convicted",
             transform=axk.transAxes, fontsize=12, color=C_TOK, ha="right",
             va="top")
    if k == 0:
        axk.annotate("convict to the right of this bar", xy=(thr, 1.04),
                     xytext=(thr + 0.10, 1.30), fontsize=11, color=INK,
                     va="center", ha="left",
                     arrowprops=dict(arrowstyle="-", color=INK, lw=0.9))
    if k == 2:
        axk.set_xlabel("average per-token margin over the audited pool",
                       fontsize=13, color=GREY, labelpad=1)
        axk.annotate("", xy=(hw.mean(), 1.22), xytext=(aw.mean(), 1.22),
                     arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.3,
                                     mutation_scale=11), zorder=5)
        axk.text(aw.mean() + 0.09, 1.22, f"the gap never moved: {GAP:.3f} nats",
                 fontsize=11, color=INK, va="center")

fig.add_artist(Rectangle((fx(x + 0.34), fy(y + F2_H - 0.14)), fw(CW - 0.68),
                         fh(CALLOUT - 0.10), facecolor=CREAM, edgecolor=MAROON,
                         lw=1.4, transform=fig.transFigure, zorder=1))
fig.text(fx(x + CW / 2), fy(y + F2_H - 0.14 - (CALLOUT - 0.10) / 2),
         f"{ZERO*100:.0f}% of individual tokens carry no signal at all.  Noise "
         f"falls as $1/\\sqrt{{b}}$, so the bill is $b = (\\delta^*/d')^2$ "
         f"tokens:\nfix the confidence an auditor needs, and $d'$ fixes the "
         f"price.", size=13, color=MAROON, weight="bold", ha="center",
         va="center", zorder=3, linespacing=1.45)

y += F2_H
y = caption(x, y + 0.10, CW, "Figure 3",
            f"The pooling law with this experiment's own measured inputs — "
            f"Qwen3-1.7B under 4-bit weights, per-token $d'$ = {dprime:.3f} "
            f"(cost_of_a_verdict_scores.npz). Against *measured* pAUC at {PSN} "
            f"pool sizes on a 56,160-token pool the law holds to a mean absolute "
            f"residual of {PSR:.3f} (pool_scaling.json) — which is what lets an "
            f"auditor price a verdict before spending a budget.")

# ---- Table: the headline grid
y = section(x, y + 0.26, CW, "WHAT CATCHES WHAT", color=C_TOK)
y = para(x, y, "Every detector is structurally blind to a different class of "
         "deviation, and which one wins is set by the *attack*, not the model. "
         "An auditor has to hold a portfolio.", CW, size=T_TAB, color=GREY)

y += 0.18
defs = HR["defenses"]
atks = HR["attacks"]
grid_h = 0.315 * (len(atks) + 1) + 0.16
axh = axes_at(x, y, CW, grid_h)
axh.set_xlim(0, len(defs) + 2.35), axh.set_ylim(0, len(atks) + 1.15)
for j, v in enumerate(defs):
    axh.text(2.35 + j + 0.5, len(atks) + 0.55, NICE_V[v], ha="center", va="center",
             fontsize=13, color=INK, weight="bold", linespacing=1.15)
for i, a in enumerate(atks):
    r = len(atks) - 1 - i
    axh.text(2.25, r + 0.5, NICE_A[a], ha="right", va="center", fontsize=13.5,
             color=INK, weight="bold")
    for j, v in enumerate(defs):
        auc = HR["cells"][a][v]["ratioed"]["auc"]
        t = np.clip((auc - 0.5) / 0.5, 0, 1)
        fc = plt.matplotlib.colors.to_rgb(C_TOK)
        col = tuple(1 - (1 - c) * (0.10 + 0.90 * t) for c in fc)
        axh.add_patch(Rectangle((2.35 + j + 0.04, r + 0.06), 0.92, 0.88,
                                facecolor=col, edgecolor="none"))
        axh.text(2.35 + j + 0.5, r + 0.5, f"{auc:.3f}", ha="center", va="center",
                 fontsize=13.5, weight="bold",
                 color=WHITE if t > 0.55 else INK)
axh.set_xticks([]), axh.set_yticks([])
y += grid_h
y = caption(x, y + 0.02, CW, "Table 2",
            "headline_ratio.json. Re-scored from the *same* per-token scores as "
            "a previously published table that resampled from too small a pool, "
            "16 of 24 cells fall, by a median of 0.137: an over-ratio pool does "
            "not exaggerate a signal, it invents one.")

# ---- Figure 4: what a verdict costs, from outside and from inside
y = section(x, y + 0.24, CW, "WHAT ONE VERDICT COSTS", color=C_TOK)
y = para(x, y, "Detectors are usually ranked by accuracy. An auditor buys "
         "neither an AUC nor a FLOP count — it buys tokens. Below: the cheapest "
         "honest verdict for each deviation, and what the *same* verdict costs "
         "if the provider is made to open up.", CW, size=T_TAB, color=GREY)

CTRL = CV["honest_control"]
BB = [v for v in CV["verifiers"] if v != "activation_difr"]


def _price(a, vs):
    best = None
    for v in vs:
        c = CV["cells"][a][v]
        if c["reachable"] and c["d_prime"] > CTRL[v]["d_prime"]:
            if best is None or c["tokens_per_verdict"] < best[1]:
                best = (v, c["tokens_per_verdict"])
    return best


F4_H = 2.72
y += 0.12
panel(x, y, CW, F4_H)
axp = axes_at(x + 2.62, y + 0.46, CW - 3.10, F4_H - 1.06, frame=True)
alist = ["quant_4bit", "kv_fp8", "bug_k32", "temp_1.1", "seed_43"]
axp.set_xscale("log"), axp.set_xlim(0.6, 2.5e6)
axp.set_ylim(-0.7, len(alist) + 0.42)
axp.set_yticks([])
axp.set_xticks([1, 1e2, 1e4]), axp.minorticks_off()
axp.set_xticklabels(["1 token", "100", "10,000"])
axp.tick_params(labelsize=12)
axp.set_xlabel("tokens of stream the auditor must buy for one verdict",
               fontsize=12.5, color=GREY, labelpad=2)
for i, a in enumerate(alist):
    r = len(alist) - 1 - i
    # the row label is drawn in DATA coords so it can never drift off its bar
    axp.text(0.45, r, NICE_A[a], fontsize=13, color=INK, weight="bold",
             ha="right", va="center", clip_on=False)
    bb_ = _price(a, BB)
    axp.barh(r, bb_[1], height=0.42, color=C_TOK, zorder=3)
    axp.text(bb_[1] * 1.30, r, f"{bb_[1]:,}", fontsize=12.5, color=C_TOK,
             weight="bold", va="center")
    axp.text(bb_[1] * 1.30 * 10 ** (0.06 * len(f"{bb_[1]:,}")), r,
             "·  " + NICE_V[bb_[0]].replace("\n", " "), fontsize=11, color=GREY,
             va="center")
    ins = CV["cells"][a]["activation_difr"]
    if (ins["reachable"] and ins["d_prime"] > CTRL["activation_difr"]["d_prime"]
            and ins["tokens_per_verdict"] < bb_[1]):
        axp.plot([ins["tokens_per_verdict"]], [r + 0.36], marker="v", ms=12,
                 color=MAROON, zorder=5, clip_on=False)
        axp.text(ins["tokens_per_verdict"] * 1.7, r + 0.37,
                 f"{ins['tokens_per_verdict']}", fontsize=12, color=MAROON,
                 weight="bold", va="center")
    else:
        axp.text(2.0e6, r, "no verdict from inside", fontsize=11,
                 color=C_DEAD, va="center", ha="right")

axp.legend(handles=[Rectangle((0, 0), 1, 1, facecolor=C_TOK, edgecolor="none",
                              label="from outside — black-box audit"),
                    Line2D([0], [0], marker="v", ms=11, lw=0,
                           markerfacecolor=MAROON, markeredgecolor=MAROON,
                           label="with inside access to activations")],
           fontsize=12, frameon=False, loc="upper right", handlelength=1.3,
           labelspacing=0.3, borderaxespad=0.05, ncol=2, columnspacing=1.6)

y += F4_H
y = caption(x, y + 0.08, CW, "Figure 4",
            f"cost_of_a_verdict.json; Qwen3-1.7B, $\\delta^*$ = "
            f"{CV['delta_star']:.2f}, 141 µs of H100 per recomputed token. "
            f"Opening the provider up is worth 2,000× against quantization and "
            f"*nothing* against a tampered sampler, which redraws from the "
            f"honest distribution. Across the whole grid 14 of 35 cells have no "
            f"price at all — and ranking by AUC hides every one of them.")
col_end(2, y)

# ==========================================================================
# COLUMN 3 -- the clock verifier, and only what was measured of it
# ==========================================================================
x, y = CX[2], Y0


def _cells(label, ctx, mode="graph", B=1, src=None):
    return [c for c in (src or CC["cells"]) if c["label"] == label
            and c["ctx"] == ctx and c["mode"] == mode and c["B"] == B]


def _avg(label, ctx, key, mode="graph", B=1):
    """Mean of `key` over every cell at this configuration -- the grid re-measures
    a few contexts in a second arm and one row must not be double-weighted."""
    return float(np.mean([c[key] for c in _cells(label, ctx, mode, B)]))


def _lsq(xv, yv):
    A = np.vstack([np.ones_like(xv), xv]).T
    return np.linalg.lstsq(A, np.asarray(yv, float), rcond=None)[0]


# the slope verifier's own cells, and its verdicts re-scored through the protocol
SV, SVC, _SV_MAIN = load_all()
DET = detection(SVC, SIGMA_MAIN, seed=0)
WCELL = lambda lab: next(c for c in SVC if c["label"] == lab)
LO, HI = WCELL("probe_lo"), WCELL("honest_hi")
WS = sorted(DET)                                # context a truncating provider keeps
CLAIM = SV["claimed_ctx"]
D_HON = HI["mean"] - LO["mean"]                 # the honest probe-pair statistic
# tokens of stream for a pAUC-0.90 verdict: (delta*/d')^2 pairs, 2 tokens a pair
TOKS = {Wk: int(round(float(clock_price(DET[Wk]["d_prime"])) * 2))
        for Wk in WS}
SHADE = {Wk: 1.0 - 0.45 * i / max(len(WS) - 1, 1) for i, Wk in enumerate(WS)}


def clk(Wk):
    """One hue for the whole deviation family, dark = keeps least of the context."""
    r, g, b = plt.matplotlib.colors.to_rgb(C_CLK)
    return tuple(1 - (1 - c) * SHADE[Wk] for c in (r, g, b))


y = section(x, y, CW, "CHANNEL 2  ·  READ WHEN IT CAME BACK", color=C_CLK)
y = lead_line(x, y, "A token cannot arrive before its bytes have moved. The "
              "saving and the evidence are the same quantity.", CW)
y = para(x, y + 0.08,
         "One token at a time a GPU is memory-bound: it must drag every weight "
         "it claims and read every position it charged for before it can emit "
         "anything, so $t_{\\mathrm{token}} \\geq$ bytes $/$ bandwidth. Nothing "
         "below is drawn — the whole column is timed on one H100.", CW,
         color=GREY)

# ---- Figure 5: the premise, measured, and the reason it is not a test
F3_H = 2.84
y += 0.14
panel(x, y, CW, F3_H)

fig.text(fx(x + 0.28), fy(y + 0.17), "A.  The channel is real: the time is in "
         "the bytes", size=14, weight="bold", color=INK, va="top")
fig.text(fx(x + CW * 0.615), fy(y + 0.17), "B.  The floor is not",
         size=14, weight="bold", color=INK, va="top")

CTX0 = min(c["ctx"] for c in CC["cells"])
MODELS = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B"]
NF4 = "Qwen3-1.7B-NF4"
BW = CC["card"]["bw_copy_tb_s"]
gb = np.array([_cells(m, CTX0)[0]["weight_bytes"] / 1e9 for m in MODELS])
gb4 = _cells(NF4, CTX0)[0]["weight_bytes"] / 1e9
XMAX = gb.max() * 1.17
# weight + KV bytes per step, exactly as Table 1 counts them one column left
BYTE_R = (_cells(MODELS[1], CTX0)[0]["bytes_read"]
          / _cells(NF4, CTX0)[0]["bytes_read"])

ax2 = axes_at(x + 1.02, y + 0.56, CW * 0.505, F3_H - 1.08, frame=True)
ms = np.array([_avg(m, CTX0, "mean") for m in MODELS])
ms4 = _avg(NF4, CTX0, "mean")
const, slope = _lsq(gb, ms)
xs = np.linspace(0, XMAX, 50)
ax2.plot(xs, const + slope * xs, color=C_TOK, lw=2.2, ls=(0, (5, 3)), zorder=2)
ax2.plot(xs, xs / BW, color=GREY, lw=1.8, zorder=2)
ax2.scatter(gb, ms, s=105, color=C_TOK, zorder=4)
for g, m, lab in zip(gb, ms, MODELS):
    ax2.annotate(lab.replace("Qwen3-", ""), (g, m), textcoords="offset points",
                 xytext=(9, -11), fontsize=12, color=INK)
# the deviation, on the same axes: it reads a 0.6B's bytes and costs more time
ax2.scatter([gb4], [ms4], s=115, marker="D", color=MAROON, zorder=5,
            edgecolor=WHITE, linewidth=1.2)
ax2.annotate("the 4-bit provider", (gb4, ms4), textcoords="offset points",
             xytext=(-2, 12), ha="center", va="bottom", fontsize=11,
             color=MAROON, weight="bold")
ax2.text(0.035, 0.975, f"observed  =  {const:.2f} ms  +  1 GB per {slope:.2f} ms\n"
         f"the weight read runs at {1/slope/BW*100:.0f}% of this card's measured "
         f"{BW:.2f} TB/s", transform=ax2.transAxes, fontsize=11.5, color=C_TOK,
         weight="bold", va="top", linespacing=1.4)
ax2.text(0.985, 0.055, "what the bus alone would give", transform=ax2.transAxes,
         fontsize=11, color=GREY, ha="right", va="bottom")
ax2.set_xlabel("weight bytes per decode step (GB)", fontsize=12, color=GREY,
               labelpad=1)
ax2.set_ylabel("ms per output token", fontsize=12, color=GREY, labelpad=2)
ax2.set_xlim(0, XMAX), ax2.set_ylim(0, ms.max() * 1.32)
ax2.tick_params(labelsize=11.5)

# the same GPU, the same weights, one stack down: the constant moves 5x
ax3 = axes_at(x + CW * 0.675, y + 0.56, CW * 0.275, F3_H - 1.08, frame=True)
msE = np.array([_avg(m, CTX0, "mean", mode="eager") for m in MODELS])
sdE = np.array([_avg(m, CTX0, "sd", mode="eager") for m in MODELS])
ms4E = _avg(NF4, CTX0, "mean", mode="eager")
constE, slopeE = _lsq(gb, msE)
ax3.errorbar(gb, msE, yerr=sdE, fmt="o", ms=9, color=C_CLK, lw=1.6, capsize=4,
             zorder=4)
ax3.scatter([gb4], [ms4E], s=110, marker="D", color=MAROON, zorder=5,
            edgecolor=WHITE, linewidth=1.2)
ax3.axhline(constE, color=C_CLK, lw=1.6, ls=(0, (5, 3)), zorder=2)
ax3.plot(xs, const + slope * xs, color=C_TOK, lw=1.8, ls=(0, (5, 3)), zorder=2)
ax3.text(0.96, constE + 0.9, f"eager: {constE:.0f} ms of stack",
         transform=ax3.get_yaxis_transform(), size=11.5, color=C_CLK,
         weight="bold", ha="right", va="bottom")
ax3.annotate(f"CUDA graphs: {const:.2f} ms",
             (XMAX * 0.55, const + slope * XMAX * 0.55),
             textcoords="offset points", xytext=(0, -13), ha="center", va="top",
             fontsize=11, color=C_TOK)
ax3.annotate("4-bit", (gb4, ms4E), textcoords="offset points", xytext=(9, 1),
             va="center", fontsize=11, color=MAROON, weight="bold")
ax3.set_xlabel("the same weight bytes (GB)", fontsize=12, color=GREY, labelpad=1)
ax3.set_ylabel("ms per output token", fontsize=11.5, color=GREY, labelpad=2)
ax3.set_xlim(0, XMAX), ax3.set_ylim(0, max(msE.max(), ms4E) * 1.14)
ax3.tick_params(labelsize=11.5)

y += F3_H
GAP_G = _avg(MODELS[1], 1024, "mean") - _avg(MODELS[0], 1024, "mean")
SD_G = _avg(MODELS[1], 1024, "sd")
GAP_E = (_avg(MODELS[1], 1024, "mean", mode="eager")
         - _avg(MODELS[0], 1024, "mean", mode="eager"))
SD_E = _avg(MODELS[1], 1024, "sd", mode="eager")
y = caption(x, y + 0.07, CW, "Figure 5",
            f"clock_channel.json, H100 PCIe, B=1, {CTX0}-token context, "
            f"{CC['reps']}×{CC['steps']} timed steps per cell. *(A)* The 4-bit provider reads "
            f"{BYTE_R:.2f}× fewer bytes and still costs "
            f"{ms4 / (const + slope * gb4):.2f}× what this line prices them at: "
            f"dequantization spends the saving. *(B)* is the same GPU and the same weights "
            f"one stack down: the additive constant moves {const:.2f} → "
            f"{constE:.0f} ms, and the 1.7B→0.6B signal falls from {GAP_G:.2f} "
            f"ms on {SD_G:.2f} ms of device noise to {GAP_E:.2f} on {SD_E:.2f}. "
            f"The slope is physics; the intercept is a stack, and it is on no "
            f"spec sheet.")

# ---- Table: what measurement did to the naive clock
y = section(x, y + 0.20, CW, "EVERY OBVIOUS CLOCK TEST FAILS", color=C_CLK)
y += 0.02
REALISED = (ms[1] - ms4) / (ms[1] - ms[1] / BYTE_R)     # of the predicted saving
B_SLOW = min(B for B in (2, 4, 8, 16, 32, 64)
             if _cells(NF4, 1024, B=B) and _avg(NF4, 1024, "mean", B=B)
             > _avg(MODELS[1], 1024, "mean", B=B))
GAP_B1 = _avg(MODELS[1], 1024, "mean", B=1) - _avg(MODELS[0], 1024, "mean", B=1)
GAP_B64 = _avg(MODELS[1], 1024, "mean", B=64) - _avg(MODELS[0], 1024, "mean", B=64)
SD_B64 = _avg(MODELS[1], 1024, "sd", B=64) / _avg(MODELS[1], 1024, "sd", B=1)
# The int4 cache's own context slope, on minima and from 1024 up: the 256-cell
# pays a one-off quanto JIT compile that is not a property of the cache.
KVS = {}
for mode in ("eager", "kvq4"):
    ks = sorted((c for c in KVQ["cells"] if c["mode"] == mode and c["ctx"] > CTX0),
                key=lambda c: c["ctx"])
    KVS[mode] = _lsq(np.array([c["ctx"] for c in ks], float),
                     [c["min"] for c in ks])[1]
ZG = CC["specdec"]["frac_gaps_zero"]
y = table(x, y, CW,
          ["what the premise seems to promise", "what 126 timing cells say"],
          [["Fewer bytes → proportionally less time",
            f"4-bit returns {REALISED*100:.0f}% of it, and is slower at B ≥ "
            f"{B_SLOW}"],
           ["Batching amortizes the weight read away",
            f"the gap is flat ({GAP_B1:.2f} → {GAP_B64:.2f} ms); the noise "
            f"grows {SD_B64:.0f}×"],
           ["Half the KV bytes → half the slope",
            f"a real int4 cache is {KVS['kvq4']/KVS['eager']:.1f}× steeper"],
           ["Jitter is positive, so take the minimum gap",
            f"{ZG*100:.0f}% of an honest server's gaps are 0 ms"]],
          [5.55, 5.02], aligns=["left", "right"], size=T_TAB - 1, head_bg=C_CLK)
y = caption(x, y + 0.08, CW, "Table 3",
            "clock_channel.json, _arch.json and _kvq.json (126 configurations, "
            "a real NF4 provider, a real quanto int4 cache, real speculative "
            "decoding). The premise survives every row; the naive *absolute* "
            "test survives none.")

# ---- Figure 6: the differential test, measured, and priced by the same law
y = section(x, y + 0.18, CW, "THE TEST THAT SURVIVES: SUBTRACT", color=C_CLK)
F6_H = 2.98
y += 0.06
panel(x, y, CW, F6_H)
fig.text(fx(x + 0.30), fy(y + 0.17), f"A.  A provider that bills for "
         f"{CLAIM:,} tokens of context and holds $W$", size=14, weight="bold",
         color=INK, va="top")
fig.text(fx(x + CW * 0.655), fy(y + 0.17), "B.  What that verdict costs",
         size=14, weight="bold", color=INK, va="top")

axa = axes_at(x + 1.00, y + 0.56, CW * 0.505, F6_H - 1.10, frame=True)
CELLS6 = [LO] + [WCELL(f"window_{Wk}") for Wk in WS] + [HI]
ectx = np.array([c["effective_ctx"] for c in CELLS6], float)
eitl = np.array([c["mean"] for c in CELLS6])
XR, YR = CLAIM * 1.34, HI["mean"] * 1.20
axa.plot(ectx, eitl, "-", color=C_TOK, lw=2.0, zorder=3)
axa.plot([LO["effective_ctx"], HI["effective_ctx"]],
         [LO["mean"], HI["mean"]], "o", ms=11, color=C_TOK, zorder=6,
         mec=WHITE, mew=1.2)
axa.plot([CLAIM, CLAIM], [0, HI["mean"]], color=RULE_C, lw=1.2, ls=(0, (4, 3)),
         zorder=1)
for ky, mk, mc, lab in (
        (0.955, "o", C_TOK, f"the probe pair: A reads {LO['effective_ctx']}, "
                            f"B bills {CLAIM:,}"),
        (0.868, "D", clk(WS[len(WS) // 2]),
         f"one server, {len(WS)} windows: reads $W$, bills {CLAIM:,}")):
    axa.plot([0.035], [ky], mk, ms=9 if mk == "o" else 8, color=mc,
             transform=axa.transAxes, clip_on=False, zorder=6)
    axa.text(0.072, ky, lab, transform=axa.transAxes, fontsize=11, color=INK,
             va="center")
# the D column: one measured probe-pair statistic per provider, decluttered so
# the two heavily truncated rows do not sit on top of each other
axa.text(CLAIM * 1.05, YR * 0.965, "$D = t_B - t_A$", fontsize=11.5, color=GREY,
         va="top")
rows = [(HI["mean"], C_TOK, f"{D_HON:.1f} ms")] + [
    (WCELL(f"window_{Wk}")["mean"], clk(Wk),
     f"{WCELL(f'window_{Wk}')['mean'] - LO['mean']:.1f}") for Wk in WS]
place, last = {}, None
for val, col, lab in sorted(rows, key=lambda r: r[0]):
    yy = val if last is None else max(val, last + YR * 0.085)
    place[lab] = (yy, col)
    last = yy
for val, col, lab in rows:
    yy, _ = place[lab]
    axa.text(CLAIM * 1.05, yy, lab, fontsize=11.5, color=col, weight="bold",
             va="center")
axa.annotate("", xy=(CLAIM * 1.015, HI["mean"]),
             xytext=(CLAIM * 1.015, LO["mean"]),
             arrowprops=dict(arrowstyle="<|-|>", color=C_TOK, lw=1.4,
                             mutation_scale=10), zorder=6)
for Wk in WS:
    c, cc = WCELL(f"window_{Wk}"), clk(Wk)
    axa.annotate("", xy=(CLAIM, c["mean"]), xytext=(c["effective_ctx"], c["mean"]),
                 arrowprops=dict(arrowstyle="-|>", color=cc, lw=1.7,
                                 linestyle=(0, (3, 2)), mutation_scale=12,
                                 shrinkA=4, shrinkB=0), zorder=4)
    axa.plot([c["effective_ctx"]], [c["mean"]], "D", ms=8, color=cc, zorder=5,
             mec=WHITE, mew=1.0)
axa.set_xlim(0, XR), axa.set_ylim(0, YR)
axa.set_xticks([0, 8192, 16384, 24576, CLAIM])
axa.set_xticklabels(["0", "8k", "16k", "24k", "32k"])
axa.set_xlabel("context the provider actually attended (tokens)", fontsize=12,
               color=GREY, labelpad=1)
axa.set_ylabel("ms per output token", fontsize=12, color=GREY, labelpad=2)
axa.tick_params(labelsize=11.5)

axb = axes_at(x + CW * 0.665, y + 0.56, CW * 0.285, F6_H - 1.10, frame=True)
BS = sorted({t.batch_size for Wk in WS for t in DET[Wk]["res"]})
for Wk in WS:
    r = DET[Wk]["res"]
    axb.plot([t.batch_size for t in r], [t.auc for t in r], "-o", ms=5.0,
             color=clk(Wk), lw=1.9, zorder=4)
# the target rule is drawn short so it clears the key rather than striking it
axb.plot([BS[2], BS[-1] * 1.3], [0.90, 0.90], color=INK, lw=1.2, ls=(0, (3, 2)),
         zorder=3)
axb.text(BS[-1] * 0.88, 0.886, "pAUC 0.90", fontsize=10.5, color=INK,
         ha="right", va="top")
axb.set_xscale("log", base=2)
axb.set_xlim(BS[0] * 0.8, BS[-1] * 1.4), axb.set_ylim(0.45, 1.14)
axb.set_yticks([0.5, 0.75, 1.0])
axb.set_xlabel("probe pairs per verdict", fontsize=12, color=GREY, labelpad=1)
axb.set_ylabel("pAUC @ FPR $\\leq$ 0.5%", fontsize=12, color=GREY, labelpad=2)
axb.tick_params(labelsize=11.5)
ly = 0.965
for Wk in WS:
    axb.add_patch(Rectangle((0.035, ly - 0.018), 0.058, 0.036, facecolor=clk(Wk),
                            edgecolor="none", transform=axb.transAxes, zorder=6))
    axb.text(0.120, ly, f"holds {Wk:,}  ·  {TOKS[Wk]} tokens",
             transform=axb.transAxes, fontsize=10.5, color=INK, va="center")
    ly -= 0.080

y += F6_H
y = caption(x, y + 0.07, CW, "Figure 6",
            f"slope_verifier_window.json, {len(LO['itl_ms']):,} timed decode "
            f"steps per cell, scored through the *same* harness.evaluate as every "
            f"token verifier. *(A)* is measurement only: the subtraction cancels "
            f"Figure 5's stack constant, and hiding from it costs the "
            f"{D_HON - (WCELL(f'window_{WS[0]}')['mean'] - LO['mean']):.0f} ms per "
            f"token that truncation saved. *(B)* adds the column's one modelled "
            f"input, σ = {SIGMA_MAIN:.0f} ms of client-side wire jitter — "
            f"pessimistic, and one-sided: a busy neighbour only pushes $D$ up, "
            f"toward the honest side, so co-tenancy costs power, never a false "
            f"accusation. Read at prefill rather than in the gap the same premise is "
            f"25,000× cheaper again on the substitution row (clock_algos.json).")

# ---- The close: what any of this buys a governance regime
y = section(x, y + 0.20, CW, "WHAT AN AUDITOR ACTUALLY GETS")
y = para(x, y,
         "*A price, not a ranking:* every verdict costed in tokens × seconds — "
         "and 14 of 35 cells cost infinity.\n"
         "*One protocol, one grid:* 8 detectors × 6 deviations; under an "
         "enforced pool ceiling, 16 of 24 cells fall.\n"
         f"*A second channel, on the same board:* {TOKS[WS[0]]}–{TOKS[WS[-1]]} "
         f"tokens of stream, no audit FLOPs, no cooperation.", CW, size=T_TAB,
         color=INK)
y = para(x, y + 0.09,
         "None of this is a treaty regime — but it prices one. A clause naming a "
         "detector cannot be enforced; one naming a pool size, a false-alarm rate "
         "and a probe pair can be. The gaps are honest: a fleet over months is "
         "not a stream over seconds; the wire is modelled, the device measured; "
         "models are 0.13B–8B; no provider here adapts.", CW, size=T_TAB,
         color=GREY)

col_end(3, y)

# ==========================================================================
# FOOTER -- one band across all three columns: acknowledgments, references, QR
# ==========================================================================
FX0, FX1 = CX[0], CX[2] + CW
rule(FX0, FOOT_TOP, FX1 - FX0, 1.6, MAROON)
fy_ = FOOT_TOP + 0.16
fig.text(fx(FX0), fy(fy_), "ACKNOWLEDGMENTS", size=T_SMALL, color=MAROON,
         weight="bold", family=SERIF, va="top")
para(FX0 + 2.55, fy_, "This work was carried out in the Machine Alignment, "
     "Transparency & Security (MATS) program, cohort 10.0. Thanks to Gabriel "
     "Kulp and Roy Rinberg for mentorship and to MATS for compute. "
     "Correspondence: jiawei2012220@gmail.com", 19.5, size=T_SMALL, color=GREY)
fig.text(fx(FX0), fy(fy_ + 0.62), "REFERENCES", size=T_SMALL, color=MAROON,
         weight="bold", family=SERIF, va="top")
para(FX0 + 2.55, fy_ + 0.62, "[1] Karvonen et al. DiFR: Inference Verification "
     "Despite Nondeterminism, 2025 — the Tier-1 methodology, reimplemented "
     "clean-room here.   [2] RoyRin, inference_verification_for_model_weight_"
     "exfil — the threat model.   [3] Every number on this board regenerates "
     "from docs/results/ via paper/make_mats_poster.py.", 19.5, size=T_SMALL,
     color=GREY)

try:
    import qrcode
    for i, (url, lab) in enumerate([
            ("https://github.com/lijiawei20161002/inference-verification/blob/main/paper/paper.pdf",
             "Paper"),
            ("https://github.com/lijiawei20161002/inference-verification",
             "Code")]):
        img = qrcode.make(url, border=1).convert("L")
        qx = FX1 - 1.15 - i * 1.55
        aq = fig.add_axes([fx(qx), fy(FOOT_TOP + 1.48), fw(1.10), fh(1.10)],
                          zorder=6)
        aq.imshow(np.asarray(img), cmap="gray", vmin=0, vmax=255)
        aq.axis("off")
        fig.text(fx(qx + 0.575), fy(FOOT_TOP + 0.14), lab, size=T_SMALL,
                 color=GREY, ha="center", va="top", weight="bold")
except Exception as exc:                                   # pragma: no cover
    print(f"  (no QR codes: {exc})")

# ==========================================================================
out_png = Path(__file__).resolve().parent / f"{args.out}.png"
out_pdf = out_png.with_suffix(".pdf")
fig.savefig(out_png, dpi=args.dpi)
fig.savefig(out_pdf)
print(f"  wrote {out_png}  ({W:.0f}x{H:.0f} in @ {args.dpi:.0f} dpi)")
print(f"  wrote {out_pdf}")
