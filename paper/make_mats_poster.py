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
grid, the verdict prices, the pooling law's inputs and residual, and the clock
roofline all come out of JSON at render time, so a rerun that moves a number
moves the poster. Figure 3 is the one drawn curve -- the pooling law's normal
approximation with this experiment's measured mean, spread and d' substituted in,
because the raw bootstrap of a score that is exactly zero on 98% of tokens is a
comb of discreteness artefacts rather than a picture of the principle.

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
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

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
SC = np.load(RES / "cost_of_a_verdict_scores.npz")

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
# COLUMN 3 -- the clock verifier
# ==========================================================================
x, y = CX[2], Y0

y = section(x, y, CW, "CHANNEL 2  ·  READ WHEN IT CAME BACK", color=C_CLK)
y = lead_line(x, y, "A token cannot arrive before its bytes have moved. The "
              "saving and the evidence are the same quantity.", CW)
y = para(x, y + 0.08,
         "One token at a time a GPU is memory-bound, not compute-bound: it must "
         "drag every weight it claims to be using across the bus before it can "
         "emit anything. Read fewer bytes and tokens arrive sooner — so the "
         "*clock* is evidence, at no audit FLOPs and no cooperation.", CW,
         color=GREY)

# ---- Figure 5: the premise, drawn; then the premise, measured
F3_H = 4.08
y += 0.18
panel(x, y, CW, F3_H)

fig.text(fx(x + 0.24), fy(y + 0.20), "A.  Why the clock knows anything",
         size=14.5, weight="bold", color=INK, va="top")
fig.text(fx(x + CW * 0.475), fy(y + 0.20), "B.  The premise, measured on an H100",
         size=14.5, weight="bold", color=INK, va="top")

ax1 = axes_at(x + 0.24, y + 0.52, CW * 0.40, F3_H - 0.98)
ax1.set_xlim(0, 100), ax1.set_ylim(0, 100)
box(ax1, 50, 94, 92, 11, "THE WEIGHTS IT CLAIMS TO RUN",
    "read once, in full, for every single token", fc="#E8EFF6", ec=C_TOK,
    lc=C_TOK, subc=C_TOK, fs=12.5, subfs=10.5)
arrow(ax1, 22, 88, 22, 80, c=INK, lw=2.0)
ax1.add_patch(Circle((22, 71), 8.5, facecolor=WHITE, edgecolor=C_CLK, lw=2.6,
                     zorder=2))
ax1.add_artist(Line2D([22, 22], [71, 77], color=C_CLK, lw=2.4, zorder=3))
ax1.add_artist(Line2D([22, 27], [71, 68], color=C_CLK, lw=2.4, zorder=3))
ax1.text(35, 71, "the bus is the bottleneck:\nHBM at 1.85 TB/s", fontsize=10.5,
         color=GREY, ha="left", va="center", linespacing=1.4)
arrow(ax1, 22, 62, 22, 55, c=INK, lw=2.0)
box(ax1, 50, 48, 92, 11, "ONE TOKEN, ON THE CLIENT'S CLOCK",
    "the only thing the client is charged for", fc=C_CLK, ec=C_CLK, lc=WHITE,
    subc="#F6E2D6", fs=12.5, subfs=10.5)
ax1.text(50, 37, "$t_{\\mathrm{token}}\\;\\geq\\;$bytes read $/$ bandwidth",
         fontsize=14, color=INK, ha="center", va="center", weight="bold")

HB = _cell("Qwen3-1.7B", 256)["bytes_read"] / 1e9
NB = _cell("Qwen3-1.7B-NF4", 256)["bytes_read"] / 1e9
ax1.text(4, 29, "bytes it actually has to read, per token", fontsize=11,
         color=INK, weight="bold", va="center")
for i, (val, lab, c) in enumerate([(HB, "honest", C_TOK), (NB, "4-bit", MAROON)]):
    w = 55 * val / HB
    ax1.add_patch(Rectangle((20, 18 - i * 10), w, 7.0, facecolor=c,
                            edgecolor="none", zorder=2))
    ax1.text(18, 21.5 - i * 10, lab, fontsize=11.5, color=c, ha="right",
             va="center", weight="bold")
    ax1.text(21.5 + w, 21.5 - i * 10, f"{val:.2f} GB", fontsize=11.5, color=c,
             va="center", weight="bold")
ax1.text(4, 2, "the cheat's whole margin is bytes it did not read",
         fontsize=11.5, color=INK, ha="left", va="bottom", weight="bold")

ax2 = axes_at(x + CW * 0.505, y + 0.60, CW * 0.435, F3_H - 1.20, frame=True)
graph = [c for c in CC["cells"] if c["mode"] == "graph" and c["B"] == 1
         and c["ctx"] == min(k["ctx"] for k in CC["cells"])]
seen, pts = set(), []
for c in graph:
    if c["label"] in seen or "NF4" in c["label"]:
        continue
    seen.add(c["label"])
    pts.append((c["weight_bytes"] / 1e9, c["min"], c["label"]))
pts.sort()
gb = np.array([p[0] for p in pts])
ms = np.array([p[1] for p in pts])
A = np.vstack([np.ones_like(gb), gb]).T
const, slope = np.linalg.lstsq(A, ms, rcond=None)[0]
xs = np.linspace(0, gb.max() * 1.15, 50)
ax2.plot(xs, const + slope * xs, color=C_TOK, lw=2.2, ls=(0, (5, 3)), zorder=2)
ax2.plot(xs, xs / 1.85, color=C_CLK, lw=2.2, zorder=2)
ax2.scatter(gb, ms, s=105, color=C_TOK, zorder=4)
for g, m, lab in pts:
    ax2.annotate(lab.replace("Qwen3-", ""), (g, m), textcoords="offset points",
                 xytext=(7, -13), fontsize=12, color=INK)
ax2.text(0.030, 0.975, f"observed  =  {const:.2f} ms\n"
         f"                +  1 GB per {slope:.2f} ms", transform=ax2.transAxes,
         fontsize=12.5, color=C_TOK, weight="bold", va="top", linespacing=1.4)
ax2.text(0.60, 0.11, "what the bus alone\nwould give", transform=ax2.transAxes,
         fontsize=11.5, color=C_CLK, linespacing=1.35)
ax2.set_xlabel("weight bytes per decode step (GB)", fontsize=12.5, color=GREY,
               labelpad=2)
ax2.set_ylabel("observed ms per output token", fontsize=12.5, color=GREY,
               labelpad=2)
ax2.set_xlim(0, gb.max() * 1.15), ax2.set_ylim(0, ms.max() * 1.25)
ax2.tick_params(labelsize=12)

y += F3_H
y = caption(x, y + 0.08, CW, "Figure 5",
            f"(B) clock_channel.json, H100 PCIe, CUDA graphs, B=1, one point "
            f"per model. Time is linear in bytes — but at "
            f"{1/slope/1.85*100:.0f}% of copy bandwidth, on a {const:.2f} ms "
            f"software constant that is on no spec sheet. Under eager PyTorch "
            f"the same GPU gives ~30 ms/token, flat in bytes: no channel at "
            f"all.")

# ---- Table: what measurement did to the naive clock
y = section(x, y + 0.22, CW, "EVERY OBVIOUS CLOCK TEST FAILS", color=C_CLK)
y += 0.02
y = table(x, y, CW,
          ["what the premise seems to promise", "what 126 timing cells say"],
          [["Fewer bytes → proportionally less time",
            "NF4 delivers 14% of it"],
           ["4-bit weights are 3.9× faster, so 6 tokens convict",
            "1.10×, and slower at B≥4"],
           ["Half the KV bytes → half the slope",
            "int4 KV is 1.6× steeper"],
           ["Jitter is positive, so take the minimum gap",
            "73% of honest gaps are 0 ms"],
           ["One clock: the inter-token gap",
            "prefill is 25,000× cheaper"]],
          [6.1, 4.47], aligns=["left", "right"], size=T_TAB - 1, head_bg=C_CLK)
y = caption(x, y + 0.08, CW, "Table 3",
            "clock_channel.json (126 configurations) and clock_algos.json. The "
            "premise survives every row; the naive *test* survives none — and "
            "every failure attacks an absolute threshold.")

# ---- Figure 6: the differential test, drawn
y = section(x, y + 0.22, CW, "THE TEST THAT SURVIVES: SUBTRACT", color=C_CLK)
F6_H = 1.76
y += 0.06
panel(x, y, CW, F6_H)
ax6 = axes_at(x, y, CW, F6_H)
ax6.set_xlim(0, 100), ax6.set_ylim(0, 100)
for cy, lab, wdt in [(74, "probe A  —  a short context", 17),
                     (34, "probe B  —  the context it billed for", 39)]:
    ax6.text(3, cy + 15, lab, fontsize=12, color=INK, weight="bold", va="center")
    ax6.add_patch(Rectangle((3, cy), wdt, 11, facecolor=C_CLK, alpha=0.9,
                            edgecolor="none", zorder=2))
    ax6.text(3 + wdt + 2, cy + 5.5, "$t_A$" if wdt < 20 else "$t_B$",
             fontsize=13, color=C_CLK, weight="bold", va="center")
ax6.text(3, 12, "each arrival time is  software + network + bytes",
         fontsize=11.5, color=GREY, va="center")
ax6.add_patch(Rectangle((49, 4), 48, 92, facecolor=CREAM, edgecolor=MAROON,
                        lw=1.5, zorder=1))
ax6.text(73, 80, "$D \\;=\\; t_B - t_A$", fontsize=17, color=MAROON,
         weight="bold", ha="center", va="center", zorder=3)
ax6.text(51.5, 55, "the software constant, the network offset and\nany padding "
         "all cancel exactly", fontsize=11.5, color=INK, va="center",
         linespacing=1.5, zorder=3)
ax6.text(51.5, 34, "a busy neighbour can only push $D$ up", fontsize=11.5,
         color=INK, va="center", zorder=3)
ax6.text(51.5, 14, "→  so the honest floor is one-sided, and the\n"
         "     test calibrates itself against the provider", fontsize=11.5,
         color=MAROON, weight="bold", va="center", linespacing=1.5, zorder=3)
y += F6_H
y = caption(x, y + 0.08, CW, "Figure 6",
            "slope_verifier.json, 42 cells, same protocol as the token channel. A "
            "provider billing for 32,768 tokens of context and holding 512 is "
            "caught at pAUC 0.996 on *36 tokens* of stream (8,192 → 62; half the "
            "context → 152) at 50 ms of wire jitter, against 2,013–35,066 for "
            "the token channel.")

# ---- The close: what any of this buys a governance regime
y = section(x, y + 0.24, CW, "WHAT AN AUDITOR ACTUALLY GETS")
y = para(x, y,
         "*A price, not a ranking:* every verdict costed in tokens × seconds — "
         "and 14 of 35 cells cost infinity.\n"
         "*One protocol, one grid:* 8 detectors × 6 deviations; under an "
         "enforced pool ceiling, 16 of 24 cells fall.\n"
         "*A second channel:* the differential clock, 36–152 tokens of stream, "
         "and no cooperation from the provider.", CW, size=T_TAB, color=INK)
y = para(x, y + 0.10,
         "None of this is a treaty regime — but it prices one. A clause that "
         "names a detector cannot be enforced; a clause that names a pool size, "
         "a false-alarm rate and a probe pair can be. The gaps are honest ones: "
         "a no-training clause is a claim about a fleet over months, not a "
         "stream over seconds; jitter here is device-side on one H100; models "
         "are 0.13B–8B, where the incentive to cheat is smallest; and no "
         "provider on this board adapts.", CW, size=T_TAB, color=GREY)

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
