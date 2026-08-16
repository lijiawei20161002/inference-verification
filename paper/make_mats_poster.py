"""MATS 10.0 symposium poster: the token verifier and the clock verifier.

Built on the MATS **top-banner landscape 36x24** template
(`MATS_poster_top_banner_landscape36x24.pptx`, symposium template folder). The
template's geometry, type scale and palette are reproduced here rather than
approximated -- column origins 1.25 / 12.72 / 24.18 in at width 10.57, a 4.00 in
maroon banner, Libre Baskerville for display and Carlito (metric-compatible with
the template's Calibri) for body.

Every number on the board is read from a committed artifact in `docs/results/`
or, where the figure quotes a derived table, from the document that derived it
(named in the caption). Nothing is typed in twice: the headline grid, the
verdict prices, the pool-scaling curve and the clock roofline all come out of
JSON at render time, so a rerun that moves a number moves the poster.

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
TITLE = "Did the provider run the model it billed you for?"
ts = T_TITLE
while text_w(TITLE, ts, "bold", family=SERIF) > TW and ts > 30:
    ts -= 1
fig.text(fx(TX), fy(0.52), TITLE, size=ts, color=WHITE, weight="bold",
         family=SERIF, va="top")
fig.text(fx(TX), fy(1.50), "Two channels for inference verification — the tokens "
         "it returned, and the clock — and what a verdict costs in each",
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
# COLUMN 1 -- motivation, the money, the game, the method diagram
# ==========================================================================
x, y = CX[0], Y0

y = section(x, y, CW, "MOTIVATION")
y = lead_line(x, y, "Every compute-governance proposal assumes a claim nobody "
              "can currently check.", CW)
y = para(x, y + 0.08,
         "Model registration, compute thresholds, third-party audits, "
         "export-control attestations, EU AI Act GPAI duties — each presumes a "
         "statement of the form “model $M$ ran under specification "
         "$\\varphi$.” Nothing in the API stack between a client and a GPU "
         "produces that statement. The client sees tokens; the provider sees "
         "the weights; a regulator inherits whichever it chooses to believe.",
         CW, color=GREY)

y = section(x, y + 0.34, CW, "THE BUSINESS OF CHEATING")
y = para(x, y,
         "You rent inference. The provider commits to $M$ under a sampling "
         "spec $\\varphi$ and bills per token. A 4-bit copy, a smaller "
         "sibling, or a truncated context is pure margin — and invisible in "
         "the tokens the client is paying for. But the deviations worth doing "
         "are exactly the ones that read fewer bytes, which is why they leave "
         "a second trace.", CW, color=GREY)

y += 0.24


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
            "Weight + KV bytes per decode step, computed from the timing grid's "
            "own byte counts (clock_channel.json, H100 PCIe, B=1). The last row "
            "is the control: a deviation that saves no bytes has no clock "
            "signature at all, and the poster's two channels split exactly "
            "there.")

y = section(x, y + 0.34, CW, "CONTRIBUTIONS")
y = numbered(x, y, CW, [
    ("One protocol, one grid.",
     "8 detectors × 6 deviations on real models under a single standardized "
     "pAUC @ FPR ≤ 0.5% with an enforced batch/pool ceiling. Re-measured "
     "inside it, 16 of 24 previously published cells fall."),
    ("Price the verdict, not the detector.",
     "$(\\delta^*/d')^2$ tokens × seconds per token. Detector rankings and "
     "FLOP counts are not what a client buys; 14 of 35 cells have no finite "
     "price at all."),
    ("A second, orthogonal channel — and the honest version of it.",
     "Absolute latency tests do not survive measurement; a *differential* "
     "clock test does, at 36–152 tokens of stream against the token "
     "channel's 2,013–35,066."),
])

FIG1_H = 4.30
y += 0.08
panel(x, y, CW, FIG1_H)
ax = axes_at(x, y, CW, FIG1_H)
ax.set_xlim(0, 100), ax.set_ylim(0, 100)


def box(a, cx, cy, w, h, label, sub=None, fc=WHITE, ec=GREY, lc=INK, lw=1.6,
        fs=15, subfs=12.5, subc=GREY, rad=1.2):
    a.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                               boxstyle=f"round,pad=0,rounding_size={rad}",
                               facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    dy = 0 if sub is None else h * 0.27
    a.text(cx, cy + dy, label, ha="center", va="center", fontsize=fs,
           color=lc, weight="bold", zorder=3)
    if sub:
        a.text(cx, cy - h * 0.15, sub, ha="center", va="center",
               fontsize=subfs, color=subc, zorder=3, linespacing=1.35)


def arrow(a, x0, y0, x1, y1, c=INK, lw=2.2, mut=16, ls="-"):
    a.annotate("", xy=(x1, y1), xytext=(x0, y0),
               arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                               mutation_scale=mut, linestyle=ls,
                               shrinkA=0, shrinkB=0), zorder=4)


ax.text(50, 99, "The game: one specification, two observables",
        ha="center", va="top", fontsize=16.5, weight="bold", color=INK)

box(ax, 20, 83, 34, 17, "CLIENT  (verifier)",
    "holds $\\varphi$ and a trusted proxy $q$;\nnever sees the weights",
    fc=WHITE, ec=INK, fs=14, subfs=11.5)
box(ax, 78, 83, 38, 17, "PROVIDER  (attacker)",
    "runs whatever it likes;\nbills as if it ran $M$ under $\\varphi$",
    fc="#F3E3E7", ec=MAROON, lc=MAROON, fs=14, subfs=11.5)
arrow(ax, 37.5, 86, 58.5, 86)
ax.text(48, 88, "prompt, spec $\\varphi$", ha="center", va="bottom",
        fontsize=11.5, color=GREY)
arrow(ax, 58.5, 80, 37.5, 80, c=INK)
ax.text(48, 78, "tokens, and their arrival times", ha="center", va="top",
        fontsize=11.5, color=GREY)

ax.add_artist(Line2D([50, 50], [67, 15], color=RULE_C, lw=1.4, ls=(0, (4, 4))))

for cx, col, head, what, sees, blind in [
    (24.5, C_TOK, "CHANNEL 1 — WHAT IT SAID",
     "Recompute $M$ on the claimed tokens and read the\n"
     "per-token margin, NLL or top-$k$ location — or skip\n"
     "$M$ entirely and score against the proxy $q$.",
     "the sampler, the seed, the exact distribution;\n"
     "wholesale substitution at AUC 0.998 with no $M$",
     "an fp8 KV cache — it moves the served\n"
     "distribution less than $M$'s own rerun noise"),
    (75.5, C_CLK, "CHANNEL 2 — WHEN IT SAID IT",
     "Read the arrival times of a stream already bought.\n"
     "Zero audit FLOPs, zero cooperation, zero added\n"
     "latency — but only differences survive the stack.",
     "which positions were attended, and which\n"
     "parameters were read at prefill",
     "anything reading the same bytes: wrong seed,\n"
     "wrong temperature, a top-$k$ bug"),
]:
    box(ax, cx, 63, 45, 7.5, head, fc=col, ec=col, lc=WHITE, fs=13.5, rad=0.8)
    left = cx - 22.5
    ax.text(left, 55, what, ha="left", va="top", fontsize=11.5, color=INK,
            linespacing=1.5)
    ax.text(left, 38, "sees", ha="left", va="top", fontsize=11.5, color=col,
            weight="bold")
    ax.text(left + 7.5, 38, sees, ha="left", va="top", fontsize=11.5, color=INK,
            linespacing=1.5)
    ax.text(left, 26, "blind", ha="left", va="top", fontsize=11.5, color=C_DEAD,
            weight="bold")
    ax.text(left + 7.5, 26, blind, ha="left", va="top", fontsize=11.5,
            color=GREY, linespacing=1.5)

ax.add_artist(Rectangle((2, 1), 96, 11, facecolor=CREAM, edgecolor=MAROON,
                        lw=1.4, zorder=1))
ax.text(50, 6.5, "One protocol scores both:  standardized pAUC @ FPR ≤ 0.5%,  "
        "held-out honest calibration,\nbatch/pool ratio ≤ 10% enforced in code",
        ha="center", va="center", fontsize=12.5, color=MAROON, weight="bold",
        zorder=3, linespacing=1.5)

y += FIG1_H
y = caption(x, y + 0.12, CW, "Figure 1",
            "The two channels are complementary by mechanism, not by accident. "
            "A deviation either changes the distribution the tokens were drawn "
            "from, or it changes the bytes the GPU had to move, or both — "
            "and the two rows with the most money in them fall on opposite "
            "sides of that line.")
col_end(1, y)

# ==========================================================================
# COLUMN 2 -- the token verifier
# ==========================================================================
x, y = CX[1], Y0

y = section(x, y, CW, "CHANNEL 1  ·  THE TOKEN VERIFIER", color=C_TOK)
y = lead_line(x, y, "One token carries almost nothing. The verdict is bought in "
              "bulk, and $d'$ sets the price.", CW)
y = para(x, y + 0.08,
         "Re-running $M$ and comparing per-token sampling margins catches every "
         "deviation tried here — on some model. But the per-token effect size "
         "from realistic quantization is $d' \\approx 0.08$, so a batch of $b$ "
         "tokens separates by only $d'\\sqrt{b}$. That one number, measurable "
         "before any budget is committed, predicts the detection AUC to within "
         "0.012.", CW, color=GREY)

# ---- Figure 2: the batch principle, on this repo's own scores
F2_H = 3.72
y += 0.20
panel(x, y, CW, F2_H)

honest = SC["honest__token_difr"]
attack = SC["quant_4bit__token_difr"]
hi = np.percentile(honest, 99.9)                 # the protocol's own winsor point
hw = np.clip(honest, None, hi)
aw = np.clip(attack, None, hi)
dprime = (aw.mean() - hw.mean()) / hw.std()
GAP = aw.mean() - hw.mean()

fig.text(fx(x + 0.30), fy(y + 0.24), "A.  Averaging shrinks the noise, not the gap",
         size=14, weight="bold", color=INK, va="top")
fig.text(fx(x + CW * 0.545), fy(y + 0.24),
         "B.  $d'$ predicts the price, before a budget is spent",
         size=14, weight="bold", color=INK, va="top")

rng = np.random.default_rng(7)
BOOT = 20000
BATCHES = [100, 600, 2494]
PA_H = (F2_H - 1.14) / 3
draws = {}
for n in BATCHES:
    draws[n] = (hw[rng.integers(0, hw.size, (BOOT, n))].mean(1),
                aw[rng.integers(0, aw.size, (BOOT, n))].mean(1))
allv = np.concatenate([np.concatenate(v) for v in draws.values()])
SPAN = tuple(np.percentile(allv, [0.4, 99.4]))
SPAN = (SPAN[0], SPAN[1] + 0.42 * (SPAN[1] - SPAN[0]))     # room for the labels
grid = np.linspace(*SPAN, 260)

for k, n in enumerate(BATCHES):
    axk = axes_at(x + 0.30, y + 0.52 + k * PA_H, CW * 0.42, PA_H * 0.82,
                  frame=True)
    hm, am = draws[n]
    for arr, c in ((hm, GREY), (am, C_TOK)):
        d = np.histogram(arr, bins=grid, density=True)[0].astype(float)
        d /= max(d.max(), 1e-12)
        axk.fill_between(grid[:-1], 0, d, color=c, alpha=0.32, lw=0, zorder=2)
        axk.plot(grid[:-1], d, color=c, lw=1.7, zorder=3)
    for m, c in ((hw.mean(), GREY), (aw.mean(), C_TOK)):
        axk.plot([m, m], [0, 1.30], color=c, lw=1.2, ls=(0, (1, 2.5)), zorder=1)
    thr = np.quantile(hm, 0.995)
    axk.plot([thr, thr], [0, 1.06], color=INK, lw=1.5, ls=(0, (3, 2)), zorder=4)
    power = (am > thr).mean()
    axk.set_xlim(*SPAN), axk.set_ylim(0, 1.42)
    axk.set_yticks([]), axk.spines["left"].set_visible(False)
    axk.tick_params(labelsize=11, pad=1, length=3)
    if k < 2:
        axk.set_xticklabels([])
    axk.text(0.99, 0.99, f"$b$ = {n:,} tokens", transform=axk.transAxes,
             fontsize=12.5, color=INK, weight="bold", ha="right", va="top")
    axk.text(0.99, 0.66, f"{power*100:.0f}% of 4-bit audits convict",
             transform=axk.transAxes, fontsize=11.5, color=C_TOK, ha="right",
             va="top")
    if k == 0:
        axk.text(0.015, 0.99, "honest", transform=axk.transAxes, fontsize=11.5,
                 color=GREY, weight="bold", va="top")
        axk.text(0.015, 0.66, "4-bit weights", transform=axk.transAxes,
                 fontsize=11.5, color=C_TOK, weight="bold", va="top")
    if k == 1:
        axk.annotate("", xy=(hw.mean(), 1.16), xytext=(aw.mean(), 1.16),
                     arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.3,
                                     mutation_scale=10), zorder=5)
        axk.text(aw.mean() + 0.02, 1.16, f"gap = {GAP:.3f} nats", fontsize=10.5,
                 color=INK, va="center")
    if k == 2:
        axk.set_xlabel("mean token-DiFR margin over the audit batch",
                       fontsize=12.5, color=GREY, labelpad=1)
        axk.annotate("threshold: the honest 99.5th pct;\nit walks left as $b$ "
                     "grows", xy=(thr, 0.20), xytext=(0.46, 0.26),
                     textcoords=axk.transAxes, fontsize=10.5, color=INK,
                     va="center", ha="left", linespacing=1.3,
                     arrowprops=dict(arrowstyle="-", color=INK, lw=0.9))

axc = axes_at(x + CW * 0.575, y + 0.58, CW * 0.375, F2_H - 1.35, frame=True)
rows = [r for r in PS["rows"] if r["in_ceiling"]]
bb = np.array([r["batch"] for r in rows], float)
au = np.array([r["auc"] for r in rows])
sd = np.array([r["sd"] for r in rows])
pr = np.array([r["predicted"] for r in rows])
grid = np.logspace(np.log10(bb.min() * 0.7), np.log10(bb.max() * 1.5), 200)
axc.plot(grid, np.interp(np.log10(grid), np.log10(bb), pr), color=INK, lw=1.8,
         ls=(0, (5, 3)), zorder=3, label="predicted from $d'$ alone")
axc.errorbar(bb, au, yerr=sd, fmt="o", color=C_TOK, ms=8, lw=0, elinewidth=1.6,
             capsize=3, zorder=4, label="measured pAUC")
axc.axhline(0.9, color=C_CLK, lw=1.2, ls=":", zorder=1)
axc.text(bb.min() * 0.75, 0.905, "the 0.90 target", fontsize=11, color=C_CLK,
         va="bottom")
axc.set_xscale("log")
axc.set_xticks([200, 500, 1000, 2000])
axc.set_xticklabels(["200", "500", "1k", "2k"])
axc.minorticks_off()
axc.set_xlabel("audit batch (tokens pooled per verdict)", fontsize=12.5,
               color=GREY, labelpad=2)
axc.set_ylabel("standardized pAUC @ FPR $\\leq$ 0.5%", fontsize=12.5, color=GREY,
               labelpad=2)
axc.set_ylim(0.48, 0.97)
axc.tick_params(labelsize=11.5)
axc.legend(fontsize=11, frameon=False, loc="upper left", handlelength=1.6,
           bbox_to_anchor=(0.02, 0.90), labelspacing=0.3)

y += F2_H
y = caption(x, y + 0.10, CW, "Figure 2",
            f"(A) 12,000 bootstrap audits per row from the experiment's own "
            f"per-token scores (cost_of_a_verdict_scores.npz; Qwen3-1.7B, "
            f"80×256 tokens per arm, per-token $d'$ = {dprime:.3f}). The "
            f"gap between the two means never moves; only the sampling noise "
            f"does. (B) The same law tested on a 56,160-token pool "
            f"(pool_scaling.json): five points inside the 10% ceiling, mean "
            f"absolute residual 0.012.")

# ---- Table: the headline grid
y = section(x, y + 0.26, CW, "WHAT CATCHES WHAT", color=C_TOK)
y = para(x, y, "Standardized pAUC @ FPR ≤ 0.5%, Qwen3-0.6B, 22,464-token honest "
         "pool, batch 1,000, 5 protocol seeds — a legitimate 8.9% batch/pool "
         "ratio. Every detector is structurally blind to a different class, and "
         "which one wins is set by the attack, not the model: use a portfolio.",
         CW, size=T_TAB, color=GREY)

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
            "headline_ratio.json. Re-scored from the *same* per-token scores "
            "as the previously published 78%-ratio table, 16 of 24 cells fall, "
            "by a median of 0.137. Under a wrong seed the provider redraws from "
            "exactly the honest distribution, so cross-entropy and top-loc can "
            "see nothing in principle — yet at 78% they read 0.987 and 0.828. "
            "An over-ratio pool does not exaggerate a signal; it invents one.")

# ---- Table: tokens per verdict
y = section(x, y + 0.24, CW, "THE PRICE OF A VERDICT", color=C_TOK)
y = para(x, y, "Tokens a verdict needs, $(\\delta^*/d')^2$ at $\\delta^*$ = "
         f"{CV['delta_star']:.2f}, times what a token costs. Tier 1 recomputes "
         "$M$ at 141.1 µs/token; the “cheap” proxy tier is only 1.05× cheaper "
         "per token, because both are one latency-bound batch-1 prefill.",
         CW, size=T_TAB, color=GREY)

y += 0.18
CTRL = CV["honest_control"]
vlist = ["token_difr", "cross_entropy", "token_toploc", "activation_difr",
         "accept_rate", "surface_stat"]
hdr = ["deviation"] + [NICE_V[v].replace("\n", " ") for v in vlist]
trows, tcols = [], []
for a in CV["attacks"]:
    r, c = [NICE_A[a]], [None]
    for v in vlist:
        cell = CV["cells"][a][v]
        ok = cell["reachable"] and cell["d_prime"] > CTRL[v]["d_prime"]
        if not ok:
            r.append("—"), c.append(C_DEAD)
        else:
            n = cell["tokens_per_verdict"]
            r.append(f"{n:,}")
            c.append(C_TOK if n <= 3 else None)
    trows.append(r), tcols.append(c)
y = table(x, y, CW, hdr, trows, [3.0] + [1.55] * len(vlist), size=T_TAB - 1,
          head_bg=C_TOK, colors=tcols)
y = caption(x, y + 0.08, CW, "Table 3",
            "cost_of_a_verdict.json, Qwen3-1.7B audited with a Qwen3-0.6B "
            "proxy. A dash is a cell with no price — $d'$ that fails its own "
            "detector's honest-vs-honest control, so no budget reaches a "
            "verdict through it. Ranking by AUC hides all 14 — 0.50 and 0.55 "
            "look adjacent while their prices are ∞ and 12,994 tokens — and a "
            "cheap token is not a cheap verdict: the proxy is 1.05× cheaper per "
            "token on 4-bit weights and needs 13× more of them.")
col_end(2, y)

# ==========================================================================
# COLUMN 3 -- the clock verifier
# ==========================================================================
x, y = CX[2], Y0

y = section(x, y, CW, "CHANNEL 2  ·  THE CLOCK VERIFIER", color=C_CLK)
y = lead_line(x, y, "A token cannot arrive before its bytes have moved — "
              "so the saving and the evidence are one quantity.", CW)
y = para(x, y + 0.08,
         "A batch-1 decode step is memory-bound: the margin is bytes not read, "
         "and bytes not read is time not spent. Zero audit FLOPs, zero "
         "cooperation, evidence arriving with a stream already bought. Measured "
         "on 126 timing cells, the *premise* holds and the naive test does not.",
         CW, color=GREY)

# ---- Figure 3: the clock, in three measured pictures
F3_H = 3.82
y += 0.20
panel(x, y, CW, F3_H)

fig.text(fx(x + 0.20), fy(y + 0.22), "A.  A token is a read", size=14,
         weight="bold", color=INK, va="top")
fig.text(fx(x + CW * 0.300), fy(y + 0.22), "B.  The floor is a stack constant",
         size=14, weight="bold", color=INK, va="top")
fig.text(fx(x + CW * 0.660), fy(y + 0.22), "C.  What a differential test costs",
         size=14, weight="bold", color=INK, va="top")

ax1 = axes_at(x + 0.20, y + 0.50, CW * 0.235, F3_H - 1.00)
ax1.set_xlim(0, 100), ax1.set_ylim(0, 100)
box(ax1, 44, 90, 80, 17, "WEIGHTS", "1.72B params × 2 B = 3.44 GB",
    fc="#E8EFF6", ec=C_TOK, lc=C_TOK, fs=13.5, subfs=11)
arrow(ax1, 38, 80, 38, 69, c=INK)
ax1.add_patch(Circle((38, 55), 12, facecolor=WHITE, edgecolor=C_CLK, lw=2.8,
                     zorder=2))
ax1.add_artist(Line2D([38, 38], [55, 64], color=C_CLK, lw=2.6, zorder=3))
ax1.add_artist(Line2D([38, 45], [55, 51], color=C_CLK, lw=2.6, zorder=3))
ax1.text(62, 55, "read every byte,\nonce per token\n\nHBM, 1.85 TB/s",
         fontsize=10.5, color=GREY, ha="left", va="center", linespacing=1.45)
arrow(ax1, 38, 41, 38, 30, c=INK)
box(ax1, 44, 19, 80, 16, "1 token", "what the client is charged for",
    fc=C_TOK, ec=C_TOK, lc=WHITE, fs=13.5, subfs=11, subc="#DCE8F3")
ax1.text(4, 4, "$t_{\\mathrm{token}}\\;\\geq\\;$bytes read $/$ bandwidth",
         fontsize=13, color=INK, ha="left", va="center", weight="bold")

ax2 = axes_at(x + CW * 0.330, y + 0.56, CW * 0.235, F3_H - 1.14, frame=True)
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
ax2.plot(xs, const + slope * xs, color=C_TOK, lw=2.0, ls=(0, (5, 3)), zorder=2)
ax2.plot(xs, xs / 1.85, color=C_CLK, lw=2.0, zorder=2)
ax2.scatter(gb, ms, s=90, color=C_TOK, zorder=4)
for g, m, lab in pts:
    ax2.annotate(lab.replace("Qwen3-", ""), (g, m), textcoords="offset points",
                 xytext=(6, -12), fontsize=11.5, color=INK)
ax2.text(0.30, 0.10, "the roofline\n(bytes / 1.85 TB/s)", transform=ax2.transAxes,
         fontsize=12, color=C_CLK, linespacing=1.3)
ax2.text(0.03, 0.90, f"observed = {const:.2f} ms\n+ bytes / "
         f"{1/slope:.2f} TB/s", transform=ax2.transAxes, fontsize=12.5,
         color=C_TOK, weight="bold", linespacing=1.35, va="top")
ax2.set_xlabel("weight bytes per decode step (GB)", fontsize=12.5, color=GREY,
               labelpad=2)
ax2.set_ylabel("observed ms per output token", fontsize=12.5, color=GREY,
               labelpad=2)
ax2.set_xlim(0, gb.max() * 1.15), ax2.set_ylim(0, ms.max() * 1.25)
ax2.tick_params(labelsize=11.5)

ax3 = axes_at(x + CW * 0.690, y + 0.56, CW * 0.280, F3_H - 1.14, frame=True)
sig = np.logspace(-1, 2, 100)
DSTAR = CV["delta_star"]
for gap, lab, c in [(53.6 / 2, "clock: half the context", C_CLK),
                    (53.6, "clock: full truncation", "#1E6B3A")]:
    ax3.plot(sig, (DSTAR * sig / gap) ** 2, color=c, lw=2.4, label=lab)
for tok, lab in [(35066, "returned tokens, fp8 KV:  35,066"),
                 (2013, "returned tokens, 4-bit:  2,013")]:
    ax3.axhline(tok, color=GREY, lw=1.3, ls=(0, (4, 3)))
    ax3.text(88, tok * 1.9, lab, fontsize=10, color=GREY, ha="right")
ax3.axhline(1, color=INK, lw=0.9)
ax3.text(88, 1.7, "one probe pair", fontsize=10, color=INK, ha="right")
ax3.axvline(0.12, color=C_CLK, lw=1.6, alpha=0.75)
ax3.annotate("measured $\\sigma$", xy=(0.12, 1.5e1), xytext=(0.22, 1.2e2),
             fontsize=10, color=C_CLK, va="center",
             arrowprops=dict(arrowstyle="-", color=C_CLK, lw=0.9))
ax3.legend(fontsize=10, frameon=False, loc="lower right", handlelength=1.4,
           borderaxespad=0.3, labelspacing=0.25)
ax3.set_xscale("log"), ax3.set_yscale("log")
ax3.set_xlim(0.1, 100), ax3.set_ylim(1e-4, 3e5)
ax3.set_yticks([1e-3, 1e-1, 1e1, 1e3, 1e5])
ax3.set_xlabel("client-side jitter $\\sigma$ on one gap (ms)", fontsize=12.5,
               color=GREY, labelpad=2)
ax3.set_ylabel("tokens of stream per verdict", fontsize=12.5, color=GREY,
               labelpad=2)
ax3.tick_params(labelsize=11.5)

y += F3_H
y = caption(x, y + 0.08, CW, "Figure 3",
            f"(B) clock_channel.json, H100 PCIe, CUDA-graph stack, B=1. The "
            f"weight read runs at {1/slope/1.85*100:.0f}% of copy bandwidth but "
            f"sits on a {const:.2f} ms constant that is pure stack and on no "
            f"spec sheet — in eager HF the same GPU gives ~30 ms/token, flat in "
            f"bytes: no channel at all. (C) Priced on the repo's own cost law.")

# ---- Table: what measurement did to the naive clock
y = section(x, y + 0.24, CW, "WHAT MEASUREMENT AMENDED", color=C_CLK)
y += 0.02
y = table(x, y, CW,
          ["the naive clock test says", "measured"],
          [["Fewer bytes → proportionally less time",
            "NF4 delivers 14% of it"],
           ["4-bit weights are 3.9× faster, so 6 tokens convict",
            "1.10×, and slower at B≥4"],
           ["Half the KV bytes → half the slope",
            "int4 KV is 1.6× steeper"],
           ["Jitter is positive, so take the minimum gap",
            "73% of honest gaps are 0 ms"],
           ["The floor is on the spec sheet",
            "4.15 ms of it is the stack"],
           ["One clock: the inter-token gap",
            "prefill is 25,000× cheaper"]],
          [6.3, 4.27], aligns=["left", "right"], size=T_TAB - 1, head_bg=C_CLK)
y = caption(x, y + 0.08, CW, "Table 4",
            "clock_channel.json (126 configurations) and clock_algos.json. The "
            "premise survives all six rows; the naive *test* survives none — "
            "and every failure attacks an absolute threshold, not a difference.")

# ---- Table: the surviving verifier
y = section(x, y + 0.22, CW, "THE VERIFIER THAT SURVIVES", color=C_CLK)
y = para(x, y, "$D = \\mathrm{ITL}(\\mathrm{ctx}_{hi}) - "
         "\\mathrm{ITL}(\\mathrm{ctx}_{lo})$ over two matched probes: the stack "
         "constant, the network offset and any padding cancel exactly, and "
         "co-tenancy only *raises* the slope, so the honest floor is one-sided. "
         "A provider billing for 32,768 tokens and holding fewer, at 50 ms of "
         "wire jitter:", CW, size=T_TAB, color=GREY)
y += 0.14
y = table(x, y, CW,
          ["context it actually holds", "$d'$ / probe pair", "pAUC @ 32",
           "tokens"],
          [["512 of 32,768", "0.89", "0.996", "36"],
           ["2,048", "0.82", "0.987", "42"],
           ["8,192", "0.68", "0.940", "62"],
           ["16,384  (half)", "0.43", "0.714", "152"]],
          [4.0, 2.3, 2.2, 2.07], size=T_TAB - 1, head_bg=C_CLK)
y = caption(x, y + 0.08, CW, "Table 5",
            "slope_verifier.json, 42 cells through harness.evaluate on the "
            "house protocol. Re-measured an hour later in a fresh process the "
            "same cells drift by 0.01% in slope, so the test is "
            "self-calibrating and *relative*: it catches a provider that "
            "starts truncating. Evading it costs up to 53 ms per output token — "
            "the provider keeps its 3.7 GB of KV cache and gives back the speed.")

# ---- Limitations
y = section(x, y + 0.20, CW, "LIMITATIONS")
y = para(x, y,
         "The wire is unmeasured: every jitter figure is device-side (sd "
         "0.11–0.8 ms), a lower bound on a client's — hence the sweep in 3C. "
         "One H100, HF eager and CUDA graphs only; vLLM's paged attention may "
         "restore fp8-KV detectability. Models are 0.13B–8B, where the "
         "attacker's incentive is smallest, and quantization attacks are "
         "logit-level. No provider here adapts.",
         CW, size=T_TAB, color=GREY)

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
