"""Conference poster, journal-figure style: verification as a cost/accuracy problem.

A restyle *and* a re-argument of `paper/make_cost_poster.py`. The house style there
is a dashboard -- hero numerals, filled cards, a two-tone headline. This one is a
scientific poster: one ink, hairline rules, numbered sections, numbered figures
with captions under them, a results table, an explicit falsification test of the
model the prices come from, and limitations printed next to the claims rather than
in a footnote.

It also carries two measurements the dashboard version does not have, both run for
this poster:

  * `verdict_price_batch.json` -- the price of a token is not a constant. It is
    measured at audit batch 1, which is where a 3.2x-fewer-FLOPs proxy looks 1.02x
    cheaper because the pass is bandwidth-bound. Swept over batch, the price falls
    9.2x and the tier gap saturates at 1.6x, still nowhere near the 12.6x token
    penalty the proxy pays.
  * `pricing_law_check.json` -- the tokens-per-verdict half of every price is a
    CLT prediction, so it is tested against the measured pAUC of the same score
    arrays at 280 (cell, batch) points, and at the three cells whose b* fits under
    the pool ceiling it is tested at b* itself.

Layout is in INCHES on a 3-column board. Text is wrapped against MEASURED glyph
widths (`text_w`), not an em heuristic, so a section cannot silently overrun.

    python paper/make_cost_poster_sci.py            # -> paper/cost_accuracy_poster_sci.png
    python paper/make_cost_poster_sci.py --size 30x40
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results"
sys.path.insert(0, str(ROOT))
from ivgym import verifiers          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--size", default="30x40")
ap.add_argument("--out", default="cost_accuracy_poster_sci")
ap.add_argument("--dpi", type=float, default=None)
args = ap.parse_args()
W, H = (float(v) for v in args.size.lower().split("x"))

# ------------------------------------------------------------------ design
# Journal ink: black text on white, hairline rules, colour only where it carries
# the tier. Slots 1-3 of the reference categorical palette re-stepped for a WHITE
# surface and validated with the skill's checker (all-pairs): lightness band,
# chroma floor, normal-vision dE 15.7, contrast >= 3:1 all PASS; worst CVD pair
# (S3 vs S2) dE 6.9 sits in the 6-8 floor band, which is legal only with a
# secondary encoding -- so tier is ALSO carried by marker fill (solid = Tier 1,
# open = Tier 0 proxy) and every S3 mark is directly labelled.
INK = "#000000"
INK2 = "#333333"
INK3 = "#666666"
RULE = "#000000"
GRID = "#d8d8d8"
S1 = "#0072B2"      # Tier 1 -- recomputes M
S2 = "#D55E00"      # Tier 0 -- cheap proxy
S3 = "#117a55"      # Tier 0 -- no model at all
SURFACE = "#ffffff"

SERIF = "STIXGeneral"
T_TITLE, T_AUTH, T_LEAD = 54, 24, 22
T_SEC, T_BODY, T_CAP, T_SMALL = 26, 20, 17, 15
T_AXIS, T_TICK, T_ANNO = 17, 16, 16
LEAD = 1.28

plt.rcParams.update({
    "font.family": SERIF, "font.size": T_BODY, "mathtext.fontset": "stix",
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.minor.size": 2, "ytick.minor.size": 2,
    "xtick.direction": "out", "ytick.direction": "out",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "pdf.fonttype": 42, "savefig.facecolor": SURFACE,
})

# -------------------------------------------------------------------- data
CV = json.loads((RES / "cost_of_a_verdict.json").read_text())
BS = json.loads((RES / "benign_shape_dprime.json").read_text())
SQ = json.loads((RES / "sequential_verdict.json").read_text())
PC = json.loads((RES / "prefix_cost_quant2bit.json").read_text())
PS = json.loads((RES / "pool_scaling.json").read_text())
PB = json.loads((RES / "verdict_price_batch.json").read_text())
LAW = json.loads((RES / "pricing_law_check.json").read_text())
SUB_TXT = (RES / "exp_spec_substitution_gpu_qwen3-4b_sub0.6b_proxy1.7b.txt").read_text()


def sub_num(pattern: str, text: str = SUB_TXT) -> float:
    m = re.search(pattern, text)
    if not m:
        raise KeyError(pattern)
    return float(m.group(1))


SUB_AUC = sub_num(r"honest vs substitute, token-batched \(b=\d+\)\s*:\s*AUC ([\d.]+)")
SUB_FLOPX = sub_num(r"([\d.]+)x FLOPs")

TIER, PRICE, CELLS = CV["tier"], CV["price"], CV["cells"]
CTRL = CV["honest_control"]
COLOR = {v: (S1 if TIER[v] == 1 else (S3 if PRICE[v]["stage"] == "decode" else S2))
         for v in CV["verifiers"]}
MARK = {"quant_4bit": "o", "kv_fp8": "s", "temp_1.1": "^", "seed_43": "D", "bug_k32": "v"}
NICE = {"quant_4bit": "int4 quant", "kv_fp8": "fp8 KV", "temp_1.1": "temp 1.1",
        "seed_43": "wrong seed", "bug_k32": "top-$k$ bug"}
SHORT_V = {"token_difr": "token\nDiFR", "cross_entropy": "cross\nentropy",
           "token_toploc": "top-loc", "activation_difr": "activation\nDiFR",
           "accept_rate": "accept\nrate", "surface_stat": "surface\nstat",
           "surface_tokens": "surface\ntokens"}
TIER1_US = PRICE["token_difr"]["sec_per_token"] * 1e6
PROXY_US = PRICE["accept_rate"]["sec_per_token"] * 1e6
DECODE_US = PRICE["surface_tokens"]["sec_per_token"] * 1e6


def priced(v, c):
    """A finite b* is not a price unless d' clears the SAME detector's
    honest-vs-honest control (`surface_tokens` separates one half of the honest
    pool from the other about as far as it separates the deviation)."""
    return c["reachable"] and c["d_prime"] > CTRL[v]["d_prime"]


REACH = [(a, v, c) for a, per in CELLS.items() for v, c in per.items() if priced(v, c)]
UNREACH = [(a, v, c) for a, per in CELLS.items() for v, c in per.items() if not priced(v, c)]
VOID = [(a, v, c) for a, v, c in UNREACH if c["reachable"]]
CHEAP_A, CHEAP_V, CHEAP_C = min(REACH, key=lambda t: t[2]["gpu_seconds_per_verdict"])

# ------------------------------------------------------------ figure + text
fig = plt.figure(figsize=(W, H))
REND = fig.canvas.get_renderer()

fx = lambda xin: xin / W
fy = lambda yin: 1.0 - yin / H
fh = lambda hin: hin / H


def text_w(s, size, weight="normal", style="normal") -> float:
    """Width of `s` in INCHES, measured with the real font metrics."""
    t = fig.text(0, 0, s, size=size, weight=weight, style=style)
    bb = t.get_window_extent(renderer=REND)
    t.remove()
    return bb.width / fig.dpi


def wrap(s, w_in, size, weight="normal", style="normal", first_indent=0.0):
    """Greedy wrap against measured widths. Returns [(indent_in, line)]."""
    out = []
    for block in s.split("\n"):
        words, line, indent = block.split(), "", first_indent if not out else 0.0
        for word in words:
            trial = f"{line} {word}".strip()
            if line and text_w(trial, size, weight, style) > w_in - indent:
                out.append((indent, line))
                line, indent = word, 0.0
            else:
                line = trial
        out.append((indent, line))
    return out


def para(x, y, s, w, size=T_BODY, color=INK2, lead=LEAD, weight="normal",
         style="normal", align="left"):
    dy = size * lead / 72
    for i, (ind, ln) in enumerate(wrap(s, w, size, weight, style)):
        if align == "just" and i < 0:
            pass
        fig.text(fx(x + ind), fy(y + i * dy), ln, size=size, color=color,
                 weight=weight, style=style, va="top", ha="left")
    return y + len(wrap(s, w, size, weight, style)) * dy


def para_h(s, w, size=T_BODY, lead=LEAD, weight="normal", style="normal"):
    return len(wrap(s, w, size, weight, style)) * size * lead / 72


def rule(x, y, w, lw=1.2, color=RULE):
    fig.add_artist(Line2D([fx(x), fx(x + w)], [fy(y), fy(y)], lw=lw, color=color,
                          transform=fig.transFigure, zorder=5))


def section(x, y, w, num, title):
    """A numbered small-caps head with a rule under it -- the poster's only chrome.

    The head is fitted to the column: shrink a point at a time, then wrap. A title
    that silently ran into the next column is exactly the failure this poster
    cannot afford, so nothing here is left to an em heuristic."""
    avail, size, head = w - 0.68, T_SEC - 4, title.upper()
    while size > T_SEC - 10 and text_w(head, size, "bold") > avail:
        size -= 1
    lines = ([head] if text_w(head, size, "bold") <= avail
             else [ln for _, ln in wrap(head, avail, size, "bold")])
    fig.text(fx(x), fy(y), f"{num}", size=T_SEC, color=INK, weight="bold", va="top",
             family=SERIF)
    dy = size * 1.14 / 72
    for i, ln in enumerate(lines):
        fig.text(fx(x + 0.62), fy(y + i * dy), ln, size=size, color=INK, va="top",
                 weight="bold", family=SERIF)
    y += (len(lines) - 1) * dy + T_SEC * 1.12 / 72
    rule(x, y, w, 1.6)
    return y + 0.16


def caption(x, y, w, label, text, size=T_CAP):
    """`Fig. 3 |` in bold, then the caption text wrapped around that indent."""
    pre = f"{label} | "
    ind = text_w(pre, size, "bold") + 0.09     # matplotlib trims the trailing space
    fig.text(fx(x), fy(y), pre, size=size, color=INK, weight="bold", va="top")
    dy = size * 1.24 / 72
    lines = wrap(text, w, size, first_indent=ind)
    for i, (extra, ln) in enumerate(lines):
        fig.text(fx(x + extra), fy(y + i * dy), ln, size=size, color=INK2, va="top")
    return y + len(lines) * dy


FOOT_Y = H - 1.55          # top of the footer band; nothing in a column may reach it


def col_end(n, y):
    """Report how much slack each column has left. A poster that overruns its own
    footer is the one defect a reader always sees, so it is checked, not eyeballed."""
    slack = (FOOT_Y - 0.30) - y
    print(f"  column {n} ends at {y:5.2f} in — {slack:+5.2f} in of slack"
          + ("   <-- OVERRUNS THE FOOTER" if slack < 0 else ""))


def axes_at(x, y, w, h):
    a = fig.add_axes([fx(x), fy(y + h), fx(w), fh(h)])
    a.set_facecolor("none")
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.tick_params(labelsize=T_TICK, pad=3)
    return a


def human_seconds(s):
    if s < 1e-3:
        return f"{s*1e6:.0f} \u00b5s"
    if s < 1:
        return f"{s*1e3:.1f} ms"
    if s < 90:
        return f"{s:.1f} s"
    if s < 5400:
        return f"{s/60:.0f} min"
    return f"{s/3600:.1f} h"


def cell_seconds(s):
    """Table 1 is 7 columns wide on a 8.9-inch column, so a cell gets ~1 inch.
    Three significant figures, no more -- the run does not support a fourth and
    the extra glyphs collide with the neighbouring column."""
    for scale, unit in ((1e-3, "µs"), (1.0, "ms")):
        if s < scale:
            v = s / scale * 1e3
            return f"{v:.0f} {unit}" if v >= 100 else f"{v:.1f} {unit}"
    if s < 90:
        return f"{s:.1f} s"
    return f"{s/60:.0f} min" if s < 5400 else f"{s/3600:.1f} h"


def human_tokens(n):
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}k"
    return f"{n:.0f}"


# ==========================================================================
#  FIGURES
# ==========================================================================
def fig_price_batch(x, y, w, h):
    """Fig. 1 -- the price of a token is a curve, not a constant (NEW RUN).

    Two stacked panels sharing the batch axis: the price itself, and the tier gap
    against the two bounds that matter (the FLOP ratio it could reach, and the
    token penalty it would have to beat to be worth buying)."""
    bs = PB["batches"]
    ref = [PB["rows"][str(b)]["reference"]["sec_per_token"] * 1e6 for b in bs]
    prox = [PB["rows"][str(b)]["proxy"]["sec_per_token"] * 1e6 for b in bs]
    gap = [PB["rows"][str(b)]["gap"] for b in bs]
    pen = PB["verdict_consequence"]["proxy_token_penalty"]
    fb = PB["flop_ratio_bound"]

    hb = h * 0.40
    ax = axes_at(x + 1.05, y, w - 1.25, h - hb - 0.62)
    ax2 = axes_at(x + 1.05, y + h - hb, w - 1.25, hb)
    for a in (ax, ax2):
        a.set_xscale("log", base=2)
        a.set_xlim(bs[0] / 1.5, bs[-1] * 1.5)
        a.grid(True, which="major", color=GRID, lw=0.7)
        a.set_axisbelow(True)

    ax.plot(bs, ref, color=S1, lw=1.8, marker="o", ms=8, mfc=S1, mec=SURFACE, mew=1.2,
            zorder=3, label="reference (1.7B), Tier 1")
    ax.plot(bs, prox, color=S2, lw=1.8, marker="s", ms=8, mfc=SURFACE, mec=S2, mew=1.8,
            zorder=3, label="proxy (0.6B), Tier 0")
    ax.set_yscale("log")
    ax.set_ylabel("price of a token\n(GPU \u00b5s, measured)", size=T_AXIS)
    # matplotlib's default log minor labels put six mantissas on a 10x range here
    ticks = [t for t in (20, 50, 100, 200, 500) if min(prox) / 1.3 <= t <= max(ref) * 1.3]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticklabels([])
    ax.legend(fontsize=T_ANNO, frameon=False, loc="lower left", handlelength=2.2,
              labelspacing=0.35, borderaxespad=0.2)
    ax.annotate(f"{ref[0]:.0f} \u00b5s", (bs[0], ref[0]), textcoords="offset points",
                xytext=(6, 8), size=T_ANNO, color=INK2)
    ax.annotate(f"{ref[-1]:.1f} \u00b5s\n\u2193 {ref[0]/ref[-1]:.1f}\u00d7 from batching",
                (bs[-1], ref[-1]), textcoords="offset points", xytext=(-6, 12),
                ha="right", size=T_ANNO, color=INK2, linespacing=1.2)

    ax2.plot(bs, gap, color=INK, lw=1.8, marker="o", ms=7, mfc=INK, zorder=3)
    ax2.axhline(fb, color=S2, lw=1.4, ls="--", zorder=2)
    ax2.axhline(pen, color=S1, lw=1.4, ls=":", zorder=2)
    ax2.axhline(1.0, color=INK3, lw=0.8, zorder=1)
    ax2.set_yscale("log")
    ax2.set_ylim(0.85, pen * 2.6)
    ax2.text(bs[0] / 1.3, pen * 1.12, f"the proxy needs {pen:.1f}\u00d7 more tokens "
             f"\u2014 the gap it must beat", size=T_ANNO, color=S1, va="bottom")
    ax2.text(bs[0] / 1.3, fb * 1.10, f"FLOP bound {fb:.2f}\u00d7", size=T_ANNO,
             color=S2, va="bottom")
    ax2.annotate(f"{max(gap):.2f}\u00d7", (bs[-1], gap[-1]), textcoords="offset points",
                 xytext=(0, 11), size=T_ANNO, color=INK, va="bottom", ha="center")
    ax2.set_ylabel("price gap\nref / proxy", size=T_AXIS)
    ax2.set_xlabel("audit batch $B$   (sequences verified per forward pass)", size=T_AXIS)
    for a in (ax, ax2):
        a.set_xticks(bs)
        a.set_xticklabels([str(b) for b in bs] if a is ax2 else [])
        a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def fig_plane(x, y, w, h):
    """Fig. 2 -- the cost plane. Rates on the axes, cost on the diagonals."""
    ax = axes_at(x + 1.15, y, w - 1.35, h)
    xs = np.array([PRICE[v]["sec_per_token"] * 1e6 for _, v, _ in REACH + UNREACH])
    ys = np.array([c["tokens_per_verdict"] for _, _, c in REACH])
    xlo, xhi = xs.min() / 2.2, xs.max() * 2.2
    ylo, ytop = max(ys.min() / 3, 1), ys.max() * 2.2
    y_break, y_rail, yhi = ytop * 2.0, ytop * 5.0, ytop * 22.0
    aw, ah = w - 1.35, h
    n_xdec, n_ydec = np.log10(xhi / xlo), np.log10(yhi / ylo)
    rot = float(np.degrees(np.arctan2(-(ah / n_ydec), aw / n_xdec)))

    for tot_s in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3):
        xx = np.geomspace(xlo, xhi, 96)
        yy = tot_s * 1e6 / xx
        m = (yy >= ylo) & (yy <= ytop)
        if m.sum() < 2:
            continue
        ax.plot(xx[m], yy[m], color=GRID, lw=1.0, zorder=1)
        ax.text(xx[m][0] * 1.05, yy[m][0] / 1.05, human_seconds(tot_s), size=T_ANNO - 1,
                color=INK3, ha="left", va="top", rotation=rot, rotation_mode="anchor")

    def to_in(px, py):
        return (np.log10(px / xlo) / n_xdec * aw, np.log10(py / ylo) / n_ydec * ah)

    def x_of_in(xin):
        return xlo * 10 ** (xin / aw * n_xdec)

    def declutter(cells, sep, span):
        out, placed = {}, []
        for atk, v, c in sorted(cells, key=lambda t: -t[2]["tokens_per_verdict"]):
            x0, y0 = to_in(PRICE[v]["sec_per_token"] * 1e6, c["tokens_per_verdict"])
            best = x0 + span
            for step in np.arange(0, span + 1e-9, sep * 0.5):
                cands = [x0 + step, x0 - step] if step else [x0]
                free = [xc for xc in cands
                        if all(np.hypot(xc - px, y0 - py) >= sep for px, py in placed)]
                if free:
                    best = free[0]
                    break
            placed.append((best, y0))
            out[(atk, v)] = best
        return out

    xpos = declutter(REACH, sep=0.34, span=1.0)
    for atk, v, c in REACH:
        solid = TIER[v] == 1
        ax.scatter(x_of_in(xpos[(atk, v)]), c["tokens_per_verdict"], s=190,
                   marker=MARK.get(atk, "o"), facecolor=COLOR[v] if solid else SURFACE,
                   edgecolor=COLOR[v] if not solid else SURFACE,
                   linewidth=1.8 if not solid else 1.2, zorder=4)
    ax.axhline(y_break, color=INK, lw=0.8, ls=(0, (6, 4)), zorder=1)
    groups: dict[float, list] = {}
    for atk, v, c in UNREACH:
        groups.setdefault(round(PRICE[v]["sec_per_token"] * 1e6, 3), []).append((atk, v))
    for px, members in groups.items():
        x0, _ = to_in(px, ylo)
        n = len(members)
        for k, (atk, v) in enumerate(sorted(members, key=lambda t: (TIER[t[1]], t[0]))):
            ax.scatter(x_of_in(x0 + (k - (n - 1) / 2) * 0.36), y_rail, s=150,
                       marker=MARK.get(atk, "o"), facecolor="none", edgecolor=INK3,
                       linewidth=1.3, zorder=4)
    ax.text(x_of_in(0.05), y_rail * 2.9,
            f"$\\infty$   no budget buys a verdict   ({len(UNREACH)} of "
            f"{len(UNREACH) + len(REACH)} cells)", size=T_ANNO, color=INK, va="center")

    lab = [min(REACH, key=lambda t: t[2]["gpu_seconds_per_verdict"]),
           max(REACH, key=lambda t: t[2]["gpu_seconds_per_verdict"])]
    lab += [t for t in REACH if COLOR[t[1]] == S3 and t not in lab]
    for atk, v, c in lab:
        xin = xpos[(atk, v)]
        _, yin = to_in(1.0, c["tokens_per_verdict"])
        left, below = xin > aw * 0.60, yin > ah * 0.62
        ax.annotate(f"{NICE.get(atk, atk)}\n{v}", (x_of_in(xin), c["tokens_per_verdict"]),
                    textcoords="offset points", xytext=(-26 if left else 26,
                                                        -14 if below else 14),
                    ha="right" if left else "left", va="top" if below else "bottom",
                    size=T_ANNO, color=INK2, linespacing=1.15,
                    arrowprops=dict(arrowstyle="-", color=INK3, lw=0.8, shrinkA=1,
                                    shrinkB=5))
    ax.legend(handles=[Line2D([], [], marker=MARK[a], color="none", markerfacecolor="none",
                              markeredgecolor=INK2, markeredgewidth=1.3, markersize=10,
                              label=NICE.get(a, a)) for a in CV["attacks"] if a in CELLS]
              + [Line2D([], [], marker="o", color="none", markerfacecolor=S1,
                        markeredgecolor=S1, markersize=10, label="Tier 1: recomputes $M$"),
                 Line2D([], [], marker="o", color="none", markerfacecolor=SURFACE,
                        markeredgecolor=S2, markeredgewidth=1.8, markersize=10,
                        label="Tier 0: proxy / no model")],
              loc="lower left", bbox_to_anchor=(0.005, 0.005), fontsize=T_ANNO,
              labelcolor=INK2, handletextpad=0.5, labelspacing=0.3, borderaxespad=0.0,
              frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_yticks([10.0 ** k for k in range(0, int(np.log10(ytop)) + 1)])
    ax.set_xlabel("price of a token $c$   (GPU \u00b5s per token, measured)", size=T_AXIS)
    ax.set_ylabel("tokens per verdict   $b^{*}=(\\delta^{*}/d')^{2}$", size=T_AXIS)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def fig_law(x, y, w, h):
    """Fig. 3 -- the pricing model, tested (NEW RUN).

    Predicted against measured standardized pAUC at 280 (cell, batch) points, and
    at the three cells whose b* fits under the pool ceiling, the pAUC that b*
    tokens actually delivers against the 0.90 it was priced for."""
    ax = axes_at(x + 1.15, y, w - 1.35, h)
    cells = LAW["cells"]
    grid = LAW["batch_grid"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "b", ["#cfe3f2", "#8fbfe0", "#4a92c8", S1, "#00466d"])
    for r in cells.values():
        for b, m, p in zip(r["batches"], r["measured"], r["predicted"]):
            f = np.log2(b / grid[0]) / np.log2(grid[-1] / grid[0])
            ax.scatter(p, m, s=46, color=cmap(f), edgecolor="none", alpha=0.9, zorder=3)
    ax.plot([0.49, 1.005], [0.49, 1.005], color=INK, lw=1.0, zorder=2)
    ax.text(0.985, 0.965, "measured = predicted", size=T_ANNO, color=INK, rotation=39,
            ha="right", va="top", rotation_mode="anchor")
    # The stars sit ON the identity line's x, so a label laid out horizontally beside
    # one crosses the line. They are indexed instead, and the key goes in the wedge
    # BELOW the line (a band [a,b] at height y is clear of y=x iff y < a).
    stars = sorted(LAW["at_b_star"].items(), key=lambda kv: -kv[1]["measured"])
    for i, (_, s) in enumerate(stars, 1):
        ax.scatter([0.90], [s["measured"]], s=165, marker="*", color=S2,
                   edgecolor=SURFACE, linewidth=0.8, zorder=5)
        low = s["measured"] < 0.55          # keep the index off the x-axis labels
        ax.annotate(str(i), (0.90, s["measured"]), textcoords="offset points",
                    xytext=(9, 9 if low else -9), ha="left",
                    va="bottom" if low else "top", size=T_ANNO, color=S2, weight="bold")
    # Upper-left, under the residual note: a band [a,b] at height y clears y=x iff
    # y > b, and every row here ends well left of its own height.
    kx, ky = 0.505, 0.915
    ax.text(kx, ky, "$\\bigstar$  what $b^{*}$ tokens actually deliver:", size=T_ANNO,
            color=S2, va="top", ha="left", zorder=6)
    for i, (key, s) in enumerate(stars, 1):
        atk, v = key.split("__")
        ax.text(kx, ky - 0.036 * i, f"{i}.  {NICE.get(atk, atk)} / {v},  "
                f"$b^{{*}}$ = {s['batch']}  →  {s['measured']:.3f}", size=T_ANNO - 1,
                color=INK2, va="top", ha="left", zorder=6)
    ax.axvline(0.90, color=S2, lw=1.0, ls="--", zorder=1)
    ax.set_xlim(0.49, 1.01)
    ax.set_ylim(0.475, 1.01)      # the 0.501 star sits on the old floor and clipped
    ax.set_xlabel("predicted standardized pAUC   $\\Phi(d'\\sqrt{b})$", size=T_AXIS)
    ax.set_ylabel("measured standardized pAUC", size=T_AXIS)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap,
                                      norm=matplotlib.colors.LogNorm(grid[0], grid[-1]))
    cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("batch $b$ (tokens)", size=T_AXIS)
    cb.ax.tick_params(labelsize=T_TICK - 1)
    cb.outline.set_linewidth(0.8)
    r = LAW["residual"]
    ax.text(0.505, 0.995, f"all {r['n']} points: mean residual {r['mean']:+.4f}, "
            f"rms {r['rms']:.4f}\nunsaturated band (pred 0.55\u20130.99, $n$="
            f"{r['band_n']}): {r['band_mean']:+.4f}, rms {r['band_rms']:.3f}",
            size=T_ANNO, color=INK2, va="top", linespacing=1.3)


def fig_floor(x, y, w, h):
    """Fig. 4 -- the benign floor: what the verifier's own schedule costs it."""
    arm = "noisy" if "noisy" in BS["arms"] else list(BS["arms"])[0]
    shapes = [s for s in BS["shapes"] if s != BS["baseline_shape"]]
    lbl = {"chunked_b4": "batched with 3 other rows", "chunked_b1_sdpa": "other attention kernel",
           "sequential": "replayed token by token"}
    q = CELLS["quant_4bit"]["token_difr"]
    ctl = CTRL["token_difr"]
    rows = [(lbl.get(s, s), BS["arms"][arm]["shapes"][s]["token_difr"]["d_prime"],
             BS["arms"][arm]["shapes"][s]["token_difr"]["ci"], S1) for s in shapes]
    rows.append(("same shape, disjoint half of pool", ctl["d_prime"],
                 ctl["d_prime_ci"], INK3))
    ax = axes_at(x + 4.35, y, w - 4.55, h)
    yp = np.arange(len(rows))[::-1]
    ax.axvspan(q["d_prime_ci"][0], q["d_prime_ci"][1], color=S2, alpha=0.12, lw=0)
    ax.axvline(q["d_prime"], color=S2, lw=1.4)
    ax.axvline(0, color=INK, lw=0.9)
    for j, (_, val, ci, col) in enumerate(rows):
        ax.plot(ci, [yp[j]] * 2, color=col, lw=1.8, solid_capstyle="butt", zorder=3)
        for e in ci:
            ax.plot([e, e], [yp[j] - 0.13, yp[j] + 0.13], color=col, lw=1.4, zorder=3)
        ax.scatter([val], [yp[j]], s=90, color=col, zorder=4, edgecolor=SURFACE, lw=0.8)
    ax.set_yticks(yp, [r[0] for r in rows], size=T_TICK)
    lim = max(q["d_prime_ci"][1] * 1.12, max(abs(r[1]) for r in rows) * 1.7, 0.01)
    ax.set_xlim(min(min(r[2][0] for r in rows) * 1.3, -lim * 0.28), lim)
    ax.set_ylim(-0.65, len(rows) - 0.35)
    ax.set_xlabel("benign $d'$ against the calibration shape (90% CI)", size=T_AXIS)
    ax.xaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.text(q["d_prime"], len(rows) - 0.5, f"  int4 quant, $d'$ = {q['d_prime']:.4f}",
            size=T_ANNO, color=S2, va="top")


def fig_prefix(x, y, w, h):
    """Fig. 5 -- the same accuracy against the real cost model."""
    ax = axes_at(x + 1.15, y, w - 1.35, h)
    tk, pf = PC["curves"]["topk"], PC["curves"]["prefix"]
    ax.plot(tk["seconds"], tk["auc"], color=S2, lw=1.8, marker="s", ms=8, mfc=SURFACE,
            mec=S2, mew=1.6, zorder=3, label="global top-$k$ (by value)")
    ax.plot(pf["seconds"], pf["auc"], color=S1, lw=1.8, marker="o", ms=8, mfc=S1,
            mec=SURFACE, mew=1.0, zorder=3, label="prefix-scheduled")
    # Both callouts are parked in a corner and led to their point: the two curves
    # cross the middle of this panel and an offset-anchored label lands on them.
    ax.annotate(f"a \u201c5%\u201d top-$k$ audit\nspends {tk['prefill_ratio'][0]:.0%} "
                f"of a full one", (tk["seconds"][0], tk["auc"][0]),
                xycoords="data", textcoords="axes fraction", xytext=(0.985, 0.985),
                ha="right", va="top", size=T_ANNO, color=INK2, linespacing=1.2,
                arrowprops=dict(arrowstyle="-", color=INK3, lw=0.8, shrinkB=5))
    ax.annotate(f"same 5% budget spends {pf['prefill_ratio'][0]:.0%},\n"
                f"{tk['seconds'][0]/pf['seconds'][0]:.0f}\u00d7 less verifier time",
                (pf["seconds"][0], pf["auc"][0]), xycoords="data",
                textcoords="axes fraction", xytext=(0.03, 0.20), ha="left", va="top",
                size=T_ANNO, color=INK2, linespacing=1.2,
                arrowprops=dict(arrowstyle="-", color=INK3, lw=0.8, shrinkB=5))
    ax.set_ylim(min(min(tk["auc"]), min(pf["auc"])) - 0.26,
                max(max(tk["auc"]), max(pf["auc"])) + 0.20)
    ax.set_xlabel("measured verifier seconds for the audit", size=T_AXIS)
    ax.set_ylabel("pAUC @ FPR $\\leq$ 0.5%", size=T_AXIS)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(fontsize=T_ANNO, frameon=False, loc="lower right", labelspacing=0.3)


def table_cells(x, y, w):
    """Table 1 -- the whole grid: GPU time per verdict, one protocol, one run."""
    atks = [a for a in CV["attacks"] if a in CELLS]
    vs = CV["verifiers"]
    stub = max([text_w(NICE.get(a, a), T_BODY - 3) for a in atks]
               + [text_w("price per token", T_CAP)]) + 0.22
    colw = (w - stub) / len(vs)
    x0 = x + stub
    row_h = 0.70
    head_h = 0.86
    # tier spans
    n1 = sum(1 for v in vs if TIER[v] == 1)
    fig.text(fx(x0 + colw * n1 / 2), fy(y), "Tier 1 \u2014 recomputes $M$", size=T_CAP,
             color=S1, ha="center", va="top")
    fig.text(fx(x0 + colw * (n1 + (len(vs) - n1) / 2)), fy(y),
             "Tier 0 \u2014 never runs $M$", size=T_CAP, color=S2, ha="center", va="top")
    rule(x0, y + 0.30, colw * n1 - 0.08, 1.0, S1)
    rule(x0 + colw * n1 + 0.08, y + 0.30, colw * (len(vs) - n1) - 0.08, 1.0, S2)
    yh = y + 0.42
    for i, v in enumerate(vs):
        for k, part in enumerate(SHORT_V[v].split("\n")):
            fig.text(fx(x0 + colw * (i + 0.5)), fy(yh + k * 0.26), part, size=T_CAP,
                     color=INK, ha="center", va="top")
    yh += head_h
    fig.text(fx(x), fy(yh - 0.30), "price per token", size=T_CAP, color=INK3, va="top")
    for i, v in enumerate(vs):
        fig.text(fx(x0 + colw * (i + 0.5)), fy(yh - 0.30),
                 f"{PRICE[v]['sec_per_token']*1e6:.1f} \u00b5s", size=T_CAP - 1,
                 color=INK3, ha="center", va="top")
    yh += 0.06
    rule(x, yh, w, 1.4)
    yy = yh + 0.16
    for a in atks:
        fig.text(fx(x), fy(yy + 0.04), NICE.get(a, a), size=T_BODY - 3, color=INK,
                 va="top")
        for i, v in enumerate(vs):
            c = CELLS[a][v]
            ok = priced(v, c)
            if ok:
                s = cell_seconds(c["gpu_seconds_per_verdict"])
                col = COLOR[v]
                wt = "bold" if (a, v) == (CHEAP_A, CHEAP_V) else "normal"
            else:
                s = "$\\infty$" + ("\u2020" if c["reachable"] else "")
                col, wt = INK3, "normal"
            if text_w(s, T_BODY - 3, wt) > colw - 0.08:
                print(f"  ! table cell overruns its column: {a}/{v} = {s!r}")
            fig.text(fx(x0 + colw * (i + 0.5)), fy(yy), s, size=T_BODY - 3, color=col,
                     ha="center", va="top", weight=wt)
        yy += row_h
        rule(x, yy - 0.10, w, 0.5, GRID)
    rule(x, yy - 0.10, w, 1.4)
    return yy + 0.06


# ==========================================================================
#  HEADER
# ==========================================================================
MARGIN, GUTTER = 1.0, 0.62
CW = (W - 2 * MARGIN - 2 * GUTTER) / 3
COLX = [MARGIN + i * (CW + GUTTER) for i in range(3)]

y = 0.92
fig.text(fx(MARGIN), fy(y), "Verification is a cost/accuracy problem:",
         size=T_TITLE, color=INK, va="top", weight="bold")
y += T_TITLE * 1.10 / 72
fig.text(fx(MARGIN), fy(y), "price the verdict, not the detector",
         size=T_TITLE, color=INK, va="top", weight="bold", style="italic")
y += T_TITLE * 1.24 / 72
fig.text(fx(MARGIN), fy(y), "Jiawei Li", size=T_AUTH, color=INK, va="top")
fig.text(fx(MARGIN + text_w("Jiawei Li", T_AUTH) + 0.25), fy(y),
         "\u00b7   Inference Verification Gym (ivgym)   \u00b7   "
         f"all numbers measured on one NVIDIA H100 with {CV['M'].split('/')[-1]} "
         f"audited by a {CV['proxy'].split('/')[-1]} proxy",
         size=T_AUTH, color=INK2, va="top", style="italic")
y += T_AUTH * 1.55 / 72
rule(MARGIN, y, W - 2 * MARGIN, 2.4)
y += 0.34

ABSTRACT = (
    "A client renting inference cannot see which model ran. Every defence against "
    "that substitution is reported as a detection accuracy and priced by a rate \u2014 "
    "FLOPs per sequence, seconds per prefill \u2014 and neither is the quantity a client "
    "buys. It buys a VERDICT at a false-accusation budget it can live with, and the "
    "price of a verdict factors exactly into how many tokens of evidence it needs and "
    "what a token of that evidence costs. We measure both factors for 5 deviations "
    "\u00d7 7 detectors on one model in one run, test the model that converts them into "
    "a price at 280 (cell, batch) points, and sweep the price of a token over audit "
    "batch. Measured this way the ranking of the defences changes, 14 of 35 cells "
    "have no finite price at all, and the engineering that matters moves off the "
    "estimator and onto the accounting.")
y = para(MARGIN, y, ABSTRACT, W - 2 * MARGIN, size=T_LEAD, color=INK) + 0.30
rule(MARGIN, y, W - 2 * MARGIN, 0.8, INK3)
BODY_TOP = y + 0.42

# ==========================================================================
#  COLUMN 1 -- the identity, the price of a token, methods
# ==========================================================================
x, y = COLX[0], BODY_TOP
y = section(x, y, CW, "1", "What a verdict costs")
EQ = r"$C(\mathrm{verdict})\;=\;b^{*}(d')\;\times\;c(\mathrm{tier})\;=\;" \
     r"(\delta^{*}/d')^{2}\;\times\;c(\mathrm{tier})$"
eqw = text_w(EQ, T_BODY + 4)
eqx = x + (CW - eqw) / 2
fig.text(fx(eqx), fy(y + 0.10), EQ, size=T_BODY + 4, color=INK, ha="left", va="top")
# the two factors, named under the terms they name
fig.text(fx(eqx + eqw * 0.60), fy(y + 0.62), "tokens of evidence", size=T_CAP,
         color=INK3, ha="center", va="top")
fig.text(fx(eqx + eqw * 0.91), fy(y + 0.62), "price per token", size=T_CAP,
         color=INK3, ha="center", va="top")
fig.text(fx(x + CW - 0.02), fy(y + 0.10), "(1)", size=T_BODY, color=INK2, ha="right",
         va="top")
y += 1.15
y = para(x, y,
         f"$d'$ is the per-token effect size of a deviation under a detector, measured "
         f"here with a sequence-level bootstrap. $\\delta^{{*}}$ = {CV['delta_star']:.3f} "
         f"is the batch separation that reaches standardized pAUC "
         f"{CV['target_pauc']:.2f} at FPR $\\leq$ {CV['max_fpr']:.1%} \u2014 the repo's "
         f"standing protocol, not a new one. Both factors are measured on the same "
         f"pool ({CV['n_prompts']}\u00d7{CV['tokens']} = "
         f"{CV['pool_tokens_per_config']:,} tokens per configuration), so their "
         f"product is a real number and not a composition of numbers from different "
         f"runs.", CW) + 0.30

y = section(x, y, CW, "2", "The price of a token is not a constant")
y = para(x, y,
         f"Batch 1 is the regime a per-sequence audit runs in, and it is also the "
         f"regime in which a {PB['flop_ratio_bound']:.1f}\u00d7-fewer-FLOPs proxy is "
         f"only {PB['rows']['1']['gap']:.2f}\u00d7 cheaper per token: a 256-token "
         f"prefill reads every weight to do a few hundred token-rows of work, so the "
         f"price tracks parameter bytes, not arithmetic. Swept over audit batch, the "
         f"price of a token falls "
         f"{PB['rows']['1']['reference']['sec_per_token']/PB['rows'][str(PB['batches'][-1])]['reference']['sec_per_token']:.1f}"
         f"\u00d7 \u2014 an order of magnitude more than the choice of tier ever buys "
         f"\u2014 while the tier gap saturates at "
         f"{max(PB['rows'][str(b)]['gap'] for b in PB['batches']):.2f}\u00d7, half of "
         f"its own FLOP bound and far below the "
         f"{PB['verdict_consequence']['proxy_token_penalty']:.1f}\u00d7 token penalty "
         f"the proxy pays for the same verdict.", CW) + 0.26
FH1 = 5.9
fig_price_batch(x, y, CW, FH1)
y += FH1 + 0.72
y = caption(x, y, CW, "Fig. 1",
            f"Measured GPU-synchronised price of one token of evidence against audit "
            f"batch (top), and the Tier-1/Tier-0 price gap against the two bounds that "
            f"decide whether the cheap tier is worth buying (bottom). "
            f"{PB['reps']} timed repeats after {PB['warmup']} warmups, bfloat16, eager "
            f"attention, full logits kept, {PB['tokens_per_row']} tokens per row. "
            f"Batching is the only lever in this figure that moves cost by an order of "
            f"magnitude. Dotted line: the token penalty from Table 1.") + 0.34

y = section(x, y, CW, "3", "Protocol")
y = para(x, y,
         f"One protocol produces every number. Detection is the standardized partial "
         f"AUC at FPR $\\leq$ {CV['max_fpr']:.1%}; the threshold comes from a held-out "
         f"honest calibration split; per-token scores are winsorized at their 99.9th "
         f"honest percentile; the batch/pool ratio is capped at 10% and the cap is "
         f"enforced in code, because batches resampled from a finite pool inflate an "
         f"AUC without leaving a sign. $d'$ carries a {CV['n_boot']}-resample "
         f"sequence-level bootstrap interval \u2014 tokens inside a sequence are not "
         f"independent, and a token bootstrap would understate the error on a price "
         f"that goes as $1/d'^{{2}}$. Prices are GPU-synchronised wall-clock on an "
         f"idle H100, each stage timed alone with the reference cache dropped, "
         f"converted at \\${CV['usd_per_gpu_hour']:.2f}/hour. A cell whose $d'$ does "
         f"not clear its OWN detector's honest-vs-honest control is void, not cheap.",
         CW, size=T_BODY - 2) + 0.30

y = section(x, y, CW, "4", "Is the price itself trustworthy?")
y = para(x, y,
         f"Half of eq. (1) is a prediction: the CLT says $b$ independent tokens "
         f"separate by $d'\\sqrt{{b}}$. Two documented effects break it — tokens "
         f"inside a sequence are correlated, and a heavy-tailed per-token score reaches "
         f"the Gaussian limit slowly — so we test it on the same score arrays the "
         f"prices come from. Over all {LAW['residual']['n']} points the model is "
         f"unbiased (mean residual {LAW['residual']['mean']:+.4f}), but that average is "
         f"carried by saturated cells; in the unsaturated band it is OPTIMISTIC by "
         f"{abs(LAW['residual']['band_mean']):.3f} pAUC, and at the one cell where the "
         f"0.90 crossing is directly observable the verdict costs "
         f"{LAW['price_check']['median']:.2f}× the tokens it was priced for. Every "
         f"price in §6 is a LOWER BOUND on cost.", CW, size=T_BODY - 2) + 0.26
FH3 = 6.1
fig_law(x, y, CW, FH3)
y += FH3 + 0.72
y = caption(x, y, CW, "Fig. 2",
            f"Predicted versus measured standardized pAUC, {len(LAW['cells'])} cells "
            f"× {len(LAW['batch_grid'])} batch sizes, every point re-measured under "
            f"the protocol of §3 at {LAW['n_batches']:,} batches per point. Stars: "
            f"the pAUC that $b^{{*}}$ tokens actually deliver, at the "
            f"{len(LAW['at_b_star'])} cells whose $b^{{*}}$ fits under the 10% pool "
            f"ceiling (max legal batch {LAW['max_legal_batch']:,}); the cheapest cell on "
            f"the board delivers 0.870, not 0.90. Every other price in Table 1 is an "
            f"extrapolation from this checkable range and is labelled one.")
col_end(1, y)

# ==========================================================================
#  COLUMN 2 -- the plane and the grid
# ==========================================================================
x, y = COLX[1], BODY_TOP
y = section(x, y, CW, "5", "Cost and accuracy do not trade off along a curve")
y = para(x, y,
         "Cost/accuracy is usually drawn as a Pareto curve of rates: cheap detector, "
         "weak; expensive detector, strong. It is not a curve of rates. Each mark below "
         "is one deviation scored by one detector, the axes are the two measured "
         "factors of eq. (1), and the product \u2014 the cost of a verdict \u2014 is "
         "carried by the diagonals. A detector 12\u00d7 cheaper per token that needs "
         "400\u00d7 more tokens is 30\u00d7 more expensive per verdict, and a detector "
         "with $d' \\leq 0$ is not on the curve at all.", CW) + 0.26
FH2 = 13.0
fig_plane(x, y, CW, FH2)
y += FH2 + 0.86
y = caption(x, y, CW, "Fig. 3",
            f"The cost plane, log-log. Grey diagonals are iso-cost: every point on one "
            f"buys a verdict for the same GPU time. Marker shape is the deviation, "
            f"colour and fill the tier. Measured price takes only three values \u2014 a "
            f"tier runs a model or it does not \u2014 and the two that run one price "
            f"within {max(TIER1_US, PROXY_US)/min(TIER1_US, PROXY_US):.2f}\u00d7 of each "
            f"other at batch 1, so the vertical axis decides almost everything; marks "
            f"are nudged sideways only where they would overlap. Above the dashed break, "
            f"the rail of cells no budget reaches: {len(UNREACH) - len(VOID)} at "
            f"$d' \\leq 0$ and {len(VOID)} whose $d'$ does not clear its own detector's "
            f"honest-vs-honest control.") + 0.40

y = section(x, y, CW, "6", "The grid, priced")
ty = table_cells(x, y, CW)
y = caption(x, ty + 0.22, CW, "Table 1",
            f"GPU time for one verdict, {len(CV['attacks'])} deviations \u00d7 "
            f"{len(CV['verifiers'])} detectors, one model, one run. $\\infty$ = no batch "
            f"size reaches pAUC {CV['target_pauc']:.2f}; \u2020 = a finite $b^{{*}}$ "
            f"that does not clear the detector's own honest control, which is not a "
            f"price. Bold: the cheapest verdict on the board, "
            f"{human_seconds(CHEAP_C['gpu_seconds_per_verdict'])} on "
            f"{CHEAP_C['tokens_per_verdict']} token(s) of activation DiFR \u2014 which "
            f"pays for it in ACCESS, not compute: it needs the provider's internal "
            f"activations, which a rented endpoint does not return. The marginal "
            f"verdict is priced here; the honest calibration pool it is scored against "
            f"(20\u00d7 $b^{{*}}$ tokens) amortizes over verdicts and is reported "
            f"separately.") + 0.34

y = para(x, y,
         f"The cheap tier is not a weak detector \u2014 it is a detector of a different "
         f"thing. On model SUBSTITUTION the same proxy reads AUC {SUB_AUC:.3f} at "
         f"{SUB_FLOPX:.1f}\u00d7 fewer FLOPs than recomputation while never running $M$. "
         f"On the numerical deviations in Table 1 it needs "
         f"{PB['verdict_consequence']['proxy_token_penalty']:.1f}\u00d7 more tokens than "
         f"a recompute for the cell it can price at all, and has no finite price for "
         f"three of the five.", CW, size=T_BODY - 2)
col_end(2, y)

# ==========================================================================
#  COLUMN 3 -- the floor, scheduling, conclusions
# ==========================================================================
x, y = COLX[2], BODY_TOP
y = section(x, y, CW, "7", "The accuracy no budget buys")
FH4 = 5.8
fig_floor(x, y, CW, FH4)
y += FH4 + 0.66
q = CELLS["quant_4bit"]["token_difr"]
pos = max(BS["arms"]["noisy"]["shapes"][s]["token_difr"]["d_prime"]
          for s in BS["shapes"] if s != BS["baseline_shape"])
y = caption(x, y, CW, "Fig. 4",
            f"Same weights, same tokens, same noise draw \u2014 only the verifier's own "
            f"forward-pass schedule differs, and every row is honest. Replaying the SAME "
            f"shape returns $d'$ = 0.000000 exactly. A benign floor reaching the orange "
            f"rule (int4 quant, $d'$ = {q['d_prime']:.4f}, band = its 90% CI) would push "
            f"that verdict's price to infinity. None does \u2014 the largest is "
            f"{pos:+.4f} \u2014 but token-by-token replay moves the null "
            f"{abs(BS['arms']['noisy']['shapes']['sequential']['token_difr']['d_prime'])/q['d_prime']:.0%} "
            f"as far as the attack does, away from it. Calibrate in the shape you "
            f"audit in.") + 0.36

y = section(x, y, CW, "8", "Cheaper tokens: schedule against the real cost")
FH5 = 6.4
fig_prefix(x, y, CW, FH5)
y += FH5 + 0.66
y = caption(x, y, CW, "Fig. 5",
            "Reading $M$ at generated position $j$ requires a prefill of everything "
            "before it, so the physical budget is PREFILL tokens: depth is cheap, "
            "breadth is expensive. Scheduling the same audit budget prefix-wise buys "
            "the same accuracy for 16\u00d7 less verifier time. No statistics in it "
            "\u2014 it replicates across deviations by construction.") + 0.40

y = section(x, y, CW, "9", "What we would deploy")
CONC = [
    ("Price the verdict, not the detector.",
     f"{len(UNREACH)} of {len(UNREACH)+len(REACH)} cells cost infinity; the cheapest "
     f"finite one, {human_seconds(CHEAP_C['gpu_seconds_per_verdict'])}, is only "
     f"available to a verifier with activation access."),
    ("Buy the exponent, not the constant.",
     "Tokens go as $1/d'^{2}$ and price per token as model size, so effect size wins "
     "every time the two compete."),
    ("Batch the audit before switching tiers.",
     f"Batching buys "
     f"{PB['rows']['1']['reference']['sec_per_token']/PB['rows'][str(PB['batches'][-1])]['reference']['sec_per_token']:.1f}"
     f"\u00d7; the cheap tier buys "
     f"{max(PB['rows'][str(b)]['gap'] for b in PB['batches']):.2f}\u00d7 and costs "
     f"{PB['verdict_consequence']['proxy_token_penalty']:.1f}\u00d7 the tokens."),
    ("Treat a predicted price as a lower bound.",
     f"Where it can be checked, the model is optimistic by "
     f"{LAW['price_check']['median']:.2f}\u00d7 in tokens."),
    ("Measure your own benign floor first.",
     "It is the price ceiling: no budget buys accuracy below it, and it is a property "
     "of your serving stack, not of the attack."),
    ("Stop early, and size the pool to say so.",
     f"Sequential testing saves {SQ['median_saving_x']:.1f}\u00d7 at matched power, but "
     f"only {SQ.get('n_within_ceiling', 0)} of "
     f"{SQ.get('n_within_ceiling', 0)+len(SQ.get('unresolvable', []))} cells have a pool "
     f"deep enough to demonstrate it."),
]
dy = T_BODY * LEAD / 72
for i, (head, rest) in enumerate(CONC, 1):
    fig.text(fx(x), fy(y), f"{i}.", size=T_BODY, color=INK, va="top")
    fig.text(fx(x + 0.42), fy(y), head, size=T_BODY, color=INK, va="top", weight="bold")
    y = para(x + 0.42, y + dy, rest, CW - 0.42, size=T_BODY - 1) + 0.20
y += 0.14

# The design note for this poster says limitations sit NEXT to the claims. They are
# a numbered section at the foot of the argument, not small print under the rule.
y = section(x, y, CW, "10", "What this does not show")
LIMITS = [
    ("The deviations are logit-level models.",
     "They are logit-space stand-ins for quantisation and sampling faults, not "
     "quantised checkpoints re-served through a real stack."),
    ("One prompt domain, one model pair, one accelerator.",
     "The cross-GPU half of the benign floor is unmeasured, so the rows of Fig. 4 are "
     "a lower bound on it."),
    ("Two transformers builds, so compare ratios and not levels.",
     "Fig. 1 re-times the tiers under a newer build than the batch-1 prices of "
     "Table 1 and runs 1.28\u00d7 slower in absolute terms; only within-figure ratios "
     "carry across the two."),
    ("Table 1 prices the marginal verdict.",
     "It excludes the honest calibration pool, which amortizes over verdicts, and 22 "
     "of 35 cells cannot be checked at $b^{*}$ under the 10% pool ceiling."),
]
for head, rest in LIMITS:
    y = para(x, y, head, CW, size=T_BODY - 2, color=INK, weight="bold")
    y = para(x + 0.42, y, rest, CW - 0.42, size=T_BODY - 2) + 0.20
col_end(3, y)

# ==========================================================================
#  FOOTER
# ==========================================================================
rule(MARGIN, FOOT_Y - 0.24, W - 2 * MARGIN, 0.8, INK3)
REFS = (
    "References. [1] Karvonen et al., DiFR: detecting inference fraud (2025). "
    "[2] Leviathan, Kalman & Matias, Fast inference via speculative decoding (2022). "
    "[3] Wald, Sequential tests of statistical hypotheses (1945).    "
    "Reproduce. python -m experiments.{exp_cost_of_a_verdict_gpu, "
    "exp_verdict_price_batch_gpu, exp_pricing_law_check, exp_benign_shape_dprime_gpu, "
    "exp_prefix_cost_gpu, exp_sequential_verdict}; figure: python "
    "paper/make_cost_poster_sci.py; artifacts and code: the inference-verification "
    "(ivgym) repository.")
para(MARGIN, FOOT_Y, REFS, W - 2 * MARGIN, size=T_SMALL, color=INK3)

out = ROOT / "paper" / args.out
dpi = args.dpi or (2400 / max(W, H))
fig.savefig(out.with_suffix(".png"), dpi=dpi)
fig.savefig(out.with_suffix(".pdf"))
print(f"wrote {out.with_suffix('.png')} at {dpi:.0f} dpi ({W:g}x{H:g} in)")
