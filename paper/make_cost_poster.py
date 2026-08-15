"""Conference poster: verification as a cost/accuracy problem.

Default size is **24 x 36 in portrait** (`--size 24x36`), two columns, in the same
house style as `paper/make_poster.py` -- title fitted to the board width, section
heads 31 pt, body 22 pt, chart labels 18-20 pt, so it reads at 1-2 m. Layout is in
INCHES throughout; panels declare a fixed height or a flex weight and each column
distributes its leftover height, so a different board re-flows instead of
overlapping. `--debug` prints the packed layout.

Every number is read from a committed run artifact -- the three new ones
(`cost_of_a_verdict.json`, `benign_shape_dprime.json`, `sequential_verdict.json`)
and three that already backed the report (`prefix_cost_quant2bit.json`,
`pool_scaling.json`, `exp_spec_substitution_gpu_*.txt`) -- so the poster cannot
drift from the experiments.

    python paper/make_cost_poster.py                 # -> paper/cost_accuracy_poster.{pdf,png}
    python paper/make_cost_poster.py --size 30x40 --debug
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results"
sys.path.insert(0, str(ROOT))
from ivgym import verifiers          # noqa: E402  -- for what a detector REQUIRES,
                                     # which is not in the run artifacts but decides
                                     # whether the cheapest cell is deployable

# ---------------------------------------------------------------- design tokens
SURFACE = "#fcfcfb"
PANEL = "#f5f4f1"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#767570"
GRID = "#e3e2de"
S1 = "#2a78d6"      # categorical slot 1 -- Tier-1, recompute M
S2 = "#eb6834"      # categorical slot 2 -- Tier-0, cheap proxy
S3 = "#1baf7a"      # categorical slot 3 -- Tier-0, no model at all
# Reference-palette slots 1-3, the documented all-pairs-safe subset (this poster's
# main figure is a scatter, so the adjacent-pair pairlist does not apply): worst
# pair CVD dE 9.2, normal-vision 24.0 on surface #fcfcfb. Slot 3 sits below 3:1
# contrast on this surface, so every S3 mark carries a visible direct label
# (the palette's relief rule). Deviation identity is carried by MARKER SHAPE and
# direct labels, never by hue -- there are more deviations than safe hues.

T_TITLE, T_SUB, T_AUTH = 62, 26, 23
T_HERO, T_HEROCAP = 52, 19
T_KICK, T_HEAD, T_BODY = 20, 31, 22
T_TICK, T_AXIS, T_NOTE, T_ANNO = 20, 20, 18, 20
T_TICK_DENSE = 18
T_FOOT = 16
LEAD = 1.30
EM = 0.60

ap = argparse.ArgumentParser()
ap.add_argument("--size", default="30x40", help="WxH in inches (default 30x40 portrait)")
ap.add_argument("--out", default="cost_accuracy_poster")
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
CV = json.loads((RES / "cost_of_a_verdict.json").read_text())
BS = json.loads((RES / "benign_shape_dprime.json").read_text())
SQ = json.loads((RES / "sequential_verdict.json").read_text())
PC = json.loads((RES / "prefix_cost_quant2bit.json").read_text())
PS = json.loads((RES / "pool_scaling.json").read_text())
SUB_TXT = (RES / "exp_spec_substitution_gpu_qwen3-4b_sub0.6b_proxy1.7b.txt").read_text()


def sub_num(pattern: str, text: str = SUB_TXT) -> float:
    m = re.search(pattern, text)
    if not m:
        raise KeyError(pattern)
    return float(m.group(1))


SUB_AUC = sub_num(r"honest vs substitute, token-batched \(b=\d+\)\s*:\s*AUC ([\d.]+)")
SUB_FLOPX = sub_num(r"([\d.]+)x FLOPs")
SUB_HONEST = sub_num(r"honest anchor:.*= ([\d.]+)")
SUB_CHEAT = sub_num(r"under cheat:.*= ([\d.]+)")

TIER = CV["tier"]
PRICE = CV["price"]
CELLS = CV["cells"]
COLOR = {v: (S1 if TIER[v] == 1 else (S3 if PRICE[v]["stage"] == "decode" else S2))
         for v in CV["verifiers"]}
# marker per deviation -- identity by SHAPE, so the hue stays free for the tier
MARK = {"quant_4bit": "o", "kv_fp8": "s", "temp_1.1": "^", "seed_43": "D",
        "bug_k32": "v", "bug_k2": "P"}
NICE = {"quant_4bit": "int4 quant", "kv_fp8": "fp8 KV cache", "temp_1.1": "temp 1.1",
        "seed_43": "wrong seed", "bug_k32": "top-k bug", "bug_k2": "top-2 bug"}
USD_HR = CV["usd_per_gpu_hour"]
TIER1_US = PRICE["token_difr"]["sec_per_token"] * 1e6
PROXY_US = PRICE["accept_rate"]["sec_per_token"] * 1e6
DECODE_US = PRICE["surface_tokens"]["sec_per_token"] * 1e6


CTRL = CV["honest_control"]


def priced(a, v, c):
    """Does this cell have a price at all?

    Two ways not to. `d' <= 0` is the obvious one. The other is `d'` at or below
    the SAME detector's honest-vs-honest control: `surface_tokens` separates one
    half of the honest pool from the other at d' = +0.081 and separates bug_k32
    from honest at +0.061, so its nominal 3,764-token verdict is the detector
    reading the split, not the deviation. A finite number is not a price if the
    null it is measured against is not null.
    """
    return c["reachable"] and c["d_prime"] > CTRL[v]["d_prime"]


def reachable():
    """(attack, verifier, cell) for every priced cell a budget can actually reach."""
    return [(a, v, c) for a, per in CELLS.items() for v, c in per.items()
            if priced(a, v, c)]


def unreachable():
    return [(a, v, c) for a, per in CELLS.items() for v, c in per.items()
            if not priced(a, v, c)]


def voided():
    """Cells with a finite b* that does not clear their own honest control."""
    return [(a, v, c) for a, per in CELLS.items() for v, c in per.items()
            if c["reachable"] and not priced(a, v, c)]


def cheapest_cell():
    r = reachable()
    return min(r, key=lambda t: t[2]["gpu_seconds_per_verdict"])


fig = plt.figure(figsize=(W, H))

# ------------------------------------------------------------- inches -> figure
fx = lambda xin: xin / W
fy = lambda yin: 1.0 - yin / H
fh = lambda hin: hin / H


def lines_of(s, w_in, size):
    n = max(8, int(w_in * 72 / (size * EM)))
    out = []
    for block in s.split("\n"):
        out += textwrap.wrap(block, n) or [""]
    return out


def fit_size(s, w_in, size):
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
    assert y_end <= y0 + h + 0.06, f"{name} overflows its panel by {y_end - y0 - h:.2f} in"


def card(x_in, y_in, w_in, h_in):
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


def series_key(x_in, y_in, labels, colors, right_edge=None, step=4.2):
    for i, (lab, c) in enumerate(zip(labels, colors)):
        last = i == len(labels) - 1
        if last and right_edge is not None:
            fig.text(fx(right_edge), fy(y_in), lab, size=T_ANNO, color=c,
                     weight="bold", va="top", ha="right")
        else:
            fig.text(fx(x_in + i * step), fy(y_in), lab, size=T_ANNO, color=c,
                     weight="bold", va="top")


def bar_h(n_series, span=0.84, gap=0.22):
    return span / n_series * (1 - gap)


def human_tokens(n):
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}k"
    return f"{n:.0f}"


def human_seconds(s):
    if s < 1e-3:
        return f"{s*1e6:.0f} µs"
    if s < 1:
        return f"{s*1e3:.0f} ms"
    if s < 90:
        return f"{s:.1f} s"
    if s < 5400:
        return f"{s/60:.0f} min"
    return f"{s/3600:.1f} h"


# ============================================================ panel renderers
def stat_row(x, y, w, items, color=S1, big=T_HERO - 8, cap=T_NOTE):
    """Three measured numbers side by side. A bar chart of three values one of
    which is 300x the others is a stat row pretending to be a plot."""
    for i, (big_s, small_s) in enumerate(items):
        sx = x + i * (w / len(items))
        fig.text(fx(sx), fy(y), big_s, size=big, weight="bold", color=color, va="top")
        fig.text(fx(sx), fy(y + big * 1.12 / 72), small_s, size=cap, color=INK2,
                 va="top", linespacing=1.45)
    return y + big * 1.12 / 72 + max(s.count("\n") + 1 for _, s in items) * cap * 1.45 / 72


def p_unit(x, y, w, h):
    """The identity that organises the poster, and the measured price of a token."""
    y0 = y
    fig.text(fx(x + 0.1), fy(y),
             r"$\mathrm{cost\ of\ a\ verdict}\;=\;(\delta^{*}/d')^{2}"
             r"\;\times\;c(\mathrm{tier})$",
             size=T_HEAD - 4, color=INK, va="top")
    fig.text(fx(x + 0.1), fy(y + 0.72),
             "                                      tokens of evidence   ×   "
             "price per token",
             size=T_NOTE, color=INK3, va="top")
    y = draw_text(x, y + 1.35,
                  f"δ* = {CV['delta_star']:.3f} is the batch separation reaching "
                  f"standardized pAUC {CV['target_pauc']:.2f} at FPR ≤ "
                  f"{CV['max_fpr']:.1%} — the repo's protocol, not a new one. Both "
                  f"factors are measured below, on one model, in one run.", w) + 0.28
    us = {v: PRICE[v]["sec_per_token"] * 1e6 for v in
          ("token_difr", "accept_rate", "surface_tokens")}
    y = stat_row(x, y, w, [
        (f"{us['token_difr']:,.0f} µs", "per token, Tier 1:\nrecompute M"),
        (f"{us['accept_rate']:,.0f} µs", "per token, Tier 0:\na cheap proxy"),
        (f"{us['surface_tokens']:,.1f} µs", "per token, Tier 0:\nno model at all")])
    ratio = us["token_difr"] / us["accept_rate"]
    used(draw_text(x, y + 0.18,
                   f"GPU-synchronised, timed alone with the reference cache dropped. "
                   f"The proxy tier is {ratio:.1f}× cheaper per token — the whole of "
                   f"its advantage, and on the smaller factor.", w, size=T_NOTE),
         y0, h, "unit")


def p_plane(x, y, w, h):
    """THE figure: price per token against tokens per verdict, with iso-cost lines.

    Cost and accuracy are usually drawn as a Pareto curve of RATES. It is not one:
    the product is what a client pays, so the honest plane puts a rate on each axis
    and lets the diagonals carry the cost.
    """
    R, U, nvoid = reachable(), unreachable(), len(voided())
    n_price = len({round(p["sec_per_token"], 12) for p in PRICE.values()})
    note = (f"Iso-cost diagonals: every point on one costs the same. Measured price "
            f"takes only {n_price} values — a tier runs a model or it does not — and "
            f"the two that do price within "
            f"{max(TIER1_US, PROXY_US)/min(TIER1_US, PROXY_US):.2f}× of each other at "
            f"batch 1. They are one column, so the vertical axis decides almost "
            f"everything; marks are nudged sideways only where they would overlap. "
            f"On the rail: {len(U) - nvoid} cells at d′ ≤ 0, plus {nvoid} whose d′ "
            f"does not clear its own detector's honest-vs-honest control.")
    nh = text_h(note, w, T_NOTE) + 0.30
    aw, ah = w - 1.3, h - nh - 2.05
    ax = axes_at(x + 0.9, y + 0.95, aw, ah)
    xs = np.array([PRICE[v]["sec_per_token"] * 1e6 for _, v, _ in R + U])
    ys = np.array([c["tokens_per_verdict"] for _, _, c in R])
    xlo, xhi = xs.min() / 5, xs.max() * 6
    ylo, ytop = max(ys.min() / 4, 1), ys.max() * 2.5
    # the unreachable band gets a real share of the axis rather than a sliver at
    # the top: a rail of 13 marks plus its caption needs about an inch of height,
    # and on a log axis that has to be bought in decades.
    y_break, y_rail, yhi = ytop * 2.2, ytop * 6.0, ytop * 30.0

    # iso-cost diagonals, drawn under the marks and clipped below the break line.
    # The label's rotation is COMPUTED from the axes geometry -- on log-log a
    # constant-product line has data slope -1, and its screen slope depends on how
    # many decades each inch of axis carries.
    n_xdec, n_ydec = np.log10(xhi / xlo), np.log10(yhi / ylo)
    rot = float(np.degrees(np.arctan2(-(ah / n_ydec), aw / n_xdec)))
    for tot_s in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3):
        tot_us = tot_s * 1e6
        xx = np.geomspace(xlo, xhi, 96)
        yy = tot_us / xx
        m = (yy >= ylo) & (yy <= ytop)
        if m.sum() < 2:
            continue
        ax.plot(xx[m], yy[m], color=GRID, lw=2.0, zorder=1)
        ax.text(xx[m][0] * 1.04, yy[m][0] / 1.06, human_seconds(tot_s), size=T_NOTE,
                color=INK3, ha="left", va="top", rotation=rot,
                rotation_mode="anchor")

    # ---- placement in INCHES, because overlap is a screen property, not a data one
    def to_in(px, py):
        return (np.log10(px / xlo) / n_xdec * aw, np.log10(py / ylo) / n_ydec * ah)

    def x_of_in(xin):
        return xlo * 10 ** (xin / aw * n_xdec)

    def declutter(cells, sep, span):
        """x positions (inches) that keep every mark `sep` inches from its
        neighbours, nudging sideways within `span` of the cell's true price.

        The two model-running tiers price within 5% of each other, so on a log
        price axis they are the same column and 21 of 22 reachable cells stack
        into it. Tier identity is carried by hue, not by that unreadable 0.07 in
        of separation, so the nudge costs nothing a reader could have used.
        """
        out, placed = {}, []
        for atk, v, c in sorted(cells, key=lambda t: -t[2]["tokens_per_verdict"]):
            x0, y0 = to_in(PRICE[v]["sec_per_token"] * 1e6, c["tokens_per_verdict"])
            best = x0
            for step in np.arange(0, span + 1e-9, sep * 0.5):
                cands = [x0 + step, x0 - step] if step else [x0]
                free = [xc for xc in cands
                        if all(np.hypot(xc - px, y0 - py) >= sep for px, py in placed)]
                if free:
                    best = free[0]
                    break
            else:
                best = x0 + span
            placed.append((best, y0))
            out[(atk, v)] = best
        return out

    xpos = declutter(R, sep=0.40, span=1.15)
    for atk, v, c in R:
        ax.scatter(x_of_in(xpos[(atk, v)]), c["tokens_per_verdict"],
                   s=340, marker=MARK.get(atk, "o"), color=COLOR[v],
                   edgecolor=SURFACE, linewidth=2.0, zorder=4)
    # the unreachable rail: cells no budget reaches, parked above a break line.
    # They are not "very expensive", they are off the scale, so they get their own
    # band rather than a large finite y that could be read as a price. Packed by
    # price column, symmetric about it, so the rail stays under its own tier.
    ax.axhline(y_break, color=GRID, lw=1.6, zorder=1)
    groups: dict[float, list] = {}
    for atk, v, c in U:
        groups.setdefault(round(PRICE[v]["sec_per_token"] * 1e6, 3), []).append((atk, v))
    for px, members in groups.items():
        x0, _ = to_in(px, ylo)
        n = len(members)
        for k, (atk, v) in enumerate(sorted(members, key=lambda t: (TIER[t[1]], t[0]))):
            off = (k - (n - 1) / 2) * 0.42
            ax.scatter(x_of_in(x0 + off), y_rail, s=230, marker=MARK.get(atk, "o"),
                       facecolor="none", edgecolor=COLOR[v], linewidth=2.4, zorder=4)
    ax.text(x_of_in(0.05), y_rail * 2.45,
            f"∞   no budget reaches a verdict   "
            f"({len(U)} of {len(U) + len(R)} cells)",
            size=T_ANNO, color=INK, va="center", ha="left", clip_on=True)

    # Selective direct labels: the two extremes of the cost range, plus every
    # slot-3 mark (the palette's relief rule -- S3 is under 3:1 on this surface).
    # Labels flip to the left inside the right third so they cannot run off the
    # panel, and carry a leader line because the mark they name has been nudged.
    show = sorted(R, key=lambda t: t[2]["gpu_seconds_per_verdict"])
    picked, seen = [], set()
    for t in [show[0], show[-1]] + [t for t in R if COLOR[t[1]] == S3]:
        if (t[0], t[1]) not in seen:
            seen.add((t[0], t[1]))
            picked.append(t)
    for atk, v, c in picked:
        xin = xpos[(atk, v)]
        _, yin = to_in(1.0, c["tokens_per_verdict"])
        left = xin > aw * 0.62
        below = yin > ah * 0.66      # near the break line, so hang the label under
        ax.annotate(f"{NICE.get(atk, atk)}\n{v}",
                    (x_of_in(xin), c["tokens_per_verdict"]),
                    textcoords="offset points",
                    xytext=(-38 if left else 38, -18 if below else 18),
                    ha="right" if left else "left",
                    va="top" if below else "bottom",
                    size=T_NOTE, color=INK2, linespacing=1.25,
                    arrowprops=dict(arrowstyle="-", color=INK3, lw=1.0,
                                    shrinkA=2, shrinkB=8))

    # shape key. Hue is the tier and shape is the deviation, and only three marks
    # carry a direct label -- without this the other 19 are undecodable. It goes in
    # the empty low-price/low-token corner, which no cell can occupy: a token that
    # cheap comes from a detector that runs no model.
    ax.legend(handles=[Line2D([], [], marker=MARK.get(a, "o"), color="none",
                              markerfacecolor="none", markeredgecolor=INK2,
                              markeredgewidth=1.8, markersize=13,
                              label=NICE.get(a, a))
                       for a in CV["attacks"] if a in CELLS],
              loc="lower left", bbox_to_anchor=(0.13, 0.015),
              fontsize=T_NOTE, labelcolor=INK2,
              handletextpad=0.5, labelspacing=0.45, borderaxespad=0.0,
              frameon=True, facecolor=PANEL, edgecolor="none", framealpha=1.0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    # no tick above the break: the rail is not a large number of tokens, it is the
    # absence of one, and a 10^8 gridline over it would read as a price.
    ax.set_yticks([10.0 ** k for k in range(0, int(np.log10(ytop)) + 1)])
    ax.set_xlabel("price of a token   (GPU µs, measured)  [log]", size=T_AXIS)
    ax.set_ylabel("tokens per verdict   (δ*/d′)²  [log]", size=T_AXIS)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(GRID)
    ax.grid(True, color=GRID, lw=1.0, alpha=0.55)
    ax.set_axisbelow(True)
    series_key(x + 0.9, y, ["Tier 1 · recompute M", "Tier 0 · proxy",
                            "Tier 0 · no model"], [S1, S2, S3],
               right_edge=x + w, step=6.0)
    fig.text(fx(x + 0.9), fy(y + 0.42),
             "shape = deviation  ·  hollow = unreachable at any budget",
             size=T_NOTE, color=INK3, va="top")
    used(draw_text(x, y + h - nh + 0.18, note, w, size=T_NOTE), y, h, "plane")


def p_cheap(x, y, w, h):
    """Where the cheap tier is cheap, and where its price is infinite."""
    note = (f"It is not a weak detector — it is a detector of a different thing. On "
            f"model substitution the same proxy reads AUC {SUB_AUC:.3f} at "
            f"{SUB_FLOPX:.1f}× fewer FLOPs than recomputation, never running M "
            f"(accept rate {SUB_HONEST:.3f} honest vs {SUB_CHEAT:.3f} cheating).")
    nh = text_h(note, w, T_NOTE) + 0.30
    atks = [a for a in CV["attacks"] if a in CELLS]
    ax = axes_at(x + 3.5, y + 0.62, w - 3.9, h - nh - 1.45)
    yp = np.arange(len(atks))[::-1]
    bh = bar_h(2, span=0.86, gap=0.20)
    finite = [c["tokens_per_verdict"] for a in atks for v, c in CELLS[a].items()
              if v in ("token_difr", "accept_rate") and c["reachable"]] or [1e3]
    top = max(finite) * 5           # where an unreachable cell's arrow runs off to
    for j, a in enumerate(atks):
        for k, v in enumerate(["token_difr", "accept_rate"]):
            c = CELLS[a][v]
            off = bh / 2 if k == 0 else -bh / 2
            if c["reachable"]:
                val = c["tokens_per_verdict"]
                ax.barh(yp[j] + off, val, height=bh, color=COLOR[v], edgecolor="none")
                ax.text(val * 1.25, yp[j] + off, human_tokens(val), va="center",
                        size=T_NOTE, color=INK2)
            else:
                # an unreachable cell is OFF the scale, not merely large: a faded
                # full-width bar would read as a finite, very big price
                ax.annotate("", xy=(top, yp[j] + off), xytext=(10, yp[j] + off),
                            arrowprops=dict(arrowstyle="-|>", color=COLOR[v], lw=2.4,
                                            shrinkA=0, shrinkB=0))
                ax.text(top * 1.3, yp[j] + off, "∞  unreachable", va="center",
                        size=T_NOTE, color=INK2)
    ax.set_yticks(yp, [NICE.get(a, a) for a in atks], size=T_TICK_DENSE)
    ax.set_xscale("log")
    ax.set_xlim(10, top * 14)
    ax.set_ylim(-0.7, len(atks) - 0.3)
    ax.set_xlabel("tokens per verdict  [log]", size=T_AXIS)
    ax.xaxis.grid(True, color=GRID, lw=1.2)
    ax.set_axisbelow(True)
    series_key(x + 3.5, y, ["recompute M (token_difr)", "cheap proxy (accept_rate)"],
               [S1, S2], right_edge=x + w)
    used(draw_text(x, y + h - nh + 0.18, note, w, size=T_NOTE), y, h, "cheap")


def p_prefix(x, y, w, h):
    """Cheaper tokens: the same accuracy, scheduled against the real cost model."""
    note = ("Budget in prefill tokens, not token counts: reading M at generated "
            "position j needs a prefill of everything before it, so depth is cheap and "
            "breadth is expensive. No statistics in it — it replicates across "
            "deviations by construction.")
    nh = text_h(note, w, T_NOTE) + 0.30
    ax = axes_at(x + 1.5, y + 0.62, w - 1.9, h - nh - 1.50)
    for key, color, lab in (("topk", S2, "global top-k"), ("prefix", S1, "prefix schedule")):
        c = PC["curves"][key]
        ax.plot(c["seconds"], c["auc"], color=color, lw=2.6, marker="o", ms=11,
                markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=3)
    tk, pf = PC["curves"]["topk"], PC["curves"]["prefix"]
    ax.annotate(f"a “5%” top-k audit spends\n{tk['prefill_ratio'][0]:.0%} of a full one",
                (tk["seconds"][0], tk["auc"][0]), textcoords="offset points",
                xytext=(-34, 54), size=T_NOTE, color=INK2, ha="right", va="bottom",
                linespacing=1.25,
                arrowprops=dict(arrowstyle="-", color=INK3, lw=1.2, shrinkB=4))
    ax.annotate(f"5% spends {pf['prefill_ratio'][0]:.0%},\n"
                f"{tk['seconds'][0]/pf['seconds'][0]:.0f}× faster",
                (pf["seconds"][0], pf["auc"][0]), textcoords="offset points",
                xytext=(18, -12), size=T_NOTE, color=INK2, va="top",
                linespacing=1.25,
                arrowprops=dict(arrowstyle="-", color=INK3, lw=1.2, shrinkB=4))
    ax.set_ylim(min(min(tk["auc"]), min(pf["auc"])) - 0.30,
                max(max(tk["auc"]), max(pf["auc"])) + 0.17)
    ax.set_xlabel("measured verifier seconds for the audit", size=T_AXIS)
    ax.set_ylabel("pAUC @ FPR ≤ 0.5%", size=T_AXIS)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(GRID)
    ax.grid(True, color=GRID, lw=1.0, alpha=0.55)
    ax.set_axisbelow(True)
    series_key(x + 1.5, y, ["global top-k", "prefix schedule"], [S2, S1],
               right_edge=x + w)
    used(draw_text(x, y + h - nh + 0.18, note, w, size=T_NOTE), y, h, "prefix")


def p_sequential(x, y, w, h):
    """Fewer tokens -- and the pool it takes to know that. Three numbers, no chart.

    The honest headline here is not the saving, it is how few cells the pool can
    resolve: a stream is bootstrapped from a finite token pool, so a design that
    draws more than 10% of it is measuring the pool, not the test.
    """
    y0 = y
    rows = [c for per in SQ["cells"].values() for c in per.values()
            if c.get("within_ceiling") and c.get("saving_x_matched")]
    unres = SQ.get("unresolvable", [])
    n_ok = SQ.get("n_within_ceiling", len(rows))
    med = SQ.get("median_saving_x")
    if not rows:
        used(draw_text(x, y,
                       f"No cell is resolvable at a {SQ['pool_tokens_per_config']:,}"
                       f"-token pool: every reachable verdict needs a batch larger "
                       f"than the {SQ['max_pool_ratio']:.0%} ceiling allows.", w),
             y0, h, "sequential")
        return
    sav = [c["saving_x_matched"] for c in rows]
    biggest = min(unres, key=lambda r: r["honest_pool_needed"], default=None)
    y = stat_row(x, y, w, [
        (f"{med:.1f}×", f"median token saving,\nmatched at the same power"),
        (f"{n_ok} of {n_ok + len(unres) + len(SQ.get('skipped', []))}",
         "cells this pool can\nresolve at all"),
        (f"{human_tokens(biggest['honest_pool_needed'])}" if biggest else "—",
         "honest tokens the next\ncell would need")])
    fpr = [c["fixed"]["fpr"] for c in rows]
    pw = [c["fixed"]["power"] for c in rows]
    used(draw_text(x, y + 0.20,
                   f"Same evidence, stopped when it is enough: {min(sav):.2f}–"
                   f"{max(sav):.2f}× fewer tokens, against a fixed design bisected to "
                   f"the same {SQ['power_target']:.0%} power rather than against "
                   f"b*(d′). That matters, because b*(d′) does not deliver its own "
                   f"spec — inside the ceiling it realizes {min(pw):.2f}–{max(pw):.2f} "
                   f"power and {min(fpr):.4f}–{max(fpr):.4f} false alarms against "
                   f"{SQ['power_target']:.2f} and {SQ['alpha']:.3f}. The other "
                   f"{len(unres)} cells are not reported: their batches run to "
                   f"{max(r['batch_pool_ratio'] for r in unres):.0%} of the pool, and "
                   f"a bootstrap that large measures the pool.", w, size=T_NOTE),
         y0, h, "sequential")


def p_floor(x, y, w, h):
    """The accuracy no budget buys: the verifier's own benign shift."""
    arm = "noisy" if "noisy" in BS["arms"] else list(BS["arms"])[0]
    shapes = [s for s in BS["shapes"] if s != BS["baseline_shape"]]
    lbl = {"chunked_b4": "replay batched with 3 other rows",
           "chunked_b1_sdpa": "a different attention kernel",
           "sequential": "replay token-by-token"}
    # The rule to beat is measured in THIS run, on the same pool, by the same
    # detector -- not carried over from pool_scaling.json, which is quoted only as
    # an independent replication of it.
    q = CELLS["quant_4bit"]["token_difr"]
    d_att, d_att_ci = q["d_prime"], q["d_prime_ci"]
    # the true zero: the same shape scored against a disjoint half of its own pool
    ctl = CV["honest_control"]["token_difr"]
    rows = [(lbl.get(s, s), BS["arms"][arm]["shapes"][s]["token_difr"]["d_prime"],
             BS["arms"][arm]["shapes"][s]["token_difr"]["ci"], S1) for s in shapes]
    rows.append(("same shape, other half of the pool",
                 ctl["d_prime"], ctl["d_prime_ci"], INK3))
    pos = max(r[1] for r in rows[:-1])
    note = (f"Same weights, same tokens, same noise draw — only the verifier's own "
            f"forward-pass schedule differs, and every row is honest. Replaying the "
            f"SAME shape twice returns d′ = 0.000000 exactly. A floor reaching the "
            f"orange rule (int4 quant, this run, d′ = {d_att:.4f}; band = its 90% CI) "
            f"would price a verdict at infinity. None does — the largest is "
            f"{pos:+.4f}. Token-by-token replay shifts the null "
            f"{abs(rows[2][1])/d_att:.0%} as far as the attack, but away from it.")
    nh = text_h(note, w, T_NOTE) + 0.30
    ax = axes_at(x + 6.3, y + 0.72, w - 6.7, h - nh - 1.85)
    yp = np.arange(len(rows))[::-1]
    ax.axvline(0, color=INK3, lw=1.4, zorder=1)
    ax.axvspan(d_att_ci[0], d_att_ci[1], color=S2, alpha=0.13, lw=0, zorder=0)
    ax.axvline(d_att, color=S2, lw=2.0, zorder=1)
    vals = [r[1] for r in rows]
    for j, (_, val, ci, col) in enumerate(rows):
        ax.plot(ci, [yp[j], yp[j]], color=col, lw=3.0, zorder=3, solid_capstyle="butt")
        ax.scatter([val], [yp[j]], s=300, color=col, zorder=4, edgecolor=SURFACE,
                   linewidth=2.0)
    ax.set_yticks(yp, [r[0] for r in rows], size=T_TICK_DENSE)
    lim = max(d_att_ci[1] * 1.15, max(abs(v) for v in vals) * 1.6, 0.01)
    left = min(min(r[2][0] for r in rows) * 1.35, -lim * 0.25)
    ax.set_xlim(left, lim)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("benign d′ against the calibration shape", size=T_AXIS)
    ax.xaxis.grid(True, color=GRID, lw=1.2)
    ax.set_axisbelow(True)
    fig.text(fx(x + 6.3), fy(y), "honest vs honest — must be ≈ 0", size=T_ANNO,
             color=S1, weight="bold", va="top")
    fig.text(fx(x + w), fy(y + 0.40), f"int4 quant: d′ = {d_att:.4f}", size=T_ANNO,
             color=S2, weight="bold", va="top", ha="right")
    used(draw_text(x, y + h - nh + 0.18, note, w, size=T_NOTE), y, h, "floor")


def p_takeaways(x, y, w, h):
    cheap_a, cheap_v, cheap_c = cheapest_cell()
    n_unreach = len(unreachable())
    needs_act = verifiers.get(cheap_v).needs_activation
    bullets = [
        ("Price the verdict, not the detector.",
         f"{n_unreach} of {n_unreach + len(reachable())} cells cost infinity; the "
         f"cheapest is {human_seconds(cheap_c['gpu_seconds_per_verdict'])}, on "
         f"{cheap_v} — which pays for it in access, not compute." if needs_act else
         f"{n_unreach} of {n_unreach + len(reachable())} cells cost infinity; the "
         f"cheapest costs {human_seconds(cheap_c['gpu_seconds_per_verdict'])}."),
        ("Buy the exponent, not the constant.",
         "Tokens go as 1/d′², price per token as model size. Effect size wins."),
        ("Spend on the accounting, not the estimator.",
         "Prefix scheduling is 16×; three better per-token signals all lost to "
         "the plain mean."),
        ("Stop early — and size the pool to say so.",
         f"Sequential testing saves {SQ['median_saving_x']:.1f}× where the pool can "
         f"resolve it, which is {SQ.get('n_within_ceiling', 0)} of "
         f"{SQ.get('n_within_ceiling', 0) + len(SQ.get('unresolvable', []))} cells."
         if SQ.get("median_saving_x") else
         "A continuously-audited provider is a sequential test — unresolvable at "
         "this pool size."),
        ("Measure your own benign floor first.",
         "It is the price ceiling. Calibrate in the shape you audit in."),
    ]
    dy = T_BODY * LEAD / 72
    y0 = y
    for head, rest in bullets:
        fig.text(fx(x), fy(y), "•", size=T_BODY, color=S1, va="top")
        fig.text(fx(x + 0.30), fy(y), head, size=T_BODY, color=INK, va="top", weight="bold")
        y = draw_text(x + 0.30, y + dy, rest, w - 0.32) + 0.16
    used(y, y0, h, "takeaways")


PANELS = [
    dict(col=0, kicker="1 · the unit of account",
         title="What a client buys is a verdict",
         draw=p_unit, h=6.75),
    dict(col=0, kicker="3 · cheaper tokens",
         title="The same accuracy, 16× cheaper",
         draw=p_prefix, flex=1.0),
    dict(col=0, kicker="5 · fewer tokens",
         title="Stop early — where the pool can prove it",
         draw=p_sequential, h=5.30),
    dict(col=0, kicker="7 · takeaways", title="What we would deploy",
         draw=p_takeaways, h=7.65),
    dict(col=1, kicker="2 · the plane",
         title="Cost and accuracy do not trade off along a curve",
         body="Each mark is one deviation scored by one detector. The axes are the two "
              "measured factors; the product is what it costs.",
         draw=p_plane, flex=1.72),
    # The two row-charts live together in column 1: five deviations and three
    # replay shapes need vertical room per row, and column 0 is text-bound.
    dict(col=1, kicker="4 · the cheap tier",
         title="A cheaper token is not a cheaper verdict",
         body="Tokens the same deviation needs under a full recompute of M and under a "
              "client-side proxy that never runs it.",
         draw=p_cheap, flex=1.02),
    dict(col=1, kicker="6 · the ceiling",
         title="The accuracy no budget buys",
         draw=p_floor, flex=1.35),
]

# ======================================================================= header
MARGIN = 0.9
GUTTER = 0.50
CW = (W - 2 * MARGIN - (NCOL - 1) * GUTTER) / NCOL
COLX = [MARGIN + i * (CW + GUTTER) for i in range(NCOL)]

TITLE = ["Verification is a cost/accuracy problem.",
         "So price the verdict, not the detector."]
t_size = min(fit_size(t, W - 2 * MARGIN, T_TITLE) for t in TITLE)
y = 0.90
fig.text(fx(MARGIN), fy(y), TITLE[0], size=t_size, weight="bold", color=INK, va="top")
y += t_size * 1.16 / 72
fig.text(fx(MARGIN), fy(y), TITLE[1], size=t_size, weight="bold", color=S1, va="top")
y += t_size * 1.34 / 72
fig.text(fx(MARGIN), fy(y),
         f"Jiawei Li   ·   Inference Verification Gym (ivgym)   ·   "
         f"new measurements on one NVIDIA H100, {CV['M'].split('/')[-1]} audited with a "
         f"{CV['proxy'].split('/')[-1]} proxy", size=T_AUTH, color=INK, va="top")
y += T_AUTH * 1.85 / 72
y = draw_text(MARGIN, y,
              "A client renting inference cannot see which model ran. Every defence "
              "against that is scored on detection accuracy and priced by a rate — "
              "FLOPs per sequence, seconds per prefill. Neither is what a client buys. "
              "It buys a VERDICT at a false-accusation budget it can live with, and a "
              "verdict's price factors into how many tokens the evidence needs and what "
              "a token of that evidence costs. Measured that way the ranking of the "
              "defences changes, three of them stop being defences at all, and the "
              "engineering that matters moves off the estimator and onto the accounting.",
              W - 2 * MARGIN, size=T_SUB, color=INK2) + 0.50

_cheap_a, _cheap_v, _cheap_c = cheapest_cell()
# the cheap tier's whole advantage, and what it costs in tokens: the one cell where
# a Tier-1 and a Tier-0 detector are both reachable on the same deviation
_both = [(a, CELLS[a]["accept_rate"], CELLS[a]["token_difr"]) for a in CELLS
         if CELLS[a]["accept_rate"]["reachable"] and CELLS[a]["token_difr"]["reachable"]]
_tokx = (max(_both, key=lambda t: t[1]["tokens_per_verdict"] / t[2]["tokens_per_verdict"])
         if _both else None)
HERO = [(human_seconds(_cheap_c["gpu_seconds_per_verdict"]),
         f"of H100 time for one verdict on\n{NICE.get(_cheap_a, _cheap_a)} "
         f"({_cheap_c['tokens_per_verdict']:,} "
         f"token{'' if _cheap_c['tokens_per_verdict'] == 1 else 's'}, {_cheap_v})"),
        (f"{len(unreachable())} of {len(unreachable())+len(reachable())}",
         "detector × deviation cells cost\ninfinity: no budget buys the verdict"),
        (f"{TIER1_US/PROXY_US:.2f}×",
         f"cheaper per token is all the cheap tier\nbuys — and it needs "
         f"{_tokx[1]['tokens_per_verdict']/_tokx[2]['tokens_per_verdict']:.0f}× "
         f"more of them" if _tokx else
         "cheaper per token is all\nthe cheap tier buys")]
hero_w = (W - 2 * MARGIN - 2 * GUTTER) / 3
hero_h = 2.25
for i, (big, small) in enumerate(HERO):
    hx = MARGIN + i * (hero_w + GUTTER)
    card(hx, y, hero_w, hero_h)
    fig.text(fx(hx + 0.38), fy(y + 0.32), big, size=T_HERO, weight="bold", color=S1,
             va="top")
    fig.text(fx(hx + 0.38), fy(y + 1.45), small, size=T_HEROCAP, color=INK2, va="top",
             linespacing=1.5)
BODY_TOP = y + hero_h + 0.60

# ================================================================ pack & render
# The footer is method + reproduce + limits, and it is long. Reserve exactly the
# height it wraps to at this board width rather than a constant, so a narrower
# board takes the space out of the panels instead of running off the bottom.
FOOT_TEXT = (
    f"Method: one protocol for every number — standardized partial AUC at "
    f"FPR ≤ 0.5%, threshold from a held-out honest calibration split, "
    f"winsorized at its 99.9th percentile, batch/pool ratio capped at 10% and "
    f"enforced in code. Prices are GPU-synchronised wall-clock on an idle H100 at "
    f"${USD_HR:.2f}/hour; d′ comes with a sequence-level bootstrap interval "
    f"({CV['n_boot']} resamples) because tokens inside a sequence are not "
    f"independent. Pools: {CV['n_prompts']}×{CV['tokens']} = "
    f"{CV['pool_tokens_per_config']:,} tokens per configuration.\n"
    "Reproduce: python -m experiments.exp_cost_of_a_verdict_gpu  ·  "
    "exp_benign_shape_dprime_gpu  ·  exp_sequential_verdict  ·  "
    "exp_prefix_cost_gpu  ·  this poster: python paper/make_cost_poster.py  "
    "·  artifacts: docs/results/{cost_of_a_verdict, benign_shape_dprime, "
    "sequential_verdict, prefix_cost_quant2bit, pool_scaling}.json\n"
    "Limits: attacks are logit-level models of quantisation, not quantised "
    "checkpoints; one prompt domain; one accelerator, so the cross-GPU half of "
    "the benign floor is still unmeasured — the shapes here are its lower "
    "bound. References: Karvonen et al., DiFR (2025)  ·  Leviathan et al. "
    "(2022), speculative decoding  ·  Wald (1945), sequential tests  ·  "
    "code and every run artifact: the inference-verification (ivgym) repository.")
FOOT_LEAD = 1.42
FOOT_H = text_h(FOOT_TEXT, W - 2 * MARGIN, T_FOOT, lead=FOOT_LEAD) + 0.62
BODY_H = H - BODY_TOP - FOOT_H
PAD = 0.42
for c in range(NCOL):
    ps = [p for p in PANELS if p["col"] == c]
    for p in ps:
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
        print(f"column {c}: body={BODY_H:.2f} fixed={fixed:.2f} gaps={gaps:.2f} "
              f"slack={slack:.2f}")
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
            fig.text(fx(x + PAD), fy(ty), p["title"], size=T_HEAD, color=INK,
                     weight="bold", va="top", linespacing=1.3)
            ty += T_HEAD * 1.30 / 72 * (p["title"].count("\n") + 1) + 0.18
        if p.get("body"):
            ty = draw_text(x + PAD, ty, p["body"], CW - 2 * PAD) + 0.18
        p["draw"](x + PAD, ty, CW - 2 * PAD, yy + h - ty - PAD * 0.55)
        yy += h + GUTTER

# ======================================================================= footer
foot_end = draw_text(MARGIN, H - FOOT_H + 0.30, FOOT_TEXT, W - 2 * MARGIN,
                     size=T_FOOT, color=INK3, lead=FOOT_LEAD)
assert foot_end <= H - 0.2, f"footer overflows the board by {foot_end - H + 0.2:.2f} in"

out = ROOT / "paper" / args.out
fig.savefig(out.with_suffix(".pdf"))
fig.savefig(out.with_suffix(".png"), dpi=int(2400 / max(W, H)))
print(f"wrote {out.with_suffix('.pdf')} ({W:g}x{H:g} in, {NCOL} columns) and .png")
