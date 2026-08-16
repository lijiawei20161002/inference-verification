"""The clock in four pictures, for a reader who has not read the other three figures.

`fig_clock_simple.png` is the argument; this is the intuition, and it deliberately
carries less. Four panels, one idea each, drawn as icons rather than as statistics:
a read takes time, a shorter read finishes early, early is a wall you cannot cross,
and only the deviations that shorten the read are on the wrong side of that wall.

  docs/figures/fig_clock_basic.png
      (1) A TOKEN IS A READ.  Weights + KV cache go through HBM once per decode
          step, so time per token >= bytes / bandwidth = 1.04 ms for bf16
          Qwen3-1.7B at 256 tokens of context. A floor, off a spec sheet.
      (2) FEWER BYTES ARRIVE EARLY.  The same five tokens on two wires: 1.04 ms
          apart honest, 0.27 ms apart at 4-bit weights. The hatched gap between
          them is the money AND the signal -- one quantity, read for free off a
          stream the client already bought.
      (3) TAKE THE MINIMUM.  Jitter is additive and positive, so it dirties the
          mean and leaves the minimum alone. One gap left of the floor is physics,
          not a p-value: d' = 1.54 per token, ~6 tokens of stream per verdict.
      (4) SCOPE.  Which deviations shorten the read (4-bit weights, wholesale
          substitution, lossless speculation, a long-context fp8 KV cache) and
          which move exactly the same bytes an honest provider does (wrong seed,
          wrong temperature, a broken top-k). The second group is not a weakness
          to fix -- it is where the returned-token detectors this repo already
          ships are cheapest, 226 tokens for a wrong seed.

MEASURED INPUTS   docs/results/cost_of_a_verdict.json, for the parameter counts
    every roofline number is computed from, delta* = 3.767, and the one measured
    price quoted in (4)'s note (226 tokens for a wrong seed, token DiFR).

MODELLED INPUTS   every clock number: bytes read / 3.35 TB/s (H100 SXM5 spec) over
    an assumed one-sided jitter sigma = 0.50 ms. Nothing in this repo has ever
    timed a provider, so this is the design of exp_clock_channel_gpu and not its
    result. The sigma sweep that shows (4)'s grouping does not move lives in
    fig_clock_channel_principle (C); the measured token-channel prices, and the
    batching hole restated in this figure's footer, live in fig_clock_simple.

    python -m experiments.plot_clock_basic
"""
from __future__ import annotations

import numpy as np

from experiments.plot_clock_channel_principle import (
    FIG_DIR, JITTER_MS, JITTER_SHAPE, kv_bytes_per_token, load_measured, roofline_ms,
)
from experiments.plot_clock_simple import CTX, SPECDEC, floor_ms, rows

# A louder cut of the house palette, with the same three meanings: slate = honest,
# blue = the deviation and the channel that reads it, orange = the time never spent.
# Every mark is direct-labelled, so nothing rests on colour alone.
SLATE, SLATE_L = "#3d4756", "#dcdfe4"
BLUE, BLUE_L = "#1668d8", "#cfe1fb"
ORANGE, ORANGE_L = "#d4551a", "#fbe1d2"
INK, INK2, MUTED, HAIR = "#101010", "#4a4a48", "#8d8b84", "#e6e5e0"

SIGMA = JITTER_MS
N_STRIP = 44                    # gaps drawn in (3): a readable strip, not a sample


def mb(s: str) -> str:
    """Plain text -> bold mathtext, so a bold phrase can sit inside a normal line."""
    body = s.replace(" ", "\\ ").replace("-", "\\!-\\!").replace("%", "\\%")
    return f"$\\bf{{{body}}}$"


# ------------------------------------------------------------------- icon shapes
def canvas(ax, w_in, h_in):
    """A blank axes whose data units are square: x in [0, 100], y in [0, YH]."""
    yh = 100.0 * h_in / w_in
    ax.set_xlim(0, 100)
    ax.set_ylim(0, yh)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return yh


def clock(ax, cx, cy, r, frac, color, *, slash=False, lw=2.6):
    """A clock face with `frac` of one revolution swept out, drawn as a wedge."""
    from matplotlib.patches import Circle, Wedge

    ax.add_patch(Circle((cx, cy), r, facecolor="white", edgecolor=color, lw=lw,
                        zorder=5))
    ax.add_patch(Wedge((cx, cy), r * 0.88, 90 - 360 * frac, 90, facecolor=color,
                       alpha=0.20, lw=0, zorder=4))
    for k in range(12):                                    # hour pips
        a = np.pi / 2 - k * np.pi / 6
        r0 = r * (0.80 if k % 3 else 0.70)
        ax.plot([cx + r0 * np.cos(a), cx + r * 0.92 * np.cos(a)],
                [cy + r0 * np.sin(a), cy + r * 0.92 * np.sin(a)],
                color=color, lw=1.0 if k % 3 else 1.8, alpha=0.65, zorder=6,
                solid_capstyle="round")
    a = np.pi / 2 - 2 * np.pi * frac                       # the sweeping hand
    ax.plot([cx, cx + r * 0.86 * np.cos(a)], [cy, cy + r * 0.86 * np.sin(a)],
            color=color, lw=lw * 0.8, zorder=7, solid_capstyle="round")
    a2 = np.pi / 2 - 2 * np.pi * (frac / 12 + 0.02)        # a short hand, for looks
    ax.plot([cx, cx + r * 0.50 * np.cos(a2)], [cy, cy + r * 0.50 * np.sin(a2)],
            color=color, lw=lw * 1.1, zorder=7, solid_capstyle="round")
    ax.plot([cx], [cy], marker="o", ms=lw * 1.5, color=color, zorder=8)
    if slash:                                              # the clock reads nothing
        d = r * 0.86
        ax.plot([cx - d, cx + d], [cy - d, cy + d], color=color, lw=lw * 1.4,
                zorder=9, solid_capstyle="round")


def token(ax, cx, cy, w, h, color, label=None, *, fs=9.0):
    """One returned token, drawn as a pill."""
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={h * 0.34:.2f}",
                                facecolor=color, edgecolor="white", lw=1.1, zorder=5))
    if label:
        ax.annotate(label, (cx, cy), ha="center", va="center", color="white",
                    fontsize=fs, weight="bold", zorder=6)


def box(ax, x0, x1, y0, y1, face, edge, lw=1.8, r=1.2, ls="-", z=4, alpha=1.0):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=face, edgecolor=edge, lw=lw, ls=ls,
                                alpha=alpha, zorder=z))


def bar(ax, x0, x1, y, h, color, alpha=1.0, z=4):
    from matplotlib.patches import Rectangle

    ax.add_patch(Rectangle((x0, y - h / 2), x1 - x0, h, facecolor=color,
                           edgecolor="white", lw=1.0, alpha=alpha, zorder=z))


def fat_arrow(ax, x0, x1, y, color, lw=9.0):
    ax.annotate("", (x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>,head_width=0.42,head_length=0.60",
                                color=color, lw=lw, shrinkA=0, shrinkB=0), zorder=5)


def badge(ax, cx, cy, r, kind, color):
    """A filled tick or a hollow cross: does this deviation read fewer bytes?"""
    from matplotlib.patches import Circle

    ok = kind == "yes"
    ax.add_patch(Circle((cx, cy), r, facecolor=color if ok else "white",
                        edgecolor=color, lw=1.9, zorder=5))
    if ok:
        ax.plot([cx - r * 0.44, cx - r * 0.08, cx + r * 0.50],
                [cy + r * 0.04, cy - r * 0.36, cy + r * 0.42], color="white", lw=2.4,
                zorder=6, solid_capstyle="round", solid_joinstyle="round")
    else:
        d = r * 0.44
        for s in (1, -1):
            ax.plot([cx - d, cx + d], [cy - s * d, cy + s * d], color=color, lw=2.2,
                    zorder=6, solid_capstyle="round")


# ============================================================ (1) a token is a read
def panel_1(ax, w_in, h_in, p):
    yh = canvas(ax, w_in, h_in)
    wb, kb = p * 2, kv_bytes_per_token(2) * CTX
    ms = roofline_ms(wb + kb)
    cy = yh * 0.685

    box(ax, 3, 27.5, cy - 7.4, cy + 7.4, BLUE_L, BLUE, r=1.4)
    ax.annotate("WEIGHTS", (15.2, cy + 4.4), ha="center", va="center", color=BLUE,
                fontsize=11.5, weight="bold", zorder=6)
    ax.annotate(f"1.72B params $\\times$ 2 bytes\n{mb(f'{wb / 1e9:.2f} GB')}",
                (15.2, cy - 2.0), ha="center", va="center", color=INK, fontsize=9.8,
                linespacing=1.7, zorder=6)
    box(ax, 3, 27.5, cy - 16.6, cy - 9.6, "white", MUTED, lw=1.3, r=1.0)
    ax.annotate(f"+  KV cache,  {mb(f'{kb / 1e6:.0f} MB')}\n256 tokens of context",
                (15.2, cy - 13.1), ha="center", va="center", color=INK2, fontsize=9.3,
                linespacing=1.6, zorder=6)

    fat_arrow(ax, 30.5, 43.0, cy, SLATE)
    ax.annotate("read every byte,\nonce per token", (36.7, cy + 5.2), ha="center",
                va="bottom", color=SLATE, fontsize=9.6, weight="bold", linespacing=1.5,
                zorder=6)
    ax.annotate("HBM, 3.35 TB/s", (36.7, cy - 5.2), ha="center", va="top", color=MUTED,
                fontsize=9.2, zorder=6)

    clock(ax, 54.5, cy, 8.6, 0.31, ORANGE)
    ax.annotate(mb(f"{ms:.2f} ms"), (54.5, cy - 11.2), ha="center", va="top",
                color=ORANGE, fontsize=13.5, zorder=6)
    ax.annotate("of reading, before anything\ncan possibly come back",
                (54.5, cy - 15.4), ha="center", va="top", color=ORANGE, fontsize=9.0,
                linespacing=1.5, zorder=6)

    fat_arrow(ax, 65.5, 76.0, cy, SLATE)
    ax.annotate("then, and\nnot before", (70.7, cy + 5.2), ha="center", va="bottom",
                color=MUTED, fontsize=9.0, linespacing=1.5, zorder=6)
    token(ax, 86.5, cy, 19.0, 10.6, BLUE, "1 token", fs=11.5)
    ax.annotate("what the client\nis charged for", (86.5, cy - 8.6), ha="center",
                va="top", color=INK2, fontsize=9.0, linespacing=1.5, zorder=6)

    ax.annotate(f"{mb('time per token')}   $\\geq$   "
                "$\\dfrac{\\bf{bytes\\ read}}{\\bf{bandwidth}}$   =   "
                f"$\\dfrac{{{(wb + kb) / 1e9:.2f}\\ \\rm{{GB}}}}"
                f"{{3.35\\ \\rm{{TB/s}}}}$   =   {mb(f'{ms:.2f} ms')}",
                (50, yh * 0.235), ha="center", va="center", color=INK, fontsize=16.0,
                zorder=6)
    ax.annotate("A batch-1 decode step is memory-bound: it cannot answer before the "
                "bytes have moved, and no cache,\nscheduler or kernel trick gets under "
                f"that.  So every serving config has a {mb('floor')} in ms per token "
                "--\n$\\it{arithmetic\\ off\\ a\\ spec\\ sheet}$, not a number "
                "calibrated from a pool.",
                (3, yh * 0.075), ha="left", va="center", color=INK2, fontsize=9.8,
                linespacing=1.75, zorder=6)

    ax.set_title("1.   A token is a read.  Reading takes time.", color=INK,
                 fontsize=14.0, weight="bold", loc="left", pad=9)


# ==================================================== (2) fewer bytes arrive early
def panel_2(ax, w_in, h_in, p):
    from matplotlib.patches import Rectangle

    yh = canvas(ax, w_in, h_in)
    f16, f4 = floor_ms(p * 2, 2, CTX), floor_ms(p * 0.5, 2, CTX)
    n = 5
    X = lambda t: 10.0 + t * 13.2                        # ms -> canvas x
    y16, y4 = yh * 0.760, yh * 0.360

    for y, lab, col, gap, note in ((y16, "honest bf16", SLATE, f16,
                                    "reads all 3.47 GB"),
                                   (y4, "4-bit weights", BLUE, f4,
                                    "reads 1/4 of the weight bytes")):
        ax.plot([8, 84.5], [y, y], color=HAIR, lw=6.0, solid_capstyle="round",
                zorder=2)
        ax.annotate(f"{mb(lab)}   {note}", (8, y + 7.4), ha="left", va="bottom",
                    color=col, fontsize=10.4, zorder=6)
        for k in range(1, n + 1):
            token(ax, X(gap * k), y, 3.1, 8.2, col)
        clock(ax, 93.0, y, 5.2, (gap * n) / 6.0, col, lw=2.0)
        tot = f"{mb(f'{gap * n:.1f} ms')}"          # the lower row's total stays on one
        if y == y16:                                # line, to clear the panel caption
            tot += f"\nfor {n} tokens"
        ax.annotate(tot, (93.0, y - 7.4), ha="center", va="top", color=col,
                    fontsize=8.8, linespacing=1.55, zorder=6)

    ax.annotate("", (X(f16), y16 - 6.2), xytext=(X(2 * f16), y16 - 6.2),
                arrowprops=dict(arrowstyle="<|-|>", color=SLATE, lw=1.6, shrinkA=0,
                                shrinkB=0), zorder=6)
    ax.annotate(f"{mb(f'{f16:.2f} ms')} between tokens", (X(1.5 * f16), y16 - 7.8),
                ha="center", va="top", color=SLATE, fontsize=9.4, zorder=6)

    ax.add_patch(Rectangle((X(f4 * n), y4 - 4.7), X(f16 * n) - X(f4 * n), 9.4,
                           facecolor=ORANGE_L, edgecolor=ORANGE, lw=1.6, hatch="///",
                           zorder=3))
    ax.annotate(mb(f"{(f16 - f4) * n:.1f} ms of reading it never did"),
                ((X(f4 * n) + X(f16 * n)) / 2, y4 + 0.4), ha="center", va="bottom",
                color=ORANGE, fontsize=11.5, zorder=7,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                          alpha=0.72))
    ax.annotate("this is the money -- and it is also the evidence",
                ((X(f4 * n) + X(f16 * n)) / 2, y4 - 1.0), ha="center", va="top",
                color=ORANGE, fontsize=9.6, zorder=7,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                          alpha=0.72))
    ax.annotate(f"{mb(f'{f4:.2f} ms')} between tokens", (X(f4 * n) + 2.5, y4 - 9.0),
                ha="left", va="top", color=BLUE, fontsize=9.4, zorder=6)
    ax.annotate("", (X(f4 * 1.6), y4 - 5.6), xytext=(X(f4 * n) + 2.2, y4 - 8.8),
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.0), zorder=6)

    ax.annotate(f"{mb(f'{f16 / f4:.1f}x')} fewer bytes read is "
                f"{mb(f'{f16 / f4:.1f}x')} less time spent -- and the deviations worth "
                "doing are exactly the ones\nthat read fewer bytes.  So the provider's "
                f"saving and the client's signal are {mb('one quantity')}: zero "
                "audit\nFLOPs, zero cooperation, evidence that arrives with a stream "
                "the client already bought.",
                (3, yh * 0.075), ha="left", va="center", color=INK2, fontsize=9.8,
                linespacing=1.75, zorder=6)

    ax.set_title("2.   Read fewer bytes and the token arrives early.", color=INK,
                 fontsize=14.0, weight="bold", loc="left", pad=9)


# ============================================================ (3) take the minimum
def panel_3(ax, w_in, h_in, p, R):
    from matplotlib.patches import Rectangle

    yh = canvas(ax, w_in, h_in)
    f16, f4 = floor_ms(p * 2, 2, CTX), floor_ms(p * 0.5, 2, CTX)
    d, n_tok = R[0]["d"], R[0]["n"]
    rng = np.random.default_rng(0)
    jit = lambda k: rng.gamma(JITTER_SHAPE, SIGMA / JITTER_SHAPE, k)
    honest, fast = f16 + jit(N_STRIP), f4 + jit(N_STRIP)

    X = lambda t: 12.0 + t * 26.3                        # ms -> canvas x, 0..3 ms
    y0, top = yh * 0.290, yh * 0.925
    yhon, yfast = yh * 0.755, yh * 0.520

    ax.add_patch(Rectangle((X(0), y0), X(f16) - X(0), top - y0, facecolor=ORANGE_L,
                           edgecolor="none", zorder=1))
    ax.plot([X(f16)] * 2, [y0, top], color=ORANGE, lw=3.6, zorder=6,
            solid_capstyle="butt")
    ax.annotate("IMPOSSIBLE", (X(f16 / 2), yh * 0.885), ha="center", va="center",
                color=ORANGE, fontsize=14.0, weight="bold", zorder=4)
    ax.annotate("no provider that reads\nall 3.47 GB is ever in here",
                (X(f16 / 2), yh * 0.848), ha="center", va="top", color=ORANGE,
                fontsize=9.0, linespacing=1.5, zorder=4)
    ax.annotate(f"{mb('the floor')}\n{f16:.2f} ms", (X(f16) + 2.0, top), ha="left",
                va="top", color=ORANGE, fontsize=10.0, linespacing=1.55, zorder=6)

    # Each row's label rides above its right-hand end, where no gap ever lands: a
    # dot at 2.8 ms is already 3.5 sigma into the jitter's tail.
    for x, y, col, lab in ((honest, yhon, SLATE, "honest bf16"),
                           (fast, yfast, BLUE, "4-bit, speedup passed on")):
        ax.plot(X(np.clip(x, 0, 2.8)), np.full(len(x), y), marker="o", ms=7.5, ls="",
                color=col, alpha=0.42, mec="white", mew=0.8, zorder=4)
        ax.annotate(f"{mb(lab)}    {mb(f'minimum = {x.min():.2f} ms')}", (96, y + 4.4),
                    ha="right", va="bottom", color=col, fontsize=10.0, zorder=6)
        ax.plot([X(x.min())], [y - 4.0], marker="^", ms=13, color=col, mec="white",
                mew=1.2, zorder=7)
    ax.annotate("just right of its own floor, as it must be",
                (X(honest.min()) + 3.2, yhon - 4.4), ha="left", va="center",
                color=SLATE, fontsize=9.4, zorder=7)
    ax.annotate(mb(f"{f16 / fast.min():.1f}x under a floor it cannot cross"),
                (X(fast.min()) + 3.2, yfast - 4.8), ha="left", va="center", color=BLUE,
                fontsize=9.8, zorder=7)

    ax.plot([X(0), 95], [y0, y0], color=MUTED, lw=1.2, zorder=5)
    for t in (0, 1, 2, 3):
        ax.plot([X(t)] * 2, [y0, y0 - 1.3], color=MUTED, lw=1.2, zorder=5)
        ax.annotate(f"{t}", (X(t), y0 - 2.4), ha="center", va="top", color=INK2,
                    fontsize=9.2, zorder=5)
    ax.annotate("time per output token as the client sees it  (ms) -- each dot is one "
                "gap", (95, y0 - 5.4), ha="right", va="top", color=INK2, fontsize=9.2,
                zorder=5)

    ax.annotate("Nuisance on the wire only $\\bf{adds}$ time -- queueing, co-tenancy, "
                "a slow hop push a gap right, never left.\nSo the mean is contaminated "
                f"and the {mb('minimum is not')}.  One gap left of the floor is physics "
                f"rather than a\np-value: $d'$ = {d:.2f} per token, so this repo's cost "
                f"law prices the verdict at {mb(f'~{round(n_tok)} tokens')} of stream.",
                (3, yh * 0.075), ha="left", va="center", color=INK2, fontsize=9.8,
                linespacing=1.75, zorder=6)

    ax.set_title("3.   So take the minimum gap.  Early is a wall you cannot cross.",
                 color=INK, fontsize=14.0, weight="bold", loc="left", pad=9)


# =========================================================== (4) what it can't see
def panel_4(ax, w_in, h_in, R):
    yh = canvas(ax, w_in, h_in)
    X0, X1 = 35.0, 70.0                                  # honest read = the full bar
    NAME = {                                             # the bar carries the rest
        "quant_4bit": "4-bit weights",
        "substitution": "0.6B served as 1.7B",
        "specdec": f"lossless speculation, {SPECDEC:.1f}x",
        "kv_long": "fp8 KV cache, 32k context",
        "seed_43": "wrong seed",
        "temp_1.1": "wrong temperature",
        "bug_k32": "top-k bug",
    }
    loud = [r for r in R if r["key"] in ("quant_4bit", "substitution", "specdec",
                                         "kv_long")]
    mute = [r for r in R if r["key"] in ("seed_43", "temp_1.1", "bug_k32")]

    ax.annotate("bytes read per token   (dashed = what an honest provider reads)",
                (X0, yh * 1.005), ha="left", va="top", color=MUTED, fontsize=8.8,
                zorder=6)
    ax.annotate("tokens of stream\nthe clock needs", (99, yh * 1.005), ha="right",
                va="top", color=MUTED, fontsize=8.8, linespacing=1.5, zorder=6)

    def group(y_head, head, sub, col, rs, seen):
        clock(ax, 4.6, y_head + 0.4, 3.4, 0.28, col, slash=not seen, lw=1.9)
        ax.annotate(head, (9.6, y_head + 2.0), ha="left", va="center", color=col,
                    fontsize=11.6, weight="bold", zorder=6)
        ax.annotate(sub, (9.6, y_head - 1.9), ha="left", va="center", color=INK2,
                    fontsize=9.2, zorder=6)
        for i, r in enumerate(rs):
            y = y_head - 5.0 - 4.1 * i
            badge(ax, 4.6, y, 2.0, "yes" if seen else "no", col)
            ax.annotate(mb(NAME[r["key"]]), (9.6, y), ha="left", va="center",
                        color=INK if seen else INK2, fontsize=9.8, zorder=6)
            frac = 1.0 / r["bytes_x"]
            box(ax, X0, X1, y - 1.7, y + 1.7, "none", MUTED, lw=1.1, r=0.5, ls="--",
                z=3)
            bar(ax, X0, X0 + (X1 - X0) * frac, y, 3.4, col if seen else SLATE_L,
                z=3 if seen else 2)
            if seen:
                ax.annotate("", (X0 + (X1 - X0) * frac, y), xytext=(X1, y),
                            arrowprops=dict(arrowstyle="<|-|>", color=ORANGE, lw=1.3,
                                            shrinkA=1, shrinkB=1), zorder=6)
                ax.annotate(f"reads {frac:.0%} of the bytes", (X1 + 2.2, y), ha="left",
                            va="center", color=INK2, fontsize=9.0, zorder=6)
                ax.annotate(mb(f"{round(r['n'])}"), (99, y), ha="right", va="center",
                            color=col, fontsize=12.5, zorder=6)
            else:
                ax.annotate("identical bytes", ((X0 + X1) / 2, y), ha="center",
                            va="center", color=SLATE, fontsize=8.8, zorder=6)
                ax.annotate("$\\bf{0}$ ms saved", (X1 + 2.2, y), ha="left",
                            va="center", color=INK2, fontsize=9.0, zorder=6)
                ax.annotate(mb("never"), (99, y), ha="right", va="center", color=col,
                            fontsize=11.0, zorder=6)

    group(yh * 0.930, "READS FEWER BYTES   \u2192   THE CLOCK SEES IT",
          "the deviation stops moving bytes, so it stops spending time", BLUE, loud,
          True)
    group(yh * 0.440, "SAME BYTES AS HONEST   \u2192   THE CLOCK IS BLIND",
          "no time is saved, so there is nothing in the arrival times to read", SLATE,
          mute, False)

    ax.annotate("Which group a deviation lands in is settled by "
                f"{mb('arithmetic')}, before any data is collected -- the clock\nasks a "
                "physical question, $\\it{did\\ you\\ read\\ these\\ bytes}$.  The "
                f"lower group is a {mb('scope, not a')}\n{mb('weakness')}: it is exactly "
                "where the returned-token detectors this repo ships are cheapest, "
                "226 tokens for a wrong seed.",
                (3, yh * 0.058), ha="left", va="center", color=INK2, fontsize=9.8,
                linespacing=1.75, zorder=6)

    ax.set_title("4.   It sees only what saves the provider money -- and that is the "
                 "whole scope.", color=INK, fontsize=14.0, weight="bold", loc="left",
                 pad=9)


# ========================================================================== main
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cost, _ = load_measured()
    p, pp = cost["m_params"], cost["proxy_params"]
    R = rows(p, pp)

    FW, FH = 19.4, 12.4
    L, RT, B, T, WS, HS = 0.021, 0.988, 0.062, 0.845, 0.060, 0.130
    fig = plt.figure(figsize=(FW, FH))
    gs = fig.add_gridspec(2, 2, left=L, right=RT, bottom=B, top=T, wspace=WS,
                          hspace=HS)
    w_in = (RT - L) * FW / (2 + WS)                      # one panel, in inches
    h_in = (T - B) * FH / (2 + HS)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    panel_1(axes[0], w_in, h_in, p)
    panel_2(axes[1], w_in, h_in, p)
    panel_3(axes[2], w_in, h_in, p, R)
    panel_4(axes[3], w_in, h_in, R)

    fig.suptitle("The clock, in four pictures: a token cannot arrive before its bytes "
                 "have moved", color=INK, fontsize=20.0, weight="bold", x=0.021,
                 ha="left", y=0.984)
    fig.text(0.021, 0.942,
             f"{mb('Bytes are time.')}  The deviations a provider makes money on are "
             f"exactly the ones that read fewer bytes -- so they arrive {mb('early')}, "
             "and early is visible in a stream the client already bought.",
             color=INK, fontsize=12.4, ha="left", va="top")
    fig.text(0.021, 0.913,
             "$\\it{Every\\ clock\\ number\\ here\\ is\\ arithmetic}$ -- bytes read / "
             "3.35 TB/s (H100 SXM5 spec) over an assumed one-sided jitter $\\sigma$ = "
             f"{SIGMA:.2f} ms, off Qwen3-1.7B's own parameter count.\nNothing in this "
             "repo has ever timed a provider, so read this as the design of "
             "$\\tt{exp\\_clock\\_channel\\_gpu}$ and not its result.  The measured "
             "token-channel prices, the $\\sigma$ sweep and the full argument are in "
             "$\\tt{fig\\_clock\\_simple.png}$.",
             color=MUTED, fontsize=9.4, ha="left", va="top", linespacing=1.65)
    fig.text(0.021, 0.012,
             f"{mb('The one hole, stated here rather than found later.')}  Continuous "
             "batching reads one copy of the weights for $B$ concurrent requests, so a "
             "$\\it{weight}$ attack's floor drops as 1/$B$ and the wall in (3) softens "
             "into a statistic past $B$ $\\approx$ 20.\nIt cannot amortize "
             "$\\it{your}$ KV cache, which is read once per token per request at any "
             "$B$ -- so the 32k-context row in (4) is the one that survives batching, "
             "and it is also the row the returned-token channel reads worst.",
             color=INK2, fontsize=9.4, ha="left", va="bottom", linespacing=1.65)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_clock_basic.png"
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out}")
    for r in R:
        n = "never" if not np.isfinite(r["n"]) else f"{round(r['n']):,}"
        print(f"  {r['key']:14s} {r['honest']:.3f} -> {r['served']:.3f} ms   "
              f"{1 / r['bytes_x']:>4.0%} of the read   clock {n:>6s} tok")


if __name__ == "__main__":
    main()
