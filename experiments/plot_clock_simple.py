"""The one-idea version of `fig_clock_vs_difr.png`: why / how / what, in three panels.

That figure argues the clock is ORTHOGONAL to DiFR, which is the second question.
This one answers the first three: why a latency channel exists at all, how you read
it, and which deviations it can and cannot see. Nothing here is new arithmetic --
every number is the same roofline and the same cost law as
`plot_clock_channel_principle` / `plot_clock_vs_difr`, imported rather than copied.

  docs/figures/fig_clock_simple.png
      (A) WHY.  A batch-1 decode step is memory-bandwidth-bound: it cannot return
          a token before the bytes have finished moving. So every serving config
          has a FLOOR in ms per output token, and it is arithmetic off a spec
          sheet -- 1.04 ms for bf16 Qwen3-1.7B at 256 tokens, 0.27 ms if the
          weights were quantized to 4 bits. A provider that cheats to save money
          saves bytes, and bytes are time. The saving is the signal.
      (B) HOW.  Time the inter-token gaps and take the MINIMUM. Every nuisance on
          the wire -- queueing, co-tenancy, network -- is additive and positive, so
          it contaminates the mean and leaves the minimum alone. One token left of
          the honest floor is a physical impossibility, not a p-value. Padding back
          to the honest mean defeats the mean test exactly, and leaves a clock too
          clean to pass a variance test.
      (C) WHAT.  Which deviations move bytes. Weight quantization, wholesale
          substitution, lossless speculation and a long-context fp8 KV cache all do,
          and cost the clock 6-12 tokens of stream. Wrong seed, wrong temperature
          and a top-k bug move exactly zero bytes -- the clock is not weak there, it
          is ABSENT, and the panel says so. The same fp8 KV cache at 256-token
          context is in the second group: the cache is 0.8% of the read, so the
          attack only becomes visible where it becomes worth doing.

MEASURED INPUTS   docs/results/cost_of_a_verdict.json -- every token-channel price
    in (C) (Qwen3-1.7B audited with Qwen3-0.6B, 80 seq x 256 tok per arm,
    standardized pAUC at FPR <= 0.5%, delta* = 3.767), plus the parameter counts
    both roofline and substitution row are computed from.

MODELLED INPUTS   the roofline (bytes read / 3.35 TB/s, H100 SXM5 spec) and the
    one-sided jitter sigma = 0.50 ms. Nothing in this repo has ever timed a
    provider, so every clock number on every panel is a PREDICTION. The load-bearing
    claim is which rows land in which group in (C), not the absolute values --
    fig_clock_channel_principle (C) sweeps sigma over a decade and the grouping does
    not move.

    python -m experiments.plot_clock_simple
"""
from __future__ import annotations

import json

import numpy as np

from experiments.plot_clock_channel_principle import (
    COST, DEV, FIG_DIR, GRID, HONEST, INK, INK2, JITTER_MS, JITTER_SHAPE, MUTED,
    N_TOK, ROOFLINE, WARN, _clean, kv_bytes_per_token, load_measured, roofline_ms,
)

SIGMA = JITTER_MS
CTX = ROOFLINE["ctx"]
CTX_LONG = ROOFLINE["ctx_long"]
SPECDEC = 2.2               # representative lossless speculation speedup, as in (A)
                            # of fig_clock_channel_principle -- a typical value, not
                            # a measurement, and labelled that way on the panel.


# ------------------------------------------------------------------------ model
def floor_ms(param_bytes: float, kv_width: int, ctx: int) -> float:
    return roofline_ms(param_bytes + kv_bytes_per_token(kv_width) * ctx)


def tokens_for(d_prime: float) -> float:
    ds = json.load(open(COST))["delta_star"]
    return np.inf if d_prime <= 0 else (ds / d_prime) ** 2


def clock_cost(honest: float, served: float) -> tuple[float, float]:
    """(per-token d', tokens of stream per verdict) for a gap of honest - served ms."""
    d = max(honest - served, 0.0) / SIGMA
    return d, tokens_for(d)


def rows(p, pp):
    """Every deviation, its floor and its two prices. Order is the panel's order."""
    h256, h32k = floor_ms(p * 2, 2, CTX), floor_ms(p * 2, 2, CTX_LONG)
    # (key, mathtext label, served floor ms, honest floor ms, note about the bytes)
    spec = (
        ("quant_4bit", "$\\bf{4\\!-\\!bit\\ weights}$", floor_ms(p * 0.5, 2, CTX),
         h256, "reads 1/4 of the weight bytes"),
        ("substitution", "$\\bf{0.6B\\ served\\ as\\ 1.7B}$", floor_ms(pp * 2, 2, CTX),
         h256, "reads a smaller model's weights"),
        ("specdec", "$\\bf{lossless\\ speculation}$", h256 / SPECDEC, h256,
         f"{SPECDEC:.1f}$\\times$ fewer target passes (a typical speedup)"),
        ("kv_long", "$\\bf{fp8\\ KV\\ cache,\\ 32k\\ context}$",
         floor_ms(p * 2, 1, CTX_LONG), h32k, "the cache is 52% of the read here"),
        ("kv_fp8", "$\\bf{fp8\\ KV\\ cache,\\ 256\\ context}$", floor_ms(p * 2, 1, CTX),
         h256, "the same attack where the cache is 0.8% of the read"),
        ("seed_43", "$\\bf{wrong\\ seed}$", h256, h256, "identical bytes"),
        ("temp_1.1", "$\\bf{wrong\\ temperature}$", h256, h256, "identical bytes"),
        ("bug_k32", "$\\bf{top}$-$\\bf{k\\ bug}$", h256, h256, "identical bytes"),
    )
    out = []
    for key, mlabel, served, honest, note in spec:
        d, n = clock_cost(honest, served)
        out.append(dict(key=key, mlabel=mlabel, served=served, honest=honest,
                        note=note, d=d, n=n, bytes_x=honest / served))
    return out


# =========================================================================== (A)
def panel_a(ax, p, pp):
    """Bytes are time, so a serving config has a floor and cheating goes under it."""
    w16, w4, w06 = p * 2, p * 0.5, pp * 2
    kv16, kv8 = kv_bytes_per_token(2), kv_bytes_per_token(1)

    # (y, label, weight bytes, kv bytes, colour) -- two contexts, two attack families.
    bars = (
        (3.6, "honest bf16", w16, kv16 * CTX, HONEST),
        (2.4, "4-bit weights", w4, kv16 * CTX, DEV),
        (1.2, "honest bf16", w16, kv16 * CTX_LONG, HONEST),
        (0.0, "fp8 KV cache", w16, kv8 * CTX_LONG, DEV),
    )
    h = 0.52
    for y, lab, wb, kb, col in bars:
        wms, kms = roofline_ms(wb), roofline_ms(kb)
        ax.barh(y, wms, height=h, color=col, alpha=0.92, zorder=4)
        ax.barh(y, kms, height=h, left=wms, color=col, alpha=0.32, zorder=4,
                edgecolor=col, lw=0.9)
        ax.annotate(f"$\\bf{{{wms + kms:.2f}\\ ms}}$", (wms + kms, y), xytext=(7, 0),
                    textcoords="offset points", ha="left", va="center", color=col,
                    fontsize=9.6, zorder=6)

    # The stolen time, drawn as the distance between the two bars of each pair.
    for (ya, yb), (side, ha) in zip(((3.6, 2.4), (1.2, 0.0)),
                                    ((1, "left"), (-1, "right"))):
        a = next(b for b in bars if b[0] == ya)
        d = next(b for b in bars if b[0] == yb)
        fa, fd = roofline_ms(a[2] + a[3]), roofline_ms(d[2] + d[3])
        ax.annotate("", xy=(fa, (ya + yb) / 2), xytext=(fd, (ya + yb) / 2),
                    arrowprops=dict(arrowstyle="<|-|>", color=WARN, lw=1.8,
                                    shrinkA=0, shrinkB=0), zorder=7)
        ax.annotate(f"$\\bf{{{fa - fd:.2f}\\ ms/token}}$ never spent  "
                    f"($\\bf{{{fa / fd:.2f}\\times}}$ the bytes)",
                    (fa if side > 0 else fd, (ya + yb) / 2),
                    xytext=(11 * side, 0), textcoords="offset points", ha=ha,
                    va="center", color=WARN, fontsize=9.0, zorder=7)

    ax.annotate("$\\bf{256\\!-\\!token\\ context}$  (this repo's audited pool)",
                (0.010, 4.34), xycoords=("axes fraction", "data"), ha="left",
                va="center", color=INK2, fontsize=9.2)
    ax.annotate("$\\bf{32k\\ context}$  (where an fp8 cache pays off)",
                (0.010, 1.80), xycoords=("axes fraction", "data"), ha="left",
                va="center", color=INK2, fontsize=9.2)
    ax.annotate("solid = weights,  pale = KV cache", (0.990, 4.34),
                xycoords=("axes fraction", "data"), ha="right", va="center",
                color=MUTED, fontsize=8.8)

    ax.annotate("A batch-1 decode step cannot return a token before it has "
                "$\\bf{read\\ the\\ weights}$.\nAt 3.35 TB/s that read has a "
                "duration, and no scheduler, cache or kernel\ntrick gets under it -- "
                "so every serving config has a $\\bf{floor}$ in ms per token,\n"
                "$\\it{computed\\ from\\ a\\ spec\\ sheet\\ rather\\ than\\ calibrated\\ "
                "from\\ a\\ pool.}$\n\n"
                "$\\bf{The\\ deviations\\ worth\\ doing\\ are\\ exactly\\ the\\ ones\\ "
                "that\\ move\\ fewer\\ bytes,}$\nbecause bytes are what the provider is "
                "paying for.  So the money and the\nsignal are the same quantity, and "
                "the clock reads it for free: the evidence\narrives with the stream "
                "the client already bought.",
                (0.010, -1.05), xycoords=("axes fraction", "data"), ha="left",
                va="top", color=INK, fontsize=9.0, linespacing=1.62)

    ax.set_xlim(0, 2.62)
    ax.set_ylim(-5.30, 4.70)
    ax.set_yticks([b[0] for b in bars], [b[1] for b in bars], fontsize=9.4)
    ax.tick_params(axis="y", length=0)
    for t, b in zip(ax.get_yticklabels(), bars):
        t.set_color(b[4])
    ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel("floor on time per output token  (ms; bytes read / 3.35 TB/s, "
                  "H100 SXM5 spec)", color=INK2, fontsize=10)
    ax.set_title("A.  WHY: fewer bytes read is less time spent", color=INK,
                 fontsize=11.5, weight="bold", loc="left", pad=8)


# =========================================================================== (B)
def panel_b(ax, p):
    """The statistic: minimum of the inter-token gaps, against the honest floor."""
    rng = np.random.default_rng(0)
    jit = lambda n: rng.gamma(JITTER_SHAPE, SIGMA / JITTER_SHAPE, n)
    f16 = floor_ms(p * 2, 2, CTX)
    f4 = floor_ms(p * 0.5, 2, CTX)

    honest, fast = f16 + jit(N_TOK), f4 + jit(N_TOK)
    pad = honest.mean() + rng.normal(0, 0.012, N_TOK)

    bins = np.linspace(0, 3.0, 120)
    top = 5.5 * max(np.histogram(honest, bins=bins)[0].max(),
                    np.histogram(fast, bins=bins)[0].max())

    ax.axvspan(0, f16, color=WARN, alpha=0.09, lw=0, zorder=0)
    ax.axvline(f16, color=WARN, lw=2.4, zorder=5)
    ax.annotate(f"$\\bf{{the\\ honest\\ floor,\\ {f16:.3f}\\ ms}}$\nmass to the left of "
                f"it is\nimpossible, not improbable",
                (f16, 0.415), xytext=(-8, 0), textcoords="offset points",
                xycoords=("data", "axes fraction"), ha="right", va="top", color=WARN,
                fontsize=8.7, linespacing=1.5, zorder=7)

    for x, col, lab, alpha in ((honest, HONEST, "honest bf16", 0.30),
                               (fast, DEV, "4-bit, speedup passed on", 0.30)):
        ax.hist(x, bins=bins, color=col, alpha=alpha, zorder=3)
        ax.hist(x, bins=bins, histtype="step", color=col, lw=1.8, zorder=3, label=lab)
    ax.hist(pad, bins=bins, color=WARN, alpha=0.95, zorder=3,
            label="4-bit, padded to that mean")

    for x, col in ((honest, HONEST), (fast, DEV)):
        ax.plot([x.min()], [top * 0.030], marker="v", ms=10, color=col, mec="white",
                mew=1.2, zorder=8)
    ax.annotate(f"$\\blacktriangledown$ = each arm's own minimum.\nThe honest arm's is "
                f"{honest.min():.3f} ms, just to\nthe right of its floor, as it must be.",
                (0.988, 0.585), xycoords="axes fraction", ha="right", va="top",
                color=INK2, fontsize=8.6, linespacing=1.5, zorder=8)
    ax.plot([fast.min()] * 2, [top * 0.055, top * 0.185], color=DEV, lw=0.9,
            ls=(0, (2, 2)), zorder=8)
    ax.annotate(f"$\\bf{{the\\ statistic}}$: min of {N_TOK:,} = {fast.min():.2f} ms,\n"
                f"{f16 / fast.min():.1f}$\\times$ under a floor it cannot cross",
                (fast.min(), top * 0.195), xytext=(-3, 0), textcoords="offset points",
                ha="left", va="bottom", color=DEV, fontsize=8.6, linespacing=1.45,
                zorder=8)

    d, n = clock_cost(f16, f4)
    frac = float((fast < f16).mean())
    ax.annotate(f"$\\bf{{1.}}$  Time the inter-token gaps.  You\nalready have them, so "
                f"the audit spends\n$\\bf{{zero}}$ FLOPs and asks the provider for\n"
                f"nothing.\n\n"
                f"$\\bf{{2.}}$  Take the $\\bf{{minimum}}$, not the mean.\nEvery "
                f"nuisance on the wire -- queueing,\nco-tenancy, a slow hop -- is "
                f"$\\it{{additive}}$\n$\\it{{and\\ positive}}$: it can only make a "
                f"provider\nlook $\\it{{slower}}$.  The mean absorbs all of it\nand the "
                f"minimum absorbs none.\n\n"
                f"$\\bf{{3.}}$  Compare it to the floor.  {frac:.0%} of\nthis provider's "
                f"tokens land left of a line\nit is not allowed to cross, and $d'$ = "
                f"gap/$\\sigma$\n= {d:.2f} per token -- so the repo's cost law\nprices "
                f"the verdict at $(\\delta^{{*}}/d')^{{2}}$ $\\approx$ "
                f"$\\bf{{{round(n)}}}$\n$\\bf{{tokens}}$ of stream.",
                (0.014, 0.980), xycoords="axes fraction", ha="left", va="top",
                color=INK, fontsize=8.8, linespacing=1.58, zorder=7)

    ax.annotate(f"$\\bf{{And\\ padding\\ is\\ not\\ an\\ escape.}}$  Sleeping\nuntil the "
                f"honest mean defeats a mean test\n$\\it{{exactly}}$ -- and leaves a "
                f"clock too $\\bf{{clean}}$ to pass a\nvariance test (sd "
                f"{pad.std():.3f} against {honest.std():.3f}).  To hide,\nthe provider "
                f"must hand back the speedup it\nstole $\\it{{and}}$ simulate the jitter "
                f"it deleted.",
                (0.560, 0.800), xycoords="axes fraction", ha="left", va="top",
                color=WARN, fontsize=8.9, linespacing=1.58, zorder=7)

    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, top)
    ax.set_xlabel("time per output token as the client sees it  (ms, modelled: floor "
                  f"+ gamma jitter, $\\sigma$ = {SIGMA:.2f} ms)", color=INK2,
                  fontsize=10)
    ax.set_ylabel(f"tokens  (of {N_TOK:,} drawn)", color=INK2, fontsize=10)
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_title("B.  HOW: time the gaps, take the minimum, compare to the floor",
                 color=INK, fontsize=11.5, weight="bold", loc="left", pad=8)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=8.6, labelspacing=0.55,
                    handlelength=1.6, borderaxespad=0.0, bbox_to_anchor=(0.988, 0.982))
    for t in leg.get_texts():
        t.set_color(INK2)


# =========================================================================== (C)
def panel_c(ax, R, measured):
    """Which deviations move bytes -- and the four that provably do not."""
    XMIN, XMAX = 1.0, 4.0e6
    ax.set_xscale("log")
    ax.set_xlim(XMIN, XMAX)

    # The token-channel price, for contrast. Best detector that reads only the
    # returned tokens; `activation_difr` is excluded because it needs the provider's
    # consent to expose activations, which is a third channel and not a passive one.
    TOK = {
        "quant_4bit": (measured["quant_4bit"]["tokens"], measured["quant_4bit"]["verifier"]),
        "kv_fp8": (measured["kv_fp8"]["tokens"], measured["kv_fp8"]["verifier"]),
        "seed_43": (measured["seed_43"]["tokens"], measured["seed_43"]["verifier"]),
        "temp_1.1": (measured["temp_1.1"]["tokens"], measured["temp_1.1"]["verifier"]),
        "bug_k32": (measured["bug_k32"]["tokens"], measured["bug_k32"]["verifier"]),
        "kv_long": (measured["kv_fp8"]["tokens"], "token_difr"),
        "substitution": (None, "AUC 0.998 at the proxy tier -- never priced on this pool"),
        "specdec": (np.inf, "$d'$ = 0 by exchangeability -- a $\\it{proof}$"),
    }
    LOUD = 4                                     # first LOUD rows are the loud ones
    GUT = 1.0                                    # gutter between the two groups
    ypos = [len(R) - 1 - i + (GUT if i < LOUD else 0) for i in range(len(R))]
    ax.set_ylim(-6.00, ypos[0] + 1.40)
    ax.axhspan(ypos[LOUD - 1] - 0.5, ypos[0] + 0.5, color=DEV, alpha=0.055, lw=0,
               zorder=0)
    ax.axhspan(-0.5, ypos[LOUD] + 0.5, color=HONEST, alpha=0.055, lw=0, zorder=0)

    for i, r in enumerate(R):
        y = ypos[i]
        n_tok, det = TOK[r["key"]]
        loud = i < LOUD
        col = DEV if loud else HONEST

        # the deviation's name rides above its own row: y-tick labels this wide would
        # spill into panel (B).
        ax.annotate(f"{r['mlabel']}   {r['note']}", (XMIN * 1.13, y), xytext=(0, 8),
                    textcoords="offset points", ha="left", va="bottom",
                    color=INK if loud else INK2, fontsize=9.2, zorder=6)

        # the clock's price
        if np.isfinite(r["n"]):
            ax.plot([XMIN, r["n"]], [y, y], color=col, lw=2.2, alpha=0.9, zorder=3)
            ax.plot([r["n"]], [y], marker="o", ms=10, color=col, mec="white", mew=1.4,
                    zorder=5)
            ax.annotate(f"$\\bf{{{round(r['n']):,}}}$", (r["n"], y), xytext=(8, 0),
                        textcoords="offset points", ha="left", va="center", color=col,
                        fontsize=9.0, zorder=6)
        else:
            ax.plot([XMIN * 1.10], [y], marker="o", ms=9, color="white", mec=col,
                    mew=2.0, zorder=5)
            ax.annotate("$d'$ = 0 $\\bf{exactly}$ -- 0 bytes moved",
                        (XMIN * 1.65, y), ha="left", va="center", color=col,
                        fontsize=8.9, zorder=6)

        # the same deviation's price in the returned-token channel, above the row
        if n_tok is None:
            ax.annotate(det, (XMAX / 1.06, y), xytext=(0, 1),
                        textcoords="offset points", ha="right", va="bottom",
                        color=MUTED, fontsize=8.4, zorder=6)
        elif np.isfinite(n_tok):
            ax.plot([n_tok], [y], marker="s", ms=7.5, color=MUTED, mec="white",
                    mew=1.0, zorder=4)
            note = "  (measured at 256-token context)" if r["key"] == "kv_long" else ""
            # Inline where there is room; below the row when the square would sit on
            # the clock's own segment; leftward when the label is long.
            on_line = np.isfinite(r["n"]) and n_tok < r["n"]
            place = dict(xytext=(0, -9), ha="center", va="top") if on_line else (
                dict(xytext=(-10, 0), ha="right", va="center") if note else
                dict(xytext=(10, 0), ha="left", va="center"))
            ax.annotate(f"{n_tok:,}  ($\\tt{{{det.replace('_', chr(92) + '_')}}}$)"
                        f"{note}", (n_tok, y), textcoords="offset points",
                        color=MUTED, fontsize=8.4, zorder=6, **place)
        else:
            ax.annotate(f"$\\infty$   {det}", (XMAX / 1.06, y), xytext=(0, 1),
                        textcoords="offset points", ha="right", va="bottom",
                        color=MUTED, fontsize=8.4, zorder=6)

    ax.annotate("$\\bf{THE\\ CLOCK\\ READS\\ THESE}$ -- the deviation stops moving "
                "bytes, so it stops spending time",
                (XMIN * 1.15, ypos[0] + 0.85), ha="left", va="bottom", color=DEV,
                fontsize=9.4, zorder=6)
    ax.annotate("$\\bf{THE\\ CLOCK\\ IS\\ ABSENT\\ HERE}$ -- same bytes as an honest "
                "provider.  Not a weakness to fix: a $\\it{scope}$",
                (XMIN * 1.15, ypos[LOUD] + 0.85), ha="left", va="bottom",
                color=HONEST, fontsize=9.4, zorder=6)

    ax.annotate("$\\bf{Read\\ the\\ two\\ groups,\\ not\\ the\\ two\\ columns.}$  The "
                "clock's scope is a $\\it{physical}$ predicate -- did you read\n"
                "these bytes -- so which group a deviation lands in is settled by "
                "arithmetic, before any data is\ncollected.  Everything that shrinks "
                "the read is in the first group; a wrong seed, a wrong temperature and "
                "a broken\ntop-$k$ shrink nothing, and are in the second.\n\n"
                "The second group is where the detectors this repo already ships are "
                "cheap: 226 tokens for a wrong seed.  The first\ncontains the two rows "
                "they are worst at -- $\\bf{4\\!-\\!bit\\ weights}$ ($d'$ = 0.075, "
                "inside every Tier-0 detector's noise) and\n$\\bf{lossless\\ "
                "speculation}$, whose price in the token channel is infinite by a "
                "$\\it{proof}$ rather than by a budget.\n\n"
                "$\\bf{Two\\ questions,\\ not\\ two\\ answers:}$  the tokens ask "
                "$\\it{did\\ you\\ compute\\ this\\ distribution}$, the clock asks "
                "$\\it{did\\ you\\ read}$\n$\\it{these\\ bytes}$.  Neither is the other's "
                "replacement, and a client can afford to ask both.",
                (0.008, 0.020), xycoords="axes fraction", ha="left", va="bottom",
                color=INK, fontsize=8.8, linespacing=1.58, zorder=7)

    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("tokens of stream per verdict, $(\\delta^{*}/d')^{2}$  (log).   "
                  "$\\bullet$ the clock, modelled   $\\blacksquare$ the best "
                  "returned-token detector, measured", color=INK2, fontsize=10)
    ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_title("C.  WHAT it catches: the deviations that save money, and only "
                 "those", color=INK, fontsize=11.5, weight="bold", loc="left", pad=8)


# ========================================================================== main
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cost, measured = load_measured()
    p, pp = cost["m_params"], cost["proxy_params"]
    R = rows(p, pp)

    fig = plt.figure(figsize=(23.0, 9.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.94, 1.02, 1.30], wspace=0.105,
                          left=0.040, right=0.995, top=0.800, bottom=0.130)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for a in axes:
        _clean(a)
    panel_a(axes[0], p, pp)
    panel_b(axes[1], p)
    panel_c(axes[2], R, measured)

    fig.suptitle("The clock: a decode step cannot return a token before it has read "
                 "the bytes -- so a provider that reads fewer bytes is early, and "
                 "early is visible",
                 color=INK, fontsize=15, weight="bold", x=0.006, ha="left", y=0.972)
    fig.text(0.006, 0.925,
             "$\\bf{The\\ whole\\ idea\\ in\\ one\\ line:}$ the deviations a provider "
             "makes money on are exactly the deviations that read fewer bytes, and "
             "at batch 1 a decode step's time is bytes / bandwidth.  So the stream's "
             "own arrival times carry evidence, for zero audit FLOPs and zero "
             "cooperation from the provider.\n"
             "$\\bf{What\\ is\\ measured\\ and\\ what\\ is\\ not:}$ the "
             "returned-token prices in (C) are $\\bf{measured}$ "
             "($\\tt{cost\\_of\\_a\\_verdict.json}$: Qwen3-1.7B audited with "
             "Qwen3-0.6B, 80$\\times$256 tokens per arm, standardized pAUC at FPR "
             "$\\leq$ 0.5%, $\\delta^{*}$ = "
             f"{cost['delta_star']:.3f}).  Every clock number is $\\bf{{arithmetic}}$ "
             "-- bytes read / 3.35 TB/s (H100 SXM5) over an assumed one-sided jitter "
             f"$\\sigma$ = {SIGMA:.2f} ms.\nNothing in this repo has ever timed a "
             "provider: $\\tt{perf\\_counter}$ appears only as the $\\it{verifier's}$ "
             "own cost.  Read this as the design of $\\tt{exp\\_clock\\_channel\\_gpu}$, "
             "not its result -- the load-bearing claim is (C)'s two groups, which "
             "survive the decade of $\\sigma$ swept in "
             "$\\tt{fig\\_clock\\_channel\\_principle}$ (C).",
             color=INK2, fontsize=9.0, ha="left", va="top", linespacing=1.6)

    fig.text(0.006, 0.016,
             "$\\bf{The\\ one\\ hole,\\ stated\\ here\\ rather\\ than\\ found\\ later.}$"
             "  Continuous batching reads one copy of the weights for $B$ concurrent "
             "requests, so a $\\it{weight}$ attack's floor drops as 1/$B$ and the "
             "impossibility in (A) softens into a statistic past $B$ $\\approx$ 20.  "
             "It cannot amortize $\\it{your}$ KV cache, which is read\nonce per token "
             "per request at any $B$ -- so the long-context row in (C) is the one that "
             "survives batching, and it is also the row the token channel reads worst "
             "($d'$ = 0.02).  Where one channel's floor moves, the other's does not, "
             "which is the argument of  fig_clock_vs_difr.png.",
             color=INK, fontsize=9.0, ha="left", va="bottom", linespacing=1.6)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_clock_simple.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"wrote {out}")
    for r in R:
        n = "inf" if not np.isfinite(r["n"]) else f"{round(r['n']):,}"
        print(f"  {r['key']:14s} floor {r['honest']:.3f} -> {r['served']:.3f} ms  "
              f"bytes {r['bytes_x']:.2f}x  d'={r['d']:.3f}  clock {n:>10s} tok")


if __name__ == "__main__":
    main()
