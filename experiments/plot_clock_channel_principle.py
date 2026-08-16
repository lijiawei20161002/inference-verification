"""Every verifier in this repo reads one channel. The clock is a second one.

A PROPOSAL figure, and it is labelled as one. Panel (A) is measured -- it is a
re-reading of `cost_of_a_verdict.json`, this repo's own committed grid, against an
arithmetic the repo has never computed: how many bytes each deviation stops moving.
Panels (B) and (C) are a MODEL. No experiment in this repo has ever timed a
provider; `time.perf_counter` appears only as the *verifier's* cost
(`hf_gpu.py`, `exp_spec_substitution_gpu.py`). So (B) and (C) are the prediction
`exp_clock_channel_gpu` would test, drawn with every assumption on its face.

  docs/figures/fig_clock_channel_principle.png
      (A) The detectors are cheapest exactly where there is no money. Measured
          token-channel price (tokens per verdict, best detector that reads only
          the returned tokens) against the decode bytes per token the deviation
          saves. Wrong seed is the cheapest verdict in the grid at 226 tokens and
          saves the provider nothing; 4-bit weights save 3.9x and cost 2 013;
          lossless speculation saves ~2.2x at a price of infinity, by a proof.
          The one y-value that is neither measured nor arithmetic is that 2.2x:
          it is a representative spec-dec speedup, and it is labelled as one.
      (B) Why the clock is not just another statistic: all of its nuisance is
          ADDITIVE and POSITIVE. Queueing, co-tenancy and network jitter can only
          make a provider look slower, so the mean is contaminated and the
          MINIMUM is not. Padding to the honest mean defeats a mean test and
          leaves the padded clock too clean to pass a variance test.
      (C) Concealment has a price, and it is quotable. Sweeping the fraction of
          the speedup the provider passes on to the client, against the same cost
          law and the same delta* = 3.767 the rest of the repo prices verdicts
          with, so the two channels land on one axis.

MEASURED INPUTS   docs/results/cost_of_a_verdict.json  (Qwen3-1.7B audited with
    Qwen3-0.6B, 80 sequences x 256 tokens per arm, 5 deviations x 7 verifiers,
    standardized pAUC at FPR <= 0.5%, delta* = 3.767)

MODELLED INPUTS   all in ROOFLINE below, all one line of arithmetic each. Batch-1
    decode is memory-bandwidth-bound, so time per output token >= bytes read /
    HBM bandwidth. Bandwidth is the H100 SXM5 spec sheet; weight bytes are the
    parameter count in the artifact above times the dtype width; KV bytes are
    Qwen3-1.7B's published attention shape. The jitter scale in (B)/(C) is an
    ASSUMPTION, swept over a decade in (C) because it is the load-bearing one.

    .venv/bin/python -m experiments.plot_clock_channel_principle
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
COST = ROOT / "docs" / "results" / "cost_of_a_verdict.json"

# Same palette and same meaning as plot_batch_principle / plot_int8_wall: blue is
# the channel being proposed, ink is the null / the incumbent. Every mark is also
# direct-labelled, so identity never rests on colour alone.
DEV, HONEST = "#2a78d6", "#52514e"
WARN = "#c2571a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8880", "#e3e2dd"

# --------------------------------------------------------------------- modelled
# Every number here is arithmetic from a spec sheet or a published model shape.
# None of it is a measurement, and nothing downstream pretends otherwise.
ROOFLINE = dict(
    hbm_bytes_per_s=3.35e12,    # H100 SXM5 HBM3, vendor spec sheet
    bf16_bytes=2,
    kv_layers=28,               # Qwen3-1.7B: 28 layers, 8 KV heads, head_dim 128
    kv_heads=8,
    head_dim=128,
    ctx=256,                    # the audited pool's own sequence length
    ctx_long=32768,             # where an fp8 KV cache actually pays off
)
# The jitter an honest stream picks up between the GPU and the client. Positive by
# construction -- queueing, co-tenancy and network delay add time, never remove it.
JITTER_MS = 0.50                # (B)'s scale; (C) sweeps 0.25 / 0.50 / 1.00
JITTER_SHAPE = 1.6              # gamma shape: right-skewed, the usual queueing tail
N_TOK = 3000                    # tokens of stream drawn in (B)
PHI_XOVER_TARGET = "quant_4bit"


def roofline_ms(param_bytes: float) -> float:
    """Lower bound on ms per output token for a batch-1 memory-bound decode."""
    return 1e3 * param_bytes / ROOFLINE["hbm_bytes_per_s"]


def kv_bytes_per_token(width: int) -> float:
    r = ROOFLINE
    return 2 * r["kv_layers"] * r["kv_heads"] * r["head_dim"] * width


# ------------------------------------------------------------------------ data
def load_measured():
    """The token-channel price of each deviation, from the repo's own grid.

    'Token channel' means what it says: the detectors that see only the returned
    tokens. `activation_difr` is excluded on purpose -- it reads projected
    activations, which is a THIRD channel and one the provider must agree to
    expose. Including it would compare a privileged-access detector against a
    passive one.
    """
    d = json.load(open(COST))
    token_channel = [v for v in d["verifiers"] if v != "activation_difr"]
    out = {}
    for atk, row in d["cells"].items():
        best, best_d = None, 0.0
        for v in token_channel:
            c = row[v]
            if c["reachable"] and c["d_prime"] > best_d:
                best, best_d = v, c["d_prime"]
        out[atk] = dict(verifier=best, d_prime=best_d,
                        tokens=row[best]["tokens_per_verdict"])
    return d, out


def bytes_saved():
    """Decode bytes per token an honest provider moves, divided by what each
    deviation moves. This is the clock signal, because a batch-1 decode is
    bandwidth-bound: fewer bytes is strictly less time."""
    r = ROOFLINE
    p = json.load(open(COST))["m_params"]
    w16 = p * r["bf16_bytes"]
    kv16 = kv_bytes_per_token(2) * r["ctx"]
    honest = w16 + kv16
    return {
        # 4-bit weights: the whole point of the deviation is to read 1/4 the bytes.
        "quant_4bit": honest / (p * 0.5 + kv16),
        # fp8 KV: at this pool's 256-token context the cache is 0.8% of the read,
        # so halving it saves ~nothing. It pays off at long context, not here.
        "kv_fp8": honest / (w16 + kv_bytes_per_token(1) * r["ctx"]),
        # These three change the sampler or the seed. They move exactly the same
        # bytes as an honest provider: there is no money in them.
        "temp_1.1": 1.0,
        "seed_43": 1.0,
        "bug_k32": 1.0,
    }


def kv_fp8_long_ctx():
    r = ROOFLINE
    p = json.load(open(COST))["m_params"]
    w16 = p * r["bf16_bytes"]
    long16 = kv_bytes_per_token(2) * r["ctx_long"]
    return (w16 + long16) / (w16 + kv_bytes_per_token(1) * r["ctx_long"])


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.set_axisbelow(True)


# =========================================================================== (A)
def panel_a(ax, measured, saved):
    ax.axhspan(1.55, 16, color=WARN, alpha=0.055, zorder=0)
    ax.annotate("deviations that save the provider real bytes -- the ones with money "
                "in them",
                (1.15e2, 13.2), color=WARN, fontsize=8.8, weight="bold",
                ha="left", va="center")
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (1, 2.5)), zorder=1)
    ax.annotate("saves nothing: no provider makes money here",
                (1.15e2, 1.045), color=INK2, fontsize=8.8, ha="left", va="bottom")

    # (name, text anchor in data coords, ha). The three zero-saving deviations sit
    # at exactly y = 1.00, so their labels go in a left-aligned column BELOW the
    # line with leader lines rather than beside the markers, where they would
    # overprint each other.
    LBL = {"quant_4bit": ("4-bit weights", (3.0e3, 4.2), "left"),
           "kv_fp8": ("fp8 KV cache", (2.5e4, 1.32), "right"),
           "seed_43": ("wrong seed", (1.15e2, 0.83), "left"),
           "bug_k32": ("top-$k$ bug", (1.15e2, 0.585), "left"),
           "temp_1.1": ("wrong temperature", (1.15e2, 0.415), "left")}
    for atk, m in measured.items():
        y = saved[atk]
        lucrative = y > 1.05
        col = DEV if lucrative else HONEST
        ax.plot([m["tokens"]], [y], marker="o", ms=11 if lucrative else 8,
                color=col, mec="white", mew=1.6, zorder=5)
        name, at, ha = LBL[atk]
        body = (f"{y:.2f}$\\times$ the bytes  |  {m['tokens']:,} tok, "
                f"$d'$ = {m['d_prime']:.3f}  ({m['verifier']})")
        lead = dict(arrowprops=dict(arrowstyle="-", color=col, lw=0.9, shrinkA=4,
                                    shrinkB=7)) if not lucrative else {}
        ax.annotate(name, xy=(m["tokens"], y), xytext=at, textcoords="data",
                    ha=ha, va="bottom", color=col, fontsize=8.4, weight="bold",
                    zorder=6, **lead)
        ax.annotate(body, xy=at, xytext=(0, -11), textcoords="offset points",
                    ha=ha, va="bottom", color=col, fontsize=8.4, zorder=6)

    # Lossless speculation: d' = 0 in this channel by exchangeability. That is a
    # proof in the same sense as this repo's wrong-seed / cross-entropy note, so
    # the price is infinite rather than large, and it goes on the right edge.
    ax.annotate("", xy=(9.4e5, 2.2), xytext=(2.4e5, 2.2),
                arrowprops=dict(arrowstyle="-|>", color=DEV, lw=2.0))
    ax.plot([2.4e5], [2.2], marker="o", ms=11, color="white", mec=DEV, mew=2.2,
            zorder=5)
    ax.annotate("lossless speculation", xy=(2.4e5, 2.2), xytext=(1.05e6, 7.4),
                textcoords="data", ha="right", va="bottom", color=DEV,
                fontsize=8.4, weight="bold",
                arrowprops=dict(arrowstyle="-", color=DEV, lw=0.9, shrinkA=4,
                                shrinkB=9))
    ax.annotate("$\\approx$2.2$\\times$ fewer target passes (typical)  |  price $\\infty$\n"
                "$d'$ = 0 by exchangeability -- a $\\it{proof}$, not a\n"
                "measurement, so no pool ever resolves it",
                xy=(1.05e6, 7.4), xytext=(0, -11), textcoords="offset points",
                ha="right", va="top", color=DEV, fontsize=8.4, linespacing=1.5)

    ax.annotate("$\\bf{The\\ one\\ exception:}$ substitution (4B served\n"
                "as 0.6B) is 6.7$\\times$ the bytes $\\it{and}$ caught at AUC\n"
                "0.998 by the proxy tier.  Its $d'$ was never\nmeasured on this pool, "
                "so it is not plotted --\nthe only cell where incentive and\n"
                "detectability agree.",
                (1.15e2, 10.6), xycoords="data", ha="left", va="top",
                color=INK, fontsize=8.5, linespacing=1.6)

    ax.set_xscale("log")
    ax.set_xlim(1.0e2, 1.15e6)
    ax.set_ylim(0.30, 16)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 4, 8], ["", "1$\\times$\n(honest)", "2$\\times$",
                                      "4$\\times$", "8$\\times$"])
    ax.set_xlabel("price in the returned-token channel  (tokens per verdict, best "
                  "detector that reads\nonly the tokens -- measured)",
                  color=INK2, fontsize=10, linespacing=1.5)
    ax.set_ylabel("decode bytes per token the deviation saves\n(arithmetic, batch 1, "
                  "256-token context)", color=INK2, fontsize=10, linespacing=1.5)
    ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_title("A.  The detectors are cheapest where there is no money",
                 color=INK, fontsize=11, weight="bold", loc="left", pad=8)


# =========================================================================== (B)
def panel_b(ax, floor16, floor4):
    rng = np.random.default_rng(0)
    jit = lambda n: rng.gamma(JITTER_SHAPE, JITTER_MS / JITTER_SHAPE, n)

    honest = floor16 + jit(N_TOK)
    fast = floor4 + jit(N_TOK)                       # speedup passed to the client
    pad = honest.mean() + rng.normal(0, 0.012, N_TOK)  # padded to the honest mean

    bins = np.linspace(0, 3.2, 128)
    # Headroom, not a scale trick: the padded arm is a near-delta and would be 9x
    # the honest peak. Everything above the histograms is annotation space.
    top = 3.6 * max(np.histogram(honest, bins=bins)[0].max(),
                    np.histogram(fast, bins=bins)[0].max())

    ax.axvspan(0, floor16, color=WARN, alpha=0.10, zorder=0)
    ax.axvline(floor16, color=WARN, lw=2.4, zorder=4)
    ax.annotate(f"$\\bf{{{floor16:.2f}\\ ms}}$: bf16 $M$ cannot beat its own\n"
                f"roofline at batch 1.  Mass left of this line\nis a physical "
                f"impossibility, not a p-value --\nno calibration set, no pool, no "
                f"ratio.",
                (floor16, 0.985), xytext=(-9, 0), textcoords="offset points",
                xycoords=("data", "axes fraction"), ha="right", va="top",
                color=WARN, fontsize=8.6, weight="bold", linespacing=1.5)
    ax.axvline(floor4, color=DEV, lw=1.4, ls=(0, (4, 2)), zorder=4)
    ax.annotate(f"4-bit\nfloor\n{floor4:.2f} ms", (floor4, 0.105),
                xytext=(-5, 0), textcoords="offset points",
                xycoords=("data", "axes fraction"), ha="right", va="center",
                color=DEV, fontsize=8.6, linespacing=1.4)

    for x, col, lab, alpha in ((honest, HONEST, "honest bf16", 0.30),
                               (fast, DEV, "4-bit, speedup passed on", 0.30)):
        ax.hist(x, bins=bins, color=col, alpha=alpha, zorder=3)
        ax.hist(x, bins=bins, histtype="step", color=col, lw=1.8, zorder=3,
                label=lab)
    ax.hist(pad, bins=bins, color=WARN, alpha=0.95, zorder=3,
            label="4-bit, padded to the honest mean")
    ax.annotate(f"clipped: {int(np.histogram(pad, bins=bins)[0].max()):,}\n"
                f"tokens in one bin",
                (pad.mean() + 0.06, 0.315), xytext=(0, 0),
                textcoords="offset points", xycoords=("data", "axes fraction"),
                ha="left", va="top", color=WARN, fontsize=8.3, weight="bold",
                linespacing=1.4)

    frac = float((fast < floor16).mean())
    ax.annotate(f"{frac:.0%} of these tokens land left of the\nfloor.  "
                f"One of them ends it.",
                (0.014, 0.405), xytext=(0, 0),
                textcoords="offset points", xycoords="axes fraction",
                ha="left", va="top", color=DEV, fontsize=8.6, weight="bold",
                linespacing=1.45)

    tab = (f"                      mean     sd     min of {N_TOK:,}\n"
           f"honest bf16          {honest.mean():5.2f}  {honest.std():5.3f}  "
           f"{honest.min():5.2f}\n"
           f"4-bit, passed on     {fast.mean():5.2f}  {fast.std():5.3f}  "
           f"{fast.min():5.2f}\n"
           f"4-bit, padded        {pad.mean():5.2f}  {pad.std():5.3f}  "
           f"{pad.min():5.2f}")
    ax.annotate(tab, (0.986, 0.985), xycoords="axes fraction", ha="right", va="top",
                color=INK, fontsize=8.0, family="monospace", linespacing=1.6)

    ax.annotate("All of the clock's nuisance is $\\bf{additive}$ and "
                "$\\bf{positive}$: queueing,\nco-tenancy and network delay make a "
                "provider look slower, never\nfaster.  So the $\\it{mean}$ is "
                "contaminated by every nuisance and the\n$\\it{minimum}$ is immune "
                "to all of them --\n$\\bf{when\\ the\\ noise\\ is\\ one\\ sided,"
                "\\ an\\ order\\ statistic\\ beats\\ a\\ mean.}$\n\n"
                "Padding defeats the mean test $\\it{exactly}$, and leaves a clock "
                "too\n$\\bf{clean}$ (sd 0.012 against 0.406) to pass a variance "
                "test.  To hide,\nthe provider must simulate the noise process it "
                "just deleted.",
                (0.014, 0.455), xycoords="axes fraction", ha="left", va="bottom",
                color=INK, fontsize=8.6, linespacing=1.6)

    ax.set_ylim(0, top)
    ax.set_xlim(0, 3.2)
    ax.set_xlabel("time per output token as the client sees it  (ms, modelled)",
                  color=INK2, fontsize=10)
    ax.set_ylabel(f"tokens  (of {N_TOK:,} drawn)", color=INK2, fontsize=10)
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_title("B.  The clock's nuisance is one-sided, so the mean is the wrong "
                 "statistic", color=INK, fontsize=11, weight="bold", loc="left",
                 pad=8)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=8.5,
                    labelspacing=0.6, handlelength=1.6, borderaxespad=0.0,
                    bbox_to_anchor=(0.986, 0.435))
    for t in leg.get_texts():
        t.set_color(INK2)


# =========================================================================== (C)
def panel_c(ax, floor16, floor4, delta_star, target_tokens, target_d):
    phi = np.linspace(0.004, 1.0, 500)
    gap = floor16 - floor4                    # ms of speedup available to steal
    served = floor4 + (1 - phi) * gap         # phi = fraction passed to the client

    ax.axhspan(0.28, target_tokens, color=DEV, alpha=0.06, zorder=0)
    ax.axhline(target_tokens, color=HONEST, lw=2.0, zorder=4)
    ax.annotate(f"$\\bf{{{target_tokens:,}\\ tokens}}$ -- what 4-bit weights actually "
                f"cost in the token\nchannel ({target_d}, $d'$ = "
                f"{delta_star / np.sqrt(target_tokens):.4f}).  Below this line the "
                f"clock is cheaper than\nevery detector in the repo -- and it spends "
                f"$\\bf{{zero}}$ audit FLOPs, because\nthe evidence arrives with the "
                f"stream you already paid for.",
                (0.985, target_tokens), xytext=(0, -10),
                textcoords="offset points", xycoords=("axes fraction", "data"),
                ha="right", va="top", color=INK, fontsize=8.6, linespacing=1.6)

    styles = ((0.25, (0, (1, 1.6)), "0.25"), (0.50, "-", "0.50"),
              (1.00, (0, (5, 2)), "1.00"))
    for sigma, ls, lab in styles:
        d_clock = phi * gap / sigma
        tokens = (delta_star / d_clock) ** 2
        ax.plot(phi, tokens, color=DEV, lw=2.4 if ls == "-" else 1.7, ls=ls,
                zorder=5, label=f"jitter sd = {lab} ms")
        # Where the clock stops being the cheaper channel.
        phi_x = delta_star / np.sqrt(target_tokens) * sigma / gap
        if phi_x < 1:
            ax.plot([phi_x], [target_tokens], marker="o", ms=7, color=DEV,
                    mec="white", mew=1.4, zorder=6)

    lo = delta_star / np.sqrt(target_tokens) * 0.25 / gap
    hi = delta_star / np.sqrt(target_tokens) * 1.00 / gap
    ax.annotate(f"$\\bf{{The\\ concealment\\ price.}}$  To make the clock cost more "
                f"than the tokens\ndo, the provider must hand back "
                f"$\\bf{{{100 * (1 - hi):.0f}}}$% to $\\bf{{{100 * (1 - lo):.0f}}}$% "
                f"of the speedup it stole.\nThe whole decade of jitter agrees, which "
                f"is why the assumption\nis swept rather than chosen.",
                (0.988, 1.2e6), xytext=(0, 0), textcoords="offset points",
                ha="right", va="top", color=INK, fontsize=8.7, linespacing=1.6)

    ax.annotate("$\\phi$ = 0: fully padded.  Invisible in the\n"
                "mean, but it forfeits the latency market\nit was cheating to win, "
                "keeps only the\nthroughput share, and must fake the\njitter it "
                "deleted (B).",
                (0.012, 0.335), xytext=(0, 0), textcoords="offset points",
                ha="left", va="bottom", color=WARN, fontsize=8.5, linespacing=1.55)
    ax.annotate("$\\phi$ = 1: the whole\n3.9$\\times$ sold to the\n"
                "client.  A handful of\ntokens convicts.",
                (0.988, 0.295), xytext=(0, 0), textcoords="offset points",
                ha="right", va="bottom", color=DEV, fontsize=8.5, linespacing=1.55)

    ax.set_yscale("log")
    ax.set_ylim(0.28, 2e6)
    ax.set_yticks([1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0\n(fully padded)", "0.25", "0.50",
                                              "0.75", "1.0\n(fully sold)"])
    ax.set_xlabel("$\\phi$ -- fraction of the 3.9$\\times$ speedup the provider "
                  "passes on to the client", color=INK2, fontsize=10,
                  linespacing=1.5)
    ax.set_ylabel("tokens of stream needed for a clock verdict\n"
                  "$(\\delta^{*}/d'_{clock})^{2}$, the repo's own cost law",
                  color=INK2, fontsize=10, linespacing=1.5)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_title("C.  Hiding the clock means giving back the thing you stole",
                 color=INK, fontsize=11, weight="bold", loc="left", pad=8)
    leg = ax.legend(loc="lower left", frameon=False, fontsize=8.5,
                    labelspacing=0.6, handlelength=2.4, borderaxespad=0.0,
                    bbox_to_anchor=(0.425, 0.022))
    for t in leg.get_texts():
        t.set_color(INK2)


# ========================================================================== main
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cost, measured = load_measured()
    saved = bytes_saved()
    delta_star = cost["delta_star"]
    p = cost["m_params"]

    floor16 = roofline_ms(p * 2 + kv_bytes_per_token(2) * ROOFLINE["ctx"])
    floor4 = roofline_ms(p * 0.5 + kv_bytes_per_token(2) * ROOFLINE["ctx"])

    tgt = measured[PHI_XOVER_TARGET]

    fig = plt.figure(figsize=(19.6, 9.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.06, 1.0, 0.96], wspace=0.205,
                          left=0.055, right=0.985, top=0.815, bottom=0.165)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for a in axes:
        _clean(a)
    panel_a(axes[0], measured, saved)
    panel_b(axes[1], floor16, floor4)
    panel_c(axes[2], floor16, floor4, delta_star, tgt["tokens"], tgt["verifier"])

    fig.suptitle("Every detector in this repo reads one channel -- the returned "
                 "tokens.  The clock is a second channel, and it is one-sided",
                 color=INK, fontsize=13.5, weight="bold", x=0.008, ha="left",
                 y=0.982)
    fig.text(0.008, 0.950,
             "(A) is $\\bf{measured}$: a re-reading of this repo's own grid "
             "($\\tt{cost\\_of\\_a\\_verdict.json}$, Qwen3-1.7B, 80$\\times$256 tokens "
             "per arm, standardized pAUC at FPR $\\leq$ 0.5%, $\\delta^{*}$ = "
             f"{delta_star:.3f}) against an arithmetic it never computed.  "
             "(B) and (C) are a $\\bf{model}$: nothing in this\nrepo has ever timed a "
             "provider -- $\\tt{perf\\_counter}$ appears only as the "
             "$\\it{verifier's}$ cost -- so they are the prediction "
             "$\\tt{exp\\_clock\\_channel\\_gpu}$ would test, not a result.  The "
             "roofline is bytes read / 3.35 TB/s (H100 SXM5 spec) at batch 1; the "
             "jitter scale is an assumption, swept over a decade in (C).",
             color=INK2, fontsize=8.5, ha="left", va="top", linespacing=1.55)

    fig.text(0.008, 0.022,
             "$\\bf{The\\ two\\ holes,\\ stated\\ here\\ rather\\ than\\ found\\ later.}$"
             "  $\\bf{(i)}$ The impossibility line in (B) is exact only under a "
             "batch-1 contract.  Continuous batching amortizes one weight read "
             "across concurrent requests, so a provider that\nhonestly batches your "
             "request with 31 others is also left of the line -- and then the clock "
             "is statistical again, against a co-tenancy null nobody here has "
             "measured.  $\\bf{(ii)}$ The floor is per-SKU and the provider picks the "
             "SKU.  Both holes have the same\nfix, and it is already half-measured: "
             "batch shape is recoverable $\\it{from\\ the\\ tokens}$ -- a batch of 4 "
             "leaves 9.5% of logit values bitwise identical and flips 0.73% of "
             "argmaxes ($\\tt{specdec\\_shape.json}$).  $\\bf{The\\ tokens\\ tell\\ you"
             "\\ the\\ shape;\\ the\\ clock\\ tells\\ you\\ the\\ bytes;\\ neither\\ is\\ "
             "interpretable\\ alone.}$",
             color=INK, fontsize=8.6, ha="left", va="bottom", linespacing=1.6)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_clock_channel_principle.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"wrote {out}")
    print(f"  bf16 floor {floor16:.3f} ms/token, 4-bit floor {floor4:.3f} ms/token")
    print(f"  fp8 KV byte saving: {saved['kv_fp8']:.4f}x at ctx 256, "
          f"{kv_fp8_long_ctx():.2f}x at ctx {ROOFLINE['ctx_long']:,}")
    for atk, m in measured.items():
        print(f"  {atk:12s} {m['tokens']:>9,} tok  d'={m['d_prime']:.4f}  "
              f"{m['verifier']:14s} bytes={saved[atk]:.3f}x")


if __name__ == "__main__":
    main()
