"""The clock is not a better DiFR. It is a second axis, and the blind spots barely overlap.

Companion to `fig_clock_channel_principle.png`, which argued that the clock is a
channel at all. This one argues the only thing a client should care about next:
whether that channel is ORTHOGONAL to the detector this repo already ships. DiFR
asks *did you compute this distribution*; the clock asks *did you read these
bytes*. Those are two different questions about one forward pass, and the repo's
own grid answers them loudly in different rows.

  docs/figures/fig_clock_vs_difr.png
      (A) Both channels in one unit -- d' per token. DiFR's axis is MEASURED
          (`cost_of_a_verdict.json`, token_difr, against its own honest-control
          CI); the clock's is ARITHMETIC off a roofline. The five deviations land
          on an anti-diagonal: the corner where both channels are loud is empty,
          and so is the corner where a client would get the same verdict twice.
      (B) What orthogonality is worth in the repo's own cost law. Independent
          channels add in d'^2, so verdict costs add in RECIPROCAL:
          1/n_joint = 1/n_DiFR + 1/n_clock. One row collapses 2 494 -> 6 tokens.
          Three rows do not move at all, and the panel says so in words.
      (C) Why that is structural rather than lucky. DiFR is blind near zero
          distributional distance; the clock is blind near honest latency. The two
          bands intersect in exactly one place -- read few bytes, bill honest time
          -- and a provider standing there has already refunded 89-97% of what it
          stole. No deviation is cheap in both currencies at once.
      (D) Where the clock's edge ends, and the one place it never does. Continuous
          batching amortizes WEIGHT reads across concurrent requests, so the clock
          on a weight attack decays as 1/B and crosses DiFR's flat floor at
          B ~ 20. It cannot amortize the KV CACHE, which is read once per token
          per request at any B -- so on a long-context KV attack the clock is flat
          in B, and that is exactly the cell DiFR reads at d' = 0.02.

MEASURED INPUTS   docs/results/cost_of_a_verdict.json  (Qwen3-1.7B audited with
    Qwen3-0.6B, 80 seq x 256 tok per arm, standardized pAUC at FPR <= 0.5%,
    delta* = 3.767) -- every DiFR d' on every panel, and the honest-control CI
    that sets DiFR's own noise floor in (A).
    docs/results/reversal_check.json -- the one long-context DiFR measurement for
    an fp8 KV cache, on a DIFFERENT model, labelled as such on the panel.
    docs/results/specdec_shape.json -- how much of the batch size the tokens give
    back, which is what (D) needs and does not have.

MODELLED INPUTS   the roofline (bytes read / 3.35 TB/s, H100 SXM5 spec) and the
    jitter scale sigma. Nothing in this repo has ever timed a provider, so every
    clock d' here is a prediction. The ROOFLINE constants and the palette are
    imported from plot_clock_channel_principle rather than copied.

    .venv/bin/python -m experiments.plot_clock_vs_difr
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.plot_clock_channel_principle import (
    COST, DEV, FIG_DIR, GRID, HONEST, INK, INK2, JITTER_MS, MUTED, ROOFLINE, WARN,
    _clean, kv_bytes_per_token, roofline_ms,
)

SHAPE = Path(COST).parent / "specdec_shape.json"
REVERSAL = Path(COST).parent / "reversal_check.json"

# The detector the clock is compared against. `token_difr` is DiFR's own default
# metric (clipped post-Gumbel margin) and the strongest verifier in the grid that
# needs nothing from the provider but the tokens and a shared seed.
DIFR = "token_difr"
# activation_difr is deliberately NOT the comparison. It wins outright where it is
# available (d' = 9.945 on quant_4bit, one token) and it is the only detector that
# needs the provider to expose projected activations. It appears in (B) as a
# hatched bar so the reader can see both the win and the price of it.
PRIVILEGED = "activation_difr"

SIGMA = JITTER_MS               # ms of one-sided jitter; swept in fig_clock_channel (C)
CTX_LONG = ROOFLINE["ctx_long"]
BATCHES = np.logspace(0, 6, 400, base=2.0)      # 1 .. 64 concurrent requests
SIGMA_SWEEP = (0.25, 0.50, 1.00)

# key -> label, weight-byte scale, KV width in bytes. Weight attacks scale the
# amortizable term; KV attacks scale the per-request term; the three sampler
# attacks touch neither, exactly and by arithmetic.
DEVIATIONS = (
    ("quant_4bit", "4-bit weights",     0.25, 2),
    ("kv_fp8",     "fp8 KV cache",      1.00, 1),
    ("bug_k32",    "top-$k$ bug",       1.00, 2),
    ("seed_43",    "wrong seed",        1.00, 2),
    ("temp_1.1",   "wrong temperature", 1.00, 2),
)
ZERO_Y, ZERO_TOP = 1.3e-3, 3.0e-3       # where a d' of exactly 0 is drawn in (A)


# ------------------------------------------------------------------------ model
def bytes_read(params, w_scale, kv_width, ctx, batch=1.0):
    """Bytes a decode step reads per output token, per request, at concurrency B.

    Continuous batching reads one copy of the weights for the whole batch, so the
    weight term amortizes as 1/B. The KV cache is this request's own history and
    does not amortize at any batch size. That asymmetry is the whole of panel (D).
    """
    return params * 2 * w_scale / batch + kv_bytes_per_token(kv_width) * ctx


def clock_d_prime(params, w_scale, kv_width, ctx, batch=1.0, sigma=SIGMA):
    """Per-token d' the clock gets when the provider passes the speedup on (phi=1).

    The deviation shifts a positive-jitter distribution, so d' = gap / sigma. Same
    cost law the token channel is priced with -- the only reason the two channels
    can share an axis."""
    honest = bytes_read(params, 1.0, 2, ctx, batch)
    fast = bytes_read(params, w_scale, kv_width, ctx, batch)
    return roofline_ms(honest - fast) / sigma


def tokens_for(d_prime, delta_star):
    return np.inf if d_prime <= 0 else (delta_star / d_prime) ** 2


def crossover_batch(params, difr_d, sigma):
    """B at which a 4-bit weight attack's clock d' erodes to DiFR's flat d'.

    gap(B) = 0.75 * W / B / bandwidth, so B = 0.75 * W / (d' * sigma * bandwidth).
    """
    return 0.75 * params * 2 / (difr_d * sigma * 1e-3 * ROOFLINE["hbm_bytes_per_s"])


def load():
    cost = json.load(open(COST))
    p, ds, ctx = cost["m_params"], cost["delta_star"], ROOFLINE["ctx"]
    rows = []
    for key, label, w, kvw in DEVIATIONS:
        cell = cost["cells"][key]
        d_difr, d_clock = cell[DIFR]["d_prime"], clock_d_prime(p, w, kvw, ctx)
        rows.append(dict(
            key=key, label=label, w=w, kvw=kvw,
            d_difr=d_difr, d_clock=d_clock,
            n_difr=tokens_for(d_difr, ds), n_clock=tokens_for(d_clock, ds),
            n_joint=tokens_for(np.hypot(max(d_difr, 0.0), d_clock), ds),
            d_priv=cell[PRIVILEGED]["d_prime"],
            n_priv=tokens_for(cell[PRIVILEGED]["d_prime"], ds),
        ))
    null_hi = cost["honest_control"][DIFR]["d_prime_ci"][1]
    return cost, p, ds, rows, null_hi


def long_ctx_difr():
    """The only long-context DiFR measurement in the repo for an fp8 KV cache."""
    for c in json.load(open(REVERSAL))["cells"]:
        if c["attack"] == "kv_fp8" and "SmolLM2" in c["model"]:
            return c["d_prime"], c["n_tok"], c["model"].split("/")[-1]
    raise LookupError("kv_fp8 / SmolLM2 cell missing from reversal_check.json")


def batch_recoverability():
    """How much of the batch size the tokens hand back. Panel (D) needs all of it."""
    rows = [r for r in json.load(open(SHAPE)) if r["tag"] == "batch_composition"]
    return sorted(rows, key=lambda r: r["batch_size"])


def gain_note(r):
    if not np.isfinite(r["n_difr"]) and not np.isfinite(r["n_clock"]):
        return "$\\bf{no\\ price\\ in\\ either}$\n$\\bf{channel}$"
    g = r["n_difr"] / r["n_joint"]
    if g < 1.01:
        return "$\\bf{unchanged}$ -- the clock\nadds nothing to this row"
    return (f"$\\bf{{{g:,.0f}}}$$\\bf{{\\times}}$ $\\bf{{cheaper}}$" if g >= 10 else
            f"$\\bf{{{g:.2f}}}$$\\bf{{\\times}}$ $\\bf{{cheaper}}$")


# =========================================================================== (A)
def panel_a(ax, p, rows, ds, null_hi):
    xlim, ylim = (3.5e-3, 12.0), (6.5e-4, 7.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Both blind bands come from numbers in the artifacts, not from the eye.
    ax.axhspan(ylim[0], ZERO_TOP, color=HONEST, alpha=0.09, zorder=0)
    ax.axvspan(xlim[0], null_hi, color=MUTED, alpha=0.13, zorder=0)
    ax.annotate(f"$\\bf{{DiFR's\\ own\\ noise\\ floor}}$\nits honest-control $d'$ CI "
                f"tops out\nat {null_hi:.3f} -- measured, same file.\nA deviation "
                f"left of this line is not\nsomething DiFR is bad at.  It is\ninside "
                f"DiFR's own null.",
                (xlim[0] * 1.10, 0.62), ha="left", va="top", color=INK2,
                fontsize=8.3, linespacing=1.55, zorder=7)
    ax.annotate("$\\bf{clock\\ signal\\ exactly\\ 0}$ -- these three read the same "
                "bytes as\nan honest provider, by arithmetic and not by measurement.\n"
                "No jitter model, no pool and no SKU changes it.  The clock\nis not "
                "weak in this band.  It is $\\it{absent}$ -- and a channel\nthat is "
                "absent cannot be oversold.",
                (0.985, 0.128), xycoords="axes fraction", ha="right", va="bottom",
                color=HONEST, fontsize=8.4, linespacing=1.55, zorder=7)

    d_long = clock_d_prime(p, 1.0, 1, CTX_LONG)
    kv = next(r for r in rows if r["key"] == "kv_fp8")

    # (anchor, ha) per point, chosen so no two text blocks share a band.
    LBL = {"quant_4bit": ((0.105, 1.5408), "left"),
           "kv_fp8": ((0.0250, 0.0200), "left"),
           "bug_k32": ((0.088, ZERO_Y), "left"),
           "seed_43": ((0.400, ZERO_Y), "left"),
           "temp_1.1": ((0.0060, ZERO_Y), "left")}
    for r in rows:
        live = r["d_clock"] > 0
        y = r["d_clock"] if live else ZERO_Y
        x = max(r["d_difr"], xlim[0] * 1.45)
        col = DEV if live else HONEST
        ax.plot([x], [y], marker="o", ms=12 if live else 8, color=col,
                mec="white", mew=1.6, zorder=6)
        at, ha = LBL[r["key"]]
        n = r["n_joint"]
        cost = "no price" if not np.isfinite(n) else f"{round(n):,} tok"
        body = (f"DiFR {max(r['d_difr'], 0):.3f}  |  clock {r['d_clock']:.3f}"
                f"  $\\rightarrow$  {cost}" if live else
                f"DiFR {max(r['d_difr'], 0):.3f}  $\\rightarrow$  {cost}")
        ax.annotate(r["label"], at, xytext=(0, 1), textcoords="offset points",
                    ha=ha, va="bottom", color=col, fontsize=9.2, weight="bold",
                    zorder=7)
        ax.annotate(body, at, xytext=(0, -2), textcoords="offset points", ha=ha,
                    va="top", color=col, fontsize=8.2, zorder=7)

    # The same KV attack at the context anyone would actually deploy it at.
    ax.annotate("", xy=(kv["d_difr"], d_long), xytext=(kv["d_difr"], kv["d_clock"]),
                arrowprops=dict(arrowstyle="-|>", color=WARN, lw=2.2, shrinkA=4,
                                shrinkB=4), zorder=5)
    ax.plot([kv["d_difr"]], [d_long], marker="o", ms=13, color="white", mec=WARN,
            mew=2.6, zorder=6)
    ax.annotate(f"same attack, 32k context\nclock $d'$ {d_long:.2f} "
                f"$\\rightarrow$ {round(tokens_for(d_long, ds))} tok",
                (kv["d_difr"] * 0.86, d_long), ha="right", va="center", color=WARN,
                fontsize=8.4, weight="bold", linespacing=1.5, zorder=7)
    ax.annotate("", xy=(kv["d_difr"] * 1.15, d_long * 0.92), xytext=(0.465, 0.548),
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-", color=WARN, lw=0.9), zorder=5)
    ax.annotate("$\\bf{Why\\ that\\ arrow\\ is\\ the\\ whole\\ argument.}$  The cache "
                "is 0.8% of\nthe read at 256 tokens and 52% of it at 32k, so the "
                "deviation\nthat is DiFR's $\\it{worst}$ cell in the entire grid "
                "(35 066 tokens) is\nthe clock's best.  Nobody deploys an fp8 KV "
                "cache at 256 tokens.",
                (0.985, 0.548), xycoords="axes fraction", ha="right", va="top",
                color=WARN, fontsize=8.5, linespacing=1.6, zorder=7)

    ax.annotate("$\\bf{The\\ empty\\ corner,\\ and\\ it\\ stays\\ empty.}$\n"
                "Orthogonality here is not a correlation near zero.\nIt is the "
                "$\\it{anti}$-diagonal: the row DiFR gets cheapest\n(wrong seed, 226 "
                "tokens) is the row the clock has\nnothing to say about, and the row "
                "DiFR nearly\nmisses (4-bit, $d'$ = 0.075, inside every Tier-0\n"
                "detector's noise) is the row the clock owns.  A client\nholding both "
                "never has to choose between them.",
                (0.985, 0.982), xycoords="axes fraction", ha="right", va="top",
                color=INK, fontsize=8.6, linespacing=1.62, zorder=7)

    ax.set_xlabel("DiFR's per-token signal  ($\\tt{token\\_difr}$ $d'$, "
                  "$\\bf{measured}$ on this repo's own grid)", color=INK2,
                  fontsize=10)
    ax.set_ylabel("the clock's per-token signal\n($d'$ = roofline gap / jitter sd; "
                  f"$\\bf{{arithmetic}}$, batch 1, $\\sigma$ = {SIGMA:.2f} ms)",
                  color=INK2, fontsize=10, linespacing=1.5)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_title("A.  Two channels in one unit: the loud one switches row by row",
                 color=INK, fontsize=11.5, weight="bold", loc="left", pad=8)


# =========================================================================== (B)
def panel_b(ax, rows, ds):
    order = sorted(rows, key=lambda r: r["n_joint"])
    h, xmax, notes_x = 0.21, 4.0e7, 1.2e7
    ax.set_xscale("log")
    ax.set_xlim(1.0, xmax)
    ax.set_ylim(-2.00, len(order) - 0.30)

    KEY = ("DiFR alone", "clock alone", "$\\bf{both\\ channels}$")
    for i, r in enumerate(order):
        base = len(order) - 1 - i
        for k, (n, col) in enumerate(((r["n_difr"], HONEST), (r["n_clock"], DEV),
                                      (r["n_joint"], WARN))):
            y = base + (1 - k) * h
            tag = f"   {KEY[k]}" if i == 0 else ""
            if np.isfinite(n):
                ax.barh(y, max(n, 1.0), height=h * 0.84, color=col, alpha=0.9,
                        zorder=3)
                ax.annotate(f"{round(n):,}{tag}", (max(n, 1.0), y), xytext=(5, 0),
                            textcoords="offset points", ha="left", va="center",
                            color=col, fontsize=8.4, weight="bold", zorder=4)
            else:
                ax.annotate(f"$\\it{{no\\ price\\ at\\ all}}${tag}", (1.15, y),
                            ha="left", va="center", color=col, fontsize=8.4,
                            zorder=4)
        ax.annotate(gain_note(r), (notes_x, base), ha="right", va="center",
                    color=INK, fontsize=8.5, linespacing=1.5, zorder=5)

    # The privileged channel, in the same units, drawn so it cannot be mistaken for
    # a deployable one.
    top = next(r for r in order if r["key"] == "quant_4bit")
    y = (len(order) - 1) - 2 * h
    ax.barh(y, max(top["n_priv"], 1.0), height=h * 0.84, color="white",
            edgecolor=INK2, lw=1.2, hatch="////", zorder=3)
    ax.annotate(f"{round(top['n_priv'])}   $\\tt{{activation\\_difr}}$, $d'$ = "
                f"{top['d_priv']:.2f} -- and the one detector that needs consent",
                (max(top["n_priv"], 1.0), y), xytext=(6, 0),
                textcoords="offset points", ha="left", va="center", color=INK2,
                fontsize=8.3, zorder=4)

    g = order[0]["n_difr"] / order[0]["n_joint"]
    ax.annotate("$\\bf{Independent\\ channels\\ add\\ in\\ }$$\\bf{d'^{2}}$"
                "$\\bf{,\\ so\\ verdict\\ costs\\ add\\ in\\ reciprocal:}$   "
                "$1/n_{joint} = 1/n_{DiFR} + 1/n_{clock}$.  The cheaper\nchannel "
                "takes the row outright, which is why three of these five rows do "
                "not move at all.  The honest headline is\ntherefore not \"the clock "
                "beats DiFR\" -- it is \"a client holding both is never worse off, "
                f"and on one row is {g:,.0f}$\\times$ better\noff, for zero audit "
                "FLOPs.\"  The hatched bar is the alternative on that row: "
                "$\\tt{activation\\_difr}$ needs a single token,\nand needs the "
                "provider to expose projected activations at batch 1.  The clock "
                "needs consent to nothing.  Cost is\nnot the only axis a channel is "
                "judged on, which is why (D) exists.",
                (0.012, 0.020), xycoords="axes fraction", ha="left", va="bottom",
                color=INK, fontsize=8.5, linespacing=1.66, zorder=6)

    ax.set_yticks(range(len(order)), [r["label"] for r in reversed(order)],
                  fontsize=9.6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(f"tokens of stream per verdict, $(\\delta^{{*}}/d')^{{2}}$ at "
                  f"$\\delta^{{*}}$ = {ds:.3f}  (log)", color=INK2, fontsize=10)
    ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_title("B.  What the orthogonality is worth, in the repo's own cost law",
                 color=INK, fontsize=11.5, weight="bold", loc="left", pad=8)


# =========================================================================== (C)
def panel_c(ax, p):
    """The evasion plane. Both axes are things the PROVIDER picks; the bands are
    what each detector cannot see. x is ordinal -- this is a mechanism diagram and
    the axis label says so."""
    rel4 = (roofline_ms(bytes_read(p, 0.25, 2, ROOFLINE["ctx"]))
            / roofline_ms(bytes_read(p, 1.0, 2, ROOFLINE["ctx"])))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.10, 1.36)
    ax.set_xticks([])

    ax.axvspan(0.0, 0.36, color=HONEST, alpha=0.10, zorder=0)
    ax.axhspan(0.94, 1.06, color=DEV, alpha=0.12, zorder=0)
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (1, 2.5)), zorder=1)
    ax.annotate("$\\bf{DiFR's\\ blind\\ band}$\nthe served distribution sits\n"
                "closer to $M$ than $M$'s own\nrun-to-run nondeterminism",
                (0.015, 1.348), ha="left", va="top", color=HONEST, fontsize=8.5,
                linespacing=1.55, zorder=6)
    ax.annotate("$\\bf{the\\ clock's\\ blind\\ band}$ -- billed at honest latency",
                (0.988, 1.113), ha="right", va="center", color=DEV, fontsize=8.7,
                zorder=6)

    # (x, y, colour, name, body, ha, name offset, body offset, va)
    pts = (
        (0.075, 1.0, HONEST, "honest bf16",
         "reads every byte,\nbills for every byte", "left", 8, -8, "center"),
        (0.105, rel4, DEV, "4-bit, speedup sold to the client",
         f"invisible to DiFR ($d'$ = 0.075) and {1 / rel4:.1f}$\\times$\nunder the "
         f"floor -- one token out of 3 000 ends it", "left", 8, -8, "center"),
        (0.285, 1.0, WARN, "4-bit, padded back to honest latency",
         "in $\\it{both}$ blind bands -- which is the point of\nthe panel: to get "
         "here it handed back 89-97%\nof the speedup it stole, kept only the\n"
         "throughput share, and still has to fake the\njitter it deleted.",
         "left", 12, 24, "bottom"),
        (0.945, 1.0, HONEST, "wrong seed / temperature / top-$k$",
         "moves the distribution,\nmoves no bytes at all.  DiFR's\nrows; the clock "
         "is absent.", "right", -12, -24, "top"),
        (0.945, 0.35, INK, "wholesale substitution (4B served as 0.6B)",
         "6.7$\\times$ the bytes $\\it{and}$ AUC 0.998 at the proxy tier --\nthe one "
         "cell where the two channels agree, and\nthe one nobody needed help with.",
         "right", -12, -24, "top"),
    )
    for x, y, col, name, body, ha, dn, db, va in pts:
        ax.plot([x], [y], marker="o", ms=11, color=col, mec="white", mew=1.6,
                zorder=6)
        dx = 14 if va == "center" else 0
        ax.annotate(name, (x, y), xytext=(dx, dn), textcoords="offset points",
                    ha=ha, va="bottom" if va == "center" else va, color=col,
                    fontsize=8.7, weight="bold", zorder=7)
        ax.annotate(body, (x, y), xytext=(dx, db), textcoords="offset points",
                    ha=ha, va="top" if va == "center" else va, color=col,
                    fontsize=8.3, linespacing=1.6, zorder=7)

    ax.annotate("", xy=(0.272, 0.972), xytext=(0.116, rel4 + 0.025),
                arrowprops=dict(arrowstyle="-|>", color=WARN, lw=2.2,
                                connectionstyle="arc3,rad=-0.28"), zorder=5)
    ax.annotate("the only way\ninto both bands", (0.157, 0.66), ha="left",
                va="center", color=WARN, fontsize=8.5, weight="bold",
                linespacing=1.45, zorder=7)

    ax.annotate("$\\bf{The\\ intersection\\ of\\ the\\ two\\ blind\\ bands\\ is\\ "
                "not\\ empty.\\ \\ It\\ is\\ a\\ refund.}$\nTo hide from DiFR you "
                "have to reproduce $M$'s distribution, which\nmeans reading $M$'s "
                "bytes.  To hide from the clock you have to\nspend $M$'s time.  A "
                "provider can still do neither and pad -- but\nthen what it keeps is "
                "the throughput share, not the latency it was\ncutting corners to "
                "sell.  $\\it{Cheap\\ in\\ one\\ currency\\ or\\ the\\ other,}$\n"
                "$\\it{never\\ both\\ at\\ once.}$",
                (0.985, 0.500), xycoords="axes fraction", ha="right", va="top",
                color=INK, fontsize=8.8, linespacing=1.7, zorder=7)

    ax.set_xlabel("what the provider serves -- distributional distance from $M$, "
                  "increasing to the right   ($\\it{ordinal:\\ a\\ mechanism\\ "
                  "diagram,\\ not\\ a\\ measurement}$)", color=INK2, fontsize=10)
    ax.set_ylabel("time per output token the client is billed at,\nrelative to the "
                  "honest batch-1 roofline", color=INK2, fontsize=10,
                  linespacing=1.5)
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_title("C.  Why it is structural: the two blind spots meet only at a "
                 "refund", color=INK, fontsize=11.5, weight="bold", loc="left",
                 pad=8)


# =========================================================================== (D)
def panel_d(ax, p, rows, ds):
    d_long_difr, n_long, model_long = long_ctx_difr()
    difr = {r["key"]: r["d_difr"] for r in rows}
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(1, 64)
    ax.set_ylim(1.6e-3, 60.0)

    for w, kvw, ctx, col, at, lab in (
            (0.25, 2, ROOFLINE["ctx"], DEV, (1.05, 0.22),
             "$\\bf{4\\!-\\!bit\\ weights}$ -- clock $d'$ $\\propto$ 1/$B$:\n"
             "weight reads amortize across the batch"),
            (1.00, 1, CTX_LONG, WARN, (13.0, 1.30),
             "$\\bf{fp8\\ KV\\ cache\\ at\\ 32k\\ context}$ -- flat in $B$:\nyour "
             "cache is read once per token per request,\nwhoever else is in flight")):
        d = np.array([clock_d_prime(p, w, kvw, ctx, b) for b in BATCHES])
        ax.plot(BATCHES, d, color=col, lw=2.8, zorder=5)
        ax.annotate(lab, at, ha="left", va="top" if w < 1 else "bottom",
                    color=col, fontsize=8.5, linespacing=1.55, zorder=7)

    ax.axhline(difr["quant_4bit"], color=HONEST, lw=1.9, zorder=4)
    ax.annotate(f"DiFR on 4-bit weights, $d'$ = {difr['quant_4bit']:.3f} -- "
                f"$\\bf{{flat\\ in}}$ $\\bf{{B}}$: a recompute\ndoes not care how "
                f"the provider batched you",
                (1.04, difr["quant_4bit"]), xytext=(0, -6),
                textcoords="offset points", ha="left", va="top", color=HONEST,
                fontsize=8.4, linespacing=1.55, zorder=6)
    ax.axhline(d_long_difr, color=MUTED, lw=1.5, ls=(0, (5, 2)), zorder=4)
    ax.annotate(f"DiFR on an fp8 KV cache at long context, $d'$ = {d_long_difr:.4f} "
                f"({model_long}, {n_long:,} tokens,\n$\\tt{{reversal\\_check.json}}$) "
                f"-- a $\\it{{different\\ model}}$, so the nearest measurement the "
                f"repo owns, not the matching one",
                (62, d_long_difr), xytext=(0, -7), textcoords="offset points",
                ha="right", va="top", color=MUTED, fontsize=8.3, linespacing=1.55,
                zorder=6)

    xo = crossover_batch(p, difr["quant_4bit"], SIGMA)
    ax.plot([xo], [difr["quant_4bit"]], marker="o", ms=11, color=DEV, mec="white",
            mew=1.8, zorder=7)
    ax.axvline(xo, color=DEV, lw=1.0, ls=(0, (2, 3)), zorder=3)
    sweep = ",  ".join(f"{crossover_batch(p, difr['quant_4bit'], s):.0f} at "
                       f"$\\sigma$ = {s:.2f}" for s in SIGMA_SWEEP)
    ax.annotate(f"$\\bf{{B\\ \\approx\\ {xo:.0f}}}$ (marked) -- past here a client "
                f"should stop reading the clock for weight\nattacks and start paying "
                f"for the recompute.  The crossover is linear in $\\sigma$, so the\n"
                f"whole swept decade agrees on the shape:  $B$ = {sweep} ms.",
                (1.04, 0.0019), ha="left", va="bottom", color=DEV, fontsize=8.4,
                linespacing=1.6, zorder=6)

    ax.annotate("$\\bf{Batching\\ is\\ the\\ clock's\\ hole,\\ and\\ it\\ is\\ only\\ "
                "half\\ a\\ hole.}$  Continuous batching reads one copy of the "
                "weights for $B$ concurrent\nrequests, so a weight attack's latency "
                "signature decays as 1/$B$ and the impossibility argument softens "
                "into a statistic.  It cannot\namortize $\\it{your}$ KV cache -- so "
                "the long-context KV attack is unbatchable-away, and it is precisely "
                "the attack DiFR reads at $d'$ = 0.02.\nWhere one channel's floor "
                "moves, the other's does not.  That is what makes the pair worth "
                "holding, rather than the better single number.",
                (0.013, 0.982), xycoords="axes fraction", ha="left", va="top",
                color=INK, fontsize=8.6, linespacing=1.65, zorder=6)

    ax.set_xticks([1, 2, 4, 8, 16, 32, 64], ["1", "2", "4", "8", "16", "32", "64"])
    ax.set_xlabel("$B$ -- concurrent requests the provider batches yours with",
                  color=INK2, fontsize=10)
    ax.set_ylabel("per-token $d'$ available to each channel\n($\\sigma$ = "
                  f"{SIGMA:.2f} ms; the clock modelled, DiFR measured)",
                  color=INK2, fontsize=10, linespacing=1.5)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_title("D.  Where the clock's edge ends -- and the one place it never does",
                 color=INK, fontsize=11.5, weight="bold", loc="left", pad=8)


# ========================================================================== main
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cost, p, ds, rows, null_hi = load()

    fig = plt.figure(figsize=(20.0, 14.2))
    gs = fig.add_gridspec(2, 2, wspace=0.175, hspace=0.265, left=0.058,
                          right=0.990, top=0.884, bottom=0.104)
    axes = [fig.add_subplot(gs[i, j]) for i in (0, 1) for j in (0, 1)]
    for a in axes:
        _clean(a)
    panel_a(axes[0], p, rows, ds, null_hi)
    panel_b(axes[1], rows, ds)
    panel_c(axes[2], p)
    panel_d(axes[3], p, rows, ds)

    fig.suptitle("The clock is not a better DiFR -- it is a second axis.  DiFR asks "
                 "whether you computed this distribution; the clock asks whether "
                 "you read these bytes",
                 color=INK, fontsize=15, weight="bold", x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.958,
             "Every DiFR number here is $\\bf{measured}$ "
             "($\\tt{cost\\_of\\_a\\_verdict.json}$: Qwen3-1.7B, proxy Qwen3-0.6B, "
             "80$\\times$256 tokens per arm, standardized pAUC at FPR $\\leq$ 0.5%, "
             f"$\\delta^{{*}}$ = {ds:.3f}).  Every clock number is "
             "$\\bf{arithmetic\\ off\\ a\\ spec\\ sheet}$ -- bytes read / 3.35 TB/s "
             "(H100 SXM5) -- divided by an\nassumed one-sided jitter $\\sigma$ = "
             f"{SIGMA:.2f} ms.  Nothing in this repo has ever timed a provider: "
             "$\\tt{perf\\_counter}$ appears only as the $\\it{verifier's}$ own cost.  "
             "So (A), (B) and (D) each put one measured axis against one predicted "
             "axis, and (C) has no measured axis at all.  Read this as the design "
             "of\n$\\tt{exp\\_clock\\_channel\\_gpu}$ rather than its result -- the "
             "load-bearing claim is the $\\it{shape}$ (which rows each channel owns, "
             "and why), which survives the decade of $\\sigma$ swept in "
             "$\\tt{fig\\_clock\\_channel\\_principle}$ (C).  The absolute $d'$ values "
             "on the clock axis do not.",
             color=INK2, fontsize=8.6, ha="left", va="top", linespacing=1.65)

    shape = batch_recoverability()
    b4 = next(r for r in shape if r["batch_size"] == 4)
    sat = [r for r in shape if r["batch_size"] >= 8]
    fig.text(0.006, 0.012,
             "$\\bf{What\\ would\\ falsify\\ this,\\ and\\ what\\ is\\ still\\ "
             "missing.}$   $\\bf{(i)}$ Panel (D) needs $B$ and the client does not "
             "know it.  The tokens carry part of it: a batch of 4 leaves "
             f"{b4['frac_exact']:.1%} of logit values bitwise identical to the "
             f"batch-1 run and flips {b4['argmax_flips']} of {b4['n_pos']} argmaxes "
             "-- but the signature\n$\\bf{saturates}$ ("
             f"{sat[0]['frac_exact']:.1%} exact at $B$ = {sat[0]['batch_size']}, "
             f"{sat[-1]['frac_exact']:.1%} at $B$ = {sat[-1]['batch_size']}, "
             "$\\tt{specdec\\_shape.json}$), so the tokens say $\\it{batched\\ or\\ "
             "not}$ and not $\\it{how\\ much}$ -- and a clock floor needs the number.  "
             "That gap is a token-channel experiment, not a clock one.   "
             "$\\bf{(ii)}$ The floor is per-SKU and the provider picks the SKU: on a "
             "B200 the honest\nfloor is lower and every impossibility in (C) softens "
             "into a statistic.   $\\bf{(iii)}$ If real jitter is heavy-tailed rather "
             "than the gamma assumed here, the clock's $d'$ falls -- but its "
             "$\\it{sign}$ does not, because the minimum stays immune to additive "
             "positive noise, so (A)'s ordering is the robust part.   $\\bf{(iv)}$ "
             "The four panels reduce to one sentence:\n$\\bf{the\\ clock\\ buys\\ "
             "exactly\\ one\\ row\\ of\\ this\\ grid,\\ it\\ buys\\ it\\ for\\ zero\\ "
             "audit\\ FLOPs\\ and\\ zero\\ provider\\ cooperation,\\ and\\ it\\ is\\ "
             "the\\ row\\ DiFR\\ is\\ worst\\ at.}$  That is a smaller claim than "
             "\"a new detector\" and a more useful one than \"another statistic.\"",
             color=INK, fontsize=8.7, ha="left", va="bottom", linespacing=1.68)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_clock_vs_difr.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"wrote {out}")
    f = lambda n: "       inf" if not np.isfinite(n) else f"{round(n):,}"
    print(f"  {'deviation':12s} {'DiFR d':>8s} {'clock d':>9s} {'n_DiFR':>10s} "
          f"{'n_clock':>11s} {'n_joint':>10s}  gain")
    for r in rows:
        g = r["n_difr"] / r["n_joint"]
        print(f"  {r['key']:12s} {r['d_difr']:8.3f} {r['d_clock']:9.4f} "
              f"{f(r['n_difr']):>10s} {f(r['n_clock']):>11s} {f(r['n_joint']):>10s}"
              f"  {g:.2f}x" if np.isfinite(g) else
              f"  {r['key']:12s} {r['d_difr']:8.3f} {r['d_clock']:9.4f} "
              f"{f(r['n_difr']):>10s} {f(r['n_clock']):>11s} {f(r['n_joint']):>10s}"
              f"  --")
    d_long = clock_d_prime(p, 1.0, 1, CTX_LONG)
    print(f"  kv_fp8 at ctx {CTX_LONG:,}: clock d' = {d_long:.3f} "
          f"({round(tokens_for(d_long, ds))} tokens), DiFR d' = "
          f"{long_ctx_difr()[0]:.4f} on {long_ctx_difr()[2]}")
    print(f"  DiFR honest-control CI upper bound: {null_hi:.4f}")
    for s in SIGMA_SWEEP:
        print(f"  4-bit clock d' == DiFR d' at B = "
              f"{crossover_batch(p, rows[0]['d_difr'], s):5.1f}  (sigma {s:.2f} ms)")


if __name__ == "__main__":
    main()
