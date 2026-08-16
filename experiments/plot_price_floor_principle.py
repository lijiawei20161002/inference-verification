"""Why "no budget buys a verdict" is a ruler, not a law.

Companion to `paper/cost_accuracy_poster_sci.png`, in the same spirit that
`plot_batch_principle.py` is a companion to `fig_int8_wall.png`: that poster
reports *that* 14 of 35 (attack, verifier) cells have no finite price, this one
asks what was actually measured when a cell is priced at infinity.

The poster's price is `b*(d') = (delta*/d')^2`, and `signal.batch_for_pauc`
returns -1 -- unreachable, no budget -- exactly when the point estimate `d' <= 0`.
That is a decision rule on the SIGN of a noisy statistic. Re-read from the same
per-token scores the poster was built from:

  docs/figures/fig_price_floor_principle.png
      (A) Every cell's `d'` with its 90% sequence-bootstrap interval, against the
          interval that verifier's own honest-vs-honest null spans at this pool.
          Not one of the 13 `d' <= 0` cells has an interval that excludes a
          detectable effect. Infinity is never measured; it is a sign flip.
      (B) The same mechanism as `fig_batch_principle` panel (B), one level up.
          There the batch mean's noise shrank as sigma/sqrt(n) while the gap
          stayed put; here the *estimate of d'* does the same as the pool grows,
          and the "threshold" it has to clear is d' = 0. kv_fp8/token_difr sits on
          the poster's board at 35,066 tokens, and reports "no budget buys a
          verdict" in 27% of quarter-pool runs of the same experiment.
      (C) The consequence, on the poster's own price plane. A pool of m sequences
          resolves `d'` no finer than `1.645 k/sqrt(m)`, so it cannot separate any
          price above `(delta* sqrt(m) / 1.645 k)^2` from infinity. The 10
          unresolved cells are not off the curve, they are off the end of the
          ruler, and the data still bounds them from below. Both the pool that
          resolves a price and the batch that buys the verdict go as 1/d'^2, so
          their ratio is a constant per verifier -- and it is smaller than the
          20 x b* honest calibration pool the verdict already has to buy.

The one real exception is in (C)'s text and it is not a measurement: `seed_43` is
the same weights at a different sampling seed, so the claimed tokens are a fresh
draw from the identical distribution and every verifier that is a function of
(prompt, claimed tokens) alone has d' = 0 by exchangeability, at any pool size.
The poster prices three of those cells at 280,666 / 1,220,994 / 1,678,019 tokens.
Those budgets buy nothing. Infinity there is provable, and the sign rule missed it
while flagging ten cells that are merely unpriced.

    .venv/bin/python -m experiments.plot_price_floor_principle
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ivgym import signal  # noqa: E402

FIG_DIR = ROOT / "docs" / "figures"
RES = ROOT / "docs" / "results"
COST = RES / "cost_of_a_verdict.json"
SCORES = RES / "cost_of_a_verdict_scores.npz"
CACHE = RES / "price_floor_principle.json"

# Same palette slots and the same meaning as plot_batch_principle.py: the blue is
# the thing under test, the null wears ink rather than a hue. Orange is reserved
# for the poster's verdict of "no budget buys this" -- it is a label, not an arm.
DEV, HONEST = "#2a78d6", "#52514e"
INF = "#d1622b"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8880", "#e3e2dd"

N_BOOT = 2000           # sequence bootstrap per cell for (A)
N_REP = 4000            # pool-subsample repeats per (pool size, cell) in (B)
Z90 = 1.6449            # one-sided 95% / the half-width of a 90% interval
POOLS = (10, 20, 40, 80)        # sequences per arm, shown in (B)
FOCUS = ("kv_fp8", "token_difr")  # the cell (B) follows: on the board at 35,066

SHORT_A = {"quant_4bit": "int4 quant", "kv_fp8": "fp8 KV", "temp_1.1": "temp 1.1",
           "seed_43": "wrong seed", "bug_k32": "top-$k$ bug"}
SHORT_V = {"token_difr": "token DiFR", "cross_entropy": "cross entropy",
           "token_toploc": "top-loc", "activation_difr": "activation DiFR",
           "accept_rate": "accept rate", "surface_stat": "surface stat",
           "surface_tokens": "surface tokens"}
# `seed_43` changes the sampling seed and nothing else, so the claimed tokens are a
# fresh draw from the SAME distribution. Any verifier whose score is a function of
# (prompt, claimed tokens) has d' = 0 identically -- no pool ever resolves it. Only
# token_difr escapes: its post-Gumbel margin is scored against the claimed seed.
SEED_BLIND = {"cross_entropy", "token_toploc", "activation_difr", "accept_rate",
              "surface_stat", "surface_tokens"}


# ------------------------------------------------------------------------ data
def _dprime(h, a):
    return signal.per_token_stats(np.concatenate(h), np.concatenate(a))["d_prime"]


def measure(meta, by_seq, n_seq):
    """Everything the three panels read, bootstrapped from the poster's own pool."""
    ver, att, dstar = meta["verifiers"], meta["attacks"], meta["delta_star"]
    honest = {v: by_seq(f"honest__{v}") for v in ver}

    # (i) how finely THIS pool can resolve d' at all, per verifier. Split the honest
    # arm against itself at several pool sizes: the spread of that null is the floor,
    # and it must fall as k/sqrt(m) or the extrapolation in (C) is not allowed.
    floor = {}
    for v in ver:
        seqs, ks = honest[v], []
        for m in (5, 10, 20, 40):
            rng = np.random.default_rng(7)
            d = [_dprime(seqs[i[:m]], seqs[i[m:2 * m]])
                 for i in (rng.permutation(n_seq) for _ in range(600))]
            ks.append(float(np.std(d, ddof=1)) * np.sqrt(m))
        floor[v] = {"k": float(np.mean(ks)), "k_by_pool": ks,
                    "d_floor": float(Z90 * np.mean(ks) / np.sqrt(n_seq)),
                    # tokens of pool per arm to resolve a price, over the b* tokens
                    # that same price costs. Both go as 1/d'^2, so this is a constant.
                    "pool_over_bstar": float(meta["tokens"] * (Z90 * np.mean(ks)) ** 2
                                             / dstar ** 2)}

    # (ii) every cell: d', its sequence-bootstrap interval, and the price the
    # interval's upper end implies -- the lower bound the data does support.
    cells = []
    for atk in att:
        for v in ver:
            h, a = honest[v], by_seq(f"{atk}__{v}")
            rng = np.random.default_rng(0)
            draws = np.array([_dprime(h[rng.integers(0, n_seq, n_seq)],
                                      a[rng.integers(0, n_seq, n_seq)])
                              for _ in range(N_BOOT)])
            dp, hi = _dprime(h, a), float(np.quantile(draws, 0.95))
            structural = atk == "seed_43" and v in SEED_BLIND
            cells.append({
                "attack": atk, "verifier": v, "d_prime": dp,
                "ci": [float(np.quantile(draws, 0.05)), hi],
                "b_poster": signal.batch_for_pauc(dp),      # -1 == the poster's inf
                "b_lower": signal.batch_for_pauc(hi),       # bigger d' -> cheaper
                "sec_per_token": meta["price"][v]["sec_per_token"],
                "structural_zero": structural,
                "regime": ("structural" if structural else
                           "unresolved" if dp <= 0 else "priced"),
            })

    # (iii) (B)'s mechanism: how often the SAME cell reports infinity, as a function
    # of how many sequences the run happened to buy.
    flip = {}
    for atk, v in [FOCUS]:
        h, a = honest[v], by_seq(f"{atk}__{v}")
        rows = {}
        for m in POOLS:
            rng = np.random.default_rng(5)
            d = np.array([_dprime(h[rng.integers(0, n_seq, m)], a[rng.integers(0, n_seq, m)])
                          for _ in range(N_REP)])
            rows[str(m)] = {"draws": d.tolist(), "p_inf": float((d <= 0).mean())}
        # the same subsampling on honest-vs-honest, which is what a null looks like
        null = {}
        for m in POOLS:
            rng = np.random.default_rng(9)
            d = np.array([_dprime(h[rng.integers(0, n_seq, m)], h[rng.integers(0, n_seq, m)])
                          for _ in range(N_REP)])
            null[str(m)] = d.tolist()
        flip[f"{atk}__{v}"] = {"cell": rows, "null": null, "d_full": _dprime(h, a)}

    # (iv) the exchangeability check behind the structural-zero claim: permute the
    # sequence labels between honest and seed_43 and ask where the observed d' lands.
    exch = {}
    for v in ver:
        h, a = honest[v], by_seq(f"seed_43__{v}")
        both = np.concatenate([h, a], 0)
        rng = np.random.default_rng(3)
        null = np.array([_dprime(both[i[:n_seq]], both[i[n_seq:]])
                         for i in (rng.permutation(2 * n_seq) for _ in range(2000))])
        dp = _dprime(h, a)
        exch[v] = {"d_prime": dp, "z": float((dp - null.mean()) / null.std(ddof=1)),
                   "p": float((np.abs(null - null.mean()) >= abs(dp - null.mean())).mean())}
    return {"floor": floor, "cells": cells, "flip": flip, "exchange": exch}


def load():
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    meta = json.loads(COST.read_text())
    z = np.load(SCORES, allow_pickle=True)
    n_seq, t = int(z["n_prompts"]), int(z["tokens"])
    out = measure(meta, lambda k: np.asarray(z[k], float).reshape(n_seq, t), n_seq)
    out["meta"] = {k: meta[k] for k in ("M", "proxy", "n_prompts", "tokens",
                                        "delta_star", "verifiers", "attacks",
                                        "pool_tokens_per_config")}
    CACHE.write_text(json.dumps(out))
    print(f"wrote {CACHE}")
    return out


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.set_axisbelow(True)


# =========================================================================== (A)
def panel_a(ax, D):
    """Every cell's effect against the noise the estimate carries. The poster's
    infinity/finite boundary is the x = 0 line, and the intervals straddle it."""
    floor, xlo, xhi = D["floor"], -0.125, 0.175
    shown = [c for c in D["cells"] if abs(c["d_prime"]) < xhi]
    off = sorted((c for c in D["cells"] if c not in shown),
                 key=lambda c: -c["d_prime"])
    shown.sort(key=lambda c: c["d_prime"])

    ax.axvspan(xlo, 0, color=INF, alpha=0.05, lw=0, zorder=0)
    ax.axvline(0, color=INK, lw=1.4, zorder=4)
    for y, c in enumerate(shown):
        f = floor[c["verifier"]]["d_floor"]
        col = {"priced": DEV, "unresolved": INF, "structural": HONEST}[c["regime"]]
        # what this verifier's own honest-vs-honest null spans at this pool: any
        # effect inside it is a coin flip on which side of infinity it lands.
        ax.plot([-f, f], [y, y], color=GRID, lw=6.5, solid_capstyle="butt", zorder=1)
        ax.plot(c["ci"], [y, y], color=col, lw=1.5, alpha=0.85, zorder=3)
        ax.plot([c["d_prime"]], [y], marker="o", ms=4.6, color=col, mec="white",
                mew=0.8, zorder=5)

    n_inf = sum(c["b_poster"] < 0 for c in D["cells"])
    n_cross = sum(c["b_poster"] < 0 and c["ci"][1] > 0 for c in D["cells"])
    ax.annotate(f"the poster prices these {n_inf} at $\\infty$",
                (0.171, sum(c["d_prime"] <= 0 for c in shown) / 2 - 0.5),
                ha="right", va="center", color=INF, fontsize=9, weight="bold")
    ax.annotate(f"{len(off)} cells off-scale right, resolved by orders of magnitude:\n"
                + ",  ".join(f"{SHORT_A[c['attack']]} / {SHORT_V[c['verifier']]}"
                             for c in off)
                + "\n" + ",  ".join(f"$d'$ = {c['d_prime']:.2f} $\\rightarrow$ "
                                    f"{c['b_poster']:,} tok" for c in off),
                (xlo + 0.003, len(shown) + 9.0), ha="left", va="top", color=INK2,
                fontsize=7.6, linespacing=1.5)
    ax.annotate("bar = 90% interval,  $\\bullet$ = $d'$,  grey = that verifier's\n"
                "own honest-vs-honest null at this pool",
                (xlo + 0.003, len(shown) + 3.2), ha="left", va="top", color=INK2,
                fontsize=8, linespacing=1.5)

    ax.annotate(f"$\\bf{{Every\\ one}}$ of those {n_inf} has an interval that\nreaches "
                f"a detectable $d'$; {n_inf - n_cross} is measured to be\n"
                f"undetectable.  The boundary between a price\nand no price is the "
                f"$x=0$ line, and the intervals\nare wider than the effects: the sign "
                f"of $d'$ is being\nread as a physical fact when it is a draw from\n"
                f"the grey bar.",
                (xlo + 0.003, -2.2), ha="left", va="top", color=INK, fontsize=9,
                linespacing=1.6)

    ax.set_yticks(range(len(shown)))
    ax.set_yticklabels([f"{SHORT_A[c['attack']]} / {SHORT_V[c['verifier']]}"
                        for c in shown], fontsize=7.4)
    ax.tick_params(axis="y", length=0, pad=2)
    for lab, c in zip(ax.get_yticklabels(), shown):
        lab.set_color({"priced": INK2, "unresolved": INF, "structural": HONEST}[c["regime"]])
    ax.set_ylim(-18.4, len(shown) + 9.6)
    ax.set_xlim(xlo, xhi)
    ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel("per-token $d'$ (the poster's own estimate, 80 sequences per arm)",
                  color=INK2, fontsize=10)
    ax.set_title("A.  Infinity is a sign, and the sign is noise", color=INK,
                 fontsize=11, weight="bold", loc="left", pad=8)


# =========================================================================== (B)
def panel_b(axes, D):
    """One cell, four pool sizes. The effect is fixed; the estimate's spread is
    k/sqrt(m); the verdict 'no budget' is everything left of zero."""
    key = f"{FOCUS[0]}__{FOCUS[1]}"
    F = D["flip"][key]
    b_post = next(c["b_poster"] for c in D["cells"]
                  if (c["attack"], c["verifier"]) == FOCUS)
    lo, hi = -0.105, 0.135
    bins = np.linspace(lo, hi, 96)
    mid = 0.5 * (bins[1:] + bins[:-1])

    for k, (ax, m) in enumerate(zip(axes, POOLS)):
        _clean(ax)
        cell = np.array(F["cell"][str(m)]["draws"])
        null = np.array(F["null"][str(m)])
        p_inf = F["cell"][str(m)]["p_inf"]
        dn, _ = np.histogram(null, bins=bins, density=True)
        dc, _ = np.histogram(cell, bins=bins, density=True)
        top = max(dn.max(), dc.max())

        ax.axvline(F["d_full"], color=DEV, lw=1.0, ls=(0, (2, 2.5)), alpha=0.75, zorder=2)
        ax.axvline(0.0, color=INK, lw=1.3, zorder=6)
        ax.fill_between(mid, 0, dn / top, color=HONEST, alpha=0.30, lw=0, zorder=3)
        ax.plot(mid, dn / top, color=HONEST, lw=1.6, zorder=4)
        ax.fill_between(mid, 0, dc / top, color=DEV, alpha=0.30, lw=0, zorder=3)
        ax.plot(mid, dc / top, color=DEV, lw=1.6, zorder=4)
        # the runs that report "no budget buys a verdict" -- same cell, same truth
        ax.fill_between(mid, 0, dc / top, where=mid <= 0, color=INF, alpha=0.95,
                        lw=0, zorder=5)

        ax.set_ylim(0, 1.30)
        ax.set_xlim(lo, hi)
        ax.set_yticks([])
        ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
        ax.spines["left"].set_visible(False)
        ax.annotate(f"$\\bf{{m = {m}}}$ sequences   sd of $\\hat{{d'}}$ = "
                    f"{np.std(cell, ddof=1):.3f}",
                    (0.985, 0.90), xycoords="axes fraction", ha="right", va="top",
                    color=INK2, fontsize=9)
        ax.annotate(f"{p_inf:.0%} of runs report\n\"no budget buys a verdict\"",
                    (0.985, 0.66), xycoords="axes fraction", ha="right", va="top",
                    color=INF, fontsize=8.5, weight="bold", linespacing=1.35)
        if k == 0:
            ax.annotate("honest\nvs honest", (0.0, 1.10), xytext=(-4, 0), ha="right",
                        va="bottom", textcoords="offset points", color=HONEST,
                        fontsize=8.5, weight="bold", linespacing=1.3,
                        annotation_clip=False)
            ax.annotate("fp8 KV", (F["d_full"], 1.10), xytext=(5, 0), ha="left",
                        va="bottom", textcoords="offset points", color=DEV,
                        fontsize=9, weight="bold", annotation_clip=False)
            ax.annotate("", xy=(F["d_full"], 1.46), xytext=(0.0, 1.46),
                        annotation_clip=False,
                        arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.2,
                                        shrinkA=0, shrinkB=0, mutation_scale=9))
            ax.annotate(f"$d'$ = {F['d_full']:.4f}, the same in every row",
                        (F["d_full"], 1.46), xytext=(8, 0), textcoords="offset points",
                        ha="left", va="center", color=INK, fontsize=9, weight="bold",
                        annotation_clip=False)
        if k == len(POOLS) - 1:
            ax.annotate("orange = the runs in which\n$b^{*}$ comes back $-1$: "
                        "everything\nleft of zero is priced $\\infty$",
                        (0.02, 0.40), xycoords="axes fraction", ha="left", va="top",
                        color=INF, fontsize=8.5, linespacing=1.35)
        if k == len(POOLS) - 1:
            ax.set_xlabel("measured $\\hat{d'}$ for one run of the experiment",
                          color=INK2, fontsize=10)
        else:
            ax.set_xticklabels([])

    axes[0].set_title("B.  The label flips with the pool, not with the provider",
                      color=INK, fontsize=11, weight="bold", loc="left", pad=44)
    axes[0].annotate(f"fp8 KV / token DiFR $-$ one cell, on the poster's board at a "
                     f"finite {b_post:,} tokens", (0.0, 1.30),
                     xycoords="axes fraction", ha="left", va="bottom", color=INK2,
                     fontsize=9, annotation_clip=False)


# =========================================================================== (C)
def panel_c(ax, D):
    """The poster's own price plane, with the ruler drawn on it."""
    import matplotlib
    dstar = D["meta"]["delta_star"]
    floor = D["floor"]
    f_lo = min(f["d_floor"] for f in floor.values())
    f_hi = max(f["d_floor"] for f in floor.values())
    r_lo = min(f["pool_over_bstar"] for f in floor.values())
    r_hi = max(f["pool_over_bstar"] for f in floor.values())
    xlo, xhi = 1.2e-3, 16.0

    d = np.geomspace(xlo, xhi, 400)
    ax.plot(d, (dstar / d) ** 2, color=INK, lw=2.2, zorder=5,
            label="the poster's price  $b^{*} = (\\delta^{*}/d')^{2}$")

    # what an 80-sequence pool can tell apart from infinity, and what 100x buys
    ax.axvspan(xlo, f_hi, color=INF, alpha=0.08, lw=0, zorder=1)
    ax.axvspan(xlo, f_hi / 10, color=INF, alpha=0.10, lw=0, zorder=1)
    for x in (f_hi, f_hi / 10):
        ax.axvline(x, color=INF, lw=1.1, ls=(0, (4, 2.5)), zorder=4)
    ax.annotate(f"80 sequences resolve $d'$ no finer than\n"
                f"{f_lo:.3f} (token DiFR) to {f_hi:.3f} (surface\n"
                f"tokens).  Left of here, this pool has\nnever measured a price.",
                (f_hi, 1.1e7), xytext=(7, 0), textcoords="offset points",
                ha="left", va="top", color=INF, fontsize=8.4, linespacing=1.5)
    ax.annotate("100$\\times$ the pool", (f_hi / 10, 1.1e7), xytext=(-5, 0),
                textcoords="offset points", ha="right", va="top", color=INF,
                fontsize=8.4)

    n_priced = sum(c["regime"] == "priced" for c in D["cells"])
    n_unres = sum(c["regime"] == "unresolved" for c in D["cells"])
    ghosts = [c for c in D["cells"] if c["structural_zero"] and c["b_poster"] > 0]
    for c in D["cells"]:
        if c["regime"] == "priced" and c["b_poster"] > 0:
            ax.plot([c["d_prime"]], [c["b_poster"]], marker="o", ms=5.5, color=DEV,
                    mec="white", mew=0.8, zorder=6)
        elif c["regime"] == "unresolved":
            # the data does not say infinity, it says "at least this much"
            ax.annotate("", xy=(c["ci"][1], c["b_lower"] * 7.0),
                        xytext=(c["ci"][1], c["b_lower"]),
                        arrowprops=dict(arrowstyle="-|>", color=INF, lw=1.3,
                                        shrinkA=0, shrinkB=0, mutation_scale=9),
                        zorder=6)
            ax.plot([c["ci"][1]], [c["b_lower"]], marker="_", ms=9, color=INF,
                    mew=2.0, zorder=7)
    for c in ghosts:                        # priced by the poster, provably infinite
        ax.plot([c["d_prime"]], [c["b_poster"]], marker="x", ms=8, color=INK,
                mew=1.8, zorder=7)
    ax.plot([], [], marker="o", ms=5.5, color=DEV, ls="none", mec="white",
            label=f"{n_priced} cells with a resolved price")
    ax.plot([], [], marker="$\\uparrow$", ms=8, color=INF, ls="none",
            label=f"{n_unres} unresolved: price $\\geq$ this")
    ax.plot([], [], marker="x", ms=8, color=INK, ls="none", mew=1.8,
            label=f"{len(ghosts)} priced, but provably $\\infty$")

    b_anchor = next(c["b_poster"] for c in D["cells"]
                    if (c["attack"], c["verifier"]) == ("quant_4bit", "token_difr"))
    ax.plot([0.075], [b_anchor], marker="o", ms=10, mfc="none", mec=INK, mew=1.4,
            zorder=8)
    ax.annotate("fig_batch_principle's $d'$ = 0.073 lives here",
                (0.075, b_anchor), xytext=(14, 12), textcoords="offset points",
                ha="left", va="bottom", color=INK, fontsize=8.8, weight="bold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(0.55, 2e7)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("per-token $d'$", color=INK2, fontsize=10)
    ax.set_ylabel("tokens per verdict", color=INK2, fontsize=10)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    sec = ax.secondary_yaxis(
        "right", functions=(lambda b: b * 141.06e-6, lambda s: s / 141.06e-6))
    sec.set_ylabel("Tier-1 GPU-seconds (141 $\\mu$s/token, measured)", color=INK2,
                   fontsize=9.5)
    sec.tick_params(colors=INK2, labelsize=8.5, length=3)
    ax.set_title("C.  Off the end of the ruler, not off the curve", color=INK,
                 fontsize=11, weight="bold", loc="left", pad=8)
    leg = ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelspacing=0.7,
                    handlelength=1.6, borderaxespad=0.7)
    for t in leg.get_texts():
        t.set_color(INK2)
    return r_lo, r_hi


# ====================================================================== footer
def footer(fig, D, r_lo, r_hi):
    """What the three panels add up to, and the one thing they do not overturn."""
    import matplotlib.pyplot as plt
    unres = [c for c in D["cells"] if c["regime"] == "unresolved"]
    b_min = min(c["b_lower"] for c in unres)
    b_max = max(c["b_lower"] for c in unres)
    s_min = min(c["b_lower"] * c["sec_per_token"] for c in unres)
    s_max = max(c["b_lower"] * c["sec_per_token"] for c in unres)
    ghosts = sorted((c for c in D["cells"] if c["structural_zero"] and c["b_poster"] > 0),
                    key=lambda c: c["b_poster"])
    p_min = min(D["exchange"][v]["p"] for v in SEED_BLIND)

    fig.add_artist(plt.Line2D([0.008, 0.992], [0.212, 0.212], color=MUTED, lw=0.9))
    cols = [
        ("1.  Infinity was never measured.",
         f"`batch_for_pauc` returns -1 on the SIGN of $d'$, and at 80 sequences that "
         f"sign is a\ncoin flip for any effect under 0.02 to 0.08.  All 13 zero-priced "
         f"cells have intervals\nthat reach a detectable $d'$; none is measured to be "
         f"undetectable.  The label is not\na property of the provider: fp8 KV / token "
         f"DiFR is on the board at 35,066 tokens\nand still comes back \"$\\infty$\" in "
         f"27% of quarter-pool runs of the same experiment,\n4% at full pool.  What "
         f"changed between those runs was the audit's budget."),
        ("2.  Price the unresolved cells; do not retire them.",
         f"The interval's upper end is a real price floor.  For the 10 merely-"
         f"unresolved cells it\nis $\\bf{{{b_min:,}}}$ to $\\bf{{{b_max:,}}}$ tokens "
         f"$-$ {s_min:.3f} to {s_max:.1f} GPU-seconds.  \"No budget buys a verdict\"\n"
         f"is, at its strongest, \"at least {s_max:.1f} seconds of an H100\".  And the "
         f"experiment that\nsettles it is cheap: resolving a price and buying it both "
         f"go as $1/d'^{{2}}$, so the pool\nneeded is a fixed {r_lo:.1f}$\\times$ to "
         f"{r_hi:.1f}$\\times$ $b^{{*}}$ $-$ always less than the $20\\times b^{{*}}$ "
         f"honest calibration\npool the verdict has to buy anyway.  Report "
         f"$b^{{*}} \\geq$ N, and say what N cost."),
        ("3.  One infinity is real $-$ and the sign test missed it.",
         f"Wrong seed is the same weights on a fresh draw, so every verifier that "
         f"reads only\n(prompt, claimed tokens) has $d'$ = 0 by exchangeability, at "
         f"$\\it{{any}}$ pool: permutation\n$p \\geq {p_min:.2f}$ for all six, while "
         f"seed-bound token DiFR sees it at $z$ = +10.1.  That is a\nproof, not a "
         f"measurement.  The poster prices three of those cells at\n"
         + " / ".join(f"{c['b_poster'] / 1e6:.2f}M" for c in ghosts)
         + f" tokens $-$ budgets that buy nothing.  A sign test on $\\hat{{d'}}$\nerrs "
           f"both ways.  The honest grid has three entries, not two: a price,\na price "
           f"floor, or an invariance argument."),
    ]
    for x, (head, body) in zip((0.010, 0.343, 0.676), cols):
        fig.text(x, 0.186, head, color=INK, fontsize=10.5, weight="bold",
                 ha="left", va="top")
        fig.text(x, 0.151, body, color=INK, fontsize=9.0, ha="left", va="top",
                 linespacing=1.62)


# ========================================================================== main
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D = load()
    m = D["meta"]

    fig = plt.figure(figsize=(17.4, 10.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.04, 0.94, 1.24], wspace=0.245,
                          left=0.098, right=0.952, top=0.800, bottom=0.255)
    ax_a = fig.add_subplot(gs[0, 0])
    _clean(ax_a)
    panel_a(ax_a, D)

    sub = gs[0, 1].subgridspec(len(POOLS), 1, hspace=0.30)
    panel_b([fig.add_subplot(sub[i, 0]) for i in range(len(POOLS))], D)

    ax_c = fig.add_subplot(gs[0, 2])
    _clean(ax_c)
    r_lo, r_hi = panel_c(ax_c, D)
    footer(fig, D, r_lo, r_hi)

    fig.suptitle("\"No budget buys a verdict\" is the ruler running out, not the "
                 "price being infinite", color=INK, fontsize=13.5, weight="bold",
                 x=0.008, ha="left", y=0.987)
    fig.text(0.008, 0.955,
             f"cost_accuracy_poster_sci retires 14 of 35 cells at an infinite price.  "
             f"That price is $b^{{*}} = (\\delta^{{*}}/d')^{{2}}$, $\\delta^{{*}}$ = "
             f"{m['delta_star']:.3f}, and batch_for_pauc returns infinity exactly when "
             f"the point estimate $d' \\leq 0$ $-$ a decision rule on the sign of a "
             f"statistic whose standard error is 0.011 to 0.049 at this pool.\n"
             f"Re-read here from the poster's own per-token scores "
             f"(cost_of_a_verdict_scores.npz: {m['M']} audited by {m['proxy']}, "
             f"{m['n_prompts']} sequences $\\times$ {m['tokens']} tokens per arm, 5 "
             f"deviations $\\times$ 7 verifiers).  The 14 are 13 cells with $d' \\leq 0$ "
             f"plus one whose finite $b^{{*}}$ fails its own honest control.\n"
             f"Intervals are {N_BOOT:,} sequence bootstraps, the poster's own protocol; "
             f"pools past {m['n_prompts']} sequences in (C) assume the observed score "
             f"tail is representative.",
             color=INK2, fontsize=8.5, ha="left", va="top", linespacing=1.5)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_price_floor_principle.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
