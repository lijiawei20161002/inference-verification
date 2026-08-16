"""What the clock actually is, once someone times it. The result of exp_clock_channel_gpu.

`fig_clock_basic.png` and its three companions are a MODEL: every floor there is
`bytes read / 3.35 TB/s` off an H100 SXM5 spec sheet, over an assumed one-sided jitter
sigma = 0.50 ms, and each figure says on its face that nothing in the repo had ever
timed a provider. `exp_clock_channel_gpu` times one: 165 cells of real batch-1..64
decode on an H100 PCIe, two stack modes, real bitsandbytes NF4 weights, and real
greedy speculation. This figure is that measurement, and it moves the channel's
headline row.

  docs/figures/fig_clock_measured.png
      (A) THE FLOOR IS NOT THE ROOFLINE. Observed ms/token against bytes read. The
          weight term is near-bandwidth-bound (BW_eff ~ 68% of this card's measured
          copy bandwidth) but sits on a large ADDITIVE STACK CONSTANT, and in an
          unoptimized stack the constant is the whole observable: eager HF decode is
          flat in bytes, so the clock there reads nothing at all. A client cannot
          know either number, and neither is on a spec sheet.
      (B) THE KV TERM IS 26x OFF THE ROOFLINE, AND PERFECTLY LINEAR. d(ITL)/d(ctx) =
          1.65 us per token of context on three model sizes, against a roofline of
          0.062 -- single-query attention runs at ~4% of peak bandwidth. It is
          reproducible to +-0.1 ms, and it is IDENTICAL for bf16 and NF4 weights, so
          the context channel and the weight channel are orthogonal by measurement.
      (C) THE BYTE-RATIO PREDICTION FAILS ON THE HEADLINE ROW. Real NF4 weights read
          ~3.4x fewer bytes and arrive 1.10x earlier, not 3.4x: dequantization spends
          most of the saving. `fig_clock_basic` panel 4 prices that row at 6 tokens
          of stream off a 3.9x ratio; the measured ratio makes it a different row.
      (D) AND IT IS GONE UNDER BATCHING, MEASURED. Serving 0.6B as 1.7B is a 1.20x
          latency gap at B=1 and 1.03x at B=64. What survives B is the per-request
          KV read, which is why (B) is the channel and (A) is not.
      (E) THE MINIMUM STATISTIC DOES NOT SURVIVE AN HONEST SERVER. Real greedy
          speculation is distribution-EXACT -- an honest provider -- and it emits
          tokens in accepted blocks, so 73% of client-visible gaps are 0 ms, i.e.
          inside the region `fig_clock_basic` panel 3 labels IMPOSSIBLE.
      (F) THE DIFFERENTIAL VERIFIER, PRICED ON THE REPO'S OWN AXIS. The paired
          statistic D = ITL(ctx_hi) - ITL(ctx_lo) cancels the stack constant of (A)
          exactly and inherits the reproducibility of (B). Against an fp8 KV cache
          (half the slope) and a context-truncating provider (no slope), swept over
          client-side jitter sigma, and compared with what the returned-token
          channel charges for the same two deviations.

MEASURED INPUTS   docs/results/clock_channel.json -- every latency number, the card's
    own copy bandwidth, and the speculation arm. docs/results/cost_of_a_verdict.json
    -- delta* = 3.767 and the token-channel prices in (F), so both channels are
    priced by one law.

MODELLED INPUTS   exactly one, and it is swept: the client-side jitter sigma in (F).
    The device-side sigma is measured (~0.1 ms) and is a LOWER bound on a real
    client's, so (F) sweeps 0.1 -> 100 ms rather than assuming a value.

    python -m experiments.plot_clock_measured
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.plot_clock_channel_principle import (
    DEV, FIG_DIR, GRID, HONEST, INK, INK2, MUTED, WARN,
)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results"
SRC = RES / "clock_channel.json"
COST = RES / "cost_of_a_verdict.json"

BLUE2, GREEN = "#7aa8e0", "#2f7d5c"
SIGMAS = np.logspace(-1, 2, 60)          # client-side jitter swept, ms
CTX_LO, CTX_HI = 256, 32768


# ------------------------------------------------------------------------- data
def load():
    d = json.load(open(SRC))
    cells = list(d["cells"])
    arch = RES / "clock_channel_arch.json"
    if arch.exists():                       # the out-of-sample KV-geometry arm
        have = {(c["label"], c["mode"], c["B"], c["ctx"]) for c in cells}
        cells += [c for c in json.load(open(arch))["cells"]
                  if (c["label"], c["mode"], c["B"], c["ctx"]) not in have]
    d["cells"] = cells
    for c in cells:
        c["key"] = (c["label"], c["mode"], c["B"], c["ctx"])
    return d, {c["key"]: c for c in cells}


def series(idx, label, mode, B, ctxs):
    """(ctx, mean_ms, sd_ms) triples that exist for this config."""
    out = [(c, idx[(label, mode, B, c)]["mean"], idx[(label, mode, B, c)]["sd"])
           for c in ctxs if (label, mode, B, c) in idx]
    return np.array(out).T if out else np.empty((3, 0))


def fit(x, y):
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, np.asarray(y, float), rcond=None)
    resid = np.asarray(y, float) - A @ coef
    return float(coef[0]), float(coef[1]), float(np.std(resid))


def price(d_prime):
    ds = json.load(open(COST))["delta_star"]
    return np.where(np.asarray(d_prime) > 0, (ds / np.maximum(d_prime, 1e-12)) ** 2,
                    np.inf)


def kv_slope(idx, label, mode="graph", B=1):
    """Measured d(ITL)/d(ctx) in ms per token of context, and the fit residual."""
    ctxs = [256, 1024, 4096, 8192, 16384, 32768]
    s = series(idx, label, mode, B, ctxs)
    if s.shape[1] < 3:
        return None
    a, b, r = fit(s[0], s[1])
    return {"intercept": a, "slope": b, "resid": r, "ctx": s[0], "itl": s[1],
            "sd": float(np.mean(s[2]))}


# ============================================================================ (A)
def panel_a(ax, d, idx):
    """The WEIGHT term alone: ITL at the shortest context, where the KV read is
    <1% of the bytes, so the x-axis is essentially the weight read."""
    bw = d["card"]["bw_copy_tb_s"] * 1e12
    labs = ("Qwen3-0.6B", "Qwen3-1.7B-NF4", "Qwen3-1.7B", "Qwen3-4B")
    for mode, col, ms in (("graph", DEV, 9.0), ("eager", MUTED, 8.0)):
        xs, ys = [], []
        for lab in labs:
            k = idx.get((lab, mode, 1, 256))
            if k:
                xs.append(k["bytes_read"] / 1e9)
                ys.append(k["mean"])
                ax.annotate(lab.replace("Qwen3-", ""), (xs[-1], ys[-1]), xytext=(0, 8),
                            textcoords="offset points", ha="center", fontsize=7.8,
                            color=col, zorder=6)
        ax.plot(xs, ys, "o", ms=ms, color=col, mec="white", mew=1.0, zorder=5)
        if mode == "graph":
            gx, gy = np.array(xs), np.array(ys)

    gb = np.linspace(0, 9.0, 50)
    ax.plot(gb, gb * 1e9 / bw * 1e3, "-", color=WARN, lw=2.4, zorder=4)
    ax.annotate(f"the roofline: bytes / {bw / 1e12:.2f} TB/s, measured on this card",
                (5.6, 5.6 * 1e9 / bw * 1e3), xytext=(0, 7),
                textcoords="offset points", ha="center", va="bottom", color=WARN,
                fontsize=8.4, rotation=6, zorder=6)

    keep = [i for i, l in enumerate(labs) if l != "Qwen3-1.7B-NF4"]
    a, b, _ = fit(gx[keep], gy[keep])
    ax.plot(gb, a + b * gb, "--", color=DEV, lw=1.9, zorder=4)
    ax.annotate(f"$\\bf{{observed}}$ = {a:.2f} ms  +  bytes / {1 / b:.2f} TB/s\n"
                f"the weight read itself runs at {(1 / b) / (bw / 1e12):.0%} of copy\n"
                f"bandwidth, but sits on a {a:.2f} ms stack constant --\n"
                "and a client can see neither term separately",
                (0.15, a + 6.0), ha="left", va="bottom", color=DEV, fontsize=8.6,
                linespacing=1.6, zorder=6)
    ey = np.mean([idx[(l, "eager", 1, 256)]["mean"] for l in labs
                  if (l, "eager", 1, 256) in idx])
    ax.annotate(f"$\\bf{{eager\\ HF}}$: {ey:.0f} ms per token and FLAT in bytes.\n"
                "the same GPU, the same weights, no clock channel at all",
                (0.15, ey * 1.22), ha="left", va="bottom", color=INK2, fontsize=8.6,
                linespacing=1.55, zorder=6)
    ax.set_ylim(0, ey * 1.45)
    ax.set_xlabel("weight bytes read per decode step  (GB)", fontsize=9.2, color=INK2)
    ax.set_ylabel("observed ms per output token", fontsize=9.2, color=INK2)
    ax.set_title("(A)  The floor is not the roofline: a stack constant, then a slope",
                 fontsize=10.6, weight="bold", color=INK, loc="left")


# ============================================================================ (B)
KV_GEOM = {                     # layers, KV heads, head_dim -- from each model's config
    "Qwen3-0.6B": (28, 8, 128), "Qwen3-1.7B": (28, 8, 128), "Qwen3-4B": (36, 8, 128),
    "Qwen3-1.7B-NF4": (28, 8, 128), "Qwen2.5-1.5B": (28, 2, 128),
    "SmolLM2-1.7B": (24, 32, 64),
}


def kv_bytes(lab):
    L, h, hd = KV_GEOM[lab]
    return 2 * L * h * hd * 2


def panel_b(ax, d, idx):
    bw = d["card"]["bw_copy_tb_s"] * 1e12
    for lab, col, ls in (("Qwen3-4B", GREEN, "-"), ("Qwen3-1.7B", DEV, "-"),
                         ("Qwen3-1.7B-NF4", WARN, "--"), ("Qwen2.5-1.5B", HONEST, "-")):
        f = kv_slope(idx, lab)
        if not f:
            continue
        ax.plot(f["ctx"], f["itl"], ls, marker="o", ms=4.4, color=col, lw=1.8,
                mec="white", mew=0.7, zorder=4)
        dy = {"Qwen3-4B": 7, "Qwen3-1.7B": 20, "Qwen3-1.7B-NF4": 7,
              "Qwen2.5-1.5B": -16}[lab]
        tail = "   (= bf16 to 0.4%: context and weight are orthogonal)" \
            if lab == "Qwen3-1.7B-NF4" else ""
        ax.annotate(f"{lab}  {f['slope'] * 1e3:.2f} $\\mu$s/tok{tail}",
                    (f["ctx"][-1], f["itl"][-1]), xytext=(-4, dy),
                    textcoords="offset points", ha="right", va="bottom", color=col,
                    fontsize=8.2, zorder=6)
    f = kv_slope(idx, "Qwen3-1.7B")
    roof = kv_bytes("Qwen3-1.7B") / bw * 1e3
    ax.plot(f["ctx"], f["itl"][0] + roof * (f["ctx"] - f["ctx"][0]), ":", color=WARN,
            lw=2.2, zorder=5)
    ax.annotate(f"the roofline: {roof * 1e3:.3f} $\\mu$s/tok\n"
                f"({f['slope'] / roof:.0f}x flatter than measured --\n"
                f"single-query attention runs at {roof / f['slope']:.0%} of peak)",
                (f["ctx"][-1] * 0.99, 9.0), ha="right", va="center", color=WARN,
                fontsize=8.2, linespacing=1.5, zorder=6)

    ins = ax.inset_axes([0.07, 0.585, 0.40, 0.355])
    xs, ys = [], []
    for lab in KV_GEOM:
        f = kv_slope(idx, lab)
        if not f:
            continue
        xs.append(kv_bytes(lab) / 1e3)
        ys.append(f["slope"] * 1e3)
        ins.annotate(lab.replace("Qwen", "Q").replace("SmolLM2", "Smol")
                     .replace("-NF4", "*"), (xs[-1], ys[-1]), xytext=(0, 5),
                     textcoords="offset points", ha="center", fontsize=6.4,
                     color=INK2)
    ins.plot(xs, ys, "o", ms=5.5, color=DEV, mec="white", mew=0.8, zorder=5)
    xr = np.linspace(min(xs) * 0.8, max(xs) * 1.1, 20)
    ins.plot(xr, np.array(ys).mean() * xr / np.mean(xs), "--", color=WARN, lw=1.4,
             zorder=4)
    ins.annotate("if it were bytes", (xr[-1], np.array(ys).mean() * xr[-1] / np.mean(xs)),
                 xytext=(-2, -2), textcoords="offset points", ha="right", va="top",
                 fontsize=6.4, color=WARN)
    ins.set_xlim(0, max(xs) * 1.15)
    ins.set_ylim(0, max(ys) * 1.35)
    ins.set_xlabel("KV bytes per token (kB)", fontsize=6.8, color=INK2, labelpad=1)
    ins.set_ylabel("slope $\\mu$s/tok", fontsize=6.8, color=INK2, labelpad=1)
    ins.tick_params(labelsize=6.2, colors=INK2, pad=1)
    f28 = [l for l in KV_GEOM if KV_GEOM[l][0] == 28 and l != "Qwen3-1.7B-NF4"
           and kv_slope(idx, l)]
    k28 = [kv_bytes(l) for l in f28]
    s28 = [kv_slope(idx, l)["slope"] for l in f28]
    ins.set_title(f"at a fixed 28 layers: {max(k28) / min(k28):.1f}x the KV bytes,\n"
                  f"{max(s28) / min(s28):.2f}x the slope", fontsize=7.2, color=INK,
                  weight="bold", pad=3, linespacing=1.4)
    for sp in ("top", "right"):
        ins.spines[sp].set_visible(False)

    ax.set_xlabel("context length (tokens)", fontsize=9.2, color=INK2)
    ax.set_ylabel("observed ms per output token", fontsize=9.2, color=INK2)
    ax.set_title("(B)  The context slope reads POSITIONS, not bytes",
                 fontsize=10.6, weight="bold", color=INK, loc="left")


# ============================================================================ (C)
def panel_c(ax, d, idx):
    """Predicted (byte ratio) vs measured (time ratio) speedup, per deviation, and the
    fraction of the predicted TIME SAVING that actually materialised."""
    rows = []

    def pair(dev_lab, hon_lab, name, ctx=256, B=1, mode="graph"):
        a, b = idx.get((dev_lab, mode, B, ctx)), idx.get((hon_lab, mode, B, ctx))
        if a and b:
            pred = b["bytes_read"] / a["bytes_read"]
            rows.append((name, pred, b["mean"] / a["mean"],
                         (b["mean"] - a["mean"]) / (b["mean"] - b["mean"] / pred)))

    pair("Qwen3-1.7B-NF4", "Qwen3-1.7B", "real NF4 weights\n(1.7B, 256 ctx)")
    pair("Qwen3-0.6B", "Qwen3-1.7B", "0.6B served\nas 1.7B")
    pair("Qwen3-1.7B", "Qwen3-4B", "1.7B served\nas 4B")
    a, b = idx.get(("Qwen3-1.7B", "graph", 1, 16384)), idx.get(("Qwen3-1.7B", "graph", 1, 32768))
    if a and b:                       # half the context attended, at 32k claimed
        pred = b["bytes_read"] / a["bytes_read"]
        rows.append(("half the context\nattended (32k->16k)", pred, b["mean"] / a["mean"],
                     (b["mean"] - a["mean"]) / (b["mean"] - b["mean"] / pred)))

    y = np.arange(len(rows))[::-1]
    h = 0.34
    for (name, pred, meas, frac), yy in zip(rows, y):
        ax.barh(yy + h / 2, pred, height=h, color=MUTED, alpha=0.55, zorder=3)
        ax.barh(yy - h / 2, meas, height=h, color=DEV, zorder=4)
        ax.annotate(f"{pred:.2f}x", (pred, yy + h / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8.4, color=INK2)
        ax.annotate(f"$\\bf{{{meas:.2f}x}}$", (meas, yy - h / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8.8, color=DEV)
        col = GREEN if frac >= 1.0 else WARN
        ax.annotate(f"{frac:.0%} of the predicted\ntime saving arrived",
                    (3.55, yy), va="center", ha="left", fontsize=8.2, color=col,
                    linespacing=1.45)
    ax.axvline(1.0, color=INK2, lw=1.0, ls=":", zorder=2)
    ax.annotate("Dequantization spends the bytes it saves, so both weight rows "
                "under-deliver. Dropping\nhalf the CONTEXT over-delivers, because "
                "attention is the inefficient part of the read.\nA genuine fp8 KV "
                "cache is NOT this row -- it keeps the positions, and (B) shows the\n"
                "slope barely reads bytes.",
                (0.0, -0.20), xycoords="axes fraction", va="top", ha="left",
                fontsize=8.2, color=INK2, linespacing=1.55)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.6)
    ax.set_xlim(0, 5.6)
    ax.set_xlabel("speedup:  grey = bytes saved (predicted),  blue = time saved "
                  "(measured)", fontsize=9.0, color=INK2)
    ax.set_title("(C)  Fewer bytes is not proportionally less time",
                 fontsize=10.6, weight="bold", color=INK, loc="left")


# ============================================================================ (D)
def panel_d(ax, idx):
    Bs = [1, 2, 4, 8, 16, 32, 64]
    got = {}
    for lab, col in (("Qwen3-1.7B", DEV), ("Qwen3-0.6B", HONEST),
                     ("Qwen3-1.7B-NF4", WARN)):
        xs = [(B, idx[(lab, "graph", B, 1024)]["mean"], idx[(lab, "graph", B, 1024)]["sd"])
              for B in Bs if (lab, "graph", B, 1024) in idx]
        if not xs:
            continue
        a = np.array(xs).T
        got[lab] = a
        ax.plot(a[0], a[1], "-o", ms=4.4, color=col, lw=1.8, mec="white", mew=0.7,
                zorder=4)
        ax.annotate(lab, (a[0][-1], a[1][-1]), xytext=(-3, 7), textcoords="offset points",
                    ha="right", fontsize=8.4, color=col, zorder=6)
    if "Qwen3-1.7B" in got and "Qwen3-0.6B" in got:
        h, s = got["Qwen3-1.7B"], got["Qwen3-0.6B"]
        n = min(h.shape[1], s.shape[1])
        gap, sd = h[1][:n] - s[1][:n], np.sqrt(h[2][:n] ** 2 + s[2][:n] ** 2)
        for i, (dx, dy) in ((0, (14, 30)), (n - 1, (-6, -60))):
            dp = gap[i] / max(sd[i], 1e-9)
            tk = price(dp)
            ax.annotate(f"B={int(h[0][i])}:  gap {gap[i]:.2f} ms  "
                        f"({h[1][i] / s[1][i]:.2f}x)\n$d'$ = {dp:.1f} device-side  "
                        f"->  {'<1' if tk < 1 else f'{tk:.0f}'} token(s)",
                        (h[0][i], h[1][i]), xytext=dx if isinstance(dx, tuple) else (dx, dy),
                        textcoords="offset points",
                        ha="left" if i == 0 else "right", va="center", fontsize=8.2,
                        color=INK2, linespacing=1.5,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        ax.annotate("the RELATIVE gap decays 1.20x -> 1.03x,\n"
                    f"but the ABSOLUTE gap does not: {gap[0]:.2f} -> {gap[n-1]:.2f} ms.\n"
                    "batching does not amortize the weight read\n"
                    "out of a client's own inter-token time --\n"
                    "it buries it in other tenants' work, and in\n"
                    "noise that grows with B.",
                    (0.02, 0.72), xycoords="axes fraction", ha="left", va="top",
                    fontsize=8.2, color=INK2, linespacing=1.6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("concurrent requests B  (continuous batching)", fontsize=9.2,
                  color=INK2)
    ax.set_ylabel("ms per output token, per request", fontsize=9.2, color=INK2)
    ax.set_title("(D)  Batching hides the weight signal in noise, not in arithmetic",
                 fontsize=10.6, weight="bold", color=INK, loc="left")


# ============================================================================ (E)
def panel_e(ax, d, idx):
    s = d.get("specdec")
    if not s:
        return
    g = np.sort(np.asarray(s["gaps_ms"]))
    ref = idx[("Qwen3-1.7B", "eager", 1, 1024)]["mean"] if \
        ("Qwen3-1.7B", "eager", 1, 1024) in idx else float(np.median(g[g > 0]))
    x = np.linspace(0, max(g.max(), ref * 1.4), 600)
    cdf = np.searchsorted(g, x, side="right") / len(g)
    ax.axvspan(0, ref, color=WARN, alpha=0.10, zorder=1)
    ax.plot(x, cdf, "-", color=DEV, lw=2.6, zorder=5)
    ax.axvline(ref, color=INK2, lw=1.6, ls="--", zorder=4)
    f0 = s["frac_gaps_zero"]
    ax.plot([0], [f0], "o", ms=9, color=WARN, mec="white", mew=1.2, zorder=6)
    ax.annotate(f"$\\bf{{{f0 * 100:.0f}\\%}}$ of the gaps are exactly 0 ms",
                (0.6, f0), xytext=(10, -4), textcoords="offset points", ha="left",
                va="top", color=WARN, fontsize=9.2, zorder=7)
    below = float((g < ref).mean())
    ax.annotate(f"$\\bf{{{below * 100:.0f}\\%}}$ of an HONEST provider's client-visible\n"
                "gaps land inside the region fig_clock_basic (3)\n"
                f"labels IMPOSSIBLE -- {s['emitted_mean']:.2f} of "
                f"{s['emitted_max']} tokens per block,\ngreedy speculation, "
                "distribution-exact, no deviation",
                (ref * 1.9, 0.40), ha="left", va="center", color=WARN, fontsize=8.5,
                linespacing=1.6, zorder=7)
    ax.annotate(f"the same model's honest per-token\nfloor in this stack: {ref:.0f} ms",
                (ref, 0.04), xytext=(8, 0), textcoords="offset points", ha="left",
                va="bottom", color=INK2, fontsize=8.4, linespacing=1.5, zorder=7)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("client-visible inter-token gap (ms)", fontsize=9.2, color=INK2)
    ax.set_ylabel("fraction of tokens with a gap this small", fontsize=9.2, color=INK2)
    ax.set_title("(E)  An honest server already lives in the 'impossible' region",
                 fontsize=10.6, weight="bold", color=INK, loc="left")


# ============================================================================ (F)
def panel_f(ax, d, idx):
    """Price the differential statistic D = ITL(ctx_hi) - ITL(ctx_lo) vs the token
    channel, sweeping the one input that is not measured: client-side jitter."""
    f = kv_slope(idx, "Qwen3-1.7B")
    dctx = CTX_HI - CTX_LO
    e_honest = f["slope"] * dctx                       # ms of separation available
    cases = (("half the context attended\n(half the slope)", 0.5 * e_honest, DEV),
             ("context truncated to 256\n(no slope at all)", e_honest, GREEN))
    for name, sep, col in cases:
        dp = sep / (SIGMAS * np.sqrt(2.0))
        y = 2 * price(dp)
        ax.plot(SIGMAS, y, "-", color=col, lw=2.2, zorder=5)
        tgt, off = (300.0, (-10, 6)) if col == DEV else (12.0, (10, -10))
        j = int(np.argmin(np.abs(y - tgt)))
        ax.annotate(name, (SIGMAS[j], y[j]), xytext=off,
                    textcoords="offset points",
                    ha="right" if off[0] < 0 else "left",
                    va="bottom" if off[1] > 0 else "top", color=col,
                    fontsize=8.5, linespacing=1.5, zorder=6)
    cost = json.load(open(COST))
    for atk, lab in (("kv_fp8", "kv_fp8"), ("quant_4bit", "quant_4bit")):
        row = cost["cells"].get(atk, {})
        best = min((c["tokens_per_verdict"] for v, c in row.items()
                    if v != "activation_difr" and c["reachable"]
                    and c["tokens_per_verdict"] > 0), default=None)
        if best:
            ax.axhline(best, color=MUTED, lw=1.3, ls="--", zorder=3)
            ax.annotate(f"returned-token channel, {lab}: {best:,} tokens",
                        (SIGMAS[-1] * 0.92, best * 1.2), ha="right", va="bottom",
                        color=INK2, fontsize=8.2, zorder=6)
    sd = f["sd"]
    ax.axvline(sd, color=WARN, lw=1.6, zorder=4)
    ax.annotate(f"measured device-side jitter,\n{sd:.2f} ms -- a LOWER bound\n"
                "on a real client's", (sd * 1.25, 2e2), ha="left", va="center", color=WARN,
                fontsize=8.3, linespacing=1.55, zorder=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 1e5)
    ax.set_xlabel("client-side jitter $\\sigma$ on one inter-token gap (ms)",
                  fontsize=9.2, color=INK2)
    ax.set_ylabel("tokens of stream per verdict", fontsize=9.2, color=INK2)
    ax.set_title("(F)  The differential verifier, priced by the repo's cost law",
                 fontsize=10.6, weight="bold", color=INK, loc="left")


# =========================================================================== main
def report(d, idx):
    """The text version, so every number in the figure is quotable from a terminal."""
    bw = d["card"]["bw_copy_tb_s"] * 1e12
    print("=" * 92)
    print(f"The clock, measured.  {d['card']['device']}, copy bandwidth "
          f"{bw / 1e12:.2f} TB/s, launch {d['card']['launch_us']:.1f} us")
    print(f"  {len(d['cells'])} cells, steps={d['steps']} reps={d['reps']}, "
          f"{d['elapsed_s']:.0f}s")
    print("=" * 92)
    print("\nContext slope  d(ITL)/d(ctx), graph mode, B=1.  Does it read bytes?")
    print(f"  {'config':16s} {'layers':>7s} {'KV B/tok':>10s} {'slope us/tok':>13s} "
          f"{'roofline':>9s} {'x roof':>7s} {'ns/tok/layer':>13s} {'resid ms':>9s}")
    for lab in KV_GEOM:
        f = kv_slope(idx, lab)
        if not f:
            continue
        L = KV_GEOM[lab][0]
        roof = kv_bytes(lab) / bw * 1e3
        print(f"  {lab:16s} {L:7d} {kv_bytes(lab):10,d} {f['slope'] * 1e3:13.3f} "
              f"{roof * 1e3:9.3f} {f['slope'] / roof:6.1f}x "
              f"{f['slope'] * 1e6 / L:13.1f} {f['resid']:9.3f}")
    have = [l for l in KV_GEOM if kv_slope(idx, l)]
    kb = [kv_bytes(l) for l in have]
    per = [kv_slope(idx, l)["slope"] * 1e6 / KV_GEOM[l][0] for l in have]
    f28 = [l for l in have if KV_GEOM[l][0] == 28 and l != "Qwen3-1.7B-NF4"]
    k28 = [kv_bytes(l) for l in f28]
    s28 = [kv_slope(idx, l)["slope"] for l in f28]
    print(f"  -> across all {len(have)} configs: KV bytes vary "
          f"{max(kb) / min(kb):.1f}x, slope PER LAYER varies {max(per) / min(per):.2f}x.")
    print(f"  -> at a fixed 28 layers ({', '.join(f28)}): KV bytes vary "
          f"{max(k28) / min(k28):.1f}x, slope varies {max(s28) / min(s28):.2f}x.")
    print("     The context term is per-position and per-layer, not per-byte, so "
          "TRUNCATION is\n     maximally visible and KV PRECISION is nearly invisible "
          "in this stack.")

    print("\nPer-request context slope vs concurrency B (Qwen3-1.7B, 1024 -> 8192 ctx)")
    for B in (1, 2, 4, 8):
        a = idx.get(("Qwen3-1.7B", "graph", B, 1024))
        b2 = idx.get(("Qwen3-1.7B", "graph", B, 8192))
        if a and b2:
            print(f"  B={B:<3d} {(b2['mean'] - a['mean']) / (8192 - 1024) * 1e3:6.3f} "
                  f"us per token of context   (co-tenancy can only RAISE this, which "
                  f"is what makes a floor test one-sided)")

    print("\nWeight term, graph mode, B=1, ctx=256")
    for lab in ("Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-1.7B-NF4"):
        k = idx.get((lab, "graph", 1, 256))
        if k:
            print(f"  {lab:18s} weights {k['weight_bytes'] / 1e9:5.2f} GB   "
                  f"ITL {k['mean']:6.2f} ms   roofline "
                  f"{k['bytes_read'] / bw * 1e3:5.2f} ms   "
                  f"observed/roofline {k['mean'] / (k['bytes_read'] / bw * 1e3):5.1f}x")

    print("\nBatching, graph mode, ctx=1024:  per-request ITL and the deviation gap")
    for B in (1, 2, 4, 8, 16, 32, 64):
        h, s = idx.get(("Qwen3-1.7B", "graph", B, 1024)), idx.get(("Qwen3-0.6B", "graph", B, 1024))
        q = idx.get(("Qwen3-1.7B-NF4", "graph", B, 1024))
        if h and s:
            line = (f"  B={B:<3d} 1.7B {h['mean']:6.2f}  0.6B {s['mean']:6.2f}  "
                    f"gap {h['mean'] - s['mean']:5.2f} ms ({h['mean'] / s['mean']:.3f}x)")
            if q:
                line += f"   NF4 {q['mean']:6.2f} ({h['mean'] / q['mean']:.3f}x)"
            print(line + f"   per-token throughput cost {h['mean'] / B:5.2f} ms")

    s = d.get("specdec")
    if s:
        print(f"\nSpeculation (honest, distribution-exact): "
              f"{s['emitted_mean']:.2f}/{s['emitted_max']} tokens per block, "
              f"{s['frac_gaps_zero']:.0%} of client-visible gaps are 0 ms, "
              f"amortized {s['amortized_ms_per_token']:.2f} ms/token")

    f = kv_slope(idx, "Qwen3-1.7B")
    sep = f["slope"] * (CTX_HI - CTX_LO)
    print(f"\nDifferential statistic D = ITL({CTX_HI}) - ITL({CTX_LO}):  honest "
          f"separation {sep:.1f} ms")
    print(f"  {'sigma (ms)':>11s} {'half ctx: d prime':>18s} {'tokens':>9s} "
          f"{'no ctx: d prime':>16s} {'tokens':>9s}")
    for sg in (f["sd"], 1.0, 10.0, 50.0, 100.0):
        d1, d2 = 0.5 * sep / (sg * np.sqrt(2)), sep / (sg * np.sqrt(2))
        print(f"  {sg:11.2f} {d1:18.2f} {2 * price(d1):9.1f} {d2:16.2f} "
              f"{2 * price(d2):9.1f}")
    cost = json.load(open(COST))
    for atk in ("kv_fp8", "quant_4bit"):
        row = cost["cells"][atk]
        best = min((c["tokens_per_verdict"] for v, c in row.items()
                    if v != "activation_difr" and c["reachable"]
                    and c["tokens_per_verdict"] > 0), default=None)
        print(f"  for comparison, returned-token channel on {atk}: {best:,} tokens")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d, idx = load()
    report(d, idx)

    fig, axes = plt.subplots(2, 3, figsize=(19.0, 10.2))
    for ax in axes.flat:
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8.4, colors=INK2)
    panel_a(axes[0, 0], d, idx)
    panel_b(axes[0, 1], d, idx)
    panel_c(axes[0, 2], d, idx)
    panel_d(axes[1, 0], idx)
    panel_e(axes[1, 1], d, idx)
    panel_f(axes[1, 2], d, idx)

    fig.suptitle("The clock channel, measured: the floor is a stack property, the "
                 "weight row is not the row, and the context slope is",
                 fontsize=15.5, weight="bold", color=INK, x=0.006, ha="left", y=0.988)
    fig.text(0.006, 0.955,
             f"{len(d['cells'])} timing cells on an {d['card']['device']} "
             f"({d['card']['bw_copy_tb_s']:.2f} TB/s measured copy bandwidth) -- "
             "real batch-1..64 decode, two stack modes, real bitsandbytes NF4 weights, "
             "real greedy speculation.  Every latency here is measured; the only "
             "modelled quantity is the client-side jitter in (F), which is swept.",
             fontsize=9.4, color=INK2, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_clock_measured.png"
    fig.savefig(out, dpi=155, facecolor="white")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
