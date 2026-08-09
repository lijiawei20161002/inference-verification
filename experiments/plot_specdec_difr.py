"""Root-cause analysis + figures for DiFR under speculative decoding.

Reads `docs/results/specdec_difr_main.json` and `specdec_difr_temps.json`
(written by `experiments/exp_specdec_difr_gpu.py --out main` / `--out temps`)
and writes `docs/figures/fig_specdec_difr_root_causes.png` and
`fig_specdec_difr_fix.png`.

    python -m experiments.plot_specdec_difr
"""
import json, math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "docs" / "results"
FIG_DIR = ROOT / "docs" / "figures"

# ---- reference palette (validated instance; light mode) ----------------------
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 9,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.titleweight": "600", "axes.titlelocation": "left", "axes.titlepad": 9,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 2.0, "lines.markersize": 5,
})


def style(ax, ygrid=True, xgrid=False, pct=False):
    ax.set_axisbelow(True)
    ax.grid(axis="y" if ygrid else "x", visible=ygrid or xgrid)
    if xgrid:
        ax.grid(axis="x", visible=True)
    if not ygrid:
        ax.grid(axis="y", visible=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if pct:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))


# ---- load -------------------------------------------------------------------
def load(path):
    rows = json.load(open(path))
    stats = {}
    for r in rows:
        if "stats" in r:
            stats[(r["T"], r["regime"], r["prompt"])] = r.pop("stats")
    return rows, stats


MAIN, MSTATS = load(RES_DIR / "specdec_difr_main.json")
TEMPS, TSTATS = load(RES_DIR / "specdec_difr_temps.json")


def sel(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def seqs(rows, regime, T=1.0, field="match"):
    """list of per-prompt arrays of the mismatch indicator (1 = DiFR flags)."""
    out = []
    for p in sorted({r["prompt"] for r in rows}):
        s = sorted(sel(rows, regime=regime, T=T, prompt=p), key=lambda r: r["pos"])
        if s:
            out.append(np.array([1 - r[field] for r in s], dtype=float))
    return out


def rate(rows, regime, T=1.0, field="match"):
    s = seqs(rows, regime, T, field)
    return float(np.mean(np.concatenate(s))) if s else float("nan")


def vif(rows, regime, T=1.0, L=20):
    """variance inflation from block autocorrelation: 1 + 2*sum rho_k."""
    s = seqs(rows, regime, T)
    acf = acorr(s, L)
    tot, k = 1.0, 0
    while k < L and acf[k] > 0:
        tot += 2 * acf[k]
        k += 1
    return tot, acf


def acorr(s, L):
    out = []
    allx = np.concatenate(s)
    mu, var = allx.mean(), allx.var()
    for lag in range(1, L + 1):
        num = np.concatenate([(a[:-lag] - mu) * (a[lag:] - mu) for a in s if len(a) > lag])
        out.append(float(num.mean() / var) if var > 0 else 0.0)
    return np.array(out)


Z99 = 2.32634787


def auc_curve(rows, null_reg, alt_reg, N):
    r0, r1 = rate(rows, null_reg), rate(rows, alt_reg)
    f0, _ = vif(rows, null_reg)
    f1, _ = vif(rows, alt_reg)
    v0, v1 = r0 * (1 - r0) * f0, r1 * (1 - r1) * f1
    d = abs(r1 - r0)
    z = np.sqrt(N) * d / math.sqrt(v0 + v1)
    from math import erf
    auc = np.array([0.5 * (1 + erf(zz / math.sqrt(2))) for zz in z])
    nstar = (Z99 * math.sqrt(v0 + v1) / d) ** 2 if d > 0 else float("inf")
    return auc, nstar, dict(r0=r0, r1=r1, f0=f0, f1=f1)


def emp_auc(rows, null_reg, alt_reg):
    a = np.array([s.sum() for s in seqs(rows, null_reg)])
    b = np.array([s.sum() for s in seqs(rows, alt_reg)])
    if not len(a) or not len(b):
        return float("nan"), 0
    w = sum((1.0 if y > x else 0.5 if y == x else 0.0) for x in a for y in b)
    return w / (len(a) * len(b)), 0


# =============================================================== REPORT
print("=" * 78)
print("DiFR mismatch rate (fraction of tokens where provider != seeded reference)")
print("=" * 78)
REGIMES = ["exact", "spec_standard", "spec_coupled", "spec_typical",
           "q4_exact", "q4_spec_standard", "q4_spec_coupled"]
LBL = {"exact": "honest, exact decode", "spec_standard": "honest, spec (rejection sampling)",
       "spec_coupled": "honest, spec (CRN-coupled)", "spec_typical": "honest, spec (typical accept)",
       "q4_exact": "int4, exact decode", "q4_spec_standard": "int4, spec (rejection sampling)",
       "q4_spec_coupled": "int4, spec (CRN-coupled)"}
tbl = {}
for r in REGIMES:
    cx, sm = rate(MAIN, r), rate(MAIN, r, field="match_same")
    f, _ = vif(MAIN, r)
    tbl[r] = dict(cross=cx, same=sm, vif=f)
    print(f"  {LBL[r]:36s} cross-stack {cx:6.3f}   same-stack {sm:6.3f}   VIF {f:4.2f}")

print("\n" + "=" * 78)
print("Detection of int4 substitution (Token-DiFR statistic = mismatch count)")
print("=" * 78)
PAIRS = [("exact", "q4_exact", "exact decode"),
         ("spec_standard", "q4_spec_standard", "spec, rejection sampling"),
         ("spec_coupled", "q4_spec_coupled", "spec, CRN-coupled")]
NST = {}
for n_, a_, name in PAIRS:
    _, nstar, info = auc_curve(MAIN, n_, a_, np.array([256.0]))
    e, _ = emp_auc(MAIN, n_, a_)
    NST[name] = nstar
    print(f"  {name:26s} r_honest={info['r0']:.3f} r_int4={info['r1']:.3f} "
          f"gap={info['r1']-info['r0']:+.4f}  tokens@AUC0.99={nstar:9.0f}  "
          f"empirical AUC@256tok={e:.3f}")

print("\n" + "=" * 78)
print("Mismatch decomposition inside a speculative round (honest weights)")
print("=" * 78)
ROLES = ["accept", "residual", "bonus"]
role_rate, role_share = {}, {}
for reg in ["spec_standard", "spec_coupled"]:
    rs = sel(MAIN, regime=reg, T=1.0)
    for role in ROLES:
        sub = [r for r in rs if r["role"] == role]
        role_rate[(reg, role)] = np.mean([1 - r["match"] for r in sub]) if sub else np.nan
        role_share[(reg, role)] = len(sub) / len(rs)
    print(f"  {reg}: " + "  ".join(
        f"{ro}={role_rate[(reg,ro)]:.3f} (share {role_share[(reg,ro)]:.2f})" for ro in ROLES))

print("\n" + "=" * 78)
print("Acceptance rate: standard rejection sampling vs CRN-coupled (same positions)")
print("=" * 78)
acc = {}
for reg in ["spec_standard", "spec_coupled", "spec_typical"]:
    ks = [k for k in MSTATS if k[1] == reg and k[0] == 1.0]
    nd = sum(MSTATS[k]["n_drafted"] for k in ks)
    na = sum(MSTATS[k]["n_accepted"] for k in ks)
    smin = np.concatenate([np.array(MSTATS[k]["sum_min"]) for k in ks]) if ks else np.array([np.nan])
    ag = np.concatenate([np.array(MSTATS[k]["coupled_agree"]) for k in ks
                         if MSTATS[k]["coupled_agree"]]) if reg == "spec_coupled" else None
    toks = len(sel(MAIN, regime=reg, T=1.0))
    rounds = sum(MSTATS[k]["n_rounds"] for k in ks)
    acc[reg] = dict(emp=na / nd, smin=float(smin.mean()), tpr=toks / rounds)
    extra = f"  coupled-agree={float(ag.mean()):.3f}" if ag is not None else ""
    print(f"  {reg:14s} accept rate={na/nd:.3f}  E[sum min(p,q)]={smin.mean():.3f}  "
          f"tokens/round={toks/rounds:.2f}{extra}")

print("\n" + "=" * 78)
print("Temperature sweep (mismatch rate)")
print("=" * 78)
TVALS = sorted({r["T"] for r in TEMPS})
tsweep = {}
for reg in ["exact", "spec_standard", "spec_coupled"]:
    tsweep[reg] = [rate(TEMPS, reg, T=t) for t in TVALS]
    print(f"  {reg:14s} " + "  ".join(f"T={t}:{v:.3f}" for t, v in zip(TVALS, tsweep[reg])))

# =============================================================== FIGURE 1
fig, ax = plt.subplots(2, 3, figsize=(13.2, 7.4))
fig.suptitle("Why Token-DiFR fails under speculative decoding — Qwen2.5-1.5B target / 0.5B draft, K=4, T=1.0",
             x=0.011, ha="left", color=INK, fontsize=12, fontweight="700")

# (a) mismatch rate by regime
a = ax[0, 0]
order = ["exact", "q4_exact", "spec_coupled", "spec_typical", "spec_standard", "q4_spec_standard"]
short = {"exact": "honest\nexact", "q4_exact": "int4\nexact", "spec_coupled": "honest\nspec-CRN",
         "spec_typical": "honest\nspec-typical", "spec_standard": "honest\nspec-RS",
         "q4_spec_standard": "int4\nspec-RS"}
vals = [tbl[r]["cross"] for r in order]
cols = [S1 if not r.startswith("q4") else S2 for r in order]
b = a.bar(range(len(order)), vals, color=cols, width=0.62,
          edgecolor=SURFACE, linewidth=2)
for i, v in enumerate(vals):
    a.text(i, v + 0.015, f"{v:.1%}", ha="center", color=INK, fontsize=8, fontweight="600")
a.set_xticks(range(len(order)), [short[r] for r in order], fontsize=7.5)
a.set_ylim(0, max(vals) * 1.22)
a.set_ylabel("DiFR mismatch rate")
a.set_title("a  Honest speculative decoding looks worse\n   than a substituted model")
style(a, pct=True)
a.legend(handles=[plt.Line2D([], [], marker="s", ls="", color=S1, label="honest fp16 weights"),
                  plt.Line2D([], [], marker="s", ls="", color=S2, label="int4 weights")],
         loc="upper left", fontsize=7.5)
a.axhline(tbl["q4_exact"]["cross"], color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)

# (b) temperature sweep
a = ax[0, 1]
for reg, c, m in [("exact", S1, "o"), ("spec_standard", S2, "s"), ("spec_coupled", S3, "^")]:
    a.plot(TVALS, tsweep[reg], color=c, marker=m, mec=SURFACE, mew=1.2, label=LBL[reg])
a.set_xlabel("sampling temperature")
a.set_ylabel("DiFR mismatch rate")
a.set_title("b  Root cause is the sampling channel:\n   the gap vanishes as T → 0")
style(a, pct=True)
a.legend(loc="upper left")

# (c) match vs top-1 prob, with collision-probability theory
a = ax[0, 2]
edges = np.array([0, .2, .4, .6, .8, .9, .95, .99, 1.0])
ctr = (edges[:-1] + edges[1:]) / 2
for reg, c, m in [("exact", S1, "o"), ("spec_standard", S2, "s")]:
    rs = sel(MAIN, regime=reg, T=1.0)
    t1 = np.array([r["top1"] for r in rs]); mt = np.array([r["match"] for r in rs])
    y = [mt[(t1 >= lo) & (t1 < hi)].mean() if ((t1 >= lo) & (t1 < hi)).sum() > 20 else np.nan
         for lo, hi in zip(edges[:-1], edges[1:])]
    a.plot(ctr, y, color=c, marker=m, mec=SURFACE, mew=1.2, label=LBL[reg])
rs = sel(MAIN, regime="spec_standard", T=1.0)
t1 = np.array([r["top1"] for r in rs]); co = np.array([r["coll"] for r in rs])
yth = [co[(t1 >= lo) & (t1 < hi)].mean() if ((t1 >= lo) & (t1 < hi)).sum() > 20 else np.nan
       for lo, hi in zip(edges[:-1], edges[1:])]
a.plot(ctr, yth, color=INK2, ls=(0, (4, 3)), lw=1.4, label=r"theory: $\sum_x p(x)^2$")
a.set_xlabel("reference top-1 probability at that position")
a.set_ylabel("DiFR match rate")
a.set_title("c  Under spec-RS the provider's token is an\n   independent draw from p, not the seeded one")
style(a, pct=True)
a.legend(loc="upper left")

# (d) role decomposition
a = ax[1, 0]
xs = np.arange(len(ROLES)); w = 0.36
for i, (reg, c) in enumerate([("spec_standard", S2), ("spec_coupled", S3)]):
    v = [role_rate[(reg, ro)] for ro in ROLES]
    a.bar(xs + (i - 0.5) * w, v, w, color=c, edgecolor=SURFACE, linewidth=2,
          label=LBL[reg])
    for x, vv in zip(xs + (i - 0.5) * w, v):
        a.text(x, vv + 0.015, f"{vv:.0%}", ha="center", color=INK, fontsize=7.5, fontweight="600")
a.set_xticks(xs, [f"accepted draft\n(share {role_share[('spec_standard',ROLES[0])]:.0%})",
                  f"rejection resample\n(share {role_share[('spec_standard',ROLES[1])]:.0%})",
                  f"bonus token\n(share {role_share[('spec_standard',ROLES[2])]:.0%})"], fontsize=7.5)
a.set_ylabel("DiFR mismatch rate")
a.set_title("d  Every slot of the round is affected —\n   worst at the rejection-resample position")
style(a, pct=True)
a.legend(loc="upper left")

# (e) autocorrelation
a = ax[1, 1]
for reg, c, m in [("exact", S1, "o"), ("spec_standard", S2, "s")]:
    f, acf = vif(MAIN, reg)
    a.plot(range(1, len(acf) + 1), acf, color=c, marker=m, mec=SURFACE, mew=1.2,
           label=f"{LBL[reg]}  (VIF {f:.2f})")
a.axhline(0, color=BASE, lw=0.9)
a.set_xlabel("lag (tokens)")
a.set_ylabel("autocorrelation of mismatch indicator")
a.set_title("e  Mismatches arrive in blocks, so DiFR's\n   per-token variance is understated")
style(a)
a.legend(loc="upper right")

# (f) AUC vs tokens
a = ax[1, 2]
N = np.unique(np.round(np.logspace(1, 6, 120)))
for (n_, a_, name), c, in zip(PAIRS, [S1, S2, S3]):
    auc, nstar, _ = auc_curve(MAIN, n_, a_, N)
    a.plot(N, auc, color=c, label=f"{name}\n  {nstar:,.0f} tokens to AUC 0.99")
    a.plot([nstar], [0.99], marker="o", color=c, mec=SURFACE, mew=1.4, ms=7)
a.axhline(0.99, color=MUTED, lw=0.9, ls=(0, (4, 3)))
a.set_xscale("log"); a.set_xlim(10, 1e6); a.set_ylim(0.45, 1.02)
a.set_xlabel("audited output tokens")
a.set_ylabel("AUC, detecting int4 substitution")
a.set_title("f  Cost of the blind spot: power to catch int4\n   collapses, and is restored by CRN coupling")
style(a)
a.legend(loc="lower right", fontsize=7, bbox_to_anchor=(1.02, -0.02))

fig.tight_layout(rect=(0, 0.005, 1, 0.945))
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / "fig_specdec_difr_root_causes.png", dpi=180)
print("\nwrote fig_specdec_difr_root_causes.png")

# =============================================================== FIGURE 2
fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.0))
fig.suptitle("Consequences and the fix", x=0.011, ha="left", color=INK,
             fontsize=12, fontweight="700")

# (a) throughput cost of CRN coupling
a = ax[0]
names = ["spec_standard", "spec_coupled"]
v = [acc[n]["emp"] for n in names]
tp = [acc[n]["tpr"] for n in names]
a.bar([0, 1], v, 0.5, color=[S2, S3], edgecolor=SURFACE, linewidth=2)
for i, (vv, tt) in enumerate(zip(v, tp)):
    a.text(i, vv + 0.012, f"{vv:.1%}\n{tt:.2f} tok/round", ha="center", color=INK,
           fontsize=8, fontweight="600")
smin = acc["spec_standard"]["smin"]
a.set_ylim(0, smin * 1.22)
a.axhline(smin, color=MUTED, lw=1.0, ls=(0, (4, 3)))
a.text(0.5, smin - 0.02, r"optimal coupling bound  $\sum_x\min(p,q)$",
       va="top", ha="center", color=INK2, fontsize=7.5)
a.set_xticks([0, 1], ["rejection sampling\n(unverifiable)", "CRN-coupled\n(DiFR-verifiable)"], fontsize=8)
a.set_xlim(-0.6, 1.6)
a.set_ylabel("draft-token acceptance rate")
a.set_title("a  Verifiability costs acceptance rate,\n   not correctness")
style(a, pct=True)

# (b) confounding: lossy accept rule vs int4
a = ax[1]
cand = ["exact", "spec_typical", "q4_exact"]
lab = ["honest\nexact decode", "honest weights\ntypical acceptance", "int4 weights\nexact decode"]
v = [tbl[c]["cross"] for c in cand]
a.bar(range(3), v, 0.5, color=[S1, S3, S2], edgecolor=SURFACE, linewidth=2)
for i, vv in enumerate(v):
    a.text(i, vv + 0.004, f"{vv:.1%}", ha="center", color=INK, fontsize=8, fontweight="600")
a.set_xticks(range(3), lab, fontsize=8)
a.set_ylabel("DiFR mismatch rate")
a.set_ylim(0, max(v) * 1.3)
a.set_title("b  A lossy accept rule is not separable\n   from weight substitution")
style(a, pct=True)

# (c) tokens to detect
a = ax[2]
ks = list(NST)
v = [NST[k] for k in ks]
XLO = 10
a.barh(range(len(ks)), [vv - XLO for vv in v], 0.5, color=[S1, S2, S3],
       edgecolor=SURFACE, linewidth=2, left=XLO)
for i, vv in enumerate(v):
    a.text(vv * 1.15, i, f"{vv:,.0f}", va="center", color=INK, fontsize=8, fontweight="600")
a.set_yticks(range(len(ks)), ks, fontsize=8)
a.invert_yaxis()
a.set_xscale("log")
a.set_xlim(XLO, max(v) * 6)
a.set_xlabel("output tokens needed to reach AUC 0.99 on int4")
a.set_title("c  Audit cost per provider")
style(a, ygrid=False, xgrid=True)

fig.tight_layout(rect=(0, 0.005, 1, 0.9))
fig.savefig(FIG_DIR / "fig_specdec_difr_fix.png", dpi=180)
print("wrote fig_specdec_difr_fix.png")
