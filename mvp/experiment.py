"""Measure the protocol on the real model. No estimates -- every FLOP count and
detection outcome comes from actually running the transformer, committing, and
auditing.

Phases:
  1. Physics:   real gen/verify FLOPs; sweep tolerance eps -> (false-positive
                rate alpha, cheat-miss rate delta) from real audits.
  2. End-to-end: full pipeline (serve -> Merkle commit -> random audit ->
                re-score + path-verify) over T trials; empirical catch prob and
                real client FLOPs vs the closed form 1-(1-q(1-delta))^{fN}.
  3. Invariant: audit count k/f is N-independent (verified by simulation driven
                by the *measured* per-cheat detection prob).
  4. Deliverable: client FLOP cost for target (f, P) using real per-req FLOPs.
"""
import time
import numpy as np
from mvp import model as M
from mvp import protocol as P
from mvp.commit import MerkleTree

np.set_printoptions(precision=4, suppress=True)

cfg = M.Config(d_model=128, n_head=4, n_layer=6, d_ff=512, vocab=256, max_ctx=256)
mdl = M.Model(cfg, seed=1)
L, T = 16, 32
QUANT = 1e-3
SIGMA = 0.01            # hardware nondeterminism std on the client re-score
EPS = 0.10             # tolerance band (chosen in Phase 1)

def make_pool(n_prompts, seed):
    rng = np.random.default_rng(seed)
    honest, cheat = [], []
    for _ in range(n_prompts):
        prompt = rng.integers(0, cfg.vocab, size=L)
        honest.append(P.serve(mdl, prompt, T, cheat=False))
        cheat.append(P.serve(mdl, prompt, T, cheat=True))
    return honest, cheat

# ---------------------------------------------------------------------------
print("=" * 74)
print("PHASE 1  --  real per-request FLOPs and tolerance calibration")
print("=" * 74)
t0 = time.time()
NP = 200
honest, cheat = make_pool(NP, seed=7)
gen_full = np.mean([t.gen_flops for t in honest])
gen_cheat = np.mean([t.gen_flops for t in cheat])
# one verify to read verify FLOPs
tree0 = P.commit(honest[:2])
rng = np.random.default_rng(0)
_, _, _, _, ver_flops = P.audit_one(mdl, honest[0], tree0, 0, tree0.root, EPS, SIGMA, rng, QUANT)
print(f"pool built in {time.time()-t0:.1f}s  ({NP} honest + {NP} cheat transcripts, L={L} T={T})")
print(f"gen FLOPs / request (honest, full {cfg.n_layer} layers) : {gen_full:,.0f}")
print(f"gen FLOPs / request (cheat, {cfg.n_layer//2} layers)      : {gen_cheat:,.0f}"
      f"   ({gen_cheat/gen_full:.2%} of full  -> provider saves {1-gen_cheat/gen_full:.1%})")
print(f"verify FLOPs / request (teacher-forced)     : {ver_flops:,.0f}")
print(f"beta = verify/gen (FLOP ratio)              : {ver_flops/gen_full:.3f}")

# sweep eps: measure alpha (honest flagged) and delta (cheat missed), real audits
print("\n  eps sweep (real audits, sigma={:.3f}, quant={:.0e}):".format(SIGMA, QUANT))
print(f"  {'eps':>8} {'alpha(FP)':>12} {'delta(miss)':>12} {'honest linf':>13} {'cheat linf':>12}")
rng = np.random.default_rng(123)
NA = 200
h_linf, c_linf, c_tokmis = [], [], []
for i in range(NA):
    tr = P.commit([honest[i]]);
    _, _, lh, _, _ = P.audit_one(mdl, honest[i], tr, 0, tr.root, 1e9, SIGMA, rng, QUANT)
    tr = P.commit([cheat[i]])
    _, _, lc, tm, _ = P.audit_one(mdl, cheat[i], tr, 0, tr.root, 1e9, SIGMA, rng, QUANT)
    h_linf.append(lh); c_linf.append(lc); c_tokmis.append(tm)
h_linf = np.array(h_linf); c_linf = np.array(c_linf)
eps_grid, eps_alpha, eps_delta = [], [], []
for eps in [0.02, 0.05, 0.10, 0.20, 0.50]:
    alpha = np.mean(h_linf > eps)
    delta = np.mean(c_linf <= eps)
    eps_grid.append(eps); eps_alpha.append(float(alpha)); eps_delta.append(float(delta))
    print(f"  {eps:>8.2f} {alpha:>12.3f} {delta:>12.3f} {h_linf.mean():>13.4f} {c_linf.mean():>12.2f}")
DELTA = float(np.mean(c_linf <= EPS))
ALPHA = float(np.mean(h_linf > EPS))
print(f"\n  operating point eps={EPS}:  measured cheat-detection (1-delta)={1-DELTA:.4f}, "
      f"false-positive alpha={ALPHA:.4f}")
print(f"  (cheat transcripts also show token mismatch in {np.mean(c_tokmis):.0%} of cases)")

# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("PHASE 2  --  full end-to-end pipeline: catch probability vs theory")
print("=" * 74)

def run_batch(N, f, q, trials, seed, cache):
    """Every audited request is really re-scored + Merkle-path-verified.

    `cache` memoizes the deterministic re-score per pool transcript so the
    forward pass isn't repeated across trials; the Merkle proof check and the
    hardware-noise draw still run for real on every audit."""
    rng = np.random.default_rng(seed)
    n_cheat = max(1, round(f * N))
    caught = 0; false_alarms = 0; client_flops = 0; binding_fail = 0
    total_audits = 0
    for _ in range(trials):
        # build a batch: pick transcripts from the pool; mark n_cheat as cheats
        idx_h = rng.integers(0, NP, size=N)
        cheat_pos = set(rng.choice(N, size=n_cheat, replace=False).tolist())
        batch = []
        for j in range(N):
            src = cheat if j in cheat_pos else honest
            batch.append(src[idx_h[j]])
        tree = P.commit(batch)
        # client audits each request independently w.p. q (private randomness,
        # chosen AFTER the root is fixed)
        audit_mask = rng.random(N) < q
        trial_caught = False
        for j in np.nonzero(audit_mask)[0]:
            total_audits += 1
            ok, flagged, linf, tm, vf = P.audit_one(
                mdl, batch[j], tree, j, tree.root, EPS, SIGMA, rng, QUANT, cache)
            client_flops += vf
            if not ok: binding_fail += 1
            if flagged:
                if batch[j].cheated: trial_caught = True
                else: false_alarms += 1
        if trial_caught: caught += 1
    return dict(caught=caught, trials=trials, P_emp=caught/trials,
                false_alarms=false_alarms, client_flops=client_flops,
                binding_fail=binding_fail, n_cheat=n_cheat, total_audits=total_audits)

N, f = 500, 0.02
n_cheat = round(f * N)
print(f"config: N={N} requests/batch, f={f} ({n_cheat} cheats/batch), "
      f"eps={EPS}, sigma={SIGMA}")
print(f"theory: P = 1-(1-q(1-delta))^(fN),  fN={n_cheat}, (1-delta)={1-DELTA:.3f}\n")
print(f"  {'q':>7} {'audits/batch':>13} {'P_empirical':>13} {'P_theory':>11} "
      f"{'false_alarms':>13} {'binding_fail':>13}")
t1 = time.time()
rescore_cache = {}                # deterministic re-scores, shared across trials
phase2_q, phase2_emp, phase2_theory = [], [], []
for q in [0.05, 0.10, 0.20, 0.35, 0.50]:
    r = run_batch(N, f, q, trials=200, seed=1000 + int(q * 1000), cache=rescore_cache)
    p_theory = 1 - (1 - q * (1 - DELTA)) ** n_cheat
    phase2_q.append(q); phase2_emp.append(r['P_emp']); phase2_theory.append(p_theory)
    print(f"  {q:>7.2f} {r['total_audits']/r['trials']:>13.1f} "
          f"{r['P_emp']:>13.3f} {p_theory:>11.3f} "
          f"{r['false_alarms']:>13d} {r['binding_fail']:>13d}")
print(f"\n  (end-to-end phase ran in {time.time()-t1:.1f}s; every audit = real "
      f"re-score + real Merkle proof check)")

# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("PHASE 3  --  invariant: audit COUNT k/f is independent of batch size N")
print("=" * 74)
print("Using the measured per-cheat detection prob (1-delta) in the inspection")
print("game. For target P, needed q = ln(1/(1-P)) / (fN*(1-delta)); audit count")
print("= qN.  Check that qN depends only on f and P, not N.\n")
Pt = 0.99
k = np.log(1 / (1 - Pt))
print(f"  target P={Pt}  ->  k=ln(1/(1-P))={k:.3f}")
print(f"  {'f':>8} {'N':>10} {'q':>10} {'audit count qN':>16}")
for (f_, N_) in [(0.01, 10_000), (0.01, 1_000_000), (0.001, 10_000), (0.001, 1_000_000)]:
    q_ = k / (f_ * N_ * (1 - DELTA))
    print(f"  {f_:>8} {N_:>10,} {q_:>10.2e} {q_*N_:>16,.1f}")

# quick MC confirmation that formula holds for two different N at fixed audit count
def mc_catch(N, f, q, trials, seed):
    rng = np.random.default_rng(seed)
    nc = max(1, round(f * N))
    caught = 0
    det = 1 - DELTA
    for _ in range(trials):
        audited = rng.random(nc) < q            # each cheat audited w.p. q
        detected = rng.random(nc) < det
        if np.any(audited & detected): caught += 1
    return caught / trials
print("\n  MC check (measured detection), target P=0.99:")
for (f_, N_) in [(0.01, 5000), (0.01, 50000)]:
    q_ = k / (f_ * N_ * (1 - DELTA))
    p = mc_catch(N_, f_, q_, 4000, seed=5)
    print(f"    f={f_} N={N_:>6,}  q={q_:.2e}  audit count={q_*N_:.0f}  "
          f"P_emp={p:.3f} (target {Pt})")

# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("PHASE 4  --  DELIVERABLE: client cost for target (f, P) w/ REAL FLOPs")
print("=" * 74)
print(f"real per-request FLOPs: full re-exec (gen) = {gen_full:,.0f}, "
      f"verify = {ver_flops:,.0f}")
print(f"client audit count = k/(f*(1-delta)); client FLOPs = count * verify_flops")
print(f"cost fraction vs full re-exec of an N-batch = (count*verify)/(N*gen)\n")
det = 1 - DELTA
print(f"  {'f':>8} {'P':>7} {'audit count':>13} {'client GFLOPs':>15} "
      f"{'frac of full re-exec':>22}")
for f_ in [0.10, 0.01, 0.001]:
    for Pt in [0.99, 0.999]:
        k = np.log(1 / (1 - Pt))
        count = k / (f_ * det)
        cflops = count * ver_flops
        N_ref = 1_000_000
        frac = cflops / (N_ref * gen_full)
        print(f"  {f_:>8} {Pt:>7.3f} {count:>13,.0f} {cflops/1e9:>15,.1f} "
              f"{frac:>21.3%}  (N={N_ref:,})")

# ---------------------------------------------------------------------------
# Figure: (A) tolerance calibration, (B) end-to-end catch prob vs theory.
# ---------------------------------------------------------------------------
def render(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.2))

    # (A) alpha (false positive) and delta (cheat miss) vs tolerance eps
    axA.plot(eps_grid, eps_alpha, "-o", color="#d62728",
             label=r"$\alpha$  false-positive (honest flagged)")
    axA.plot(eps_grid, eps_delta, "-s", color="#1f77b4",
             label=r"$\delta$  cheat miss (early-exit not flagged)")
    axA.axvline(EPS, ls="--", color="#555", lw=1.2,
                label=f"operating point $\\epsilon$={EPS}")
    axA.set_xscale("log")
    axA.set_xlabel(r"tolerance band  $\epsilon$  (logit $L_\infty$)")
    axA.set_ylabel("rate")
    axA.set_ylim(-0.03, 1.03)
    axA.set_title(f"(A) tolerance calibration\nhonest $L_\\infty$≈{h_linf.mean():.3f}, "
                  f"cheat $L_\\infty$≈{c_linf.mean():.2f}  (σ={SIGMA}, quant={QUANT:.0e})",
                  fontsize=10)
    axA.grid(alpha=0.25); axA.legend(fontsize=8.5, loc="center right")

    # (B) empirical catch prob vs closed form across audit rate q
    axB.plot(phase2_q, phase2_theory, "-", color="#333", lw=1.6,
             label=r"theory  $1-(1-q(1-\delta))^{fN}$")
    axB.scatter(phase2_q, phase2_emp, s=80, color="#2ca02c", zorder=3,
                label="empirical (real re-score + Merkle proof)")
    axB.set_xlabel("per-request audit rate  q")
    axB.set_ylabel("batch catch probability  P")
    axB.set_ylim(-0.03, 1.03)
    axB.set_title(f"(B) end-to-end catch probability vs theory\n"
                  f"N={N}, f={f} ({n_cheat} cheats/batch), measured "
                  f"(1−δ)={1-DELTA:.3f}", fontsize=10)
    axB.grid(alpha=0.25); axB.legend(fontsize=8.5, loc="lower right")

    fig.suptitle("Inference-verification protocol: real-model measurement",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

from pathlib import Path
try:
    figpath = Path(__file__).resolve().parents[1] / "docs" / "figures" / "fig_protocol_measurement.png"
    render(figpath)
    print(f"\nwrote figure: {figpath}")
except Exception as e:
    print(f"\n(skipped figure: {e})")

print("\nDONE.")
