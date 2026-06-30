"""Closing the loop: black-box detectability is bounded by the proxy-CE shift,
measured in units of the honest KL(M‖proxy) budget.

`exp_family_correlation.py` measured the within-family agreement and argued that
the proxy detector's *entire* discriminative budget is KL(M‖proxy) -- because
E[surface_stat | honest] = H(M) + KL(M‖proxy). This experiment tests the
consequence directly, on the real model + the real cheap proxy + the real harness:

    surface_stat detectability(attack)  rises with   |ΔCE| / KL(M‖proxy)
        where ΔCE = mean proxy-CE(attack tokens) − mean proxy-CE(honest tokens)

and is at its honest-null FLOOR exactly when an attack preserves M's distribution
(ΔCE ≈ 0), no matter how high token_difr (recomputation of M) scores it. That is
the same seed_43 story as `exp_io_detector_gpu.py`, now with the mechanism made
quantitative: the proxy literally has no budget to spend on a distribution it
agrees should look identical.

Uses HFGPUBackend with a REAL same-family proxy (Option B): M = the reference, a
smaller same-family model = the cheap proxy. surface_stat reads the proxy; token_difr
recomputes M. Both score the SAME attack pools.

Run (validated on a single H100-80GB):
    IVGYM_M=Qwen/Qwen3-4B IVGYM_PROXY=Qwen/Qwen3-0.6B \
        /root/.venv/bin/python -m experiments.exp_detectability_vs_kl

Env overrides:
  IVGYM_M        reference model M       (default Qwen/Qwen3-4B)
  IVGYM_PROXY    same-family cheap proxy (default Qwen/Qwen3-0.6B)
  IVGYM_PROMPTS  prompts per pool        (default 16; honest+null use [0,N),[N,2N))
  IVGYM_TOKENS   tokens per sequence     (default 96)
  IVGYM_BATCH    batch size for S        (default 200)
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import attacks, defenses, harness, io_detectors
from ivgym.backends.hf_gpu import HFGPUBackend, DEFAULT_PROMPTS
from ivgym.core import SamplingSpec
from ivgym.sampling import log_softmax

M_NAME = os.environ.get("IVGYM_M", "Qwen/Qwen3-4B")
PROXY = os.environ.get("IVGYM_PROXY", "Qwen/Qwen3-0.6B")
N = int(os.environ.get("IVGYM_PROMPTS", 16))
T = int(os.environ.get("IVGYM_TOKENS", 96))
BATCH = int(os.environ.get("IVGYM_BATCH", 200))
N_BATCHES = 400
ATTACKS = ("quant_4bit", "kv_fp8", "temp_1.1", "seed_43", "bug_k2", "bug_k32")


def _detect(auc: float) -> float:
    """Symmetric detectability = max(AUC, 1-AUC) (a reversed signal still separates)."""
    return max(auc, 1.0 - auc)


@dataclass
class Row:
    attack: str
    ce_shift: float       # mean proxy-CE(attack) - mean proxy-CE(honest), nats (signed)
    ratio: float          # |ce_shift| / KL(M||proxy)
    ss_detect: float      # surface_stat (proxy) detectability
    td_detect: float      # token_difr (recompute M) detectability


def honest_kl(backend, seqs, spec) -> float:
    """Mean per-token KL(M ‖ proxy) over the honest pool -- the proxy detector's
    entire budget. Read straight off the fresh caches (ref = M, proxy = the cheap
    model), so it must run BEFORE any attack pool overwrites the per-prompt cache."""
    temp = max(spec.temperature, 1e-6)
    vals = []
    for seq in seqs:
        for s in seq.steps:
            lr = log_softmax(backend.reference_logits(seq.prompt_id, s.position) / temp)
            lp = log_softmax(backend.proxy_logits(seq.prompt_id, s.position) / temp)
            vals.append(float((np.exp(lr) * (lr - lp)).sum()))
    return float(np.mean(vals))


def run():
    t0 = time.time()
    print(f"loading M={M_NAME} + proxy={PROXY} ...", flush=True)
    backend = HFGPUBackend(model_name=M_NAME, proxy_model_name=PROXY)
    ratio_params = backend.proxy_n_params / backend.n_params
    print(f"  loaded ({time.time()-t0:.1f}s)  M={backend.n_params/1e9:.2f}B  "
          f"proxy={backend.proxy_n_params/1e9:.2f}B  ({1/ratio_params:.1f}x fewer params)",
          flush=True)

    spec = SamplingSpec()
    td = defenses.get("token_difr")
    ss = io_detectors.get("surface_stat")

    def io_scores(seqs):
        return harness.io_verify(backend, seqs, spec, [ss]).scores[ss.name]

    def auc_ss(honest_io, a_io):
        h = harness.TokenScores("honest", {ss.name: honest_io})
        a = harness.TokenScores("attack", {ss.name: a_io})
        return _detect(harness.evaluate(h, a, [ss], [BATCH], n_batches=N_BATCHES,
                                        winsor_pct=99.9, seed=7)[0].auc)

    def auc_td(honest_td, a_td):
        return _detect(harness.evaluate(honest_td, a_td, [td], [BATCH],
                                        n_batches=N_BATCHES, winsor_pct=99.9, seed=7)[0].auc)

    # --- honest pool: scores + KL budget, materialised while the cache is fresh ---
    honest = harness.generate_dataset(backend, attacks.get("honest"), spec, N, T)
    honest_td = harness.verify(backend, honest, spec, [td])
    honest_io = io_scores(honest)
    KL = honest_kl(backend, honest, spec)
    honest_ce = float(honest_io.mean())
    print(f"  honest pool: KL(M‖proxy)={KL:.3f} nats  mean proxy-CE={honest_ce:.3f} "
          f"({time.time()-t0:.1f}s)", flush=True)

    # --- honest-null floors, one per detector's actual null hypothesis ---
    # surface_stat is DISTRIBUTION-based, so its null is a SAME-PROMPT resample (a
    # second honest draw from the SAME distribution with a different RNG seed). This
    # calibrates finite-sample proxy-CE noise WITHOUT leaking prompt-content variance.
    # Note seed_43 IS exactly such a resample, so it must land on this floor -- that is
    # the whole point: a wrong seed is INSIDE the proxy detector's null.
    resample = [backend.generate(p, T, spec.replace(seed=99), attacks.get("honest"),
                                 False, 123, 32) for p in range(N)]
    resample_io = io_scores(resample)
    floor_ss = auc_ss(honest_io, resample_io)
    # token_difr scores the seed-synced Gumbel margin (≈0 for any honest token,
    # prompt-agnostic), so a disjoint-prompt honest draw under the SAME seed is its
    # proper floor -- a wrong SEED is OUTSIDE its null (it is the deviation it catches).
    null = [backend.generate(p, T, spec, attacks.get("honest"), False, 123, 32)
            for p in range(N, 2 * N)]
    null_td = harness.verify(backend, null, spec, [td])
    floor_td = auc_td(honest_td, null_td)
    print(f"  null floors: surface_stat(resample)={floor_ss:.3f}  "
          f"token_difr(disjoint)={floor_td:.3f} ({time.time()-t0:.1f}s)", flush=True)

    rows = []
    for name in ATTACKS:
        seqs = harness.generate_dataset(backend, attacks.get(name), spec, N, T)
        a_td = harness.verify(backend, seqs, spec, [td])
        a_io = io_scores(seqs)
        shift = float(a_io.mean()) - honest_ce
        rows.append(Row(attack=name, ce_shift=shift, ratio=abs(shift) / max(KL, 1e-9),
                        ss_detect=auc_ss(honest_io, a_io), td_detect=auc_td(honest_td, a_td)))
        print(f"  {name:>12}: ΔCE={shift:+.3f}  |ΔCE|/KL={rows[-1].ratio:.3f}  "
              f"surface_stat={rows[-1].ss_detect:.3f}  token_difr={rows[-1].td_detect:.3f} "
              f"({time.time()-t0:.1f}s)", flush=True)

    return backend, KL, floor_ss, floor_td, rows, time.time() - t0


def main():
    backend, KL, floor_ss, floor_td, rows, elapsed = run()

    print(f"\nDETECTABILITY vs PROXY-CE SHIFT   [M={M_NAME}, proxy={PROXY}, "
          f"{N}x{T} tok, batch={BATCH}]")
    print(f"honest KL(M‖proxy) = {KL:.3f} nats  (the proxy detector's ENTIRE budget)")
    print(f"null floors: surface_stat={floor_ss:.3f}  token_difr={floor_td:.3f}\n")
    h = (f"{'attack':>12} | {'ΔCE(nats)':>10} {'|ΔCE|/KL':>9} | "
         f"{'surface_stat':>12} {'token_difr':>11}  regime")
    print(h + "\n" + "-" * len(h))
    for r in sorted(rows, key=lambda r: r.ratio):
        # Recompute-dominance is read from the GAP token_difr - surface_stat, which is
        # robust to the (prompt-variance-)inflated absolute floors: an attack that
        # spends little proxy budget (low ratio) yet is caught far better by recompute.
        gap = r.td_detect - r.ss_detect
        reg = ("RECOMPUTE-DOMINANT (proxy spends ~no budget; only recompute sees it)"
               if r.ratio < 0.20 and gap > 0.15
               else "output-visible (cheap proxy already separates it)")
        print(f"{r.attack:>12} | {r.ce_shift:>+10.3f} {r.ratio:>9.3f} | "
              f"{r.ss_detect:>12.3f} {r.td_detect:>11.3f}  {reg}")

    print("\nREADING IT:")
    print("  * surface_stat detectability tracks |ΔCE|/KL: an attack is catchable by the")
    print("    cheap proxy only to the extent it spends the proxy's KL budget. seed_43")
    print("    preserves M's distribution -> ΔCE≈0 -> proxy pinned at its floor.")
    print("  * token_difr (recompute of M) does NOT need the budget: it stays high even at")
    print("    ΔCE≈0, because it reads M's own seed-synced Gumbel margin, not a proxy gap.")

    try:
        out = Path(__file__).resolve().parents[1] / "docs" / "figures" / "fig_detectability_vs_kl.png"
        render(rows, KL, floor_ss, floor_td, out)
        print(f"\nwrote figure: {out}")
    except Exception as e:
        print(f"\n(skipped figure: {e})")
    print(f"\ntotal {elapsed:.1f}s")


def render(rows, KL, floor_ss, floor_td, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 5.6))

    xs = [r.ratio for r in rows]
    ax.scatter(xs, [r.ss_detect for r in rows], s=90, color="#2ca02c", zorder=3,
               label="surface_stat  (cheap proxy, KL-bounded)")
    ax.scatter(xs, [r.td_detect for r in rows], s=90, marker="^", color="#1f77b4",
               zorder=3, label="token_difr  (recompute M)")
    for r in rows:
        ax.annotate(r.attack, (r.ratio, r.ss_detect), textcoords="offset points",
                    xytext=(6, -10), fontsize=8, color="#1a661a")

    ax.axhline(floor_ss, ls=":", color="#2ca02c", lw=1.4,
               label=f"surface_stat null floor ({floor_ss:.2f})")
    ax.axhline(floor_td, ls=":", color="#1f77b4", lw=1.2,
               label=f"token_difr null floor ({floor_td:.2f})")
    ax.set_xlabel("proxy budget spent by the attack:  |ΔCE| / KL(M‖proxy)\n"
                  f"(ΔCE = proxy-CE shift vs honest;  honest KL = {KL:.2f} nats)")
    ax.set_ylabel("detectability   (max(AUC, 1−AUC))")
    ax.set_ylim(0.4, 1.03)
    ax.set_title("Black-box detectability is bounded by the KL budget; recomputation is not\n"
                 "left edge (ΔCE≈0, e.g. seed_43): proxy blind, token_difr still high",
                 fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
