"""Where did the `token_difr` baseline go? Diagnosing detection HEADROOM.

`docs/TRIAGE_AND_AUDIT_COST.md` reports the learned-triage experiment as
inconclusive for one reason: in that configuration full recompute itself scores
only **AUC 0.570** on `quant_4bit`, against the 0.85-1.00 the same detector scores
in `exp_gpu.py`. With the ceiling at 0.57 there is no signal for a triage policy
to allocate, so every value signal lands inside noise. Before tuning any head,
that gap has to be explained. This experiment does that, and then locates a
configuration with real headroom.

Four candidate explanations, and the sweep that separates them
-------------------------------------------------------------
1. **The batch/pool ratio (the accounting artifact).** `harness.batch_means`
   resamples `batch_size` tokens WITHOUT replacement from a FIXED token pool. As
   `batch_size` approaches the pool size every batch mean converges to the pool
   mean, honest variance collapses, and the AUC measures "do these two particular
   pools have different means" -- which is nearly deterministic -- rather than
   "would a fresh batch be flagged". `EvalConfig` documents a ~10% ceiling for
   exactly this. `exp_gpu.py`'s default (12x48 tokens, batch 200) runs at **69%**
   of the honest eval split, and the README headline (20x128, batch 1000) at
   **78%**. The triage run (64x128, batch 200) sat at 4.9% -- inside the ceiling.
   So: sweep the pool size at FIXED batch. If this is the explanation, a big pool
   reproduces 0.57 and a small one reproduces 0.85+, with the same model, the same
   attack and the same per-token scores.
2. **Batch size itself (legitimate power).** A batch of `b` tokens carries
   `sqrt(b)` times the per-token effect size, so a bigger `b` genuinely detects
   more. Separated from (1) by sweeping `b` with the pool GROWN to hold the ratio
   at 10%.
3. **The model.** Per-token effect size may simply be smaller on Qwen3-1.7B (the
   triage config) than on Qwen3-0.6B (the README config). Measured directly as
   the per-token d-prime on both.
4. **Attack strength.** `attacks.Quantization`'s sigma sets the deviation
   magnitude; swept as a ladder.

Everything after generation is pure numpy over cached per-token score arrays, so
the batch/pool sweeps are free -- and every AUC is reported as a mean +- sd over
`N_SEED` independent `EvalConfig` seeds, which is what tells you whether a
0.57-vs-0.64 gap is a result or noise.

Writes `docs/results/baseline_headroom.json` and, via `experiments/plot_headroom.py`,
`docs/figures/fig_baseline_headroom.png`.

    IVGYM_M=Qwen/Qwen3-1.7B python -m experiments.exp_baseline_headroom_gpu
Env: IVGYM_MODELS(Qwen/Qwen3-1.7B,Qwen/Qwen3-0.6B), IVGYM_PROMPTS(117),
     IVGYM_TOKENS(192), IVGYM_SEEDS(5), IVGYM_SIGMAS(0.09,0.18,0.36),
     IVGYM_LADDER_MODEL(Qwen/Qwen3-1.7B).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import attacks, harness, signal, verifiers
from ivgym.backends.hf_gpu import HFGPUBackend
from ivgym.core import SamplingSpec
from experiments import plot_headroom

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "docs" / "results"
FIG_DIR = ROOT / "docs" / "figures"

MODELS = [m for m in os.environ.get(
    "IVGYM_MODELS", "Qwen/Qwen3-1.7B,Qwen/Qwen3-0.6B").split(",") if m]
N = int(os.environ.get("IVGYM_PROMPTS", 117))
T = int(os.environ.get("IVGYM_TOKENS", 192))
N_SEED = int(os.environ.get("IVGYM_SEEDS", 5))
SIGMAS = [float(s) for s in os.environ.get("IVGYM_SIGMAS", "0.09,0.18,0.36").split(",")]
LADDER_MODEL = os.environ.get("IVGYM_LADDER_MODEL", MODELS[0])
ATTACK = os.environ.get("IVGYM_ATTACK", "quant_4bit")

# batch sizes for the two batch sweeps, and pool sizes for the ratio sweep
BATCHES = [50, 100, 200, 400, 800, 1600, 3200]
POOLS = [576, 1152, 2304, 4608, 9216, 18432]
RATIO_BATCH = [200, 1000]       # the two batch sizes the repo's own runs used
TARGET_AUC = 0.90


# --------------------------------------------------------------------- helpers
def _scores(h: np.ndarray, a: np.ndarray, name: str) -> tuple:
    """Wrap flat per-token arrays as the `TokenScores` pair `evaluate` wants."""
    td = verifiers.get("token_difr")
    return (harness.TokenScores("honest", {td.name: h}),
            harness.TokenScores(name, {td.name: a}), [td])


def auc_of(h: np.ndarray, a: np.ndarray, batch: int, name: str = ATTACK,
           n_seed: int = N_SEED) -> tuple[float, float]:
    """AUC@FPR<=0.5% as mean +- sd over `n_seed` independent protocol seeds.

    The seed drives the honest calib/eval split AND the batch resampling, i.e.
    every source of Monte-Carlo noise inside `evaluate`. A single-seed number
    cannot tell a 0.57-vs-0.64 difference from noise; this can.
    """
    hs, as_, defs = _scores(h, a, name)
    vals = [harness.evaluate(hs, as_, defs, [batch], seed=s)[0].auc
            for s in range(n_seed)]
    return float(np.mean(vals)), float(np.std(vals))


def subsample(h: np.ndarray, a: np.ndarray, n: int, seed: int = 0):
    """A random size-`n` sub-pool of both score arrays (pool-size axis)."""
    rng = np.random.default_rng(seed)
    if n >= min(len(h), len(a)):
        return h, a
    return h[rng.choice(len(h), n, replace=False)], a[rng.choice(len(a), n, replace=False)]


def per_token_stats(h: np.ndarray, a: np.ndarray, winsor_pct: float = 99.9) -> dict:
    """Per-token effect size, on exactly the scale `evaluate` scores, plus the
    token-level AUC.

    The arithmetic lives in `ivgym.signal.per_token_stats` (winsorize at the honest
    percentile the way `evaluate` does, then `d' = (mean_a - mean_h) / sd_h`); this
    wrapper adds the token-level AUC the tables here print alongside it.
    """
    cap = np.percentile(h[np.isfinite(h)], winsor_pct)
    stats = signal.per_token_stats(h, a, winsor_pct)
    stats["token_auc"] = float(harness.roc_auc(np.minimum(h, cap), np.minimum(a, cap)))
    return stats


def batch_for_target(d_prime: float, target_auc: float = TARGET_AUC) -> int:
    """Batch size at which `d' * sqrt(b)` reaches standardized pAUC@FPR<=0.5% =
    `target_auc` -- `ivgym.signal.batch_for_pauc`.

    The separation that target needs is *solved* (3.7673 at FPR <= 0.5%) rather
    than hard-coded as the rounded 3.767 two files used to carry independently, so
    this prediction and panel B of `plot_headroom` cannot drift apart. Purely a
    prediction from the per-token number -- the measured `fixed_ratio` sweep below
    is what confirms it.
    """
    return signal.batch_for_pauc(d_prime, target_auc)


# ------------------------------------------------------------------- generation
def token_scores_for(backend, spec, attack, ids, chunk: int = 24) -> np.ndarray:
    """Flat per-token `token_difr` scores for one config, generated in chunks.

    Chunked because the backend's reference/served caches are per-prompt-id and
    hold a `[T, V]` float32 row each -- at these pool sizes that is tens of GB if
    every prompt is kept live at once. A Tier-1 per-token score depends only on
    its own row, so chunking is exactly equivalent to one big call.
    """
    td = verifiers.get("token_difr")
    out = []
    for i in range(0, len(ids), chunk):
        part = ids[i:i + chunk]
        seqs = [backend.generate(p, T, spec, attack) for p in part]
        out.append(harness.verify(backend, seqs, spec, [td]).scores[td.name])
        backend.drop_reference_cache()
        backend._served_cache.clear()
    return np.concatenate(out)


# ------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    RES_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("Baseline headroom diagnostic: why is full-recompute token_difr at 0.570?")
    print(f"  pool = {N} prompts x {T} tokens = {N*T} tokens/config, "
          f"{N_SEED} protocol seeds per AUC")
    print("=" * 78, flush=True)

    spec = SamplingSpec()
    ids = list(range(N))
    payload = {"n_prompts": N, "tokens": T, "n_seed": N_SEED, "attack": ATTACK,
               "batches": BATCHES, "pools": POOLS, "ratio_batch": RATIO_BATCH,
               "target_auc": TARGET_AUC, "models": {}, "sigma_ladder": {}}

    for model in MODELS:
        print(f"\n{'='*78}\nM = {model}\n{'='*78}", flush=True)
        backend = HFGPUBackend(model_name=model)
        h = token_scores_for(backend, spec, attacks.get("honest"), ids)
        print(f"  honest pool: {len(h)} tokens [{time.time()-t0:.0f}s]", flush=True)
        a = token_scores_for(backend, spec, attacks.get(ATTACK), ids)
        print(f"  {ATTACK} pool: {len(a)} tokens [{time.time()-t0:.0f}s]", flush=True)

        pt = per_token_stats(h, a)
        b_pred = batch_for_target(pt["d_prime"])
        print(f"\n  per-token effect size d' = {pt['d_prime']:.4f}  "
              f"(token-level AUC {pt['token_auc']:.4f})")
        print(f"  => predicted batch for AUC {TARGET_AUC:.2f}: b = {b_pred} "
              f"(needs a >= {20*b_pred}-token honest pool to stay inside the 10% ceiling)",
              flush=True)

        # --- 1. POOL-SIZE sweep at FIXED batch: the accounting artifact --------
        print(f"\n  [1] fixed batch, growing pool -- the batch/pool ratio artifact")
        print(f"  {'pool':>7}{'ratio':>9}{'AUC':>16}   (batch)")
        ratio_sweep = {}
        for b in RATIO_BATCH:
            rows = []
            for n_pool in POOLS + [len(h)]:
                if n_pool > len(h):
                    continue
                hs, as_ = subsample(h, a, n_pool)
                null = 0.5 * n_pool          # evaluate() splits honest 50/50
                if b > null:
                    continue
                m, s = auc_of(hs, as_, b)
                rows.append({"pool": int(n_pool), "ratio": float(b / null),
                             "auc": m, "auc_sd": s})
                print(f"  {n_pool:>7}{b/null:>8.1%}{m:>11.3f} +-{s:>4.3f}   ({b})",
                      flush=True)
            ratio_sweep[str(b)] = rows

        # --- 2. BATCH sweep, ratio pinned at 10%: legitimate power -------------
        print(f"\n  [2] batch sweep with the pool grown to hold ratio = 10% "
              f"(and, for contrast, on the full pool)")
        print(f"  {'batch':>7}{'pool@10%':>10}{'AUC@10%':>16}"
              f"{'ratio_full':>12}{'AUC_full':>16}")
        batch_sweep = []
        for b in BATCHES:
            need = 20 * b                       # pool s.t. b = 10% of the null half
            row = {"batch": b, "pool_needed": need}
            if need <= len(h):
                hs, as_ = subsample(h, a, need)
                m, s = auc_of(hs, as_, b)
                row.update(auc_fixed=m, auc_fixed_sd=s)
            if b <= 0.5 * len(h):
                m2, s2 = auc_of(h, a, b)
                row.update(auc_full=m2, auc_full_sd=s2,
                           ratio_full=float(b / (0.5 * len(h))))
            batch_sweep.append(row)
            f = (f"{row['auc_fixed']:>11.3f} +-{row['auc_fixed_sd']:>4.3f}"
                 if "auc_fixed" in row else f"{'--':>16}")
            g = (f"{row['auc_full']:>11.3f} +-{row['auc_full_sd']:>4.3f}"
                 if "auc_full" in row else f"{'--':>16}")
            rf = (f"{row['ratio_full']:>11.1%}" if "ratio_full" in row
                  else f"{'--':>12}")
            print(f"  {b:>7}{need:>10}{f}{rf}{g}", flush=True)

        payload["models"][model] = {
            "n_tokens": int(len(h)), "per_token": pt, "batch_for_target": b_pred,
            "ratio_sweep": ratio_sweep, "batch_sweep": batch_sweep,
        }
        if model == LADDER_MODEL:
            # --- 3. ATTACK-STRENGTH ladder, at a fixed properly-powered point --
            # Batch must stay inside the honest eval half or `batch_means` clamps
            # it to the pool size, every batch mean becomes the pool mean, and the
            # AUC degenerates to exactly 1.0 -- the artifact this experiment is about.
            b_lad = int(min(200, 0.1 * 0.5 * len(h)))
            print(f"\n  [3] attack-strength ladder (quant sigma), batch {b_lad} "
                  f"at the full pool")
            print(f"  {'sigma':>7}{'d_prime':>10}{'b for 0.90':>12}"
                  f"{f'AUC@{b_lad}':>16}")
            for sg in SIGMAS:
                atk = attacks.Quantization(name=f"quant_s{sg}", extra_sigma=sg,
                                           bias_sigma=sg / 3.0, act_sigma=sg * 5 / 3)
                a_s = token_scores_for(backend, spec, atk, ids)
                pts = per_token_stats(h, a_s)
                m, s = auc_of(h, a_s, b_lad, name=atk.name)
                payload["sigma_ladder"][str(sg)] = {
                    "model": model, "per_token": pts, "batch": b_lad,
                    "auc": m, "auc_sd": s,
                    "batch_for_target": batch_for_target(pts["d_prime"])}
                print(f"  {sg:>7.2f}{pts['d_prime']:>10.4f}"
                      f"{batch_for_target(pts['d_prime']):>12}"
                      f"{m:>11.3f} +-{s:>4.3f}   [{time.time()-t0:.0f}s]", flush=True)

        del backend
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()

    out = RES_DIR / "baseline_headroom.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")
    plot_headroom.main([])
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
