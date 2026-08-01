"""Does the d' arithmetic predict the batch at which a real deviation is caught?

`docs/TRIAGE_AND_AUDIT_COST.md` section 3 measured the per-token effect size of
`quant_4bit` on Qwen3-1.7B (d' = 0.0716) and turned it into a prediction: a batch
of ~2 766 tokens reaches AUC 0.90, which needs a >= 55 000-token honest pool to
stay inside the ~10% batch/pool ratio ceiling. It could not test that, because its
pool was 22 464 tokens -- so the two batches that did reach 0.715 / 0.925 were
already at 14% and 28% ratios, i.e. buying the number with overlap. Item 1 of that
document's open list calls a big enough pool "the most useful place to spend the
next block of GPU-hours".

This spends them. The pool is 117 prompts x 480 tokens = 56 160 tokens per config,
which puts the predicted batch 2 766 at a 9.8% ratio -- inside the ceiling. The
batch is then swept with the pool FIXED, so every point below the ceiling is a
legitimate measurement of the same underlying d' and the curve is a direct test of
`signal.predicted_pauc`. Points above the ceiling are measured too, and reported
as such, so the artifact and the real curve appear on one axis.

The prediction is fixed before the measurement: `signal.batch_for_pauc(d')` is the
library's own forward model, not a fit to these points.

    IVGYM_M=Qwen/Qwen3-1.7B python -m experiments.exp_pool_scaling_gpu
Env: IVGYM_M(Qwen/Qwen3-1.7B), IVGYM_ATTACK(quant_4bit), IVGYM_PROMPTS(117),
     IVGYM_TOKENS(480), IVGYM_SEEDS(5).

Writes `docs/results/pool_scaling.json`.
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

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results"

M = os.environ.get("IVGYM_M", "Qwen/Qwen3-1.7B")
ATTACK = os.environ.get("IVGYM_ATTACK", "quant_4bit")
N = int(os.environ.get("IVGYM_PROMPTS", 117))
T = int(os.environ.get("IVGYM_TOKENS", 480))
N_SEED = int(os.environ.get("IVGYM_SEEDS", 5))
DEFENSE = os.environ.get("IVGYM_DEFENSE", "token_difr")
RATIO_CEILING = 0.10


def main() -> None:
    t0 = time.time()
    print("=" * 78)
    print(f"Pool scaling: is the predicted batch the batch?   M={M}  attack={ATTACK}")
    print(f"  pool = {N} prompts x {T} tokens, batch swept at FIXED pool, "
          f"{N_SEED} protocol seeds")
    print("=" * 78, flush=True)

    backend = HFGPUBackend(model_name=M)
    spec = SamplingSpec()
    d = verifiers.get(DEFENSE)

    honest = harness.verify(
        backend, harness.generate_dataset(backend, attacks.get("honest"), spec, N, T),
        spec, [d])
    print(f"  honest pool: {len(honest.scores[DEFENSE])} tokens "
          f"[{time.time()-t0:.0f}s]", flush=True)
    attacked = harness.verify(
        backend, harness.generate_dataset(backend, attacks.get(ATTACK), spec, N, T),
        spec, [d])
    print(f"  {ATTACK} pool: {len(attacked.scores[DEFENSE])} tokens "
          f"[{time.time()-t0:.0f}s]", flush=True)

    h, a = honest.scores[DEFENSE], attacked.scores[DEFENSE]
    n_tok = len(h)
    null = 0.5 * n_tok                      # evaluate splits honest 50/50
    st = signal.per_token_stats(h, a)
    dp = st["d_prime"]
    b_pred = signal.batch_for_pauc(dp)
    print(f"\n  per-token d' = {dp:.4f}  (honest mean {st['honest_mean']:.4f}, "
          f"sd {st['honest_sd']:.4f})")
    print(f"  => predicted batch for AUC 0.90: {b_pred}"
          f"  (needs a >= {int(np.ceil(b_pred/RATIO_CEILING*2))}-token pool at a "
          f"{RATIO_CEILING:.0%} ratio; this pool is {n_tok})")
    print(f"  ceiling batch for this pool: {int(RATIO_CEILING*null)}", flush=True)

    batches = sorted({200, 400, 800, 1600, max(b_pred, 1), 5600, 11200})
    rows = []
    print(f"\n{'batch':>7} {'ratio':>7} {'in ceiling':>11} {'measured AUC':>18} "
          f"{'predicted':>10} {'measured TPR':>13}")
    for b in batches:
        if b >= 0.9 * n_tok:
            continue
        aucs, tprs = [], []
        for s in range(N_SEED):
            r = harness.evaluate(honest, attacked, [d], [b], n_batches=2000, seed=s)[0]
            aucs.append(r.auc)
            tprs.append(r.tpr)
        ratio = b / null
        row = {"batch": int(b), "ratio": float(ratio),
               "in_ceiling": bool(ratio <= RATIO_CEILING),
               "auc": float(np.mean(aucs)), "sd": float(np.std(aucs)),
               "tpr": float(np.mean(tprs)),
               "predicted": float(signal.predicted_pauc(dp, b)),
               "is_prediction_point": bool(b == b_pred)}
        rows.append(row)
        mark = "yes" if row["in_ceiling"] else "OVER"
        star = "  <- predicted b for 0.90" if row["is_prediction_point"] else ""
        print(f"{b:>7} {ratio:>6.1%} {mark:>11}   {row['auc']:.3f} +-{row['sd']:.3f}"
              f"      {row['predicted']:>8.3f} {row['tpr']:>12.3f}{star}", flush=True)

    inside = [r for r in rows if r["in_ceiling"]]
    best = max(inside, key=lambda r: r["auc"]) if inside else None
    if best:
        print(f"\n  best legitimate point: batch {best['batch']} at a "
              f"{best['ratio']:.1%} ratio -> AUC {best['auc']:.3f} +-{best['sd']:.3f} "
              f"(predicted {best['predicted']:.3f})")
    err = [abs(r["auc"] - r["predicted"]) for r in inside]
    if err:
        print(f"  |measured - predicted| over the {len(inside)} in-ceiling points: "
              f"mean {np.mean(err):.3f}, max {np.max(err):.3f}")

    payload = {"model": M, "attack": ATTACK, "defense": DEFENSE, "n_prompts": N,
               "tokens": T, "n_seed": N_SEED, "eval_tokens": int(n_tok),
               "null_split": float(null), "ratio_ceiling": RATIO_CEILING,
               "per_token": st, "d_prime": float(dp), "batch_pred_090": int(b_pred),
               "rows": rows, "elapsed_s": time.time() - t0}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "pool_scaling.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {RES/'pool_scaling.json'}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
