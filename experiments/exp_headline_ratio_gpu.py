"""The README headline grid, re-measured inside the batch/pool ratio ceiling.

`docs/TRIAGE_AND_AUDIT_COST.md` § 3 found that `harness.batch_means` resamples a
batch *without replacement from a fixed token pool*, so once the batch is a large
fraction of the honest eval split the batches overlap, honest variance collapses,
and the AUC stops measuring "would a fresh batch be flagged". It showed this for
ONE cell (`token_difr` vs `quant_4bit`: 0.977 at a 69% ratio, 0.530 at 1.8%) and
closed by naming what was left open -- "the README headline table (78%) and
`exp_gpu.py`'s default (69%) both sit above the ceiling ... those tables have not
been re-measured."

This re-measures them. Both arms read the SAME per-token scores from one
generation pass, so the only thing that differs between them is the pool the
batches are drawn from:

  * `ratioed` -- the full 117 x 192 pool (22 464 tokens/config), batch 1000,
    which is 8.9% of the honest eval split: inside the documented ceiling.
  * `readme`  -- the first 20 sequences x first 128 tokens of that same pool
    (2 560 tokens), batch 1000 = 78% of its eval split: the README's
    configuration, reproduced as a sub-pool so the contrast is controlled.

Every cell is a mean +- sd over `IVGYM_SEEDS` protocol seeds (the README's table
is a single draw), so "these two numbers differ" is a claim with an error bar on
it. Also reports the per-token effect size d' behind each column, which is the
pool-independent quantity, and the batch a properly-ratioed pool would need to
reach AUC 0.90 on it.

    python -m experiments.exp_headline_ratio_gpu
Env: IVGYM_MODEL(Qwen/Qwen3-0.6B), IVGYM_PROMPTS(117), IVGYM_TOKENS(192),
     IVGYM_BATCH(1000), IVGYM_SEEDS(5), IVGYM_SUB_PROMPTS(20), IVGYM_SUB_TOKENS(128).

Writes `docs/results/headline_ratio.json`.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import attacks, harness, metrics, verifiers
from ivgym.backends.hf_gpu import HFGPUBackend
from ivgym.core import SamplingSpec

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results"

MODEL = os.environ.get("IVGYM_MODEL", "Qwen/Qwen3-0.6B")
N_PROMPTS = int(os.environ.get("IVGYM_PROMPTS", 117))
T = int(os.environ.get("IVGYM_TOKENS", 192))
BATCH = int(os.environ.get("IVGYM_BATCH", 1000))
N_SEED = int(os.environ.get("IVGYM_SEEDS", 5))
SUB_P = int(os.environ.get("IVGYM_SUB_PROMPTS", 20))
SUB_T = int(os.environ.get("IVGYM_SUB_TOKENS", 128))

ATTACKS = ["quant_4bit", "kv_fp8", "temp_1.1", "seed_43", "bug_k2", "bug_k32"]
DEFENSES = ["token_difr", "cross_entropy", "activation_difr", "token_toploc"]

# pAUC@FPR<=0.5% = 0.90 for a Gaussian pair needs this much separation; see
# docs/TRIAGE_AND_AUDIT_COST.md section 3.
SEP_FOR_090 = 3.767


def sub_index(seq_lens: list[int], n_seq: int, n_tok: int) -> np.ndarray:
    """Flat indices of the first `n_tok` tokens of the first `n_seq` sequences.

    `verify` flattens sequence-major, so a sub-pool is a gather on the same array
    rather than a second generation -- which is what makes the two arms below the
    same measurement at two ratios."""
    off, idx = 0, []
    for i, L in enumerate(seq_lens):
        if i < n_seq:
            idx.extend(range(off, off + min(n_tok, L)))
        off += L
    return np.asarray(idx, dtype=int)


def auc_seeds(h: np.ndarray, a: np.ndarray, batch: int) -> tuple[float, float, float]:
    """(mean, sd, mean TPR) of the standardized pAUC over N_SEED protocol seeds."""
    hs = harness.TokenScores("honest", {"s": h})
    as_ = harness.TokenScores("attack", {"s": a})
    v = [verifiers.get("token_difr")]                    # only `.name` is read here
    v[0] = type("V", (), {"name": "s", "tier": 1})()
    aucs, tprs = [], []
    for s in range(N_SEED):
        # over_ratio="allow": this experiment measures the INFLATED arm on
        # purpose (that is the whole comparison), so the ceiling guard in
        # `EvalConfig` is opted out of explicitly rather than silently.
        r = harness.evaluate(hs, as_, v, [batch], seed=s,
                             config=harness.EvalConfig(n_batches=2000,
                                                       over_ratio="allow"))[0]
        aucs.append(r.auc)
        tprs.append(r.tpr)
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(tprs))


def d_prime(h: np.ndarray, a: np.ndarray) -> float:
    """Per-token standardized effect size -- the pool-independent quantity."""
    sd = float(np.std(h))
    return float((np.mean(a) - np.mean(h)) / sd) if sd > 0 else 0.0


def main() -> None:
    t0 = time.time()
    print("=" * 78)
    print(f"The headline grid at two batch/pool ratios   M={MODEL}")
    print(f"  pool = {N_PROMPTS} prompts x {T} tokens; README sub-pool = "
          f"{SUB_P} x {SUB_T}; batch {BATCH}; {N_SEED} protocol seeds")
    print("=" * 78, flush=True)

    backend = HFGPUBackend(model_name=MODEL)
    spec = SamplingSpec()
    defs = [verifiers.get(d) for d in DEFENSES]

    honest_seqs = harness.generate_dataset(
        backend, attacks.get("honest"), spec, N_PROMPTS, T, record_activations=True)
    honest = harness.verify(backend, honest_seqs, spec, defs)
    seq_lens = [len(s.steps) for s in honest_seqs]
    n_tok = len(honest.scores[DEFENSES[0]])
    sub = sub_index(seq_lens, SUB_P, SUB_T)
    arms = {
        "ratioed": (np.arange(n_tok), BATCH / (0.5 * n_tok)),
        "readme": (sub, BATCH / (0.5 * len(sub))),
    }
    print(f"honest pool: {n_tok} tokens -> ratioed arm batch/null = "
          f"{arms['ratioed'][1]:.1%};  README sub-pool: {len(sub)} tokens -> "
          f"{arms['readme'][1]:.1%} [{time.time()-t0:.0f}s]", flush=True)

    payload = {
        "model": MODEL, "n_prompts": N_PROMPTS, "tokens": T, "batch": BATCH,
        "n_seed": N_SEED, "sub_prompts": SUB_P, "sub_tokens": SUB_T,
        "eval_tokens": int(n_tok), "sub_tokens_total": int(len(sub)),
        "ratio": {k: v[1] for k, v in arms.items()},
        "attacks": ATTACKS, "defenses": DEFENSES, "cells": {},
    }

    for aname in ATTACKS:
        seqs = harness.generate_dataset(
            backend, attacks.get(aname), spec, N_PROMPTS, T, record_activations=True)
        ascores = harness.verify(backend, seqs, spec, defs)
        row = {}
        for d in DEFENSES:
            h_all, a_all = honest.scores[d], ascores.scores[d]
            cell = {"d_prime": d_prime(h_all, a_all)}
            cell["batch_for_090"] = (
                int(np.ceil((SEP_FOR_090 / abs(cell["d_prime"])) ** 2))
                if abs(cell["d_prime"]) > 1e-9 else None)
            cell["token_auc"] = float(metrics.roc_auc(h_all, a_all))
            for arm, (idx, _) in arms.items():
                m, s, tpr = auc_seeds(h_all[idx], a_all[idx], BATCH)
                cell[arm] = {"auc": m, "sd": s, "tpr": tpr}
            row[d] = cell
        payload["cells"][aname] = row
        print(f"  {aname:>12} " + "  ".join(
            f"{d.split('_')[0][:5]}: {row[d]['readme']['auc']:.3f}->"
            f"{row[d]['ratioed']['auc']:.3f}" for d in DEFENSES)
            + f"  [{time.time()-t0:.0f}s]", flush=True)

    for arm in arms:
        print(f"\n{'='*78}\n{arm.upper()} arm  (batch/null = {arms[arm][1]:.1%})"
              f"   AUC @ FPR <= 0.5%, mean +- sd over {N_SEED} seeds\n{'='*78}")
        print(f"{'attack':>12} | " + " ".join(f"{d:>18}" for d in DEFENSES))
        for aname in ATTACKS:
            print(f"{aname:>12} | " + " ".join(
                f"{payload['cells'][aname][d][arm]['auc']:>11.3f} "
                f"+-{payload['cells'][aname][d][arm]['sd']:.3f}" for d in DEFENSES))

    print(f"\n{'='*78}\nPer-token effect size d' (pool-independent) and the batch a "
          f"10%-ratio pool\nwould need for AUC 0.90\n{'='*78}")
    print(f"{'attack':>12} | " + " ".join(f"{d:>18}" for d in DEFENSES))
    for aname in ATTACKS:
        print(f"{aname:>12} | " + " ".join(
            f"{payload['cells'][aname][d]['d_prime']:>+8.4f} "
            f"b={str(payload['cells'][aname][d]['batch_for_090']):>7}"
            for d in DEFENSES))

    payload["elapsed_s"] = time.time() - t0
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "headline_ratio.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {RES/'headline_ratio.json'}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
