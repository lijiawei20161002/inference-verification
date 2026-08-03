"""Attack x verifier detection-AUC sweep on a REAL model on a GPU.

The standard sweep: every built-in attack scored by every built-in Tier-1
verifier, with logits and activations from a real LLM (default Qwen/Qwen3-0.6B)
on CUDA via `ivgym.backends.hf_gpu`.

**The batch is chosen from the pool, not fixed.** The batch statistic is a mean
of `b` tokens drawn WITHOUT replacement from a fixed token pool, so once `b`
approaches the pool size every batch mean converges to the pool mean, the honest
variance collapses, and the AUC stops answering "would a fresh batch be
flagged?". This script used to default to a 69% batch/pool ratio, and the numbers
it printed were inflated by it. It now sizes the batch at
`EvalConfig.max_pool_ratio` of the honest eval split and prints the ratio in the
header, so the table is interpretable as printed. Set IVGYM_BATCH to override --
`harness` will warn if that puts you over the ceiling.

The consequence is worth stating plainly rather than hiding behind a bigger
number: at the default 3-minute pool the batch is small, so subtle deviations
(`quant_4bit`, `kv_fp8`) correctly read near chance. Detecting those needs a
bigger POOL, not a bigger batch -- see `exp_pool_scaling_gpu` for the 56,160-token
run where `quant_4bit` does arrive, exactly when its per-token effect size says
it should.

Run:  .venv/bin/python -m experiments.exp_gpu
Env overrides: IVGYM_MODEL, IVGYM_PROMPTS, IVGYM_TOKENS, IVGYM_BATCH.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import attacks, harness, signal, verifiers
from ivgym.backends.hf_gpu import HFGPUBackend
from ivgym.core import SamplingSpec

MODEL = os.environ.get("IVGYM_MODEL", "Qwen/Qwen3-0.6B")
N_PROMPTS = int(os.environ.get("IVGYM_PROMPTS", 12))
N_TOKENS = int(os.environ.get("IVGYM_TOKENS", 48))
BATCH = int(os.environ["IVGYM_BATCH"]) if os.environ.get("IVGYM_BATCH") else None
ATTACKS = ["quant_4bit", "kv_fp8", "temp_1.1", "seed_43", "bug_k2", "bug_k32"]
VERIFIERS = ["token_difr", "cross_entropy", "activation_difr", "token_toploc"]


def main():
    t0 = time.time()
    print(f"loading {MODEL} ...", flush=True)
    backend = HFGPUBackend(model_name=MODEL)
    print(
        f"loaded in {time.time()-t0:.1f}s | vocab={backend.vocab} hidden={backend.hidden_dim} "
        f"| {N_PROMPTS} prompts x {N_TOKENS} tokens",
        flush=True,
    )
    spec = SamplingSpec()
    defs = [verifiers.get(d) for d in VERIFIERS]
    cfg = harness.EvalConfig()

    honest_seqs = harness.generate_dataset(
        backend, attacks.get("honest"), spec, N_PROMPTS, N_TOKENS, record_activations=True
    )
    honest = harness.verify(backend, honest_seqs, spec, defs)

    # The honest pool `evaluate` will draw its null batches from, and the largest
    # batch that stays inside the ceiling on it.
    pool = N_PROMPTS * N_TOKENS
    eval_split = int(pool * (1.0 - cfg.calib_frac))
    batch = BATCH if BATCH is not None else max(1, int(eval_split * cfg.max_pool_ratio))
    ratio = batch / eval_split

    header = f"{'attack':>12} | " + " ".join(f"{d:>16}" for d in VERIFIERS)
    print(f"\nReal-model detection AUC @ FPR<=0.5%   (batch={batch} tokens, "
          f"honest eval pool={eval_split}, batch/pool={ratio:.1%}"
          f"{'' if ratio <= cfg.max_pool_ratio else '  << OVER THE CEILING'})")
    print(header)
    print("-" * len(header))
    scored = {}                       # attack -> TokenScores, generated once
    for aname in ATTACKS:
        seqs = harness.generate_dataset(
            backend, attacks.get(aname), spec, N_PROMPTS, N_TOKENS, record_activations=True
        )
        scored[aname] = harness.verify(backend, seqs, spec, defs)
        res = harness.evaluate(honest, scored[aname], defs, [batch], config=cfg)
        by_def = {r.defense: r for r in res}
        row = " ".join(f"{by_def[d].auc:>16.4f}" for d in VERIFIERS)
        print(f"{aname:>12} | {row}", flush=True)

    # The pool-independent read of the SAME scores: d' does not depend on how the
    # batches were drawn, so it says what this pool could show and what it cannot.
    print("\nPer-token effect size d'  /  batch it nominates for AUC 0.90 "
          "(pool-independent)")
    print(header)
    print("-" * len(header))
    for aname in ATTACKS:
        cells = []
        for d in VERIFIERS:
            dp = signal.per_token_stats(honest.scores[d],
                                        scored[aname].scores[d],
                                        cfg.winsor_pct)["d_prime"]
            need = signal.batch_for_pauc(dp, 0.90, cfg.max_fpr)
            cells.append(f"{dp:+.3f} / {'--' if need < 0 or need > 99999 else need}")
        print(f"{aname:>12} | " + " ".join(f"{c:>16}" for c in cells))
    print(f"\n  A cell needing more than this pool's {eval_split}-token eval split "
          f"x {cfg.max_pool_ratio:.0%} = {batch} tokens per batch cannot be resolved "
          f"here.\n  Grow the pool (IVGYM_PROMPTS / IVGYM_TOKENS), not the batch.")

    print(f"\ntotal {time.time()-t0:.1f}s on {MODEL}")


if __name__ == "__main__":
    main()
