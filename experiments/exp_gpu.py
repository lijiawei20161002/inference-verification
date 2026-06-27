"""Attack x defense detection-AUC sweep on a REAL model on a GPU.

The standard sweep: every built-in attack scored by every built-in defense, with
logits and activations from a real LLM (default Qwen/Qwen3-0.6B) on CUDA via
ivgym.backends.hf_gpu.

Run:  .venv/bin/python -m experiments.exp_gpu
Env overrides: IVGYM_MODEL, IVGYM_PROMPTS, IVGYM_TOKENS, IVGYM_BATCH.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import attacks, defenses, harness
from ivgym.backends.hf_gpu import HFGPUBackend
from ivgym.core import SamplingSpec

MODEL = os.environ.get("IVGYM_MODEL", "Qwen/Qwen3-0.6B")
N_PROMPTS = int(os.environ.get("IVGYM_PROMPTS", 12))
N_TOKENS = int(os.environ.get("IVGYM_TOKENS", 48))
BATCH = int(os.environ.get("IVGYM_BATCH", 200))
ATTACKS = ["quant_4bit", "kv_fp8", "temp_1.1", "seed_43", "bug_k2", "bug_k32"]
DEFENSES = ["token_difr", "cross_entropy", "activation_difr"]


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
    defs = [defenses.get(d) for d in DEFENSES]

    honest_seqs = harness.generate_dataset(
        backend, attacks.get("honest"), spec, N_PROMPTS, N_TOKENS, record_activations=True
    )
    honest = harness.verify(backend, honest_seqs, spec, defs)

    header = f"{'attack':>12} | " + " ".join(f"{d:>16}" for d in DEFENSES)
    print(f"\nReal-model DiFR detection AUC @ batch={BATCH} tokens\n" + header)
    print("-" * len(header))
    for aname in ATTACKS:
        atk = attacks.get(aname)
        seqs = harness.generate_dataset(
            backend, atk, spec, N_PROMPTS, N_TOKENS, record_activations=True
        )
        ascores = harness.verify(backend, seqs, spec, defs)
        res = harness.evaluate(honest, ascores, defs, [BATCH], n_batches=400, winsor_pct=99.9)
        by_def = {r.defense: r for r in res}
        row = " ".join(f"{by_def[d].auc:>16.4f}" for d in DEFENSES)
        print(f"{aname:>12} | {row}", flush=True)

    print(f"\ntotal {time.time()-t0:.1f}s on {MODEL}")


if __name__ == "__main__":
    main()
