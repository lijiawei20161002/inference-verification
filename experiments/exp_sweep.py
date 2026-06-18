"""Full attack x defense sweep: the main DiFR results grid.

Builds an honest pool, then for each registered attack reports detection AUC
at a fixed batch size for every defense. Mirrors Table 2 / Figure 1.

Run:  .venv/bin/python -m experiments.exp_sweep
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import attacks, defenses, harness
from ivgym.backends.synthetic import SyntheticBackend
from ivgym.core import SamplingSpec

N_PROMPTS = 60
N_TOKENS = 256
BATCH = 1000
ATTACKS = ["quant_4bit", "kv_fp8", "temp_1.1", "seed_43", "bug_k2", "bug_k32"]
DEFENSES = ["token_difr", "cross_entropy", "activation_difr"]


def main():
    backend = SyntheticBackend(vocab=512)
    spec = SamplingSpec()
    defs = [defenses.get(d) for d in DEFENSES]

    honest_seqs = harness.generate_dataset(
        backend, attacks.get("honest"), spec, N_PROMPTS, N_TOKENS, record_activations=True)
    honest = harness.verify(backend, honest_seqs, spec, defs)

    header = f"{'attack':>12} | " + " ".join(f"{d:>16}" for d in DEFENSES)
    print(f"AUC @ batch={BATCH} tokens (FPR target 1%)\n" + header)
    print("-" * len(header))
    for aname in ATTACKS:
        atk = attacks.get(aname)
        seqs = harness.generate_dataset(backend, atk, spec, N_PROMPTS, N_TOKENS,
                                        record_activations=True)
        ascores = harness.verify(backend, seqs, spec, defs)
        res = harness.evaluate(honest, ascores, defs, [BATCH], n_batches=400, winsor_pct=99.9)
        by_def = {r.defense: r for r in res}
        row = " ".join(f"{by_def[d].auc:>16.4f}" for d in DEFENSES)
        print(f"{aname:>12} | {row}")


if __name__ == "__main__":
    main()
