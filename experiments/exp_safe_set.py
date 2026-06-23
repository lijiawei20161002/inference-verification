"""Seed-synced regime: the SAFE-set substitution attack.

Demonstrates an attack that deviates from the honest sample yet stays
indistinguishable to *every* defense (Token-DiFR, cross-entropy,
activation-DiFR), and quantifies the price of that indistinguishability:

  * the achievable deviation rate (the SAFE-set capacity), and
  * how the disguise degrades as `epsilon` grows and as the batch grows --
    Token-DiFR's per-token score is one-sided (>= 0, = 0 only at the honest
    token), so any systematic bias is eventually surfaced by averaging.

`run_sweep` returns the structured grid (reused by `experiments/figures.py`);
`main` just prints it.

Run:  python -m experiments.exp_safe_set
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import attacks, defenses, harness
from ivgym.backends.synthetic import SyntheticBackend
from ivgym.core import SamplingSpec
from ivgym.harness import EvalResult

from examples.safe_set_strategies import SafeSetSubstitution

# Defaults. The largest batch stays well under the test split
# (N_PROMPTS*N_TOKENS/2) so `batch_means` samples a small fraction without
# replacement -- otherwise every batch collapses to the full-set mean and AUC
# degenerates to 0/1.
N_PROMPTS = 120
N_TOKENS = 256
BATCH_SIZES = (100, 500, 1000, 4000)
EPSILONS = (0.01, 0.05, 0.2, 0.5, 1.0)
DEFENSES = ("token_difr", "cross_entropy", "activation_difr")


@dataclass
class SafeSetRow:
    """One epsilon: its measured deviation rate and the per-(defense, batch) grid."""

    epsilon: float
    deviation_rate: float                       # fraction of tokens that differ from honest
    results: dict[tuple[str, int], EvalResult]  # (defense, batch_size) -> EvalResult


def _deviation_rate(honest_seqs, atk_seqs) -> float:
    """Fraction of tokens where the attack actually claimed a different token."""
    diff = tot = 0
    for hs, as_ in zip(honest_seqs, atk_seqs):
        for h, a in zip(hs.steps, as_.steps):
            tot += 1
            diff += int(h.claimed_token != a.claimed_token)
    return diff / max(tot, 1)


def run_sweep(epsilons=EPSILONS, batch_sizes=BATCH_SIZES,
              n_prompts=N_PROMPTS, n_tokens=N_TOKENS) -> list[SafeSetRow]:
    """Sweep substitution aggressiveness; score every defense at every batch size."""
    backend = SyntheticBackend(vocab=512)
    spec = SamplingSpec()
    defs = [defenses.get(d) for d in DEFENSES]

    honest_seqs = harness.generate_dataset(
        backend, attacks.get("honest"), spec, n_prompts, n_tokens, record_activations=True)
    honest = harness.verify(backend, honest_seqs, spec, defs)

    rows = []
    for eps in epsilons:
        atk = SafeSetSubstitution(name=f"safe_set_eps{eps}", epsilon=eps, logit_eps=eps)
        seqs = harness.generate_dataset(backend, atk, spec, n_prompts, n_tokens,
                                        record_activations=True)
        ascores = harness.verify(backend, seqs, spec, defs)
        res = harness.evaluate(honest, ascores, defs, list(batch_sizes),
                               n_batches=400, winsor_pct=99.9)
        rows.append(SafeSetRow(
            epsilon=eps,
            deviation_rate=_deviation_rate(honest_seqs, seqs),
            results={(r.defense, r.batch_size): r for r in res},
        ))
    return rows


def main():
    rows = run_sweep()
    print(f"SAFE-set substitution vs every defense   (honest pool: "
          f"{N_PROMPTS}x{N_TOKENS} tokens)")
    print("each cell = AUC (TPR@1%);  ~0.50 (0.01) = indistinguishable, "
          "attacker wins;  ->1.0 = caught\n")
    for row in rows:
        print(f"epsilon={row.epsilon:<4}  deviation rate = {row.deviation_rate*100:5.2f}% "
              f"of tokens (this many differ from the honest sample)")
        head = "  " + "batch".rjust(7) + " | " + " ".join(d.rjust(22) for d in DEFENSES)
        print(head + "\n  " + "-" * (len(head) - 2))
        for b in BATCH_SIZES:
            cells = (f"{(r := row.results[(d, b)]).auc:.3f} ({r.tpr_at_1pct:.2f})".rjust(22)
                     for d in DEFENSES)
            print(f"  {b:>7} | " + " ".join(cells))
        print()


if __name__ == "__main__":
    main()
