"""Reproduce DiFR Figure 2 (qualitatively) on the synthetic backend.

An adversary quantizes the model, then tunes its sampling temperature until
its mean cross-entropy matches the honest configuration. Cross-entropy
detection collapses toward chance; seed-synchronized Token-DiFR still flags it.

Run:  .venv/bin/python -m experiments.exp_adversarial_temp
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from difr import attacks, defenses, harness
from difr.attacks import AdversarialTemp
from difr.backends.synthetic import SyntheticBackend
from difr.core import SamplingSpec

N_PROMPTS = 80
N_TOKENS = 256
BATCH_SIZES = [1, 10, 100, 1000, 5000]


def mean_ce(backend, attack, spec) -> float:
    seqs = harness.generate_dataset(backend, attack, spec, 30, N_TOKENS)
    ts = harness.verify(backend, seqs, spec, [defenses.get("cross_entropy")])
    return float(ts.scores["cross_entropy"].mean())


# Strong quantization: meaningfully degrades token quality (raises CE), so the
# adversary must lower temperature to disguise it.
QUANT = dict(extra_sigma=0.55, bias_sigma=0.15, act_sigma=0.30)


def tune_adv_temp(backend, spec, target_ce: float) -> float:
    """Find provider temperature so quantized attack matches honest mean CE."""
    best_t, best_gap = 1.0, 1e9
    for t in np.linspace(0.4, 1.0, 25):
        atk = AdversarialTemp(provider_temp=float(t), **QUANT)
        gap = abs(mean_ce(backend, atk, spec) - target_ce)
        if gap < best_gap:
            best_t, best_gap = float(t), gap
    return best_t


def main():
    backend = SyntheticBackend(vocab=512)
    spec = SamplingSpec()
    defs = [defenses.get("token_difr"), defenses.get("cross_entropy")]

    honest = harness.generate_dataset(backend, attacks.get("honest"), spec, N_PROMPTS, N_TOKENS)
    honest_scores = harness.verify(backend, honest, spec, defs)
    target_ce = honest_scores.scores["cross_entropy"].mean()

    adv_t = tune_adv_temp(backend, spec, target_ce)
    atk = AdversarialTemp(provider_temp=adv_t, **QUANT)
    atk_seqs = harness.generate_dataset(backend, atk, spec, N_PROMPTS, N_TOKENS)
    atk_scores = harness.verify(backend, atk_seqs, spec, defs)

    print(f"honest mean CE      = {target_ce:.4f}")
    print(f"adversary temp      = {adv_t:.3f}")
    print(f"adversary mean CE   = {atk_scores.scores['cross_entropy'].mean():.4f}  "
          f"(matched -> CE detector blinded)")
    print(f"adversary mean DiFR = {atk_scores.scores['token_difr'].mean():.4f}  "
          f"(honest = {honest_scores.scores['token_difr'].mean():.4f})\n")

    results = harness.evaluate(honest_scores, atk_scores, defs, BATCH_SIZES,
                               n_batches=400, winsor_pct=99.9)

    print(f"{'batch':>7} | {'token_difr AUC':>15} {'TPR@1%':>8} | "
          f"{'cross_ent AUC':>14} {'TPR@1%':>8}")
    print("-" * 64)
    by_b: dict[int, dict[str, harness.EvalResult]] = {}
    for r in results:
        by_b.setdefault(r.batch_size, {})[r.defense] = r
    for b in BATCH_SIZES:
        td = by_b[b]["token_difr"]
        ce = by_b[b]["cross_entropy"]
        print(f"{b:>7} | {td.auc:>15.4f} {td.tpr_at_1pct:>8.3f} | "
              f"{ce.auc:>14.4f} {ce.tpr_at_1pct:>8.3f}")


if __name__ == "__main__":
    main()
