"""Seed-free regime: how much compute can a cheat hide?

Without seed-sync the verifier has only seed-free scores (cross_entropy,
activation_difr, and a TOPLOC-style rank check topk_overlap). For a sweep of
quantization strengths we tune the provider temperature to match honest mean CE,
spoof the activation fingerprint, and report detection across every defense --
including the seed-synced Token-DiFR, shown for contrast.

Two findings drop out:
  * Against {cross_entropy, activation_difr} the attack is indistinguishable
    (AUC ~ 0.5, TPR@1% ~ chance) while genuinely quantizing -- and Token-DiFR
    catches the very same attack the moment seed-sync is available.
  * A rank detector (topk_overlap) is one-sided like Token-DiFR, so it resists
    forward-pass cheating even without seed-sync: it sets a *frontier* on how
    much quantization stays hidden.

`run_sweep` returns the structured grid (reused by `experiments/figures.py`);
`main` just prints it.

Run:  python -m experiments.exp_seed_free
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import attacks, defenses, harness
from ivgym.backends.synthetic import SyntheticBackend
from ivgym.core import SamplingSpec
from ivgym.harness import EvalResult

import examples.custom_strategies     # registers topk_overlap        # noqa: F401
from examples.seed_free_strategies import AdvQuantTempSpoof

N_PROMPTS = 80
N_TOKENS = 256
BATCH = 1000
QUANTS = (0.1, 0.2, 0.3, 0.4, 0.7, 1.0, 1.4)   # compute-saving strength multiplier
TEMP_GRID = np.linspace(0.7, 1.0, 13)          # temperature tuning grid (coarse is plenty)
TUNE_PROMPTS = 20                              # prompts used only for temp tuning

# A quant strength `q` scales both the random and systematic logit perturbation.
EXTRA_BASE, BIAS_BASE = 0.30, 0.10
HIDDEN_TPR = 0.05                              # TPR@1% at/below this counts as "hidden"

# Verifier panels.
SEED_FREE = ("cross_entropy", "activation_difr")  # the natural "no seed sync" panel
RANK = "topk_overlap"                             # seed-free rank detector (TOPLOC-style)
SEED_SYNC = "token_difr"                          # needs seed sync; shown for contrast
ALL_DEFS = (SEED_SYNC, *SEED_FREE, RANK)


def _attack(q: float, temp: float, *, spoof: bool, name: str) -> AdvQuantTempSpoof:
    return AdvQuantTempSpoof(name=name, extra_sigma=EXTRA_BASE * q, bias_sigma=BIAS_BASE * q,
                             provider_temp=temp, spoof_activation=spoof)


@dataclass
class SeedFreeRow:
    """One quant strength: tuned temperature + detection by every defense."""

    quant: float
    tuned_temp: float
    spoof: dict[str, EvalResult]    # detection when the fingerprint is spoofed
    honest_act: EvalResult          # activation_difr when the real fingerprint is reported

    @property
    def hidden(self) -> bool:
        """True if every seed-free defense (incl. rank) stays at ~chance."""
        return all(self.spoof[d].tpr_at_1pct <= HIDDEN_TPR for d in (*SEED_FREE, RANK))


def _mean_ce(backend, atk, spec) -> float:
    seqs = harness.generate_dataset(backend, atk, spec, TUNE_PROMPTS, N_TOKENS)
    ts = harness.verify(backend, seqs, spec, [defenses.get("cross_entropy")])
    return float(ts.scores["cross_entropy"].mean())


def _tune_temp(backend, spec, target_ce: float, q: float) -> float:
    """Pick the provider temperature whose mean CE best matches honest."""
    best_t, best_gap = 1.0, 1e9
    for t in TEMP_GRID:
        gap = abs(_mean_ce(backend, _attack(q, float(t), spoof=True, name="_tune"), spec) - target_ce)
        if gap < best_gap:
            best_t, best_gap = float(t), gap
    return best_t


def run_sweep(quants=QUANTS, batch=BATCH, n_prompts=N_PROMPTS, n_tokens=N_TOKENS):
    """Returns (honest_mean_ce, list[SeedFreeRow]). Tunes temperature per quant."""
    backend = SyntheticBackend(vocab=512)
    spec = SamplingSpec()
    defs = [defenses.get(d) for d in ALL_DEFS]
    act_difr = defenses.get("activation_difr")

    honest_seqs = harness.generate_dataset(
        backend, attacks.get("honest"), spec, n_prompts, n_tokens, record_activations=True)
    honest = harness.verify(backend, honest_seqs, spec, defs)
    target_ce = float(honest.scores["cross_entropy"].mean())

    def evaluate(atk, score_defs):
        seqs = harness.generate_dataset(backend, atk, spec, n_prompts, n_tokens,
                                        record_activations=True)
        scores = harness.verify(backend, seqs, spec, score_defs)
        res = harness.evaluate(honest, scores, score_defs, [batch], n_batches=400, winsor_pct=99.9)
        return {r.defense: r for r in res}

    rows = []
    for q in quants:
        t = _tune_temp(backend, spec, target_ce, q)
        spoof = evaluate(_attack(q, t, spoof=True, name=f"spoof_q{q}"), defs)
        honest_act = evaluate(_attack(q, t, spoof=False, name=f"hact_q{q}"), [act_difr])
        rows.append(SeedFreeRow(quant=q, tuned_temp=t, spoof=spoof,
                                honest_act=honest_act["activation_difr"]))
    return target_ce, rows


def main():
    target_ce, rows = run_sweep()
    cell = lambda r: f"{r.auc:.2f}({r.tpr_at_1pct:.2f})"
    cols = (SEED_SYNC, *SEED_FREE, RANK)

    print(f"Seed-free regime  (honest mean CE={target_ce:.3f}, batch={BATCH})")
    print("cells = AUC(TPR@1%);  ~0.5(0.01)=indistinguishable | seed-free panel = "
          "{cross_entropy, activation_difr}\n")
    head = (f"{'quant q':>8} {'tunedT':>7} | {'token_difr*':>14} | "
            + " ".join(f"{c:>15}" for c in cols[1:]) + "   act(honest)")
    print(head + "\n" + "-" * len(head))
    for row in rows:
        print(f"{row.quant:>8} {row.tuned_temp:>7.3f} | {cell(row.spoof[SEED_SYNC]):>14} | "
              + " ".join(f"{cell(row.spoof[c]):>15}" for c in cols[1:])
              + f"   {cell(row.honest_act)}")

    print("\n* token_difr needs seed-sync (unavailable to this verifier); shown to "
          "contrast.\n  act(honest) = activation_difr if the provider reports its "
          "REAL quantized activation.")
    hidden = [r.quant for r in rows if r.hidden]
    if hidden:
        print(f"\nIndistinguishable compute-cheating frontier: quant q<={max(hidden)} stays "
              f"hidden from\nevery seed-free defense incl. the rank check. Beyond it, "
              f"topk_overlap (rank) fires.")
    else:
        print("\nNo quant strength in the sweep stayed hidden from the rank detector.")


if __name__ == "__main__":
    main()
