"""End-to-end demo of the SELECTIVE-recompute verifier tier on the real backend.

Exercises `harness.verify(budget<1)` -- the value-directed cost-aware tier --
through the SAME HFGPUBackend / attacks / verifiers / evaluate pipeline as
`exp_gpu.py`, so it shows the tier is a first-class citizen alongside a full
recompute (budget=1.0) and a pure Tier-0 (no-recompute) run. For each recompute
budget it reports the realized recompute ratio and the detection AUC (honest vs
attack), calibrating the honest reference with the SAME budget.

    IVGYM_M=Qwen/Qwen3-1.7B IVGYM_PROXY=Qwen/Qwen3-0.6B \
        .venv/bin/python -m experiments.exp_selective_verify_gpu
Env: IVGYM_M, IVGYM_PROXY, IVGYM_PROMPTS(24), IVGYM_TOKENS(96), IVGYM_BATCH(200),
     IVGYM_ATTACK(quant_4bit).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import attacks, harness, verifiers
from ivgym.backends.hf_gpu import HFGPUBackend
from ivgym.core import SamplingSpec

M = os.environ.get("IVGYM_M", "Qwen/Qwen3-1.7B")
PROXY = os.environ.get("IVGYM_PROXY", "Qwen/Qwen3-0.6B")
N = int(os.environ.get("IVGYM_PROMPTS", 24))
T = int(os.environ.get("IVGYM_TOKENS", 96))
BATCH = int(os.environ.get("IVGYM_BATCH", 200))
ATTACK = os.environ.get("IVGYM_ATTACK", "quant_4bit")
BUDGETS = [0.1, 0.25, 0.5, 1.0]


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"Selective-recompute verifier tier (real backend)  M={M}  proxy={PROXY}")
    print("=" * 78)
    backend = HFGPUBackend(model_name=M, proxy_model_name=PROXY)
    print(f"loaded  M={backend.n_params/1e9:.2f}B  proxy={backend.proxy_n_params/1e9:.2f}B "
          f"[{time.time()-t0:.0f}s]", flush=True)
    spec = SamplingSpec()
    td = verifiers.get("token_difr")
    # The tie-margin value signal is the one this selective-recompute Pareto was
    # built around (proxy near-tie-ness, where quant/fp8 flips concentrate). The
    # library default is now entropy H(q); pass value_fn="entropy" to compare.
    VALUE_FN = os.environ.get("IVGYM_VALUE_FN", "tie_margin")

    # HFGPUBackend caches per-prompt logits and OVERWRITES them on each
    # generate_dataset, so every backend read (verify / value / selective) must
    # happen while that config's cache is fresh -- i.e. before generating the next
    # config. So materialise all of honest's scores, THEN all of the attack's.
    honest = harness.generate_dataset(backend, attacks.get("honest"), spec, N, T)
    honest_full = harness.verify(backend, honest, spec, [td])
    h_val = harness.token_values(backend, honest, spec, VALUE_FN)
    h_sel = {b: harness.verify(backend, honest, spec, [td], budget=b,
                               value_fn=VALUE_FN, values=h_val) for b in BUDGETS}

    attack = harness.generate_dataset(backend, attacks.get(ATTACK), spec, N, T)
    attack_full = harness.verify(backend, attack, spec, [td])
    a_val = harness.token_values(backend, attack, spec, VALUE_FN)
    a_sel = {b: harness.verify(backend, attack, spec, [td], budget=b,
                               value_fn=VALUE_FN, values=a_val) for b in BUDGETS}
    full = harness.evaluate(honest_full, attack_full, [td], [BATCH], seed=7)[0]
    print(f"generated + verified [{time.time()-t0:.0f}s]\n", flush=True)

    print(f"attack = {ATTACK};  full-recompute token_difr AUC = {full.auc:.3f} "
          f"(TPR@1% {full.tpr_at_1pct:.2f})")
    print(f"\n  {'budget':>7}{'recompute ratio':>17}{'selective AUC':>15}{'TPR@1%':>9}")
    print("  " + "-" * 46)
    for bud in BUDGETS:
        h_scores = h_sel[bud]; r_h = h_scores.recompute_ratio
        a_scores = a_sel[bud]; r_a = a_scores.recompute_ratio
        res = harness.evaluate(h_scores, a_scores, [td], [BATCH], seed=7)[0]
        print(f"  {bud*100:>6.0f}%{(0.5*(r_h+r_a))*100:>16.1f}%"
              f"{res.auc:>15.3f}{res.tpr_at_1pct:>9.2f}", flush=True)

    print(f"\n  proxy-triaged recompute recovers the full-recompute AUC at a fraction of the")
    print(f"  M-calls; the ratio column is the measured fraction of tokens re-run on M.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
