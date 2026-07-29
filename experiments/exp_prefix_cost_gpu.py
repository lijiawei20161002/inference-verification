"""What a selective audit actually costs -- and a prefix scheduler that fixes it.

The selective tier reports `recompute_ratio`: the fraction of TOKENS a Tier-1
verifier scored. That number silently assumes recompute is billed per token. It
is not. Reading M's logits at generated position `j` requires a prefill of
`[prompt + claimed[:j]]`, so:

  * a sequence's audit cost is set by the DEEPEST position audited in it, and
  * every sequence touched at all pays for its prompt again.

A global top-k over tokens scatters the audit across every sequence, so it pays
nearly the full prefill however small the token fraction looks. This experiment
measures that -- with `HFGPUBackend(lazy_reference=True)`, which refuses to serve
a reference row the verifier has not paid a real prefill for -- and then compares
against `harness.select_prefix_scheduled`, the port of DSpark's hardware-aware
prefix scheduler (arXiv:2607.05147): greedy admission over a globally sorted pool
where the sort key is value per unit of MARGINAL COST, and admitting a request
commits to a contiguous prefix.

Three curves, all under the standardized `EvalConfig`:

  1. nominal token budget  ->  realized prefill cost   (the accounting gap)
  2. detection AUC         ->  realized prefill cost   (the honest Pareto)
  3. measured GPU seconds  ->  realized prefill cost   (that the model is real)

    IVGYM_M=Qwen/Qwen3-1.7B IVGYM_PROXY=Qwen/Qwen3-0.6B \
        python -m experiments.exp_prefix_cost_gpu
Env: IVGYM_M, IVGYM_PROXY, IVGYM_PROMPTS(32), IVGYM_TOKENS(96), IVGYM_BATCH(200),
     IVGYM_ATTACK(quant_4bit), IVGYM_VALUE_FN(tie_margin).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import attacks, harness, verifiers
from ivgym.backends.hf_gpu import HFGPUBackend
from experiments import plot_triage
from ivgym.core import SamplingSpec

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
RES_DIR = ROOT / "docs" / "results"

M = os.environ.get("IVGYM_M", "Qwen/Qwen3-1.7B")
PROXY = os.environ.get("IVGYM_PROXY", "Qwen/Qwen3-0.6B")
N = int(os.environ.get("IVGYM_PROMPTS", 32))
T = int(os.environ.get("IVGYM_TOKENS", 96))
BATCH = int(os.environ.get("IVGYM_BATCH", 200))
ATTACK = os.environ.get("IVGYM_ATTACK", "quant_4bit")
VALUE_FN = os.environ.get("IVGYM_VALUE_FN", "tie_margin")

BUDGETS = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]
SCHEDULERS = ["topk", "prefix"]

def gen(backend, attack, spec, ids):
    return [backend.generate(pid, T, spec, attack) for pid in ids]


def measured_verify(backend, seqs, spec, td, budget, values, scheduler):
    """One selective audit, priced. Drops the reference cache first so this
    budget pays its own prefills rather than inheriting a deeper run's."""
    backend.drop_reference_cache()
    s0 = backend.timed_seconds["reference"]
    c0 = backend.timed_calls["reference"]
    out = harness.verify(backend, seqs, spec, [td], budget=budget, values=values,
                         scheduler=scheduler)
    return out, backend.timed_seconds["reference"] - s0, backend.timed_calls["reference"] - c0


# ------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"Prefill-cost accounting + prefix scheduler   M={M}  proxy={PROXY}")
    print("=" * 78, flush=True)

    backend = HFGPUBackend(model_name=M, proxy_model_name=PROXY, lazy_reference=True)
    spec = SamplingSpec()
    td = verifiers.get("token_difr")
    ids = list(range(N))
    print(f"loaded  M={backend.n_params/1e9:.2f}B  proxy={backend.proxy_n_params/1e9:.2f}B "
          f"(lazy reference: nothing prefilled during generation) "
          f"[{time.time()-t0:.0f}s]", flush=True)

    seq_lens = [T] * N
    honest = gen(backend, attacks.get("honest"), spec, ids)
    prompt_lens = [backend.prompt_len(p) for p in ids]
    full_cost = harness.full_prefill_cost(seq_lens, prompt_lens)
    print(f"full-audit prefill cost = {full_cost} reference-forward tokens "
          f"({N} prompts x (prompt {np.mean(prompt_lens):.1f} avg + {T-1} generated))",
          flush=True)

    h_val = harness.token_values(backend, honest, spec, VALUE_FN)
    h = {}
    for s in SCHEDULERS:
        for b in BUDGETS:
            h[(s, b)] = measured_verify(backend, honest, spec, td, b, h_val, s)
    print(f"honest priced [{time.time()-t0:.0f}s]", flush=True)

    a_seqs = gen(backend, attacks.get(ATTACK), spec, ids)
    a_val = harness.token_values(backend, a_seqs, spec, VALUE_FN)
    a = {}
    for s in SCHEDULERS:
        for b in BUDGETS:
            a[(s, b)] = measured_verify(backend, a_seqs, spec, td, b, a_val, s)
    print(f"attack priced [{time.time()-t0:.0f}s]\n", flush=True)

    curves = {s: {k: [] for k in
                  ("budget", "token_ratio", "prefill_ratio", "auc", "seconds",
                   "prefill_tokens", "n_prefills")} for s in SCHEDULERS}
    print(f"attack={ATTACK}  value_fn={VALUE_FN}")
    for s in SCHEDULERS:
        print(f"\n  scheduler = {s}")
        print(f"  {'budget':>7}{'tokens audited':>16}{'PREFILL cost':>14}"
              f"{'prefills':>10}{'sec':>8}{'AUC':>8}")
        print("  " + "-" * 63)
        for b in BUDGETS:
            hs, h_sec, h_calls = h[(s, b)]
            as_, a_sec, a_calls = a[(s, b)]
            res = harness.evaluate(hs, as_, [td], [BATCH], seed=7)[0]
            tok = 0.5 * (hs.recompute_ratio + as_.recompute_ratio)
            pre = 0.5 * (hs.prefill_ratio + as_.prefill_ratio)
            sec = 0.5 * (h_sec + a_sec)
            curves[s]["budget"].append(b)
            curves[s]["token_ratio"].append(tok)
            curves[s]["prefill_ratio"].append(pre)
            curves[s]["auc"].append(res.auc)
            curves[s]["seconds"].append(sec)
            curves[s]["prefill_tokens"].append(0.5 * (hs.prefill_tokens + as_.prefill_tokens))
            curves[s]["n_prefills"].append(0.5 * (h_calls + a_calls))
            print(f"  {b*100:>6.0f}%{tok*100:>15.1f}%{pre*100:>13.1f}%"
                  f"{0.5*(h_calls+a_calls):>10.0f}{sec:>8.2f}{res.auc:>8.3f}", flush=True)

    payload = {"M": M, "proxy": PROXY, "n": N, "tokens": T, "batch": BATCH,
               "attack": ATTACK, "value_fn": VALUE_FN, "budgets": BUDGETS,
               "schedulers": SCHEDULERS, "full_prefill_cost": full_cost,
               "prompt_lens": prompt_lens, "curves": curves}
    out = RES_DIR / "prefix_cost.json"
    out.write_text(json.dumps(payload, indent=2))

    tk, px = curves["topk"], curves["prefix"]
    print(f"\n  At nominal budget 10%, top-k really costs {px and tk['prefill_ratio'][1]:.1%} "
          f"of a full audit; the prefix schedule at the SAME real cost is the "
          f"comparison panel B makes.")
    print(f"\nwrote {out}")
    plot_triage.main([])
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
