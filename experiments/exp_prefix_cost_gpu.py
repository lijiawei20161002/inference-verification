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

*What the scheduler admits against.* The two axes are independent: `scheduler`
decides the SHAPE of the audit (scattered top-k vs contiguous prefixes) and
`value_fn` decides WHICH tokens look worth auditing. This experiment crosses them,
so `IVGYM_VALUE_FNS=tie_margin,learned` asks whether value-density admission
improves when the value signal is the calibrated confidence head instead of a
hand-picked guess. The head is not refit here -- it is loaded from
`docs/results/confidence_head_params.json`, exactly the head
`exp_confidence_head_gpu.py` fit and calibrated, and this run then uses that
file's `eval_prompt_ids` so the head is scoring prompts it never trained on (and
the same ones the Pareto figure reports).

    IVGYM_M=Qwen/Qwen3-1.7B IVGYM_PROXY=Qwen/Qwen3-0.6B \
        python -m experiments.exp_prefix_cost_gpu
Env: IVGYM_M, IVGYM_PROXY, IVGYM_PROMPTS(64), IVGYM_TOKENS(128), IVGYM_BATCH(400),
     IVGYM_ATTACK(quant_2bit), IVGYM_VALUE_FNS(tie_margin,entropy,learned),
     IVGYM_SEEDS(5),
     IVGYM_PROMPT_OFFSET(auto: the head's eval range when a learned head is used).
The defaults reproduce the doc's headline run (~40 min on an H100-80GB); they are
chosen to keep the batch under 10% of the honest split, which § 3 of the doc shows
is the difference between a real AUC and an inflated one.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import attacks, harness, triage, verifiers
from ivgym.backends.hf_gpu import HFGPUBackend
from experiments import plot_triage
from ivgym.core import SamplingSpec

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
RES_DIR = ROOT / "docs" / "results"
HEAD_PARAMS = RES_DIR / "confidence_head_params.json"

M = os.environ.get("IVGYM_M", "Qwen/Qwen3-1.7B")
PROXY = os.environ.get("IVGYM_PROXY", "Qwen/Qwen3-0.6B")
# Defaults ARE the headline configuration of docs/TRIAGE_AND_AUDIT_COST.md: batch
# 400 is 9.8% of the 8192-token honest split (under the ~10% ratio ceiling that
# § 3 of that doc shows is what inflated this repo's older AUCs), and quant_2bit
# is the deviation strong enough to leave headroom at that ratio. Weakening any of
# these reproduces a superseded run -- see § 3 before overriding them.
N = int(os.environ.get("IVGYM_PROMPTS", 64))
T = int(os.environ.get("IVGYM_TOKENS", 128))
BATCH = int(os.environ.get("IVGYM_BATCH", 400))
N_SEED = int(os.environ.get("IVGYM_SEEDS", 5))
ATTACK = os.environ.get("IVGYM_ATTACK", "quant_2bit")
# Historically one value signal (`IVGYM_VALUE_FN`); still honoured.
VALUE_FNS = [v for v in os.environ.get(
    "IVGYM_VALUE_FNS", os.environ.get("IVGYM_VALUE_FN",
                                      "tie_margin,entropy,learned")).split(",") if v]

BUDGETS = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]
SCHEDULERS = ["topk", "prefix"]


def load_head_value_fns(names):
    """Register any requested `learned`/`oracle` value fn from the saved head.

    Returns the prompt-id offset this run must use: the confidence-head
    experiment's OWN eval range. The head was fit on honest data from prompts
    `train_prompt_ids`, so scoring it on those prompts here would be measuring a
    head on its training set -- and the offset also makes the two experiments
    report on the identical prompts, which is what lets their Pareto curves be
    read against each other.
    """
    wanted = [n for n in names if n in ("learned", "oracle")]
    if not wanted:
        return int(os.environ.get("IVGYM_PROMPT_OFFSET", 0)), None
    if not HEAD_PARAMS.exists():
        raise SystemExit(
            f"value_fn {wanted} needs a fitted head at {HEAD_PARAMS}; run\n"
            f"  IVGYM_M={M} IVGYM_PROXY={PROXY} "
            f"python -m experiments.exp_confidence_head_gpu\nfirst.")
    p = json.loads(HEAD_PARAMS.read_text())
    if (p["M"], p["proxy"]) != (M, PROXY):
        raise SystemExit(f"head was fit for M={p['M']} proxy={p['proxy']}, "
                         f"but this run uses M={M} proxy={PROXY}")
    for n in wanted:
        verifiers.register_value_fn(
            n, triage.head_value_fn(triage.ConfidenceHead.from_dict(p[n])))
    off = int(os.environ.get("IVGYM_PROMPT_OFFSET", min(p["eval_prompt_ids"])))
    return off, p


def gen(backend, attack, spec, ids):
    return [backend.generate(pid, T, spec, attack) for pid in ids]


def auc_ms(h_sc, a_sc, td):
    """AUC@FPR<=0.5% as (mean, sd) over `N_SEED` protocol seeds -- the AUC axis of
    panel B is the noisy one, so it is reported with its noise."""
    v = [harness.evaluate(h_sc, a_sc, [td], [BATCH], seed=7 + s)[0].auc
         for s in range(N_SEED)]
    return float(np.mean(v)), float(np.std(v))


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

    offset, head_meta = load_head_value_fns(VALUE_FNS)
    backend = HFGPUBackend(model_name=M, proxy_model_name=PROXY, lazy_reference=True)
    spec = SamplingSpec()
    td = verifiers.get("token_difr")
    ids = list(range(offset, offset + N))
    assert offset + N <= len(backend.prompts), (
        f"prompt bank holds {len(backend.prompts)}; this run needs ids "
        f"{offset}..{offset+N-1}")
    print(f"loaded  M={backend.n_params/1e9:.2f}B  proxy={backend.proxy_n_params/1e9:.2f}B "
          f"(lazy reference: nothing prefilled during generation) "
          f"[{time.time()-t0:.0f}s]")
    print(f"value signals: {VALUE_FNS}   prompts {ids[0]}..{ids[-1]}"
          + (f"   (head fit on {head_meta['train_prompt_ids'][0]}.."
             f"{head_meta['train_prompt_ids'][-1]}, disjoint)" if head_meta else ""),
          flush=True)

    seq_lens = [T] * N
    honest = gen(backend, attacks.get("honest"), spec, ids)
    prompt_lens = [backend.prompt_len(p) for p in ids]
    full_cost = harness.full_prefill_cost(seq_lens, prompt_lens)
    print(f"full-audit prefill cost = {full_cost} reference-forward tokens "
          f"({N} prompts x (prompt {np.mean(prompt_lens):.1f} avg + {T-1} generated))",
          flush=True)

    # Price every (value signal, scheduler, budget) cell. The value signals are
    # Tier-0 reads of the same cached proxy rows, so adding one costs no forward
    # pass of M -- only the audits themselves are paid for.
    h_val = {v: harness.token_values(backend, honest, spec, v) for v in VALUE_FNS}
    h = {}
    for v in VALUE_FNS:
        for s in SCHEDULERS:
            for b in BUDGETS:
                h[(v, s, b)] = measured_verify(backend, honest, spec, td, b, h_val[v], s)
    print(f"honest priced [{time.time()-t0:.0f}s]", flush=True)

    a_seqs = gen(backend, attacks.get(ATTACK), spec, ids)
    a_val = {v: harness.token_values(backend, a_seqs, spec, v) for v in VALUE_FNS}
    a = {}
    for v in VALUE_FNS:
        for s in SCHEDULERS:
            for b in BUDGETS:
                a[(v, s, b)] = measured_verify(backend, a_seqs, spec, td, b, a_val[v], s)
    print(f"attack priced [{time.time()-t0:.0f}s]\n", flush=True)

    keys = ("budget", "token_ratio", "prefill_ratio", "auc", "auc_sd", "seconds",
            "prefill_tokens", "n_prefills")
    curves = {v: {s: {k: [] for k in keys} for s in SCHEDULERS} for v in VALUE_FNS}
    print(f"attack={ATTACK}  batch={BATCH}  AUC = mean +- sd over {N_SEED} seeds")
    for v in VALUE_FNS:
        for s in SCHEDULERS:
            print(f"\n  value_fn = {v}   scheduler = {s}")
            print(f"  {'budget':>7}{'tokens audited':>16}{'PREFILL cost':>14}"
                  f"{'prefills':>10}{'sec':>8}{'AUC':>16}")
            print("  " + "-" * 71)
            for b in BUDGETS:
                hs, h_sec, h_calls = h[(v, s, b)]
                as_, a_sec, a_calls = a[(v, s, b)]
                m, sd = auc_ms(hs, as_, td)
                tok = 0.5 * (hs.recompute_ratio + as_.recompute_ratio)
                pre = 0.5 * (hs.prefill_ratio + as_.prefill_ratio)
                sec = 0.5 * (h_sec + a_sec)
                c = curves[v][s]
                c["budget"].append(b)
                c["token_ratio"].append(tok)
                c["prefill_ratio"].append(pre)
                c["auc"].append(m)
                c["auc_sd"].append(sd)
                c["seconds"].append(sec)
                c["prefill_tokens"].append(0.5 * (hs.prefill_tokens + as_.prefill_tokens))
                c["n_prefills"].append(0.5 * (h_calls + a_calls))
                print(f"  {b*100:>6.0f}%{tok*100:>15.1f}%{pre*100:>13.1f}%"
                      f"{0.5*(h_calls+a_calls):>10.0f}{sec:>8.2f}"
                      f"{m:>11.3f}+-{sd:>4.3f}", flush=True)

    payload = {"M": M, "proxy": PROXY, "n": N, "tokens": T, "batch": BATCH,
               "attack": ATTACK, "value_fn": VALUE_FNS[0], "value_fns": VALUE_FNS,
               "budgets": BUDGETS, "n_seed": N_SEED,
               "schedulers": SCHEDULERS, "full_prefill_cost": full_cost,
               "prompt_ids": ids, "prompt_lens": prompt_lens,
               "head_fit_on": (head_meta or {}).get("train_prompt_ids"),
               # `curves` keeps the flat single-value-signal shape the figure and
               # every existing reader expect; `curves_by_value` carries the cross.
               "curves": curves[VALUE_FNS[0]], "curves_by_value": curves}
    out = RES_DIR / "prefix_cost.json"
    out.write_text(json.dumps(payload, indent=2))

    tk = curves[VALUE_FNS[0]]["topk"]
    print(f"\n  At nominal budget 10%, top-k really costs {tk['prefill_ratio'][1]:.1%} "
          f"of a full audit; the prefix schedule at the SAME real cost is the "
          f"comparison panel B makes.")
    if len(VALUE_FNS) > 1:
        print("\n  Prefix-scheduled AUC by value signal (step 5: does admitting "
              "against the\n  calibrated head beat admitting against a guessed signal?)")
        print(f"  {'budget':>7}" + "".join(f"{v:>16}" for v in VALUE_FNS))
        for i, b in enumerate(BUDGETS):
            print(f"  {b*100:>6.0f}%" + "".join(
                f"{curves[v]['prefix']['auc'][i]:>11.3f}"
                f"+-{curves[v]['prefix']['auc_sd'][i]:>4.3f}" for v in VALUE_FNS))
    print(f"\nwrote {out}")
    plot_triage.main([])
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
