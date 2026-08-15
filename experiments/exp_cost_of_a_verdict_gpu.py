"""What does one verdict cost? -- the exchange rate between compute and accuracy.

Every other cost number in this repo is a *rate*: FLOPs per sequence
(`exp_io_detector_gpu`), seconds per prefill (`exp_spec_substitution_gpu`),
prefill tokens per audit (`exp_prefix_cost_gpu`). None of them is the quantity a
client actually buys, which is **a verdict**: "is this provider serving what it
promised, at a false-accusation rate I can live with?"

A verdict has a price, and it factors exactly into two measured terms:

    tokens per verdict   b*(d') = (delta* / d')^2         [ivgym.signal]
    price per token      c(tier) = measured GPU-seconds / token
    ------------------------------------------------------------------
    cost of a verdict    = b*(d') x c(tier)

with `delta* = 3.767` the batch separation that reaches standardized pAUC 0.90 at
FPR <= 0.5% -- the repo's protocol, not a new one. Both factors are measured here
on one model, in one run, so the product is a real number and not a composition of
numbers from different configs:

  * `d'` per (attack, verifier) from the honest and deviating token pools, on
    exactly the scale `harness.evaluate` scores (`signal.per_token_stats`), with a
    **sequence-level bootstrap** CI -- tokens inside a sequence are not independent,
    so a token bootstrap would understate the error on `d'` and therefore on price.
  * `c(tier)` from the backend's GPU-synchronised timers: the full-M reference
    prefill (Tier-1), the cheap-proxy prefill (Tier-0 accept_rate / surface_stat),
    and the tokenizer decode (surface_tokens, the zero-forward-pass floor).

Why the product is the point. Cost and accuracy are usually reported as if a
verifier picks a point on a Pareto curve of *rates* -- cheap detector, weak;
expensive detector, strong. It is not a curve of rates. A detector 12x cheaper per
token that needs 400x more tokens is 30x more *expensive* per verdict, and a
detector with `d' <= 0` is not on the curve at all: its price is infinite, no
budget reaches a verdict. This experiment produces that grid.

Two prices are reported per cell, because a deployment pays both:

  marginal verdict : b*(d') tokens at the tier's price.
  first verdict    : plus the honest calibration pool the verdict is scored
                     against. `EvalConfig.max_pool_ratio` caps batch/pool at 10%
                     and `evaluate` splits honest 50/50 into calibration and eval
                     null, so a legitimate batch of b* needs 20*b* honest tokens.
                     That pool amortizes over many verdicts; it is reported
                     separately rather than folded in.

And one deployment ratio, which is the number that decides whether continuous
verification is affordable at all: if a client wants one verdict per `1M` served
tokens, the audit costs `b*/1e6` of its traffic at `FLOPs(tier)/FLOPs(serving)`
each -- reported as `overhead_frac`.

    IVGYM_M=Qwen/Qwen3-1.7B IVGYM_PROXY=Qwen/Qwen3-0.6B \
        python -m experiments.exp_cost_of_a_verdict_gpu

Env: IVGYM_M(Qwen/Qwen3-1.7B), IVGYM_PROXY(Qwen/Qwen3-0.6B),
     IVGYM_PROMPTS(80), IVGYM_TOKENS(256), IVGYM_BOOT(400),
     IVGYM_GPU_USD_PER_HR(2.50), IVGYM_ATTACKS(comma-separated),
     IVGYM_TAG(suffix for the output artifacts, so a deep-pool run over a few
     attacks does not overwrite the wide grid).

Writes `docs/results/cost_of_a_verdict.json` and, for
`exp_sequential_verdict.py` to re-analyse without a GPU,
`docs/results/cost_of_a_verdict_scores.npz` (the per-token score arrays).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import attacks, harness, signal, verifiers
from ivgym.backends.hf_gpu import HFGPUBackend
from ivgym.core import SamplingSpec

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results"

M = os.environ.get("IVGYM_M", "Qwen/Qwen3-1.7B")
PROXY = os.environ.get("IVGYM_PROXY", "Qwen/Qwen3-0.6B")
N = int(os.environ.get("IVGYM_PROMPTS", 80))
T = int(os.environ.get("IVGYM_TOKENS", 256))
N_BOOT = int(os.environ.get("IVGYM_BOOT", 400))
USD_PER_HR = float(os.environ.get("IVGYM_GPU_USD_PER_HR", 2.50))
ATTACKS = os.environ.get(
    "IVGYM_ATTACKS", "quant_4bit,kv_fp8,temp_1.1,seed_43,bug_k32").split(",")
# A run at a different pool size or attack set is a different artifact, not a
# replacement for this one: `exp_sequential_verdict` needs a pool large enough that
# a batch stays under the 10% ceiling, which is a much deeper pool over far fewer
# attacks, and that run must not overwrite the wide grid the cost figures rest on.
TAG = os.environ.get("IVGYM_TAG", "")
STEM = f"cost_of_a_verdict{('_' + TAG) if TAG else ''}"

# Tier-1 recomputes M; Tier-0 never does. The tier fixes the price per token.
TIER1 = ["token_difr", "cross_entropy", "token_toploc", "activation_difr"]
TIER0 = ["accept_rate", "surface_stat", "surface_tokens"]
ALL = TIER1 + TIER0
# which measured timer pays for each verifier's evidence
PRICED_BY = {**{v: "reference" for v in TIER1},
             "accept_rate": "proxy", "surface_stat": "proxy",
             "surface_tokens": "decode"}
TARGET_PAUC = 0.90


def d_prime_ci(h_by_seq: list[np.ndarray], a_by_seq: list[np.ndarray],
               n_boot: int, seed: int = 0) -> tuple[float, float, float]:
    """`d'` and a 90% sequence-bootstrap interval.

    Resamples whole SEQUENCES with replacement on both sides, because tokens
    within a sequence share a prompt and a prefix and are visibly correlated; a
    token-level bootstrap would report an interval several times too narrow and
    the price of a verdict -- which goes as `1/d'^2` -- would inherit it.
    """
    h, a = np.concatenate(h_by_seq), np.concatenate(a_by_seq)
    point = signal.per_token_stats(h, a)["d_prime"]
    rng = np.random.default_rng(seed)
    nh, na = len(h_by_seq), len(a_by_seq)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        hb = np.concatenate([h_by_seq[j] for j in rng.integers(0, nh, nh)])
        ab = np.concatenate([a_by_seq[j] for j in rng.integers(0, na, na)])
        draws[i] = signal.per_token_stats(hb, ab)["d_prime"]
    return point, float(np.quantile(draws, 0.05)), float(np.quantile(draws, 0.95))


def measure_prices(backend, prompt_ids: list[int], warmup: int = 3) -> dict:
    """GPU-synchronised seconds per [prompt+claimed] sequence for each priced
    stage, measured on the same sequences the detectors were scored on.

    Not read off the generation pass: that one interleaves the provider's own
    forward passes with the reference prefill, and a verifier does not pay for the
    provider's. Here each stage is timed alone, after a warmup, with the reference
    cache dropped so the prefill is really recomputed.
    """
    for pid in prompt_ids[:warmup]:                      # kernels/autotune warmup
        backend.drop_reference_cache()
        backend.prefill_reference(pid, T)
    backend.reset_cost()
    for pid in prompt_ids:
        backend.drop_reference_cache()
        backend.prefill_reference(pid, T)                # timed: "reference"
    for pid in prompt_ids:
        backend._populate_proxy_cache(pid, backend._prompt_ids(pid),
                                      backend._claimed[pid], T)   # timed: "proxy"
    for pid in prompt_ids:
        backend.decode(backend._claimed[pid])            # timed: "decode"
    sec, calls = backend.timed_seconds, backend.timed_calls
    return {k: sec[k] / max(calls[k], 1) for k in ("reference", "proxy", "decode")}


def main() -> None:
    t0 = time.time()
    print("=" * 84)
    print(f"The cost of a verdict   M={M}  proxy={PROXY}")
    print(f"  pool = {N} prompts x {T} tokens per config; "
          f"target = standardized pAUC {TARGET_PAUC} @ FPR <= 0.5%")
    print("=" * 84, flush=True)

    backend = HFGPUBackend(model_name=M, proxy_model_name=PROXY)
    spec = SamplingSpec()
    dets = [verifiers.get(n) for n in ALL]
    prompt_ids = list(range(N))

    def pool(attack_name: str):
        seqs = harness.generate_dataset(backend, attacks.get(attack_name), spec, N, T,
                                        record_activations=True)
        ts = harness.verify(backend, seqs, spec, dets)
        print(f"  {attack_name:>12}: {len(ts.scores['token_difr'])} tokens "
              f"[{time.time()-t0:.0f}s]", flush=True)
        return ts

    honest = pool("honest")
    attacked = {name: pool(name) for name in ATTACKS}

    # ---- the price side: measured seconds and analytic FLOPs per token -------
    print("\nmeasuring verifier prices (timed alone, cache dropped) ...", flush=True)
    sec_per_seq = measure_prices(backend, prompt_ids)
    n_embed = backend.vocab * backend.hidden_dim
    m_nonembed = max(backend.n_params - n_embed, 0)
    q_hidden = int(backend.proxy_model.config.hidden_size)
    q_nonembed = max(backend.proxy_n_params - backend.vocab * q_hidden, 0)
    # FLOPs to VERIFY one token = one prefill row, 2*N_non_embed. FLOPs to SERVE
    # one token is the same 2*N_non_embed of M, so the ratio below is the fraction
    # of the provider's own compute an audit of that token costs.
    flops_tok = {"reference": 2.0 * m_nonembed, "proxy": 2.0 * q_nonembed,
                 "decode": 0.0}
    serve_flops_tok = 2.0 * m_nonembed
    price = {}
    for v in ALL:
        stage = PRICED_BY[v]
        s_tok = sec_per_seq[stage] / T
        price[v] = {"stage": stage, "sec_per_token": s_tok,
                    "flops_per_token": flops_tok[stage],
                    "usd_per_1m_tokens": s_tok * 1e6 / 3600.0 * USD_PER_HR,
                    "flops_vs_serving": flops_tok[stage] / serve_flops_tok}

    print(f"\n{'verifier':>16} {'tier':>5} {'stage':>10} {'ms/seq':>9} {'us/token':>9} "
          f"{'GFLOP/tok':>10} {'$/1M tok':>9}")
    for v in ALL:
        p = price[v]
        print(f"{v:>16} {(1 if v in TIER1 else 0):>5} {p['stage']:>10} "
              f"{sec_per_seq[p['stage']]*1e3:>9.2f} {p['sec_per_token']*1e6:>9.1f} "
              f"{p['flops_per_token']/1e9:>10.2f} {p['usd_per_1m_tokens']:>9.4f}")

    # ---- the accuracy side: d' per cell, and the tokens it nominates ---------
    def by_seq(ts, name):
        return list(np.asarray(ts.scores[name], float).reshape(N, T))

    delta_star = signal.delta_for_pauc(TARGET_PAUC)
    cells: dict[str, dict] = {}
    print(f"\nper-token d' (sequence-bootstrap 90% CI), tokens per verdict, and price "
          f"[delta* = {delta_star:.3f}]")
    for atk in ATTACKS:
        cells[atk] = {}
        print(f"\n  {atk}")
        print(f"    {'verifier':>16} {'d prime':>20} {'tokens/verdict':>16} "
              f"{'GPU-s':>9} {'USD':>10} {'overhead':>10}")
        for v in ALL:
            dp, lo, hi = d_prime_ci(by_seq(honest, v), by_seq(attacked[atk], v), N_BOOT)
            b = signal.batch_for_pauc(dp, TARGET_PAUC)
            b_lo = signal.batch_for_pauc(hi, TARGET_PAUC)   # bigger d' -> cheaper
            b_hi = signal.batch_for_pauc(lo, TARGET_PAUC)
            p = price[v]
            reach = b > 0
            sec = b * p["sec_per_token"] if reach else float("inf")
            cells[atk][v] = {
                "d_prime": dp, "d_prime_ci": [lo, hi],
                "tokens_per_verdict": int(b), "tokens_ci": [int(b_lo), int(b_hi)],
                "reachable": bool(reach),
                "gpu_seconds_per_verdict": sec,
                "usd_per_verdict": (sec / 3600.0 * USD_PER_HR) if reach else float("inf"),
                # honest tokens the verdict must be scored against: batch/pool <= 10%
                # of the eval null, and evaluate splits honest 50/50 -> 20 x batch.
                "calibration_tokens": int(20 * b) if reach else -1,
                # fraction of the provider's own compute, if one verdict is wanted
                # per 1M served tokens
                "overhead_frac": (b / 1e6 * p["flops_vs_serving"]) if reach else float("inf"),
            }
            c = cells[atk][v]
            tok = f"{b:,}" if reach else "unreachable"
            sec_s = f"{sec:.2f}" if reach else "inf"
            usd_s = f"{c['usd_per_verdict']:.2e}" if reach else "inf"
            ovh = f"{c['overhead_frac']:.2%}" if reach else "inf"
            print(f"    {v:>16} {dp:>+8.4f} [{lo:+.4f},{hi:+.4f}] {tok:>16} "
                  f"{sec_s:>9} {usd_s:>10} {ovh:>10}", flush=True)

    # ---- the control: honest against honest must price at infinity ----------
    rng = np.random.default_rng(11)
    perm = rng.permutation(N)
    ctrl = {}
    for v in ALL:
        seqs = by_seq(honest, v)
        h1 = [seqs[i] for i in perm[: N // 2]]
        h2 = [seqs[i] for i in perm[N // 2:]]
        dp, lo, hi = d_prime_ci(h1, h2, N_BOOT)
        ctrl[v] = {"d_prime": dp, "d_prime_ci": [lo, hi],
                   "tokens_per_verdict": int(signal.batch_for_pauc(dp, TARGET_PAUC))}
    print("\n  control: honest vs honest (disjoint halves of the SAME pool) -- a "
          "verdict must be unreachable here")
    for v in ALL:
        c = ctrl[v]
        print(f"    {v:>16} d' = {c['d_prime']:+.4f} "
              f"[{c['d_prime_ci'][0]:+.4f},{c['d_prime_ci'][1]:+.4f}]  "
              f"tokens = {c['tokens_per_verdict'] if c['tokens_per_verdict'] > 0 else 'unreachable'}")

    payload = {
        "M": M, "proxy": PROXY, "n_prompts": N, "tokens": T,
        "pool_tokens_per_config": N * T, "n_boot": N_BOOT,
        "usd_per_gpu_hour": USD_PER_HR, "target_pauc": TARGET_PAUC,
        "max_fpr": 0.005, "delta_star": float(delta_star),
        "m_params": int(backend.n_params), "proxy_params": int(backend.proxy_n_params),
        "serve_flops_per_token": serve_flops_tok,
        "seconds_per_sequence": sec_per_seq,
        "price": price, "attacks": ATTACKS, "verifiers": ALL,
        "tier": {v: (1 if v in TIER1 else 0) for v in ALL},
        "cells": cells, "honest_control": ctrl,
        "elapsed_s": time.time() - t0,
    }
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"{STEM}.json").write_text(json.dumps(payload, indent=1))

    # per-token scores, so the sequential-test analysis needs no GPU
    arrays = {f"honest__{v}": honest.scores[v] for v in ALL}
    for atk in ATTACKS:
        arrays.update({f"{atk}__{v}": attacked[atk].scores[v] for v in ALL})
    np.savez_compressed(RES / f"{STEM}_scores.npz", n_prompts=N, tokens=T, **arrays)
    print(f"\nwrote {RES/f'{STEM}.json'} and {RES/f'{STEM}_scores.npz'}  "
          f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
