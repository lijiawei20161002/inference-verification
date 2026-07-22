"""One point on the performance-vs-cost curve: detection AUC vs verifier cost for
ONE reference model M (optionally policed by ONE same-family cheap proxy q).

This is the per-model engine behind the systematic sweep (`run_cost_curve_sweep`)
and the mega figure (`plot_mega_cost_curve`). It runs the SAME experiment for
every (family, size) so the resulting curves are apples-to-apples:

  * **performance** = the repo's headline detection metric, standardized partial
    AUC @ FPR<=0.5% (`harness.EvalConfig`), honest vs each attack, for two
    verifiers:
      - `token_difr`  -- the FULL RECOMPUTE of M (expensive, correct).
      - `surface_stat` -- a cheap same-family PROXY q reading its own logits
        (cheap, weaker). Only scored when a real proxy is configured.
  * **cost** = verifier FLOPs / sequence (`2 * N_non_embed * T`, T = prompt +
    claimed tokens): the full-M recompute for `token_difr`, the proxy prefill for
    `surface_stat`. This is the deterministic x-axis (measured GPU seconds are
    also recorded, as a cross-check, but vary with host load).

Everything is driven off `ivgym.model_registry` so family / label / params are
never re-typed here. The result for one model lands as
`docs/results/cost_curve/<tag>.json`.

Run (one model; the sweep sets these per model):
    IVGYM_M=Qwen/Qwen3-4B IVGYM_PROXY=Qwen/Qwen3-0.6B \
        python -m experiments.exp_cost_curve_gpu

Env overrides:
  IVGYM_M        reference model M HF id      (required)
  IVGYM_PROXY    same-family cheap proxy q    (optional; omit for the smallest
                 model in a family, which has no smaller sibling to police it)
  IVGYM_PROMPTS  prompts per pool             (default 16; honest+null use
                 disjoint ranges [0,N),[N,2N), so keep 2N <= prompt bank)
  IVGYM_TOKENS   tokens per sequence          (default 80)
  IVGYM_BATCH    batch size for the statistic (default 48)
  IVGYM_NBATCH   resampled batches per split  (default 4000)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import attacks, harness, verifiers
from ivgym.backends.hf_gpu import HFGPUBackend, DEFAULT_PROMPTS
from ivgym.core import SamplingSpec
from ivgym.model_registry import identity

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "results" / "cost_curve"

M_NAME = os.environ["IVGYM_M"]
PROXY = os.environ.get("IVGYM_PROXY") or None
N_PROMPTS = int(os.environ.get("IVGYM_PROMPTS", 16))
N_TOKENS = int(os.environ.get("IVGYM_TOKENS", 80))
BATCH = int(os.environ.get("IVGYM_BATCH", 48))
N_BATCHES = int(os.environ.get("IVGYM_NBATCH", 4000))

# The paper's canonical attacks (the same set exp_gpu / exp_io_detector sweep).
CORE_ATTACKS = ("quant_4bit", "kv_fp8", "temp_1.1", "seed_43", "bug_k2", "bug_k32")


def _tag(m) -> str:
    """Filesystem-friendly per-model tag, e.g. 'qwen3-4b', matching the label."""
    return m.label.lower().replace(" ", "-")


def flops_2nt(n_params: int, vocab: int, hidden: int, T: int) -> float:
    """FLOPs for one prefill of `T` tokens, 2*N_non_embed*T. N_non_embed excludes
    the embedding table (index reads, not matmuls) -- the standard transformer
    forward-FLOP approximation, identical to exp_io_detector_gpu.compute_flops."""
    non_embed = max(n_params - vocab * hidden, 0)
    return 2.0 * non_embed * T


def run() -> dict:
    if 2 * N_PROMPTS > len(DEFAULT_PROMPTS):
        print(f"  NOTE: 2*PROMPTS ({2*N_PROMPTS}) > prompt bank ({len(DEFAULT_PROMPTS)}); "
              "honest and null pools will share text.", flush=True)

    m_id = identity(M_NAME)
    proxy_id = identity(PROXY) if PROXY else None
    if proxy_id is not None and proxy_id.tokenizer != m_id.tokenizer:
        raise ValueError(
            f"proxy {PROXY} tokenizer '{proxy_id.tokenizer}' != M {M_NAME} "
            f"tokenizer '{m_id.tokenizer}'; a proxy detector reads M's claimed "
            "token ids against the proxy's logits, so they must share a tokenizer.")

    t0 = time.time()
    print(f"loading M={M_NAME}" + (f"  + proxy={PROXY}" if PROXY else "") + " ...",
          flush=True)
    backend = HFGPUBackend(model_name=M_NAME, proxy_model_name=PROXY)
    T = backend.max_prompt_tokens + N_TOKENS
    print(f"loaded in {time.time()-t0:.1f}s | vocab={backend.vocab} "
          f"hidden={backend.hidden_dim} | M={backend.n_params/1e9:.3f}B"
          + (f"  proxy={backend.proxy_n_params/1e9:.3f}B "
             f"({backend.n_params/backend.proxy_n_params:.1f}x fewer)" if PROXY else "")
          + f" | {N_PROMPTS} prompts x {N_TOKENS} tok, T={T}", flush=True)

    spec = SamplingSpec()
    td = verifiers.get("token_difr")           # the full recompute of M
    have_proxy = backend.proxy_model is not None
    ss = verifiers.get("surface_stat") if have_proxy else None
    dets = [td] + ([ss] if ss else [])

    def score_pool(seqs):
        return harness.verify(backend, seqs, spec, dets)

    def aucs(honest_scores, attack_scores):
        """Headline partial AUC @ FPR<=0.5% (+ full-range AUC) per verifier."""
        res = {r.defense: r for r in harness.evaluate(
            honest_scores, attack_scores, dets, [BATCH], n_batches=N_BATCHES,
            winsor_pct=99.9, seed=7)}
        return {name: {"auc": res[name].auc, "auc_full": res[name].auc_full,
                       "tpr": res[name].tpr}
                for name in res}

    # honest reference pool
    honest = score_pool(harness.generate_dataset(
        backend, attacks.get("honest"), spec, N_PROMPTS, N_TOKENS))
    print(f"  honest pool scored ({time.time()-t0:.1f}s)", flush=True)

    # honest-null floor: a second independent honest draw on a DISJOINT prompt
    # range -- both verifiers must sit near 0.5 here.
    null_seqs = [backend.generate(p, N_TOKENS, spec, attacks.get("honest"), False, 123, 32)
                 for p in range(N_PROMPTS, 2 * N_PROMPTS)]
    null_floor = aucs(honest, score_pool(null_seqs))
    print(f"  null floor scored ({time.time()-t0:.1f}s)", flush=True)

    per_attack = {}
    for name in CORE_ATTACKS:
        a = score_pool(harness.generate_dataset(
            backend, attacks.get(name), spec, N_PROMPTS, N_TOKENS))
        per_attack[name] = aucs(honest, a)
        print(f"  {name:>12} scored ({time.time()-t0:.1f}s)", flush=True)

    recompute_flops = flops_2nt(backend.n_params, backend.vocab, backend.hidden_dim, T)
    proxy_flops = (flops_2nt(backend.proxy_n_params, backend.vocab,
                             int(backend.proxy_model.config.hidden_size), T)
                   if have_proxy else None)

    sec, n = backend.timed_seconds, backend.timed_calls
    meas = {k: (sec[k] / n[k] if n[k] else None) for k in sec}

    return {
        "tag": _tag(m_id), "family": m_id.family, "label": m_id.label,
        "org": m_id.org, "params": m_id.params, "generation": m_id.generation,
        "vocab": backend.vocab, "hidden": backend.hidden_dim, "T": T,
        "recompute_flops": recompute_flops,
        "recompute_seconds": meas["reference"],
        "proxy": (None if not have_proxy else {
            "tag": _tag(proxy_id), "label": proxy_id.label, "params": proxy_id.params,
            "flops": proxy_flops, "seconds": meas["proxy"]}),
        "null_floor": null_floor,
        "attacks": per_attack,
        "config": {"n_prompts": N_PROMPTS, "n_tokens": N_TOKENS, "batch": BATCH,
                   "n_batches": N_BATCHES, "max_fpr": 0.005,
                   "core_attacks": list(CORE_ATTACKS)},
        "elapsed_s": time.time() - t0,
    }


def main():
    result = run()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{result['tag']}.json"
    out.write_text(json.dumps(result, indent=2))

    # A compact human-readable summary of the two curve endpoints (mean over the
    # core attacks): cheap proxy vs full recompute, performance and cost.
    def mean_auc(kind):
        vals = [result["attacks"][a][kind]["auc"] for a in result["config"]["core_attacks"]
                if kind in result["attacks"][a]]
        return sum(vals) / len(vals) if vals else float("nan")

    print(f"\n=== {result['label']}  ({result['family']}, "
          f"{result['params']/1e9:.3f}B) ===")
    print(f"  full recompute (token_difr): mean AUC={mean_auc('token_difr'):.3f}  "
          f"cost={result['recompute_flops']/1e9:.1f} GFLOPs/seq")
    if result["proxy"]:
        print(f"  cheap proxy  ({result['proxy']['label']}, surface_stat): "
              f"mean AUC={mean_auc('surface_stat'):.3f}  "
              f"cost={result['proxy']['flops']/1e9:.1f} GFLOPs/seq  "
              f"({result['recompute_flops']/result['proxy']['flops']:.1f}x cheaper)")
    print(f"  wrote {out}  ({result['elapsed_s']:.1f}s)")


if __name__ == "__main__":
    main()
