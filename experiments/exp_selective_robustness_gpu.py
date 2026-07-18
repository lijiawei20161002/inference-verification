"""Selective-vs-full recompute ROBUSTNESS across families, sizes, attack types.

`exp_robustness_gpu.py` shows the *unified registry* generalises across a model
matrix, and reports selective recompute at two fixed budgets with one value
signal (proxy entropy). `exp_tie_triage_margin.py` shows the load-bearing control
-- information-directed triage beats an EQUAL-COST random subsample -- but only on
ONE model pair (Qwen3-1.7B/0.6B) and only for quant. This experiment crosses the
two: it runs the triaged-vs-random Pareto for the whole selective-recompute tier
on a MATRIX of real models (families x sizes) and EVERY attack, so the question
"is selective recompute robustly worth it, and does the best triage signal hold?"
is answered per cell rather than as a single aggregate.

Efficient by construction. A Tier-1 verifier's per-token score depends only on
that token's own recomputed logits (`Tier1Verifier.evidence` scores audited rows
independently, leaves the rest at `neutral`). So a token's audited score is the
SAME at any budget -- we compute the FULL per-token scores + every cheap value
signal ONCE on the GPU (one pass per model x attack, exactly as the full sweep),
then derive every budget x value-fn x {triage, random} curve in pure numpy. The
triage path reuses the shipped `harness.select_triaged`, so the numbers are the
shipped `harness.verify(budget<1, value_fn=...)` tier, with an equal-cost random
subsample as the control it lacks.

For each (family, size) x attack x Tier-1 verifier it records:
  * full-recompute AUC (budget 1.0);
  * triaged AUC vs recompute ratio, one curve per value signal
    (entropy / tie_margin / surprisal), token_difr headline;
  * equal-cost random-subsample AUC vs ratio (mean +/- std over seeds);
  * recompute ratio to reach a target AUC, and the triage-vs-random SAVING factor.

    .venv/bin/python -m experiments.exp_selective_robustness_gpu
    IVGYM_REPLOT=1 .venv/bin/python -m experiments.exp_selective_robustness_gpu  # curves from cache, no GPU

Env: IVGYM_PROMPTS(16) IVGYM_TOKENS(96) IVGYM_BATCH(128) IVGYM_NBATCH(200)
     IVGYM_BOOT(6) IVGYM_TARGET(0.95) IVGYM_MODELS(csv of tags) IVGYM_MAX_PARAMS
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ivgym import attacks, harness, verifiers  # noqa: E402
from ivgym.core import SamplingSpec  # noqa: E402
from ivgym.harness import select_triaged  # noqa: E402
from ivgym.metrics import roc_auc  # noqa: E402

# ---------------------------------------------------------------------------
# Representative subset: a size ladder in each of 3 families plus a 4th family,
# every model with a REAL same-family proxy (a legitimate value signal, not the
# noised-M fallback). Families: Qwen3, Llama-3.2, SmolLM2 (GPT tokenizer, Llama
# arch), Pythia (GPT-NeoX). Size ladders: qwen3 1.7/4, smollm2 0.36/1.7,
# pythia 0.41/1.4. Superset of exp_robustness_gpu.MODEL_MATRIX (with proxies).
# ---------------------------------------------------------------------------
MODEL_MATRIX = [
    dict(tag="qwen3-1.7b", ref="Qwen/Qwen3-1.7B", proxy="Qwen/Qwen3-0.6B", params=1.7e9),
    dict(tag="qwen3-4b",   ref="Qwen/Qwen3-4B",   proxy="Qwen/Qwen3-0.6B", params=4.0e9),
    dict(tag="llama3.2-3b", ref="unsloth/Llama-3.2-3B-Instruct",
         proxy="unsloth/Llama-3.2-1B-Instruct", params=3.2e9),
    dict(tag="smollm2-360m", ref="HuggingFaceTB/SmolLM2-360M-Instruct",
         proxy="HuggingFaceTB/SmolLM2-135M-Instruct", params=0.36e9),
    dict(tag="smollm2-1.7b", ref="HuggingFaceTB/SmolLM2-1.7B-Instruct",
         proxy="HuggingFaceTB/SmolLM2-135M-Instruct", params=1.7e9),
    dict(tag="pythia-410m", ref="EleutherAI/pythia-410m",
         proxy="EleutherAI/pythia-160m", params=0.41e9),
    dict(tag="pythia-1.4b", ref="EleutherAI/pythia-1.4b",
         proxy="EleutherAI/pythia-160m", params=1.4e9),
]

ATTACKS = ["quant_4bit", "kv_fp8", "temp_1.1", "seed_43", "bug_k2", "bug_k32",
           "adv_quant_temp"]
# forward-pass corruptions flip the sampled token only at near-tie positions --
# where a cheap proxy CAN point; sampling-only changes have no such structure, so
# the triage rationale is expected to hold on the first group and not the second.
FORWARD_PASS = {"quant_4bit", "kv_fp8", "adv_quant_temp"}
SAMPLING_ONLY = [a for a in ATTACKS if a not in FORWARD_PASS]
VALUE_FNS = ["entropy", "tie_margin", "surprisal"]   # triage signals (uniform == random-ish, excluded)
HEADLINE = "token_difr"                              # per-value-fn curves only for the headline verifier

N_PROMPTS = int(os.environ.get("IVGYM_PROMPTS", 16))
N_TOKENS = int(os.environ.get("IVGYM_TOKENS", 96))
BATCH = int(os.environ.get("IVGYM_BATCH", 128))
N_BATCH = int(os.environ.get("IVGYM_NBATCH", 200))
BOOT = int(os.environ.get("IVGYM_BOOT", 6))          # random-subsample selection seeds
TARGET = float(os.environ.get("IVGYM_TARGET", 0.95))  # absolute AUC target
# On subtle forward-pass attacks even FULL recompute can sit below 0.95 at a given
# pool, making the absolute cost-to-target n/a for everyone -- uninformative for
# exactly the attacks triage targets. The relative target -- recompute ratio to
# recover REL_FRAC of full-recompute's OWN AUC -- is always defined (both curves
# hit full AUC at ratio 1.0) and is the cleaner selective-vs-full question.
REL_FRAC = float(os.environ.get("IVGYM_REL_FRAC", 0.95))
WINSOR = 99.9
RHOS = np.unique(np.round(np.geomspace(0.02, 1.0, 16), 4))
MAX_PARAMS = float(os.environ.get("IVGYM_MAX_PARAMS", 0)) or None

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "results"
OUT_JSON = OUT_DIR / "selective_robustness.json"
CACHE_DIR = ROOT / "experiments" / "difr_data" / "selective_robustness"


def _pick_models():
    want = os.environ.get("IVGYM_MODELS")
    matrix = MODEL_MATRIX
    if want:
        keep = {t.strip() for t in want.split(",") if t.strip()}
        matrix = [m for m in matrix if m["tag"] in keep]
    if MAX_PARAMS:
        matrix = [m for m in matrix if m["params"] <= MAX_PARAMS]
    return matrix


def _free(backend):
    try:
        del backend.model
        if getattr(backend, "proxy_model", None) is not None:
            del backend.proxy_model
    except Exception:
        pass
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


# --------------------------------------------------------------------------- GPU
def compute_features(cfg) -> dict:
    """One GPU pass per model: FULL per-token Tier-1 scores + every cheap value
    signal, for honest and each attack. Cached to npz for GPU-free re-analysis."""
    from ivgym.backends.hf_gpu import HFGPUBackend
    tag, ref, proxy = cfg["tag"], cfg["ref"], cfg["proxy"]
    t0 = time.time()
    print(f"\n{'='*78}\n[{tag}] loading ref={ref} proxy={proxy}", flush=True)
    backend = HFGPUBackend(model_name=ref, proxy_model_name=proxy)
    print(f"[{tag}] loaded {time.time()-t0:.1f}s vocab={backend.vocab} "
          f"params={backend.n_params/1e9:.2f}B proxy={backend.proxy_n_params/1e9:.2f}B", flush=True)

    all_v = verifiers.all_verifiers()
    tier1 = [v for v in all_v.values() if v.tier == 1]
    tier1_names = [v.name for v in tier1]
    need_act = any(v.needs_activation for v in tier1)
    spec = SamplingSpec()

    def gen(atk):
        return harness.generate_dataset(backend, atk, spec, N_PROMPTS, N_TOKENS,
                                        record_activations=need_act)

    def full_scores(seqs):
        ts = harness.verify(backend, seqs, spec, tier1)          # budget 1.0 == full recompute
        return {n: ts.scores[n] for n in tier1_names}

    def values(seqs):
        return {vf: harness.token_values(backend, seqs, spec, vf) for vf in VALUE_FNS}

    feat = {
        "tag": tag, "ref": ref, "proxy": proxy, "params": backend.n_params,
        "proxy_params": backend.proxy_n_params, "vocab": backend.vocab,
        "tier1": tier1_names, "neutral": {v.name: float(v.neutral) for v in tier1},
    }
    honest_seqs = gen(attacks.get("honest"))
    feat["honest_scores"] = full_scores(honest_seqs)
    feat["honest_values"] = values(honest_seqs)
    feat["attacks"] = {}
    for aname in ATTACKS:
        seqs = gen(attacks.get(aname))
        feat["attacks"][aname] = {"scores": full_scores(seqs), "values": values(seqs)}
        td = feat["attacks"][aname]["scores"].get(HEADLINE)
        print(f"[{tag}] {aname:>14}  {HEADLINE} full-mean(hon={feat['honest_scores'][HEADLINE].mean():.3f} "
              f"att={td.mean():.3f})", flush=True)

    _save_cache(feat)
    feat["seconds"] = time.time() - t0
    _free(backend)
    return feat


def _cache_path(tag):
    return CACHE_DIR / f"{tag}.npz"


def _save_cache(feat):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    flat = {}
    for n, a in feat["honest_scores"].items():
        flat[f"hs::{n}"] = a
    for vf, a in feat["honest_values"].items():
        flat[f"hv::{vf}"] = a
    for atk, d in feat["attacks"].items():
        for n, a in d["scores"].items():
            flat[f"as::{atk}::{n}"] = a
        for vf, a in d["values"].items():
            flat[f"av::{atk}::{vf}"] = a
    meta = {k: feat[k] for k in ("tag", "ref", "proxy", "params", "proxy_params",
                                 "vocab", "tier1", "neutral")}
    np.savez(_cache_path(feat["tag"]), _meta=json.dumps(meta), **flat)


def _load_cache(tag) -> dict | None:
    p = _cache_path(tag)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    meta = json.loads(str(z["_meta"]))
    feat = dict(meta)
    feat["honest_scores"] = {}
    feat["honest_values"] = {}
    feat["attacks"] = {a: {"scores": {}, "values": {}} for a in ATTACKS}
    for k in z.files:
        if k == "_meta":
            continue
        kind, *rest = k.split("::")
        if kind == "hs":
            feat["honest_scores"][rest[0]] = z[k]
        elif kind == "hv":
            feat["honest_values"][rest[0]] = z[k]
        elif kind == "as":
            feat["attacks"][rest[0]]["scores"][rest[1]] = z[k]
        elif kind == "av":
            feat["attacks"][rest[0]]["values"][rest[1]] = z[k]
    return feat


# ----------------------------------------------------------- numpy AUC machinery
def _batch_means(x, batch, n_batches, rng):
    n = len(x)
    b = min(batch, n)
    keys = rng.random((n_batches, n))
    idx = np.argpartition(keys, b - 1, axis=1)[:, :b]     # b random tokens/batch, no replacement
    return x[idx].mean(axis=1)


def _auc(h, a, seed):
    """Replicate harness.evaluate for a single verifier: split, winsor on honest
    train, batch-mean, ROC AUC. `h`/`a` are already the (neutral-padded) per-token
    score arrays for the chosen audit set."""
    rng = np.random.default_rng(seed)
    hi, ai = rng.permutation(len(h)), rng.permutation(len(a))
    h_tr, h_te = h[hi[:len(h) // 2]], h[hi[len(h) // 2:]]
    a_te = a[ai[len(a) // 2:]]
    finite = h_tr[np.isfinite(h_tr)]
    cap = np.percentile(finite, WINSOR) if len(finite) else np.inf
    h_te = np.minimum(h_te, cap)
    a_te = np.minimum(a_te, cap)
    return roc_auc(_batch_means(h_te, BATCH, N_BATCH, rng),
                   _batch_means(a_te, BATCH, N_BATCH, rng))


def _mask_to(full, mask, neutral):
    out = np.full(len(full), neutral, float)
    out[mask] = full[mask]
    return out


def _triage_auc(hs, as_, hv, av, rho, neutral):
    """Shipped path: rank honest/attack tokens by their OWN value signal, audit the
    top `rho`, pad the rest to neutral (== harness.verify(budget=rho, value_fn))."""
    h = _mask_to(hs, select_triaged(hv, rho), neutral)
    a = _mask_to(as_, select_triaged(av, rho), neutral)
    return _auc(h, a, seed=0)


def _random_auc(hs, as_, rho, neutral):
    """Equal-cost control: audit a RANDOM rho-fraction. Mean +/- std over seeds."""
    aucs = []
    for s in range(BOOT):
        rng = np.random.default_rng(2000 + s)
        kh = max(1, round(rho * len(hs)))
        ka = max(1, round(rho * len(as_)))
        hm = np.zeros(len(hs), bool); hm[rng.choice(len(hs), kh, replace=False)] = True
        am = np.zeros(len(as_), bool); am[rng.choice(len(as_), ka, replace=False)] = True
        aucs.append(_auc(_mask_to(hs, hm, neutral), _mask_to(as_, am, neutral), seed=100 + s))
    return float(np.mean(aucs)), float(np.std(aucs))


def _cost_to_target(rhos, aucs, target):
    """Smallest recompute ratio whose curve reaches `target` AUC (linear interp)."""
    aucs = np.asarray(aucs)
    for i in range(len(rhos)):
        if aucs[i] >= target:
            if i == 0:
                return float(rhos[0])
            x0, x1, y0, y1 = rhos[i - 1], rhos[i], aucs[i - 1], aucs[i]
            return float(x0 + (target - y0) * (x1 - x0) / max(y1 - y0, 1e-9))
    return None


def compute_curves(feat) -> dict:
    """All triaged/random/full curves + cost-to-target, per attack x verifier."""
    tier1 = feat["tier1"]
    neutral = feat["neutral"]
    rhos = RHOS.tolist()
    cells = {}
    for atk in ATTACKS:
        cells[atk] = {}
        adat = feat["attacks"][atk]
        for vn in tier1:
            hs = np.asarray(feat["honest_scores"][vn], float)
            as_ = np.asarray(adat["scores"][vn], float)
            nz = neutral[vn]
            full = _auc(hs, as_, seed=0)
            # per-value-fn triage only for the headline verifier; entropy for the rest
            vfs = VALUE_FNS if vn == HEADLINE else ["entropy"]
            triaged = {}
            cost = {}
            for vf in vfs:
                hv = np.asarray(feat["honest_values"][vf], float)
                av = np.asarray(adat["values"][vf], float)
                curve = [_triage_auc(hs, as_, hv, av, r, nz) for r in rhos]
                triaged[vf] = curve
                cost[f"triage:{vf}"] = _cost_to_target(rhos, curve, TARGET)
            rmean, rstd = [], []
            for r in rhos:
                m, s = _random_auc(hs, as_, r, nz)
                rmean.append(m); rstd.append(s)
            cost["random"] = _cost_to_target(rhos, rmean, TARGET)
            # relative target: recover REL_FRAC of full recompute's OWN AUC. Always
            # defined (a full audit reaches full AUC at ratio 1.0). Floor at 0.5 so a
            # near-chance full AUC does not make the target trivially met at ratio 0.
            rel = max(0.5, REL_FRAC * full)
            cost_rel = {f"triage:{vf}": _cost_to_target(rhos, triaged[vf], rel) for vf in vfs}
            cost_rel["random"] = _cost_to_target(rhos, rmean, rel)
            # best value fn ranked by the always-defined relative cost (cheaper wins)
            best_vf = min(vfs, key=lambda vf: cost_rel[f"triage:{vf}"] or 9.0)
            ct_a, cr_a = cost[f"triage:{best_vf}"], cost["random"]
            ct_r, cr_r = cost_rel[f"triage:{best_vf}"], cost_rel["random"]
            cells[atk][vn] = {
                "full_auc": full, "rel_target": rel, "triaged": triaged,
                "random_mean": rmean, "random_std": rstd,
                "cost": cost, "cost_rel": cost_rel, "best_value_fn": best_vf,
                "saving_factor": (cr_a / ct_a) if (ct_a and cr_a) else None,
                "saving_rel": (cr_r / ct_r) if (ct_r and cr_r) else None,
            }
    return {
        "tag": feat["tag"], "ref": feat["ref"], "proxy": feat["proxy"],
        "params": feat["params"], "proxy_params": feat["proxy_params"],
        "vocab": feat["vocab"], "tier1": tier1, "rhos": rhos, "cells": cells,
    }


def _print_model(res):
    tag = res["tag"]
    print(f"\n[{tag}] {HEADLINE}: full AUC | ratio to {int(REL_FRAC*100)}%-of-full "
          f"(triage-best / random) | rel-saving  [abs-saving @AUC {TARGET}]", flush=True)
    for atk in ATTACKS:
        c = res["cells"][atk][HEADLINE]
        grp = "FWD " if atk in FORWARD_PASS else "samp"
        vf = c["best_value_fn"]
        ct, cr = c["cost_rel"][f"triage:{vf}"], c["cost_rel"]["random"]
        print(f"  [{grp}] {atk:>14}  full={c['full_auc']:.3f}  "
              f"triage[{vf}]={_fmt(ct)}  random={_fmt(cr)}  "
              f"rel-saving={_fmt(c['saving_rel'],'x')}  [abs={_fmt(c['saving_factor'],'x')}]",
              flush=True)


def _fmt(x, suf=""):
    return "  n/a" if x is None else f"{x:.3f}{suf}"


def main():
    replot = os.environ.get("IVGYM_REPLOT") == "1"
    models = _pick_models()
    print(f"selective-robustness over {len(models)} models: {[m['tag'] for m in models]}")
    print(f"config: {N_PROMPTS}x{N_TOKENS} tok, batch={BATCH}, n_batch={N_BATCH}, "
          f"boot={BOOT}, target={TARGET}, value_fns={VALUE_FNS}, replot={replot}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results, t0 = [], time.time()
    for cfg in models:
        try:
            feat = _load_cache(cfg["tag"]) if replot else None
            if feat is None:
                if replot:
                    print(f"[{cfg['tag']}] no cache; skipping (replot mode)", flush=True)
                    continue
                feat = compute_features(cfg)
            res = compute_curves(feat)
            _print_model(res)
            results.append(res)
        except Exception as e:
            print(f"[{cfg['tag']}] FAILED: {e}", flush=True)
            traceback.print_exc()
            results.append({"tag": cfg["tag"], "ref": cfg["ref"], "error": str(e)})
        payload = {"config": {"prompts": N_PROMPTS, "tokens": N_TOKENS, "batch": BATCH,
                              "n_batch": N_BATCH, "boot": BOOT, "target": TARGET,
                              "rel_frac": REL_FRAC,
                              "rhos": RHOS.tolist(), "value_fns": VALUE_FNS,
                              "forward_pass": sorted(FORWARD_PASS),
                              "sampling_only": SAMPLING_ONLY, "attacks": ATTACKS},
                   "models": results}
        OUT_JSON.write_text(json.dumps(payload, indent=2, default=float))
        print(f"[progress] {len([r for r in results if 'error' not in r])}/{len(models)} "
              f"ok, {time.time()-t0:.0f}s -> {OUT_JSON}", flush=True)
    print(f"\ndone in {time.time()-t0:.0f}s -> {OUT_JSON}")


if __name__ == "__main__":
    main()
