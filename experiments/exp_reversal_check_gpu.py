"""Are the robustness sweep's reversal cells real, or the ratio artifact again?

`docs/results/robustness_report.md` is the basis of the repo's second headline
claim -- "no single recompute detector is uniformly robust", evidenced by
`token_difr` dropping to AUC 0.040 on `smollm2-1.7b`/`kv_fp8` and 0.086 on
`pythia-410m`/`kv_fp8`. But that sweep ran 16 prompts x 96 tokens at batch 400,
which is a **52% batch/pool ratio** -- five times over the ceiling that
`docs/TRIAGE_AND_AUDIT_COST.md` section 3 established. A collapsed-variance
measurement is pushed toward BOTH extremes (it reports whether these two
particular pools differ, which is nearly deterministic), and "some cells read 1.0
and some read 0.04" is exactly what that looks like. So the evidence for the claim
has the same defect as the evidence it was contrasted against.

This re-measures the extreme cells at a legitimate ratio, and adds the test that
settles it regardless of pool size: the **sign of the per-token effect size d'**.
Reversal is a mechanistic claim -- the attack scores *lower* than honest under this
detector -- so a real reversal has d' < 0 on the per-token scores, which is
pool-independent. An artifact does not.

Each cell is reported three ways from ONE generation pass: d' (pool-independent),
AUC at a ~10% ratio (legitimate), and AUC on a 16 x 96 sub-pool at batch 400 (the
sweep's own 52% configuration, reproduced as a sub-pool so the contrast is
controlled).

    python -m experiments.exp_reversal_check_gpu
Env: IVGYM_PROMPTS(64), IVGYM_TOKENS(128), IVGYM_BATCH(400), IVGYM_SEEDS(5),
     IVGYM_CELLS("smollm2-1.7b/kv_fp8,pythia-410m/kv_fp8,...").

Writes `docs/results/reversal_check.json`.
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

N = int(os.environ.get("IVGYM_PROMPTS", 64))
T = int(os.environ.get("IVGYM_TOKENS", 128))
BATCH = int(os.environ.get("IVGYM_BATCH", 400))
N_SEED = int(os.environ.get("IVGYM_SEEDS", 5))
SUB_P, SUB_T, SUB_B = 16, 96, 400          # the robustness sweep's own configuration

# The cells the "no single detector is uniformly robust" claim rests on, plus a
# control cell the same sweep reported at ~1.0 so the check is two-sided.
CELLS = os.environ.get("IVGYM_CELLS", ",".join([
    "HuggingFaceTB/SmolLM2-1.7B-Instruct/kv_fp8/0.040",
    "EleutherAI/pythia-410m/kv_fp8/0.086",
    "EleutherAI/pythia-410m/quant_4bit/0.146",
    "Qwen/Qwen3-0.6B/bug_k2/0.123",
    "Qwen/Qwen3-0.6B/seed_43/1.000",
])).split(",")
DEFENSE = "token_difr"


def sub_index(seq_lens, n_seq, n_tok):
    off, idx = 0, []
    for i, L in enumerate(seq_lens):
        if i < n_seq:
            idx.extend(range(off, off + min(n_tok, L)))
        off += L
    return np.asarray(idx, dtype=int)


def auc_seeds(h, a, batch):
    d = type("V", (), {"name": "s", "tier": 1})()
    hs = harness.TokenScores("h", {"s": h})
    as_ = harness.TokenScores("a", {"s": a})
    out = [harness.evaluate(hs, as_, [d], [batch], n_batches=2000, seed=s)[0].auc
           for s in range(N_SEED)]
    return float(np.mean(out)), float(np.std(out))


def main() -> None:
    t0 = time.time()
    print("=" * 78)
    print("Reversal check: real detector reversal, or the batch/pool artifact?")
    print(f"  {N} prompts x {T} tokens at batch {BATCH}; sweep sub-pool "
          f"{SUB_P} x {SUB_T} at batch {SUB_B}; {N_SEED} seeds; defense={DEFENSE}")
    print("=" * 78, flush=True)

    spec = SamplingSpec()
    d = verifiers.get(DEFENSE)
    out = []
    for spec_str in CELLS:
        parts = spec_str.rsplit("/", 2)
        model, aname, published = parts[0], parts[1], float(parts[2])
        print(f"\n--- {model}  x  {aname}   (sweep reported {published:.3f})", flush=True)
        try:
            backend = HFGPUBackend(model_name=model)
            honest = harness.verify(backend, harness.generate_dataset(
                backend, attacks.get("honest"), spec, N, T, record_activations=False),
                spec, [d])
            seqs = harness.generate_dataset(
                backend, attacks.get(aname), spec, N, T, record_activations=False)
            atk = harness.verify(backend, seqs, spec, [d])
        except Exception as e:
            print(f"    SKIPPED: {type(e).__name__}: {e}", flush=True)
            out.append({"model": model, "attack": aname, "published": published,
                        "error": f"{type(e).__name__}: {e}"})
            continue

        h, a = honest.scores[DEFENSE], atk.scores[DEFENSE]
        lens = [len(s.steps) for s in seqs]
        sub = sub_index(lens, SUB_P, SUB_T)
        st = signal.per_token_stats(h, a)
        big, big_sd = auc_seeds(h, a, BATCH)
        small, small_sd = auc_seeds(h[sub], a[sub], SUB_B)
        rec = {"model": model, "attack": aname, "published": published,
               "n_tok": int(len(h)), "d_prime": st["d_prime"],
               "honest_mean": st["honest_mean"], "attack_mean": st["attack_mean"],
               "ratioed": {"auc": big, "sd": big_sd,
                           "ratio": BATCH / (0.5 * len(h))},
               "sweepcfg": {"auc": small, "sd": small_sd,
                            "ratio": SUB_B / (0.5 * len(sub))},
               "reversed_per_token": bool(st["d_prime"] < 0)}
        out.append(rec)
        print(f"    d' = {st['d_prime']:+.4f}  "
              f"(honest mean {st['honest_mean']:.4f} vs attack {st['attack_mean']:.4f})"
              f" -> per-token reversal: {'YES' if rec['reversed_per_token'] else 'no'}")
        print(f"    AUC @ {rec['ratioed']['ratio']:.1%} ratio : {big:.3f} +-{big_sd:.3f}"
              f"   |   AUC @ {rec['sweepcfg']['ratio']:.0%} ratio (sweep cfg): "
              f"{small:.3f} +-{small_sd:.3f}   [{time.time()-t0:.0f}s]", flush=True)
        del backend

    print(f"\n{'='*78}\n{'model / attack':<44}{'pub':>7}{'sweep':>8}{'ratioed':>9}"
          f"{'d prime':>10}{'reversed':>10}\n{'='*78}")
    for r in out:
        if "error" in r:
            print(f"{r['model']+'/'+r['attack']:<44}{r['published']:>7.3f}   (skipped)")
            continue
        print(f"{r['model'].split('/')[-1]+'/'+r['attack']:<44}{r['published']:>7.3f}"
              f"{r['sweepcfg']['auc']:>8.3f}{r['ratioed']['auc']:>9.3f}"
              f"{r['d_prime']:>+10.4f}{('YES' if r['reversed_per_token'] else 'no'):>10}")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "reversal_check.json").write_text(json.dumps(
        {"n_prompts": N, "tokens": T, "batch": BATCH, "n_seed": N_SEED,
         "sub": [SUB_P, SUB_T, SUB_B], "defense": DEFENSE, "cells": out,
         "elapsed_s": time.time() - t0}, indent=1))
    print(f"\nwrote {RES/'reversal_check.json'}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
