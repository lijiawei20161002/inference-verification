"""Does the SHAPE of the forward pass change the logits? (bf16 / fp16 / fp32)

Same prefix, same weights, same mathematically identical quantity -- the
next-token logits at each of the last gamma+1 positions -- computed three ways:

  sequential      prefill, then gamma single-token decode steps through a KV cache
                  (plain autoregressive serving; THE REFERENCE)
  chunked         one forward pass over the whole window, logits read at all
                  positions at once (the speculative-decoding VERIFY shape)
  rerun control   the sequential pass, run twice

Speculative decoding uses the chunked shape exactly where plain serving uses the
sequential one, so the gap between them is the price of the shape, with the model
and the arithmetic held fixed. The rerun control is what makes that readable: it
establishes that at a FIXED shape this backend is bitwise deterministic, so any
nonzero chunked-vs-sequential difference is the shape and nothing else.

A second sweep asks the same question of batch composition: the same row, scored
alone and scored inside a batch of unrelated filler rows. Rows cannot influence
each other through attention, so any difference is the kernel's reduction
schedule reacting to its neighbours -- the batch-invariance problem, measured on
the same prompts.

Why this belongs to a verification testbed
------------------------------------------
`ivgym`'s honest null is "the same model, run twice, on one H100" -- the rerun
control, which is exactly 0. A deployed verifier does not get that null. It
recomputes `M` in whatever shape its own serving stack picks, against tokens the
provider generated in a different one, and this experiment measures how far apart
those two are before any attack is applied. It is the cheapest arm of
`NEXT_EXPERIMENTS.md` item 1 (cross-GPU benign variation), the one runnable on a
single card.

Writes `docs/results/specdec_shape.json`. Figures come from
`python -m experiments.plot_specdec`, which needs all four `exp_specdec_*`
artifacts.

    python -m experiments.exp_specdec_shape_gpu
Env: IVGYM_SPECDEC_TARGET(Qwen/Qwen2.5-1.5B-Instruct), IVGYM_PROMPTS(24),
     IVGYM_DTYPES(bfloat16,float16,float32), IVGYM_GAMMAS(1,2,4,8,16),
     IVGYM_BATCHES(1,2,4,8,16,32), IVGYM_BATCH_GAMMA(8).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import specdec_common as sc

TARGET = os.environ.get("IVGYM_SPECDEC_TARGET", "Qwen/Qwen2.5-1.5B-Instruct")
N_PROMPTS = int(os.environ.get("IVGYM_PROMPTS", 24))
DTYPE_NAMES = [d for d in os.environ.get(
    "IVGYM_DTYPES", "bfloat16,float16,float32").split(",") if d]
GAMMAS = [int(g) for g in os.environ.get("IVGYM_GAMMAS", "1,2,4,8,16").split(",")]
BATCHES = [int(b) for b in os.environ.get("IVGYM_BATCHES", "1,2,4,8,16,32").split(",")]
BATCH_GAMMA = int(os.environ.get("IVGYM_BATCH_GAMMA", 8))
DEV = "cuda"


def _windows(tok, prompts, gamma):
    """Chat-formatted prompt ids long enough to carry a gamma-token window."""
    for p in prompts:
        ids = sc.encode_chat(tok, p, DEV)
        if ids.shape[1] >= gamma + 8:
            yield ids


def sweep_dtypes(tok, prompts, results):
    """gamma x dtype: how much of the disagreement is the numeric format?"""
    for name in DTYPE_NAMES:
        model = sc.load_model(TARGET, sc.DTYPES[name], DEV)
        for gamma in GAMMAS:
            seq, chk, ctl = [], [], []
            for ids in _windows(tok, prompts, gamma):
                n_prefill, n_logits = ids.shape[1] - gamma, gamma + 1
                seq.append(sc.sequential_logits(model, ids, n_prefill, n_logits))
                chk.append(sc.chunked_logits(model, ids, n_prefill, n_logits))
                ctl.append(sc.sequential_logits(model, ids, n_prefill, n_logits))
            ref = torch.cat(seq, 0)
            r = sc.compare(ref, torch.cat(chk, 0), "chunked_vs_sequential")
            c = sc.compare(ref, torch.cat(ctl, 0), "rerun_control")
            for row in (r, c):
                row.update(dtype=name, gamma=gamma)
            results += [r, c]
            print(f"{name} g={gamma:2d}  chunk-vs-seq max|d|={r['max_abs']:.3e} "
                  f"exact={r['frac_exact']*100:.1f}% flips={r['argmax_flips']}/{r['n_pos']}"
                  f"   | control max|d|={c['max_abs']:.3e} "
                  f"exact={c['frac_exact']*100:.1f}%", flush=True)
        del model
        torch.cuda.empty_cache()


def sweep_batch(tok, prompts, results):
    """Batch composition at bf16: the same row, different neighbours."""
    model = sc.load_model(TARGET, torch.bfloat16, DEV)
    gamma = BATCH_GAMMA
    for bs in BATCHES:
        solo, batched = [], []
        for ids in _windows(tok, prompts, gamma):
            n_prefill, n_logits = ids.shape[1] - gamma, gamma + 1
            solo.append(sc.chunked_logits(model, ids, n_prefill, n_logits))
            # Seed varies with batch size so no two conditions share filler rows;
            # the committed artifact was measured at these seeds.
            batched.append(sc.chunked_logits(model, ids, n_prefill, n_logits,
                                             batch_size=bs, seed=1234 + bs))
        r = sc.compare(torch.cat(solo, 0), torch.cat(batched, 0), "batch_composition")
        r.update(dtype="bfloat16", gamma=gamma, batch_size=bs)
        results.append(r)
        print(f"bf16 bs={bs:2d} batch-vs-solo max|d|={r['max_abs']:.3e} "
              f"exact={r['frac_exact']*100:.1f}% flips={r['argmax_flips']}/{r['n_pos']}",
              flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    tok = AutoTokenizer.from_pretrained(TARGET)
    prompts = sc.load_prompts(N_PROMPTS)
    results: list[dict] = []
    sweep_dtypes(tok, prompts, results)
    sweep_batch(tok, prompts, results)
    sc.write_result("specdec_shape.json", results)


if __name__ == "__main__":
    main()
