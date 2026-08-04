"""The per-token argmax flip rate, measured on real decode trajectories.

`exp_specdec_shape_gpu` measures the shape gap on short prompts. This measures
the thing that actually costs an output token, on the sequences a server would
really produce: for each prompt take prompt + its own 128 greedy tokens, then
recompute the next-token logits at all 128 generated positions three ways. All
three are the same mathematical quantity.

  sequential  prefill the prompt, then 128 single-token decode steps (REFERENCE)
  chunked     ONE forward pass over the whole sequence  (the spec-dec verify shape)
  batched     the whole sequence inside a batch of 8    (the batch-invariance problem)
  control     the sequential pass again                 (fixed-shape determinism)

The flip rate is only interpretable next to the argmax margin, so the
top1-top2 gap at all 3,968 reference positions is recorded alongside it: a flip
needs the perturbation to exceed the margin, and the margin distribution is what
says whether a last-bit difference is rare or routine.

Why this belongs to a verification testbed
------------------------------------------
This is the benign false-positive floor for any token-comparison verifier, stated
in the units that matter. A detector asked to flag a provider whose tokens differ
from a local recompute will see this rate on a perfectly honest provider that
merely schedules its forward passes differently -- with no attack, no
quantization and no sampler deviation anywhere in the system.

Writes `docs/results/specdec_fliprate.json`. Figures come from
`python -m experiments.plot_specdec`.

    python -m experiments.exp_specdec_fliprate_gpu
Env: IVGYM_SPECDEC_TARGET(Qwen/Qwen2.5-1.5B-Instruct), IVGYM_PROMPTS(all 32),
     IVGYM_TOKENS(128), IVGYM_BATCH(8).
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
N_PROMPTS = int(os.environ["IVGYM_PROMPTS"]) if "IVGYM_PROMPTS" in os.environ else None
NGEN = int(os.environ.get("IVGYM_TOKENS", 128))
BATCH = int(os.environ.get("IVGYM_BATCH", 8))
DEV = "cuda"


def main():
    tok = AutoTokenizer.from_pretrained(TARGET)
    model = sc.load_model(TARGET, torch.bfloat16, DEV)
    prompts = sc.load_prompts(N_PROMPTS)
    keys = ["chunked", "batched", "control"]
    stats = {k: dict(n=0, flips=0, maxd=0.0, exact=0, tot=0) for k in keys}
    margins: list[float] = []

    for pi, p in enumerate(prompts):
        ids = sc.encode_chat(tok, p, DEV)
        gen = model.generate(ids, max_new_tokens=NGEN, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        if gen.shape[1] - ids.shape[1] < NGEN:
            continue                              # hit EOS early; no full window
        full = gen[:, :ids.shape[1] + NGEN]
        n_prefill = ids.shape[1]

        ref = sc.sequential_logits(model, full, n_prefill, NGEN)
        variants = dict(
            chunked=sc.chunked_logits(model, full, n_prefill, NGEN),
            batched=sc.chunked_logits(model, full, n_prefill, NGEN,
                                      batch_size=BATCH, seed=7),
            control=sc.sequential_logits(model, full, n_prefill, NGEN),
        )
        s = ref.sort(-1, descending=True).values
        margins += (s[:, 0] - s[:, 1]).tolist()
        for k, v in variants.items():
            st = stats[k]
            st["n"] += ref.shape[0]
            st["flips"] += (ref.argmax(-1) != v.argmax(-1)).sum().item()
            st["maxd"] = max(st["maxd"], (ref - v).abs().max().item())
            st["exact"] += (ref == v).sum().item()
            st["tot"] += ref.numel()
        if pi % 8 == 0:
            print(f"[{pi:2d}] " + "  ".join(
                f"{k}: {stats[k]['flips']}/{stats[k]['n']}" for k in keys), flush=True)

    for k, st in stats.items():
        st["flip_rate"] = st["flips"] / st["n"]
        st["frac_exact"] = st["exact"] / st["tot"]
        print(f"{k:10s} flip {st['flips']:5d}/{st['n']} = {st['flip_rate']*100:.2f}%  "
              f"max|d|={st['maxd']:.3f}  bitwise-identical logits="
              f"{st['frac_exact']*100:.1f}%")
    sc.write_result("specdec_fliprate.json", dict(stats=stats, margins=margins))


if __name__ == "__main__":
    main()
