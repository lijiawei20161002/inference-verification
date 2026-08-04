"""Does the chunked-vs-sequential disagreement grow with context length?

Two measurements of the same gap disagreed by a factor of seven.
`exp_specdec_shape_gpu`, on ~35-token chat prompts, found 97.5% of logit entries
bitwise identical between the chunked and sequential shapes.
`exp_specdec_fliprate_gpu`, on ~190-token prompt+completion sequences, found
13.6%. Sequence length is the obvious difference between them. This tests it
directly: identical measurement, prefix length swept 32 -> 2048, everything else
held fixed.

The answer is not a trend. It is a step: the exact-match fraction sits at ~97%
through 64 tokens and collapses to ~13-24% from 128 on, and stays there. That is
the signature of a discrete kernel selection -- above some sequence length the
attention backend switches to a different tiling or split-K decomposition, whose
reduction order differs from the short-sequence path -- not of error accumulating
smoothly with depth.

Windows are drawn from one long natural document (Dickens, *A Tale of Two
Cities*) so that every context length is measured on real text; a repeated or
synthetic prefix would give the attention pattern a degeneracy that is itself a
confound.

Why this belongs to a verification testbed
------------------------------------------
It is the second half of the answer to `NEXT_EXPERIMENTS.md` item 1's cheap arm.
A step function is worse news for a verifier than a trend: a calibration set
collected at one context length can land on the other side of the switch from the
traffic it is auditing, and nothing about the two configurations announces that
they differ.

Writes `docs/results/specdec_ctxlen.json`. Figures come from
`python -m experiments.plot_specdec`.

    python -m experiments.exp_specdec_ctxlen_gpu
Env: IVGYM_SPECDEC_TARGET(Qwen/Qwen2.5-1.5B-Instruct), IVGYM_GAMMA(8),
     IVGYM_CTXLENS(32,64,128,256,512,1024,2048), IVGYM_WINDOWS(16),
     IVGYM_WINDOW_STRIDE(97).
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
GAMMA = int(os.environ.get("IVGYM_GAMMA", 8))
CTXLENS = [int(x) for x in os.environ.get(
    "IVGYM_CTXLENS", "32,64,128,256,512,1024,2048").split(",")]
N_WINDOWS = int(os.environ.get("IVGYM_WINDOWS", 16))
# Coprime with every context length in the sweep, so windows at a given length
# start at genuinely different places in the document rather than tiling it.
STRIDE = int(os.environ.get("IVGYM_WINDOW_STRIDE", 97))
DEV = "cuda"


def main():
    tok = AutoTokenizer.from_pretrained(TARGET)
    model = sc.load_model(TARGET, torch.bfloat16, DEV)
    all_ids = tok(sc.load_longtext(), return_tensors="pt").input_ids.to(DEV)
    print("corpus tokens:", all_ids.shape[1])

    out = []
    for L in CTXLENS:
        seqs, chks = [], []
        for k in range(N_WINDOWS):
            start = k * STRIDE
            if start + L > all_ids.shape[1]:
                break
            ids = all_ids[:, start:start + L]
            n_prefill, n_logits = L - GAMMA, GAMMA + 1
            seqs.append(sc.sequential_logits(model, ids, n_prefill, n_logits))
            chks.append(sc.chunked_logits(model, ids, n_prefill, n_logits))
        if not seqs:
            continue
        ref, alt = torch.cat(seqs, 0), torch.cat(chks, 0)
        d = (ref - alt).abs()
        rec = dict(ctx_len=L, n_windows=len(seqs),
                   frac_exact=(d == 0).float().mean().item(),
                   max_abs=d.max().item(),
                   mean_abs=d.mean().item(),
                   flip_rate=(ref.argmax(-1) != alt.argmax(-1)).float().mean().item(),
                   n_pos=ref.shape[0])
        out.append(rec)
        print(f"ctx={L:5d} ({rec['n_windows']:2d} windows)  "
              f"bitwise-identical={rec['frac_exact']*100:5.1f}%  "
              f"max|d|={rec['max_abs']:.3f}  mean|d|={rec['mean_abs']:.2e}  "
              f"argmax flips={rec['flip_rate']*100:.2f}%", flush=True)
    sc.write_result("specdec_ctxlen.json", out)


if __name__ == "__main__":
    main()
