"""Lossless greedy speculative decoding vs plain greedy decoding: where they diverge.

Greedy speculative decoding is *provably* output-identical to greedy decoding. A
drafted token is accepted iff it equals the target's argmax; otherwise the
target's argmax is substituted. Same token sequence, by construction -- in exact
arithmetic. This runs both, on a real target/draft pair, and finds the index of
the first token where they differ.

Target: Qwen2.5-1.5B-Instruct   Draft: Qwen2.5-0.5B-Instruct   dtype: bf16

The losslessness proof is not wrong; its premise is. It assumes the target's
argmax is the same quantity whether the target scores one token or a block of
gamma at once. `exp_specdec_shape_gpu` shows it is not, to the last bit, and
`exp_specdec_fliprate_gpu` shows the last bit crosses the argmax margin about 1
token in 113. This experiment is the end-to-end consequence: how long two
"identical" decoders stay identical in practice.

Why this belongs to a verification testbed
------------------------------------------
`ivgym`'s Tier-0 proxy detector reads the acceptance rate `1 - TV(p, q)`
(`ivgym/spec_decode.py`); this measures the realized acceptance rate of a real
draft against a real target, which is the honest operating point that detector
calibrates against. More sharply: a verifier that recomputes `M` and compares
tokens will see these divergences on a provider that is running *exactly the
lossless algorithm it promised*. The divergence curve is a false-positive source
with no attack in it.

Writes `docs/results/specdec_divergence.json` (per-(prompt, gamma) first
divergence index, realized accept rate, and both decoded texts, so a
disagreement can be read rather than only counted). Figures come from
`python -m experiments.plot_specdec`.

    python -m experiments.exp_specdec_divergence_gpu
Env: IVGYM_SPECDEC_TARGET(Qwen/Qwen2.5-1.5B-Instruct),
     IVGYM_SPECDEC_DRAFT(Qwen/Qwen2.5-0.5B-Instruct), IVGYM_PROMPTS(all 32),
     IVGYM_TOKENS(128), IVGYM_GAMMAS(2,4,8).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import specdec_common as sc

TARGET = os.environ.get("IVGYM_SPECDEC_TARGET", "Qwen/Qwen2.5-1.5B-Instruct")
DRAFT = os.environ.get("IVGYM_SPECDEC_DRAFT", "Qwen/Qwen2.5-0.5B-Instruct")
N_PROMPTS = int(os.environ["IVGYM_PROMPTS"]) if "IVGYM_PROMPTS" in os.environ else None
MAXNEW = int(os.environ.get("IVGYM_TOKENS", 128))
GAMMAS = [int(g) for g in os.environ.get("IVGYM_GAMMAS", "2,4,8").split(",")]
DEV = "cuda"


@torch.no_grad()
def plain_greedy(model, ids, max_new, collect_margin=None):
    """The reference: one token at a time, KV cache, batch size 1.

    `collect_margin` accumulates the top1-top2 logit gap at each step -- the
    distance a perturbation has to cover to change this token.
    """
    cache = DynamicCache()
    out = model(ids, past_key_values=cache, use_cache=True)
    toks = []
    for _ in range(max_new):
        lg = out.logits[:, -1, :].float()
        if collect_margin is not None:
            s = lg[0].sort(descending=True).values
            collect_margin.append((s[0] - s[1]).item())
        t = lg.argmax(-1, keepdim=True)
        toks.append(t.item())
        out = model(t, past_key_values=cache, use_cache=True)
    return toks


@torch.no_grad()
def spec_greedy(model, draft, ids, max_new, gamma):
    """Standard lossless greedy speculative decoding. Returns (tokens, accept_counts).

    The draft proposes `gamma` tokens one at a time; the target verifies all of
    them in ONE chunked forward pass. That asymmetry is the whole point -- it is
    what makes speculation a speedup, and it is what puts the target's logits in
    a different numeric regime from `plain_greedy`'s.
    """
    L = ids.shape[1]
    tcache, dcache = DynamicCache(), DynamicCache()
    prev_last = model(ids, past_key_values=tcache, use_cache=True).logits[0, -1, :].float()
    dout = draft(ids, past_key_values=dcache, use_cache=True)
    emitted, accepts = [], []

    def step(m, cache, tok):
        return m(torch.tensor([[tok]], device=DEV), past_key_values=cache, use_cache=True)

    while len(emitted) < max_new:
        n0 = len(emitted)

        props = []
        for _ in range(gamma):
            t = dout.logits[0, -1, :].argmax(-1).item()
            props.append(t)
            dout = step(draft, dcache, t)

        tout = model(torch.tensor([props], device=DEV), past_key_values=tcache, use_cache=True)
        # Scoring logits for block positions 0..gamma-1: position 0 is scored by
        # the logits already in hand from before the block, the rest by the
        # chunked pass shifted one left.
        cand = torch.cat([prev_last.unsqueeze(0), tout.logits[0, :-1, :].float()], 0)
        tgt_argmax = cand.argmax(-1).tolist()

        n_acc = 0
        while n_acc < gamma and tgt_argmax[n_acc] == props[n_acc]:
            n_acc += 1
        accepts.append(n_acc)

        if n_acc == gamma:
            emitted += props
            emitted.append(tout.logits[0, -1, :].float().argmax(-1).item())   # bonus token
        else:
            emitted += props[:n_acc]
            emitted.append(tgt_argmax[n_acc])                                 # correction

        # Both caches must hold exactly the committed prefix minus its last token,
        # which is then re-fed to produce `prev_last` for the next block.
        tcache.crop(L + len(emitted) - 1)
        dcache.crop(L + len(emitted) - 1)
        prev_last = step(model, tcache, emitted[-1]).logits[0, -1, :].float()
        dout = step(draft, dcache, emitted[-1])
        assert len(emitted) > n0, "speculation made no progress"

    return emitted[:max_new], accepts


def main():
    tok = AutoTokenizer.from_pretrained(TARGET)
    model = sc.load_model(TARGET, torch.bfloat16, DEV)
    draft = sc.load_model(DRAFT, torch.bfloat16, DEV)
    prompts = sc.load_prompts(N_PROMPTS)
    rows, margins = [], []

    for pi, p in enumerate(prompts):
        ids = sc.encode_chat(tok, p, DEV)
        m: list[float] = []
        ref = plain_greedy(model, ids, MAXNEW, collect_margin=m)
        margins += m
        # Without this the whole experiment is unreadable: a divergence would be
        # ambiguous between the shape and plain run-to-run noise.
        assert ref == plain_greedy(model, ids, MAXNEW), "reference not run-to-run deterministic!"
        line = []
        for gamma in GAMMAS:
            spec, acc = spec_greedy(model, draft, ids, MAXNEW, gamma)
            div = next((i for i, (a, b) in enumerate(zip(ref, spec)) if a != b), None)
            rows.append(dict(prompt_idx=pi, gamma=gamma,
                             divergence_idx=-1 if div is None else div,
                             n_match=MAXNEW if div is None else div,
                             accept_rate=sum(acc) / (len(acc) * gamma),
                             ref_text=tok.decode(ref), spec_text=tok.decode(spec)))
            line.append(f"g={gamma}:{rows[-1]['divergence_idx']:>4}"
                        f"(acc{rows[-1]['accept_rate']:.2f})")
        print(f"[{pi:2d}] first divergence  " + "  ".join(line), flush=True)

    sc.write_result("specdec_divergence.json", dict(rows=rows, margins=margins))


if __name__ == "__main__":
    main()
