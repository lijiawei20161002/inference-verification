"""Measuring the SAME next-token logits under two different forward-pass shapes.

Shared by the four `exp_specdec_*_gpu.py` experiments. It lives in `experiments/`
rather than `ivgym/` on purpose: the numpy-only core (`sampling`, `verifiers`,
`harness`, `metrics`, `signal`, `spec_decode`) must stay importable without
torch, and everything here is torch.

The quantity
------------
Fix a token sequence. The next-token logit vector at each position is a
mathematically well-defined function of the weights and the prefix -- it does not
depend on *how* the forward pass is scheduled. Three schedules compute it:

  sequential : prefill a prefix, then feed the remaining tokens one at a time
               through a KV cache. This is plain autoregressive serving, and it
               is the REFERENCE here because it is what an honest single-stream
               provider does.
  chunked    : one forward pass over the whole sequence, reading the logits off
               every position at once. This is the speculative-decoding VERIFY
               shape, and it is also a prefill.
  batched    : the chunked pass, but sharing a batch with unrelated rows.

In exact arithmetic all three agree exactly. In floating point they do not,
because the shape selects the kernel: a different reduction order, tile size or
split-K decomposition gives a different last bit. `compare()` quantifies the gap
and `ulp_diff()` puts it on the only scale where "a different last bit" is
literally what it means.

Why this repo cares
-------------------
Every AUC in `ivgym` is measured against an honest null of "the same model, run
twice, in the same shape" -- which `compare()` confirms is bitwise identical
(`rerun_control`, frac_exact = 1.0). A deployed verifier does not get that null:
it recomputes `M` in whatever shape its own hardware picks, against tokens the
provider produced in a different one. The gap measured here is the floor under
that false-positive rate, and it is on the same scale as the deviations the
detectors are trying to catch. See `NEXT_EXPERIMENTS.md` item 1.

The API is deliberately symmetric -- `sequential_logits` and `chunked_logits`
take the same `(n_prefill, n_logits)` window and return the same
`[n_logits, vocab]` fp32 tensor, indexed from position `n_prefill - 1` -- so a
call site cannot accidentally compare two different sets of positions. The three
original scripts each carried their own copy of these two functions, offset from
one another by one position; that is precisely the drift this module removes.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "experiments" / "data"
RES_DIR = ROOT / "docs" / "results"

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_model(name: str, dtype: torch.dtype, device: str = "cuda"):
    """Eval-mode causal LM, `sdpa` attention pinned.

    The attention implementation is pinned rather than left to
    transformers' auto-selection because it is one of the knobs under study:
    an experiment that silently changed backend between conditions would be
    measuring the backend, not the shape.
    """
    return AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, attn_implementation="sdpa").to(device).eval()


def load_prompts(n: int | None = None) -> list[str]:
    """The 32-prompt bank these experiments were measured on.

    Deliberately NOT `backends.hf_gpu.DEFAULT_PROMPTS`: these are chat-formatted
    instruction prompts run through the tokenizer's chat template, which is what
    makes a real 0.5B draft model's proposals worth accepting in
    `exp_specdec_divergence_gpu`. The committed artifacts are measured on this
    file, so changing it invalidates them.
    """
    prompts = json.loads((DATA_DIR / "specdec_prompts.json").read_text())
    return prompts if n is None else prompts[:n]


def load_longtext() -> str:
    """One long natural document: Dickens, *A Tale of Two Cities* (public domain).

    `exp_specdec_ctxlen_gpu` sweeps context length over windows of this, so every
    length is measured on real text rather than on a repeated or synthetic prefix
    whose attention pattern would be degenerate.
    """
    return (DATA_DIR / "specdec_longtext.txt").read_text()


def encode_chat(tok, prompt: str, device: str = "cuda") -> torch.Tensor:
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").input_ids.to(device)


# --------------------------------------------------------------- the two shapes
@torch.no_grad()
def sequential_logits(model, ids: torch.Tensor, n_prefill: int, n_logits: int) -> torch.Tensor:
    """THE REFERENCE. Prefill `ids[:, :n_prefill]`, then decode one token at a time.

    Returns `[n_logits, vocab]` fp32: the next-token logits at positions
    `n_prefill - 1 ... n_prefill + n_logits - 2`. The first comes off the prefill;
    each subsequent one comes off a single-token forward through the KV cache,
    consuming `ids[:, n_prefill + i]`. Note the last token of the window is never
    consumed -- its own next-token logits are not part of the window.
    """
    cache = DynamicCache()
    out = model(ids[:, :n_prefill], past_key_values=cache, use_cache=True)
    lg = [out.logits[:, -1, :].float()]
    for i in range(n_logits - 1):
        out = model(ids[:, n_prefill + i: n_prefill + i + 1],
                    past_key_values=cache, use_cache=True)
        lg.append(out.logits[:, -1, :].float())
    return torch.cat(lg, 0)


@torch.no_grad()
def chunked_logits(model, ids: torch.Tensor, n_prefill: int, n_logits: int,
                   batch_size: int = 1, seed: int = 7) -> torch.Tensor:
    """One forward pass over the whole window; read the SAME `n_logits` positions.

    `batch_size > 1` pads the batch with unrelated random filler rows and still
    reads row 0. The filler cannot influence row 0 through attention -- rows are
    independent -- so any difference it makes is the batch-invariance problem:
    the kernel's reduction schedule depends on who else is in the batch.
    """
    x = ids
    if batch_size > 1:
        g = torch.Generator(device="cpu").manual_seed(seed)
        vocab_hi = min(int(model.config.vocab_size), 30000)
        filler = torch.randint(1000, vocab_hi, (batch_size - 1, ids.shape[1]),
                               generator=g).to(ids.device)
        x = torch.cat([ids, filler], 0)
    out = model(x, use_cache=False)
    return out.logits[0, n_prefill - 1: n_prefill - 1 + n_logits, :].float()


# ------------------------------------------------------------------ comparisons
def ulp_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distance in fp32 representable steps (ULPs).

    The natural unit for this question: 1 ULP is "the last bit differs", which is
    the smallest disagreement two schedules of the same arithmetic can possibly
    have. Reinterprets the bits as sign-magnitude ordered integers so that the
    count is exact across the sign boundary.
    """
    def ordered(t):
        i = t.contiguous().view(torch.int32).to(torch.int64)
        return torch.where(i < 0, torch.tensor(-2147483648, dtype=torch.int64,
                                               device=i.device) - i, i)
    return (ordered(a) - ordered(b)).abs().float()


def compare(ref: torch.Tensor, alt: torch.Tensor, tag: str) -> dict:
    """Summarize `alt` against the sequential reference `ref`, both `[pos, vocab]`.

    `frac_exact` is the headline: the share of logit *values* that are bitwise
    identical. `argmax_flips` is what actually changes an output token, and
    `margin_*` records the top1-top2 gap on the reference, because a flip needs
    the perturbation to exceed the margin -- the two numbers are only
    interpretable together.
    """
    d = (ref - alt).abs()
    s = ref.sort(dim=-1, descending=True).values
    margin = s[:, 0] - s[:, 1]
    # Counted in int64 and divided in python, NOT as `(d == 0).float().mean()`.
    # These tensors run to ~3e7 entries, and an fp32 mean over that many ones
    # accumulates to 1 - 2^-24: the committed artifact records the rerun control
    # at frac_exact = 0.99999994 for exactly this reason, when the truth (its
    # max_abs is exactly 0.0) is 1. A statistic whose whole job is to say
    # "bitwise identical" must not itself round.
    n_exact = int((d == 0).sum().item())
    return dict(
        tag=tag,
        max_abs=d.max().item(),
        mean_abs=d.double().mean().item(),
        p99_abs=d.flatten().kthvalue(int(0.99 * d.numel())).values.item(),
        frac_exact=n_exact / d.numel(),
        max_ulp=ulp_diff(ref, alt).max().item(),
        argmax_flips=(ref.argmax(-1) != alt.argmax(-1)).sum().item(),
        n_pos=ref.shape[0],
        margin_min=margin.min().item(),
        margin_median=margin.median().item(),
    )


def write_result(name: str, payload) -> Path:
    """Write `docs/results/<name>` and say so. Every artifact lands in one place."""
    RES_DIR.mkdir(parents=True, exist_ok=True)
    path = RES_DIR / name
    path.write_text(json.dumps(payload, indent=1))
    print(f"wrote {path.relative_to(ROOT)}")
    return path
