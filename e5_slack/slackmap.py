"""
Core of E5: honestly generate a continuation and, at every position, measure the model's
local decision *slack* using the GLS scorer.

Slack metric (per position):
  * |SAFE| = number of tokens with GLS > threshold for a given Gumbel draw — the set a
    covert sender could pick from, i.e. tokens indistinguishable from the honest sample.
    This depends on the random Gumbel noise, so we average |SAFE| over `n_seed_avg`
    independent draws to get a stable expectation, and report slack_bits = log2(mean|SAFE|).
  * entropy_bits = Shannon entropy of the (temperature/top-k/top-p) sampling distribution
    at that position — the model's intrinsic next-token uncertainty, noise-independent.

The honest token emitted is the Gumbel-max competitor c* under the canonical draw (seed k=0),
exactly an honest server. We also record its raw-logit rank and the syntactic category of
the emitted token so slack can be sliced by where it occurs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import torch

from e4_stego.gls import GLSParams, score_position, apply_top_k_top_p
from e4_stego.codec import _logits_full, GumbelStream

# Common English function words (closed-class). Used only to split word-initial tokens into
# function vs content for the by-category slice; nothing downstream depends on the exact list.
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "because", "as", "of", "to",
    "in", "on", "at", "by", "for", "with", "from", "into", "about", "over", "under", "is",
    "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "have", "has",
    "had", "will", "would", "can", "could", "shall", "should", "may", "might", "must", "that",
    "this", "these", "those", "it", "its", "he", "she", "they", "them", "his", "her", "their",
    "we", "you", "i", "me", "my", "our", "your", "not", "no", "yes", "than", "such", "which",
    "who", "whom", "what", "when", "where", "how", "there", "here", "also", "very", "more",
    "most", "some", "any", "all", "each", "both", "either", "neither",
}

_PUNCT_RE = re.compile(r"^[^\w\s]+$")
_NUM_RE = re.compile(r"^\d[\d.,]*$")


def categorize(tok_str: str) -> str:
    """Syntactic bucket for a decoded token string."""
    core = tok_str.strip()
    if core == "":
        return "whitespace"
    if _PUNCT_RE.match(core):
        return "punct"
    if _NUM_RE.match(core):
        return "number"
    leading_space = tok_str[:1].isspace()
    if leading_space:
        return "function-word" if core.lower() in STOPWORDS else "content-word"
    return "word-cont"  # subword continuation (no leading space)


@dataclass
class TokenSlack:
    pos: int
    token_id: int
    token_str: str
    category: str
    entropy_bits: float
    safe_size_seed0: int
    safe_size_mean: float
    slack_bits: float            # log2(mean|SAFE|)
    nondanger_size_mean: float   # SAFE+SUSPICIOUS (rank<=thr) band, for reference
    top1_prob: float
    emitted_rank: int            # raw-logit rank of the honest token (0 = argmax)


@dataclass
class MapResult:
    prompt_idx: int
    gen_ids: list[int]
    tokens: list[TokenSlack] = field(default_factory=list)


def _entropy_bits(logits_V: torch.Tensor, p: GLSParams) -> float:
    device = logits_V.device
    tl = logits_V / p.temperature if p.temperature > 0 else logits_V
    tk = torch.tensor([p.top_k], device=device) if p.top_k is not None else None
    tp = torch.tensor([p.top_p], device=device)
    masked = apply_top_k_top_p(tl[None, :], tk, tp).squeeze(0)
    logp = torch.log_softmax(masked.float(), dim=-1)
    p_ = logp.exp()
    ent_nats = -(p_ * logp).masked_fill(~torch.isfinite(logp), 0.0).sum()
    return float(ent_nats.item() / np.log(2.0))


def generate_and_map(model, tokenizer, prompt_ids: list[int], params: GLSParams, *,
                     seed: int, max_tokens: int, n_seed_avg: int = 4,
                     eos_id: int | None = None, prompt_idx: int = 0) -> MapResult:
    device = model.device
    vocab = model.config.vocab_size
    # n_seed_avg independent Gumbel streams; stream 0 is the canonical (honest-emit) one.
    streams = [GumbelStream(seed + k, vocab, device) for k in range(n_seed_avg)]

    res = MapResult(prompt_idx=prompt_idx, gen_ids=[])
    for pos in range(max_tokens):
        logits_V = _logits_full(model, prompt_ids + res.gen_ids)[-1]
        draws = [s.next() for s in streams]

        sc0 = score_position(logits_V, draws[0], params)          # canonical: honest emit + rank
        safe_sizes = []
        nondanger_sizes = []
        for d in draws:
            sc = score_position(logits_V, d, params)
            safe_sizes.append(int(sc.safe_mask.sum().item()))
            nondanger_sizes.append(int(sc.admissible_mask.sum().item()))
        safe_mean = float(np.mean(safe_sizes))
        nond_mean = float(np.mean(nondanger_sizes))

        tok = sc0.competitor                                       # honest Gumbel-max sample
        tok_str = tokenizer.decode([tok])
        top1_prob = float(torch.softmax(logits_V.float(), dim=-1).max().item())

        res.tokens.append(TokenSlack(
            pos=pos, token_id=tok, token_str=tok_str, category=categorize(tok_str),
            entropy_bits=_entropy_bits(logits_V, params),
            safe_size_seed0=int(sc0.safe_mask.sum().item()),
            safe_size_mean=safe_mean,
            slack_bits=float(np.log2(max(1.0, safe_mean))),
            nondanger_size_mean=nond_mean,
            top1_prob=top1_prob,
            emitted_rank=int(sc0.ranks[tok].item()),
        ))
        res.gen_ids.append(tok)
        if eos_id is not None and tok == eos_id:
            break
    return res
