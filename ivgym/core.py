"""Core data structures: specification, traces, and the verify context.

These are backend-agnostic. The HF-GPU backend and a future vLLM adapter both
produce `Sequence` objects and answer `recompute_logits` / activations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SamplingSpec:
    """The specification phi the provider claims to follow."""

    temperature: float = 1.0
    top_k: int | None = 50
    top_p: float | None = 0.95
    seed: int = 42

    def replace(self, **kw) -> "SamplingSpec":
        return SamplingSpec(**{**self.__dict__, **kw})


@dataclass
class TokenStep:
    """One generated token, as emitted by the provider."""

    position: int
    claimed_token: int            # t* the provider says it sampled
    sampling: SamplingSpec        # sampling params the *verifier* should use (= phi)
    fingerprint: np.ndarray | None = None  # Activation-DiFR: provider's projected activation


@dataclass
class Sequence:
    prompt_id: int
    config_name: str              # which attack/honest config produced this
    steps: list[TokenStep] = field(default_factory=list)


@dataclass
class VerifyContext:
    """Everything a Defense needs to score one token, computed by the verifier."""

    claimed_token: int
    ref_logits: np.ndarray        # verifier's trusted recomputed logits, shape [V]
    gumbel: np.ndarray            # shared Gumbel noise for this position, shape [V]
    sampling: SamplingSpec
    fingerprint: np.ndarray | None = None      # provider's activation fingerprint
    ref_fingerprint: np.ndarray | None = None  # verifier's recomputed fingerprint


@dataclass
class IOContext:
    """Everything an input-output (black-box) detector may see for one whole
    `Sequence`. The contrast with `VerifyContext` is the point of the abstraction:

      * `VerifyContext` is *per-token* and is handed `ref_logits` -- the verifier
        re-ran the reference model M's forward pass (the white-box / recomputation
        defenses).
      * `IOContext` is *per-sequence* and carries **no** `ref_logits` and **no**
        `ref_fingerprint`. An I/O detector decides "is this a faithful sample from
        M under spec phi?" from the prompt and the claimed tokens alone, *without*
        recomputing M. That "no recompute of M" boundary is the whole reason this
        is a separate context.

    `proxy_logits` is the one allowed escape hatch and does NOT break the boundary:
    it is the output of a *different, cheap* model (a small proxy LM), never M's
    own forward pass. It is the cheap end of the cost/accuracy Pareto ("a cheap
    model polices the expensive model"). Detectors that want it set
    `needs_proxy = True`; the harness fills it by calling `backend.proxy_logits`.
    """

    prompt_id: int
    claimed_tokens: list[int]
    sampling: SamplingSpec
    prompt_text: str | None = None             # raw prompt (text backends only)
    proxy_logits: np.ndarray | None = None     # [T, V] CHEAP-proxy logits, not M's
