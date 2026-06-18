"""Core data structures: specification, traces, and the verify context.

These are backend-agnostic. The synthetic backend and a future vLLM adapter
both produce `Sequence` objects and answer `recompute_logits` / activations.
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
