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
class VContext:
    """The single, per-sequence context every `Verifier` scores against.

    This replaces the old `VerifyContext` (per-token, white-box) / `IOContext`
    (per-sequence, black-box) split. The split was really along ONE axis -- *which
    distribution the verifier paid to obtain* -- so we keep one context and let
    each field be present or absent by cost tier:

      Tier-0 (cheap, NO recompute of the reference model M). Always safe for a
      black-box / I/O verifier:
        * `proxy_logits`  -- a *different, cheap* proxy LM's logits `q` (never M's
          own forward pass); the cheap end of the cost/accuracy Pareto. It is also
          the substrate for the per-token `value` (informativeness) signal that
          directs where the expensive Tier-1 recompute is spent.
        * `served_logits` -- the distribution `p` the provider served under (what
          a provider returning logprobs exposes); drives the acceptance-rate
          fingerprint (`1 - TV(p, q)`).
        * `prompt_text`, `fingerprints` -- raw I/O / provider activations.

      Tier-1 (expensive: the verifier re-ran M's forward pass). Filled by the
      driver ONLY on audited rows (`audit_mask`), so a partial-budget run pays for
      recompute only where the cheap `value` signal says it matters:
        * `ref_logits`, `ref_fingerprints`, `gumbel`.

    Every array is `[T, V]` (or a length-`T` list) aligned to `claimed_tokens`.
    Tier-1 arrays carry valid rows only where `audit_mask[t]` is True (other rows
    are never read). A Tier-0 verifier ignores the Tier-1 fields entirely.
    """

    prompt_id: int
    claimed_tokens: list[int]
    sampling: SamplingSpec
    # --- Tier-0: cheap, no recompute of M ---
    proxy_logits: np.ndarray | None = None       # [T, V] cheap proxy q
    served_logits: np.ndarray | None = None      # [T, V] provider's served target p
    prompt_text: str | None = None               # raw prompt (text backends only)
    fingerprints: list | None = None             # provider activations per token
    # --- Tier-1: expensive, verifier recomputed M (audited rows only) ---
    ref_logits: np.ndarray | None = None         # [T, V] recomputed M logits
    ref_fingerprints: list | None = None         # verifier's recomputed activations
    gumbel: np.ndarray | None = None             # [T, V] shared Gumbel noise
    audit_mask: np.ndarray | None = None         # [T] bool: which rows got Tier-1 recompute
