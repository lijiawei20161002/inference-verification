"""Defenses = per-token divergence-from-reference scores.

Higher score => more divergent from the reference (so it can be thresholded
directly). This is the extension point for "different versions of defense":
subclass `Defense`, register it, and the harness scores every config with it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import VerifyContext
from .sampling import filtered_logits, log_softmax

_REGISTRY: dict[str, "Defense"] = {}


def register(d: "Defense") -> "Defense":
    _REGISTRY[d.name] = d
    return d


def get(name: str) -> "Defense":
    return _REGISTRY[name]


def all_defenses() -> dict[str, "Defense"]:
    return dict(_REGISTRY)


class Defense:
    name: str = "base"
    needs_seed: bool = True            # requires shared Gumbel noise?
    needs_activation: bool = False

    def score(self, ctx: VerifyContext) -> float:
        raise NotImplementedError


# ----------------------------------------------------------------------------
@dataclass
class TokenDiFR(Defense):
    """Clipped post-Gumbel logit margin (DiFR's default metric).

    delta = z[v_token] - z[claimed]; v_token = verifier's argmax under shared
    noise. 0 if tokens agree, grows with divergence, +Delta_max if the claimed
    token was filtered out by top-k/top-p.
    """

    name: str = "token_difr"
    delta_max: float = 30.0
    needs_seed: bool = True

    def score(self, ctx: VerifyContext) -> float:
        s = ctx.sampling
        filt = filtered_logits(ctx.ref_logits, s.top_k, s.top_p)
        z = filt + s.temperature * ctx.gumbel
        v_token = int(np.argmax(z))
        if filt[ctx.claimed_token] <= -1e29:
            return self.delta_max          # claimed token filtered out
        diff = float(z[v_token] - z[ctx.claimed_token])
        return min(diff, self.delta_max)


@dataclass
class CrossEntropy(Defense):
    """Negative log-likelihood of the claimed token under the verifier softmax.
    No seed synchronization required -- and defeatable by temperature tuning."""

    name: str = "cross_entropy"
    needs_seed: bool = False

    def score(self, ctx: VerifyContext) -> float:
        s = ctx.sampling
        logp = log_softmax(ctx.ref_logits / max(s.temperature, 1e-6))
        return float(-logp[ctx.claimed_token])


@dataclass
class ActivationDiFR(Defense):
    """L2 distance between provider and verifier projected activations."""

    name: str = "activation_difr"
    needs_seed: bool = False
    needs_activation: bool = True

    def score(self, ctx: VerifyContext) -> float:
        if ctx.fingerprint is None or ctx.ref_fingerprint is None:
            return 0.0
        return float(np.linalg.norm(ctx.fingerprint - ctx.ref_fingerprint))


for d in [TokenDiFR(), CrossEntropy(), ActivationDiFR()]:
    register(d)
