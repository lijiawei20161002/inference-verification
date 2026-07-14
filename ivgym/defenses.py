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


def register(d):
    """Register a defense. Accepts either a `Defense` *instance* or a `Defense`
    *subclass* (instantiated with its defaults) so it works as a class
    decorator:

        @register
        class MyDefense(Defense):
            name = "my_defense"
            def score(self, ctx): ...

    Returns its argument unchanged, so it is decorator-safe.
    """
    if isinstance(d, type):
        inst = d()
        # Restore a subclass-declared `name` in case a dataclass __init__ reset
        # it (mirrors ivgym.attacks.register).
        inst.name = d.name
        d_inst = inst
    else:
        d_inst = d
    _REGISTRY[d_inst.name] = d_inst
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


@dataclass
class TokenTOPLOC(Defense):
    """TOPLOC-flavoured score: rank of the claimed token in the verifier's
    filtered (top-k/top-p) distribution. 0 means the claimed token is the
    verifier's argmax; larger means the provider emitted something the
    verifier ranks lower. Needs no shared Gumbel noise -- it only reads the
    reference logits -- so it is a cheap, seed-free contrast to Token-DiFR.

    A real TOPLOC (Sun et al., "TOPLOC: A Locality Sensitive Hashing Scheme
    for Trustless Verifiable Inference") commits to top-k index/value pairs
    per token and checks overlap against that commitment; this defense plays
    the same top-k-rank role within `ivgym`'s scalar-per-token-score harness,
    without the separate hashing/commitment layer (out of scope for a
    `Defense.score(ctx) -> float` contract).
    """

    name: str = "token_toploc"
    needs_seed: bool = False
    needs_activation: bool = False
    rank_cap: float = 50.0      # clip so a single filtered token can't dominate

    def score(self, ctx: VerifyContext) -> float:
        s = ctx.sampling
        filt = filtered_logits(ctx.ref_logits, s.top_k, s.top_p)
        if filt[ctx.claimed_token] <= -1e29:
            return self.rank_cap            # claimed token isn't even in top-k/p
        # how many tokens the verifier ranks strictly above the claimed one
        rank = float(np.sum(filt > filt[ctx.claimed_token]))
        return min(rank, self.rank_cap)


for d in [TokenDiFR(), CrossEntropy(), ActivationDiFR(), TokenTOPLOC()]:
    register(d)
