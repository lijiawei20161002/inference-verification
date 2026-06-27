"""Example custom strategies -- a template you can copy.

Drop your own attacks/defenses in a file like this, then point the runner at it:

    .venv/bin/python -m experiments.run --strategies examples/custom_strategies.py

No library files are edited: importing this module runs the `@register`
decorators, which add the strategies to the same registries the harness and
every backend already use. The two strategies below run unchanged on a real
model on a GPU (the default `hf_gpu` backend).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ivgym.attacks import Attack, register as register_attack
from ivgym.defenses import Defense, register as register_defense
from ivgym.sampling import filtered_logits


# ---------------------------------------------------------------------------
# A custom ATTACK: a rare, large logit "spike" (e.g. a buggy fused kernel that
# occasionally corrupts a few logits). Most tokens look honest; a small fraction
# get a big structured perturbation -- the kind of low-rate deviation batch
# statistics are meant to catch.
# ---------------------------------------------------------------------------
@register_attack
@dataclass
class LogitSpike(Attack):
    name: str = "logit_spike"
    rate: float = 0.03          # fraction of positions that get spiked
    spike: float = 6.0          # magnitude of the spike
    n_spiked: int = 3           # how many logits get hit when it fires

    def perturb_logits(self, base, rng):
        out = base + rng.normal(0.0, self.benign_sigma, base.shape)
        if rng.random() < self.rate:
            idx = rng.choice(base.shape[-1], size=self.n_spiked, replace=False)
            out[idx] += self.spike
        return out


# ---------------------------------------------------------------------------
# A custom DEFENSE: rank of the claimed token in the verifier's filtered
# distribution (a TOPLOC-flavoured score). 0 means the claimed token is the
# verifier's argmax; larger means the provider emitted something the verifier
# ranks lower. Needs no shared Gumbel noise -- it only reads the reference
# logits -- so it is a cheap, seed-free contrast to Token-DiFR.
# ---------------------------------------------------------------------------
@register_defense
@dataclass
class TopKOverlap(Defense):
    name: str = "topk_overlap"
    needs_seed: bool = False
    needs_activation: bool = False
    rank_cap: float = 50.0      # clip so a single filtered token can't dominate

    def score(self, ctx) -> float:
        s = ctx.sampling
        filt = filtered_logits(ctx.ref_logits, s.top_k, s.top_p)
        if filt[ctx.claimed_token] <= -1e29:
            return self.rank_cap            # claimed token isn't even in top-k/p
        # how many tokens the verifier ranks strictly above the claimed one
        rank = float(np.sum(filt > filt[ctx.claimed_token]))
        return min(rank, self.rank_cap)
