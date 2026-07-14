"""Example custom strategies -- a template you can copy.

Drop your own attacks/defenses in a file like this, then point the runner at it:

    .venv/bin/python -m experiments.run --strategies examples/custom_strategies.py

No library files are edited: importing this module runs the `@register`
decorators, which add the strategies to the same registries the harness and
every backend already use. The two strategies below run unchanged on a real
model on a GPU (the default `hf_gpu` backend).

(The rank-based TOPLOC-flavoured defense that used to live here as a demo has
since been promoted to a first-class built-in: `ivgym.defenses.TokenTOPLOC`,
name `"token_toploc"`. The defense below is a distinct, deliberately-naive
example kept purely to show the registration mechanism.)
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
# A custom DEFENSE: is the claimed token the verifier's argmax under the
# filtered (top-k/top-p) distribution? A deliberately naive top-1-only check
# (no seed sync needed) -- weaker than the built-in rank-based `token_toploc`,
# here only to demonstrate the registration mechanism.
# ---------------------------------------------------------------------------
@register_defense
@dataclass
class Top1Mismatch(Defense):
    name: str = "top1_mismatch_toy"
    needs_seed: bool = False
    needs_activation: bool = False

    def score(self, ctx) -> float:
        s = ctx.sampling
        filt = filtered_logits(ctx.ref_logits, s.top_k, s.top_p)
        if filt[ctx.claimed_token] <= -1e29:
            return 1.0               # claimed token isn't even in top-k/p
        return 0.0 if int(np.argmax(filt)) == ctx.claimed_token else 1.0
