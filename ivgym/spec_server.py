"""The PROVIDER side of speculative decoding -- and what it does to verification.

Everything else in this repo treats speculative decoding as a *verifier* tool: the
client drafts with a cheap proxy it owns and reads the acceptance rate as
`1 - TV(p, q)` (`ivgym/spec_decode.py`). This module is the other half, and it is
adversarially the more interesting one: **the provider** drafts with a small model
and uses the reference model `M` only to accept, reject or resample. That is
standard, honest, distribution-preserving practice -- and it silently invalidates
the assumption a seed-replaying verifier is built on.

Why it breaks a seed-replaying verifier
---------------------------------------
`ivgym/sampling.py` implements DiFR's contract: provider and verifier draw the
*same* Gumbel vector from a public per-position seed, so the sampled token is a
deterministic function `t = argmax(filt(logits) + T*g)` of the logits alone, and
`token_difr`'s margin is 0 for an honest provider up to numerical noise. The
guarantee is not "the provider sampled from `p`" -- it is "the provider sampled
from `p` *via this exact map*".

A speculative-decoding server samples from the same `p` via a different map. Its
token is `x ~ q` accepted with probability `min(1, p(x)/q(x))`, else a draw from
the normalized residual `(p - q)_+`. The marginal law is *exactly* `p`
(`emit_token`'s docstring carries the two-line proof, and
`tests/test_spec_server.py` measures it), but the token is a function of the
draft's randomness and the server's acceptance uniforms -- quantities the verifier
does not hold and cannot reconstruct. Replaying the seed therefore recovers a
token the honest server had no reason to emit, and the margin is positive on most
positions.

So an honest speculative server is, to `token_difr`, indistinguishable in *kind*
from `attacks.WrongSeed` -- the deviation the README calls the clean case for
recomputation, caught at AUC 1.000. That is the false positive
`experiments/exp_spec_decode_difr_gpu.py` measures.

There is a second, independent mechanism that no seed bookkeeping can fix: the
target scores `gamma` drafted tokens in ONE batched forward pass, while an
ordinary decode reaches the same positions one token at a time and a verifier
reaches them with a full prefill. Those are three different reduction orders over
the same arithmetic, so they return different floats.
`experiments/exp_spec_batch_numerics_gpu.py` measures that separately, with the
simulated benign noise switched off so the number is real.

What is here
------------
* `emit_token` / `speculative_round` -- the pure-numpy sampler core. No torch, no
  backend: `tests/test_spec_server.py` Monte-Carlos it on CPU.
* `AcceptRule` -- `exact` (distribution-preserving), `lenient` (a cheaper,
  distribution-*shifting* server with a real economic motive), `topk_match`.
* `SpecTrace` -- the per-token record of how a token came to be emitted, which is
  what a *spec-aware* verifier would need the provider to disclose
  (`experiments/exp_spec_aware_verifier_gpu.py` uses it, and shows what it costs).
* `SpecDecode` attacks -- registered in `ivgym.attacks`'s registry so the harness
  generates and scores them like any other config. `honest_spec` is registered
  next to the cheats *even though it is honest*, because the whole finding is that
  a Tier-1 verifier cannot tell the difference.

Sources
-------
* Leviathan et al., Fast Inference from Transformers via Speculative Decoding
  (arXiv:2211.17192) -- Algorithm 1, the accept/reject rule and its correctness.
* Chen et al., Accelerating LLM Decoding with Speculative Sampling
  (arXiv:2302.01318) -- the residual-resample form used here.
* DiFR: Inference Verification Despite Nondeterminism (arXiv:2511.20621) -- the
  seed-replaying verifier whose assumption this breaks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .attacks import Attack, register
from .core import SamplingSpec
from .sampling import filtered_logits

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Numerics: every draw is an inverse-CDF read of ONE uniform.
#
# This is not a style choice. A spec-aware verifier can only replay the server's
# sampler if every random decision is a deterministic function of a *seedable*
# scalar; `rng.choice(p=...)` consumes an implementation-defined number of raw
# draws and is not replayable across a version bump. Taking each decision from a
# single named uniform is what makes `AcceptRule.seeded` possible at all -- see
# `position_uniforms`.
# ---------------------------------------------------------------------------
def categorical_from_uniform(probs: np.ndarray, u: float) -> int:
    """Inverse-CDF sample from `probs` using a single uniform `u` in [0, 1)."""
    cdf = np.cumsum(probs)
    total = float(cdf[-1])
    if total <= _EPS:                     # degenerate row: fall back to the mode
        return int(np.argmax(probs))
    return int(np.searchsorted(cdf, u * total, side="right"))


def spec_probs(logits: np.ndarray, spec: SamplingSpec) -> np.ndarray:
    """The distribution a position is actually sampled from under `spec`.

    Must agree *exactly* with what `sampling.gumbel_max_sample` draws from, or the
    two arms of the experiment would differ for a trivial reason instead of the
    one under study. That function returns `argmax(filt + T*g)` with `g` standard
    Gumbel, which is a draw from `softmax(filt / T)` -- so this is that softmax.
    """
    filt = filtered_logits(logits, spec.top_k, spec.top_p)
    z = filt / max(spec.temperature, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    e[filt <= -1e29] = 0.0                # keep filtered-out tokens at exactly 0
    return e / e.sum()


def residual_probs(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Normalized residual `(p - q)_+ / ||(p - q)_+||_1` -- the distribution a
    rejected draft is corrected from. Falls back to `p` if the residual is empty
    (only possible when `q == p`, in which case nothing is ever rejected)."""
    r = np.maximum(p - q, 0.0)
    s = float(r.sum())
    return (r / s) if s > _EPS else p


# ---------------------------------------------------------------------------
# Acceptance rules
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AcceptRule:
    """How a server decides whether to keep a drafted token.

    `exact` is the only rule that preserves `p`. The others are here because they
    are *cheaper* -- they accept more drafts, so they emit more tokens per target
    forward pass -- which is precisely the incentive a provider has to deviate, and
    they are easy to mistake for an implementation detail rather than a change of
    served distribution.
    """

    kind: str = "exact"
    threshold: float = 0.3      # `lenient`: accept when p(x)/q(x) >= threshold
    top_k: int = 8              # `topk_match`: accept when x is in p's top-k

    def accepts(self, p: np.ndarray, q: np.ndarray, x: int, u: float) -> bool:
        if self.kind == "exact":
            return u <= min(1.0, p[x] / max(q[x], _EPS))
        if self.kind == "lenient":
            return (p[x] / max(q[x], _EPS)) >= self.threshold
        if self.kind == "topk_match":
            return bool(np.sum(p > p[x]) < self.top_k)
        raise ValueError(f"unknown accept rule {self.kind!r}")

    def correction(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Distribution a rejected position is resampled from. Only `exact`'s
        residual makes the round distribution-preserving; resampling from `p`
        directly (the natural-looking shortcut) does not, because the rejection
        event is already correlated with `x`."""
        return residual_probs(p, q) if self.kind == "exact" else p


# ---------------------------------------------------------------------------
# Per-token provenance: what a spec-aware protocol would have to disclose.
# ---------------------------------------------------------------------------
@dataclass
class SpecTrace:
    """How one emitted token came to be emitted.

    `experiments/exp_spec_aware_verifier_gpu.py` treats this as the provider's
    *disclosure*: everything a verifier would need to replay the accept/reject
    decision. It is recorded here honestly; that experiment then asks what happens
    when a dishonest provider fabricates it, which is the point -- every field
    except `position` is provider-asserted.
    """

    position: int
    token: int                  # the token actually emitted
    draft_token: int            # what the draft proposed at this position
    accepted: bool              # was the draft kept?
    draft_prob: float           # q(draft_token), the draft's own probability
    target_prob: float          # p(draft_token) under the target the server used
    u_accept: float             # the acceptance uniform it drew
    u_sample: float             # the uniform used for the correction/bonus draw
    round_index: int            # which speculative round emitted it
    slot: int                   # index within the round (0 .. gamma)
    is_bonus: bool              # emitted from p_gamma after a full-accept round


def position_uniforms(seed: int, n: int = 2) -> np.ndarray:
    """The `n` uniforms a *seeded* speculative server uses at one position.

    A server that derives its acceptance and correction draws from the public
    per-position seed is replayable, which is what makes a spec-aware verifier
    possible (`AcceptRule` unchanged -- only the source of `u` moves). It is also a
    deployment burden: the server must commit to the seed schedule *and* to the
    draft model, and `exp_spec_aware_verifier_gpu` measures what that buys.
    """
    return np.random.default_rng(seed).random(n)


# ---------------------------------------------------------------------------
# The sampler core (pure numpy, no backend)
# ---------------------------------------------------------------------------
def emit_token(p: np.ndarray, q: np.ndarray, x: int, u_accept: float,
               u_sample: float, rule: AcceptRule = AcceptRule()) -> tuple[int, bool]:
    """One speculative position: return `(emitted_token, accepted)`.

    Correctness of `rule.kind == "exact"` (Leviathan et al. Thm. 1 / Chen et al.):
    for any token `t`, drafting `x ~ q` and accepting with `min(1, p/q)` emits `t`
    on the accept path with probability `q(t) * min(1, p(t)/q(t)) = min(p(t), q(t))`.
    Rejection happens with probability `1 - sum_x min(p, q) = ||(p - q)_+||_1`, and
    the correction draw then yields `t` with probability
    `(p(t) - q(t))_+ / ||(p - q)_+||_1`. Summing the two paths gives
    `min(p(t), q(t)) + (p(t) - q(t))_+ = p(t)`. Exactly `p`, for every `t`.

    So the *output* of an honest speculative server is unimpeachable. What it does
    not preserve is the map from randomness to token, which is the object a
    seed-replaying verifier actually checks.
    """
    if rule.accepts(p, q, x, u_accept):
        return int(x), True
    return categorical_from_uniform(rule.correction(p, q), u_sample), False


def speculative_round(p_rows: np.ndarray, q_rows: np.ndarray, drafted: list[int],
                      uniforms: np.ndarray, rule: AcceptRule = AcceptRule(),
                      ) -> tuple[list[int], int, list[dict]]:
    """One speculative round over `gamma` drafted tokens.

    `p_rows` is `[gamma + 1, V]`: the target distribution at the position each
    drafted token occupies, plus the bonus position reached only if every draft is
    accepted. `q_rows` is `[gamma, V]`. `uniforms` is `[gamma + 1, 2]`.

    Returns `(emitted, n_accepted, details)`. A round always emits
    `n_accepted + 1` tokens -- the accepted prefix plus either the correction for
    the first rejected draft or, on a full-accept round, a free bonus token drawn
    from `p_gamma`. That "+1" is where the speedup comes from and it is also why
    the emitted token count is not a multiple of anything a verifier can see.
    """
    gamma = len(drafted)
    emitted: list[int] = []
    details: list[dict] = []
    for i in range(gamma):
        tok, ok = emit_token(p_rows[i], q_rows[i], drafted[i],
                             float(uniforms[i, 0]), float(uniforms[i, 1]), rule)
        emitted.append(tok)
        details.append({"draft_token": int(drafted[i]), "accepted": bool(ok),
                        "draft_prob": float(q_rows[i][drafted[i]]),
                        "target_prob": float(p_rows[i][drafted[i]]),
                        "u_accept": float(uniforms[i, 0]),
                        "u_sample": float(uniforms[i, 1]), "is_bonus": False})
        if not ok:
            return emitted, i, details
    bonus = categorical_from_uniform(p_rows[gamma], float(uniforms[gamma, 1]))
    emitted.append(bonus)
    details.append({"draft_token": -1, "accepted": False, "draft_prob": float("nan"),
                    "target_prob": float(p_rows[gamma][bonus]),
                    "u_accept": float(uniforms[gamma, 0]),
                    "u_sample": float(uniforms[gamma, 1]), "is_bonus": True})
    return emitted, gamma, details


def expected_accept_rate(p: np.ndarray, q: np.ndarray) -> float:
    """`sum_x min(p, q) = 1 - TV(p, q)` -- the exact rule's acceptance rate, and
    the same identity the client-side detector in `spec_decode.py` reads. Here it
    is the provider's *speedup*: tokens emitted per target forward pass is
    `(1 - a^(gamma+1)) / (1 - a)` at acceptance rate `a`."""
    return float(np.minimum(p, q).sum())


# ---------------------------------------------------------------------------
# Registered configs. `spec_mode` is the only new field the backend reads; an
# attack without it generates by ordinary sequential decoding, so every existing
# config is untouched.
# ---------------------------------------------------------------------------
@dataclass
class SpecDecode(Attack):
    """A speculative-decoding server. HONEST by default -- `perturb_logits` is the
    base class's benign-noise-only implementation and the accept rule preserves
    `p`. It is a subclass of `Attack` only because that is this repo's word for
    "a way a provider can generate", and registering it here is what lets the
    harness score it against every verifier with no special-casing."""

    name: str = "honest_spec"
    spec_mode: str = "exact"       # AcceptRule.kind
    gamma: int = 4                 # draft length per round
    threshold: float = 0.3
    top_k_match: int = 8
    seeded: bool = False           # derive uniforms from the public position seed?

    def accept_rule(self) -> AcceptRule:
        return AcceptRule(kind=self.spec_mode, threshold=self.threshold,
                          top_k=self.top_k_match)


@dataclass
class SpecDecodeSeeded(SpecDecode):
    """The honest server made *replayable*: acceptance and correction uniforms come
    from the public per-position seed instead of the server's own entropy. Output
    distribution is identical (the uniforms are still uniform); the difference is
    that a verifier holding the draft model can now reproduce the decision.
    `exp_spec_aware_verifier_gpu` uses it as the constructive fix -- and then shows
    what the fix still does not buy."""

    name: str = "honest_spec_seeded"
    seeded: bool = True


@dataclass
class SpecDecodeLenient(SpecDecode):
    """A **cheating** speculative server: accept whenever `p(x)/q(x) >= threshold`
    rather than with probability `min(1, p/q)`, and correct a rejection from `p`
    instead of the residual. Both shortcuts raise the acceptance rate -- fewer
    target forward passes per emitted token, i.e. real money -- and both pull the
    served distribution toward the small draft model.

    This is the attack that matters once honest speculative decoding is on the
    table: it is a *deviation shaped like the excuse*. A verifier that stopped
    running `token_difr` because honest speculation false-positives has to catch
    this with what is left."""

    name: str = "spec_lenient"
    spec_mode: str = "lenient"
    threshold: float = 0.3


@dataclass
class SpecDecodeTopK(SpecDecodeLenient):
    """The other common shortcut: accept the draft whenever it lands in the
    target's top-k. Deterministic, so its acceptance rate is high and its served
    distribution is sharpened as well as shifted."""

    name: str = "spec_topk"
    spec_mode: str = "topk_match"
    top_k_match: int = 8


for _a in [SpecDecode(), SpecDecodeSeeded(), SpecDecodeLenient(), SpecDecodeTopK()]:
    register(_a)
