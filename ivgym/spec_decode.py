"""Client-side speculative verification: a cheap proxy that approximates recompute.

Motivation
----------
The strong inference verifier (``defenses.token_difr``) recomputes the *reference
model* ``M`` for every claimed token -- correct, but as expensive as re-running
inference. This module implements the **cheap** regime: the client holds its own
small **proxy** model ``q`` (a genuinely different, smaller model that shares
``M``'s tokenizer -- e.g. Qwen3-0.6B for Qwen3-4B) and uses the
**speculative-decoding acceptance rate** as a cheap approximation of full
recomputation.

Crucially this is **entirely client-side**: nothing here trusts a provider
self-report, and nothing asks the provider (vLLM) to emit a trace or a
commitment. The client runs the proxy itself. That is the whole point -- the
provider is the untrusted party, so a verifier that depends on the provider's
cooperation is one it can stub out.

The acceptance-rate signal (the theory)
---------------------------------------
Speculative decoding accepts a token ``x ~ q`` with probability
``min(1, p(x)/q(x))``. Averaging over ``x ~ q`` gives the expected acceptance
rate::

    E_{x~q}[ min(1, p(x)/q(x)) ]  =  Σ_x min(p(x), q(x))  =  1 − TV(p, q)

so the realized acceptance rate is a **direct measurement of ``1 − TV(p, q)``**
-- the agreement between the served target distribution ``p`` and the client's
proxy ``q``. Because the client owns ``q`` it is a **trusted anchor**: when a
provider serves a corrupted target (quantized / fp8 / fewer-layer weights,
``p* → p̂``), the acceptance rate moves from ``1 − TV(p*, q)`` to
``1 − TV(p̂, q)`` and the client sees the shift **without recomputing ``M``**,
by comparing to a one-time offline honest reference for this ``(M, proxy, spec)``
(``ProxyReference``). See ``docs/ACCEPTANCE_RATE_FINGERPRINT.md`` and
``docs/SPEC_DECODING_AND_PROXY_DETECTION.md``.

What the cheap proxy can and cannot see
---------------------------------------
The acceptance-rate signal is the **best of the no-recompute options, not a
silver bullet** (same recompute-dominant boundary the DiFR detectors hit):

* it needs a one-time trusted honest reference (a single offline ``M`` run);
* it only fires once the corruption moves ``TV(p, q)`` past honest run-to-run
  variance -- **small quantization is invisible** to it; and
* it does not dominate the full recompute (``recompute_divergence`` /
  ``token_difr``), which stays at AUC ~ 1.0.

What it buys is a **shrink in how often the exact recompute must fire**, and a
draft-anchored axis that survives the temperature-retune evasion an entropy /
cross-entropy fingerprint does not (matching entropy does not restore
``TV(p̂, q)``).

Everything here is pure numpy and backend-agnostic. ``synthetic_positions``
fabricates correlated ``(p, q)`` pairs so the whole verifier runs on CPU for
tests; a real backend feeds actual ``M`` / proxy logprob rows instead (see
``ivgym/backends/hf_gpu.py`` -- ``reference_logits`` supplies ``p`` for the
one-time reference, ``proxy_logits`` supplies ``q`` -- and
``experiments/exp_proxy_spec_verify.py`` for the real-proxy run).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import SamplingSpec

# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------
_EPS = 1e-12


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    return z - np.log(np.exp(z).sum())


def tv(p: np.ndarray, q: np.ndarray) -> float:
    """Total-variation distance ½·Σ|p−q| between two distributions."""
    return float(0.5 * np.abs(p - q).sum())


def accept_rate(p: np.ndarray, q: np.ndarray) -> float:
    """Expected speculative-decoding acceptance rate of proxy ``q`` against
    target ``p``:  ``E_{x~q}[min(1, p/q)] = Σ min(p, q) = 1 − TV(p, q)``."""
    return float(np.minimum(p, q).sum())


def per_token_accept_prob(p: np.ndarray, q: np.ndarray, x: int) -> float:
    """Speculative-decoding accept probability for a *specific* served token
    ``x``:  ``min(1, p(x)/q(x))``."""
    return float(min(1.0, p[x] / max(q[x], _EPS)))


# ---------------------------------------------------------------------------
# Position: the true target/proxy distributions at one position.
# ---------------------------------------------------------------------------
@dataclass
class Position:
    """One position's distributions, as the client sees them.

    ``target_logprobs`` is log ``p`` -- the distribution the provider **served
    under** (equal to the true model for an honest provider, or a corrupted
    ``p̂`` for a quantized one). ``proxy_logprobs`` is log ``q`` -- the client's
    own cheap proxy, always trustworthy because the client runs it.
    """

    target_logprobs: np.ndarray   # log p (served distribution), shape [V]
    proxy_logprobs: np.ndarray    # log q (client proxy),        shape [V]


def synthetic_positions(rng: np.random.Generator, n: int, vocab: int = 64,
                        agreement: float = 0.85, sharpness: float = 2.0) -> list[Position]:
    """Fabricate ``n`` correlated (target, proxy) distribution pairs on CPU.

    ``agreement`` in [0,1] blends the proxy toward the target (higher -> higher
    acceptance rate, as for a within-family proxy); ``sharpness`` scales the
    target logits. Pure numpy -- a real backend supplies these from actual model
    forward passes instead (``target = M``, ``proxy = the cheap model``)."""
    out = []
    for _ in range(n):
        t = rng.standard_normal(vocab) * sharpness
        t_lp = _log_softmax(t)
        noise = rng.standard_normal(vocab) * sharpness
        d = agreement * t + (1.0 - agreement) * noise
        d_lp = _log_softmax(d)
        out.append(Position(target_logprobs=t_lp, proxy_logprobs=d_lp))
    return out


# ---------------------------------------------------------------------------
# The client-side sample: served tokens + the (p, q) rows the client scored,
# plus the hidden true target used ONLY by the recompute baseline oracle.
# ---------------------------------------------------------------------------
@dataclass
class ProxySample:
    """One served sequence as the client verifier processes it.

    ``positions`` carries, per emitted token, the served target ``p`` and the
    client proxy ``q``. ``served_tokens`` are the tokens the provider actually
    emitted (drawn from the served target ``p``). ``_truth`` is the *true*
    (uncorrupted) target logprobs per position -- NOT observable to the cheap
    client; it stands in for the trusted forward pass that the recompute
    baseline (``recompute_divergence``) would run on a sampled subset.
    """

    provider_name: str
    spec: SamplingSpec
    positions: list[Position] = field(default_factory=list)
    served_tokens: list[int] = field(default_factory=list)
    _truth: list[np.ndarray] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Provider deviations (client-side view): how the served target was produced.
# Only forward-pass / output deviations remain -- the sampler-procedure cheats
# of the old provider-trace design assumed the provider emitted a trace to break.
# ---------------------------------------------------------------------------
_CHEATS: dict[str, "CheatStrategy"] = {}


def register_cheat(c):
    """Register a cheat (instance or subclass, like ``ivgym.attacks.register``)."""
    if isinstance(c, type):
        inst = c()
        inst.name = c.name
    else:
        inst = c
    _CHEATS[inst.name] = inst
    return c


def get_cheat(name: str) -> "CheatStrategy":
    return _CHEATS[name]


def all_cheats() -> dict[str, "CheatStrategy"]:
    return dict(_CHEATS)


@dataclass
class CheatStrategy:
    """A provider, honest by default. ``served_target_logprobs`` returns the
    distribution the provider actually served under, given the *true* target
    ``p*``. An honest provider serves ``p*`` itself; a forward-pass cheat serves
    a corrupted ``p̂``. The served token is then sampled from that distribution.
    """

    name: str = "honest"

    def served_target_logprobs(self, rng, p_true_lp):
        return p_true_lp


@dataclass
class QuantTarget(CheatStrategy):
    """Forward-pass deviation (the SD analogue of ``quant_4bit`` / ``kv_fp8``):
    the provider serves under a **quantized target**, so the true target logits
    are perturbed to ``p̂``. It samples honestly from ``p̂`` -- the corruption is
    in the distribution, not the procedure -- so nothing about the served tokens
    is internally inconsistent. Only the draft-anchored acceptance rate (moved by
    ``TV(p̂, q) ≠ TV(p*, q)``) or a full recompute of ``p*`` can see it."""

    name: str = "quant_target"
    sigma: float = 0.5        # quantization logit-error scale
    temp: float = 1.0         # temperature retune on the quantized target

    def served_target_logprobs(self, rng, p_true_lp):
        noisy = p_true_lp + rng.normal(0.0, self.sigma, p_true_lp.shape)
        return _log_softmax(noisy / max(self.temp, 1e-6))


@dataclass
class AdvQuantTemp(QuantTarget):
    """Quantize the target, then **retune temperature** so the served
    distribution's entropy matches the honest reference again. This defeats the
    generic *entropy / cross-entropy* fingerprint (a no-recompute baseline) but
    NOT the draft-anchored acceptance rate: retuning temperature does not restore
    ``TV(p̂, q)``, which the trusted proxy still measures. See
    ``experiments/exp_proxy_spec_verify.py``."""

    name: str = "adv_quant_temp"
    sigma: float = 0.5
    temp: float = 0.85        # tuned at runtime to match honest entropy


# Register the built-ins.
for _c in [CheatStrategy(), QuantTarget(), AdvQuantTemp()]:
    register_cheat(_c)


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------
def generate_sample(rng: np.random.Generator, cheat: CheatStrategy,
                    positions: list[Position],
                    spec: SamplingSpec | None = None) -> ProxySample:
    """Run one provider (honest or cheating) over ``positions``: at each position
    it serves under ``cheat.served_target_logprobs`` and emits a token sampled
    from that served distribution. Returns the ``ProxySample`` the client scores.
    """
    spec = spec or SamplingSpec()
    sample = ProxySample(provider_name=cheat.name, spec=spec)
    for pos in positions:
        p_true_lp = pos.target_logprobs
        served_lp = cheat.served_target_logprobs(rng, p_true_lp)
        served_p = np.exp(served_lp)
        served_p = served_p / served_p.sum()          # guard tiny renorm drift
        x = int(rng.choice(len(served_p), p=served_p))
        sample.positions.append(Position(target_logprobs=served_lp,
                                         proxy_logprobs=pos.proxy_logprobs))
        sample.served_tokens.append(x)
        sample._truth.append(p_true_lp)
    return sample


def positions_from_rows(target_logprobs: np.ndarray,
                        proxy_logprobs: np.ndarray) -> list[Position]:
    """Build ``Position`` rows from two aligned ``[T, V]`` logprob arrays -- the
    adapter a real backend uses: ``target`` = ``M``'s (or the served) rows,
    ``proxy`` = the cheap model's rows over the same ``[prompt + claimed]``
    sequence (``ivgym.backends.hf_gpu`` ``reference_logits`` / ``proxy_logits``).
    """
    return [Position(target_logprobs=t, proxy_logprobs=q)
            for t, q in zip(target_logprobs, proxy_logprobs)]


# ---------------------------------------------------------------------------
# No-recompute features of a served sample (the cheap client-side statistics).
# ---------------------------------------------------------------------------
def sequence_features(sample: ProxySample) -> dict[str, float]:
    """Aggregate statistics of one served sample, all computed from the served
    target ``p`` and the client proxy ``q`` (no recompute of the true ``M``):

    * ``accept_rate``    mean over positions of ``1 − TV(p, q)`` -- the
                         SD-specific, draft-anchored signal;
    * ``mean_entropy``   mean entropy of the served target (the generic
                         entropy-fingerprint baseline, gameable by temp retune);
    * ``mean_surprisal`` mean proxy surprisal ``−log q(x)`` of the served tokens
                         (the ``surface_stat`` / cross-entropy cousin).
    """
    acc, ent, sur = [], [], []
    for pos, x in zip(sample.positions, sample.served_tokens):
        p = np.exp(pos.target_logprobs)
        q = np.exp(pos.proxy_logprobs)
        acc.append(accept_rate(p, q))
        ent.append(float(-np.sum(p * np.log(p + _EPS))))
        sur.append(float(-pos.proxy_logprobs[x]))
    return {
        "accept_rate": float(np.mean(acc)) if acc else 0.0,
        "mean_entropy": float(np.mean(ent)) if ent else 0.0,
        "mean_surprisal": float(np.mean(sur)) if sur else 0.0,
    }


class ProxyReference:
    """The one-time offline honest anchor. ``fit`` characterizes honest served
    samples (per-feature mean/std -- a single trusted ``M`` run, amortized over
    every later verification). ``score`` returns a per-feature standardized
    distance ``|value − mean| / std`` (higher = more anomalous) plus an
    aggregate. Detection is a threshold on that distance.

    It requires this external anchor by design: the acceptance rate of a served
    sample is meaningful only relative to what honest ``(M, proxy)`` produces.
    """

    def __init__(self, features: tuple[str, ...] = ("accept_rate", "mean_entropy",
                                                    "mean_surprisal")):
        self.features = features
        self.mean: dict[str, float] = {}
        self.std: dict[str, float] = {}

    def fit(self, honest_samples: list[ProxySample]) -> "ProxyReference":
        rows = [sequence_features(s) for s in honest_samples]
        for f in self.features:
            vals = np.array([r[f] for r in rows], float)
            self.mean[f] = float(vals.mean())
            self.std[f] = float(vals.std() + _EPS)
        return self

    def score(self, sample: ProxySample) -> dict[str, float]:
        feats = sequence_features(sample)
        out = {f: abs(feats[f] - self.mean[f]) / self.std[f] for f in self.features}
        out["aggregate"] = float(max(out.values())) if out else 0.0
        return out


# ---------------------------------------------------------------------------
# The expensive baseline the cheap proxy approximates: recompute the true M.
# ---------------------------------------------------------------------------
def recompute_divergence(sample: ProxySample) -> float:
    """The full-recompute reference line (the DiFR ``token_difr`` analogue): the
    max total-variation distance between the served target ``p`` and the **true**
    target ``p*`` recomputed from ``M``, over the served positions. Honest ->
    ~0; any forward-pass corruption (quant/fp8) -> >0. This is what the cheap
    proxy accept-rate approximates and what it is measured against; a deployed
    verifier runs it only on a sampled subset. In the CPU harness the truth is
    stored on the sample (``_truth``); a real client recomputes it with ``M``."""
    if not sample.positions:
        return 0.0
    worst = 0.0
    for pos, p_true_lp in zip(sample.positions, sample._truth):
        worst = max(worst, tv(np.exp(pos.target_logprobs), np.exp(p_true_lp)))
    return float(worst)


# ---------------------------------------------------------------------------
# The client-side verifier: calibrate a threshold on honest samples, then flag.
# ---------------------------------------------------------------------------
@dataclass
class ProxyVerdict:
    provider_name: str
    feature: str
    score: float
    threshold: float
    flagged: bool

    def summary(self) -> str:
        head = (f"provider={self.provider_name!r}  ->  "
                + ("FLAGGED (served distribution diverges from honest "
                   f"{self.feature})" if self.flagged else "consistent"))
        return (head + f"\n    {'FLAG' if self.flagged else ' ok '}  "
                f"{self.feature:<14} z={self.score:.4g}  thr={self.threshold:.4g}")


class ProxySpecVerifier:
    """Client-side acceptance-rate verifier. Runs only the cheap proxy: it scores
    a served sample's ``feature`` (default ``accept_rate``) against the offline
    ``ProxyReference`` and flags when the standardized distance exceeds a
    threshold calibrated to a target honest false-positive rate.

    ``feature="accept_rate"`` is the draft-anchored signal that survives the
    temperature-retune evasion; ``"mean_entropy"`` reproduces the generic
    fingerprint that does not.
    """

    def __init__(self, feature: str = "accept_rate"):
        self.feature = feature
        self.reference: ProxyReference | None = None
        self.threshold: float = np.inf

    def calibrate(self, honest_samples: list[ProxySample], fpr: float = 0.01
                  ) -> "ProxySpecVerifier":
        """Fit the reference on the honest set and set the flag threshold to its
        ``(1 − fpr)`` standardized-distance quantile."""
        self.reference = ProxyReference(features=(self.feature,)).fit(honest_samples)
        zs = np.array([self.reference.score(s)[self.feature] for s in honest_samples])
        zs = zs[np.isfinite(zs)]
        self.threshold = float(np.quantile(zs, 1.0 - fpr)) if len(zs) else np.inf
        return self

    def verify(self, sample: ProxySample) -> ProxyVerdict:
        if self.reference is None:
            raise RuntimeError("ProxySpecVerifier must be .calibrate(...) before verify")
        z = self.reference.score(sample)[self.feature]
        return ProxyVerdict(sample.provider_name, self.feature, z, self.threshold,
                            z > self.threshold)
