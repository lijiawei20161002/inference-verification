"""Attacks = provider deviations from specification phi.

Every attack is a transform applied during provider-side generation. The
verifier never sees the attack; it always recomputes under the reference spec.
This is the extension point for "different versions of attack": subclass
`Attack`, register it, and the whole harness picks it up.

Modeling note (synthetic backend): real misconfigurations (quantization,
fp8 KV cache) perturb the forward pass and therefore the *logits*. We model
that as extra zero-mean logit noise of a configurable scale plus an optional
systematic bias, which is enough to reproduce the qualitative DiFR results.
On the vLLM backend these same attacks map to real config flags
(`quantization=...`, `kv_cache_dtype=...`, temperature, seed) instead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import SamplingSpec

_REGISTRY: dict[str, "Attack"] = {}


def register(attack: "Attack") -> "Attack":
    _REGISTRY[attack.name] = attack
    return attack


def get(name: str) -> "Attack":
    return _REGISTRY[name]


def all_attacks() -> dict[str, "Attack"]:
    return dict(_REGISTRY)


@dataclass
class Attack:
    """Base attack = honest behaviour (only benign noise, no deviation)."""

    name: str = "honest"
    benign_sigma: float = 0.02     # benign numerical noise on logits (always present)

    # --- hooks an attack may override ---
    def provider_spec(self, ref: SamplingSpec) -> SamplingSpec:
        """Sampling spec the provider actually uses (verifier still uses `ref`)."""
        return ref

    def logit_bias_sigma(self) -> tuple[float, float]:
        """(extra_noise_sigma, systematic_bias_sigma) added to provider logits."""
        return 0.0, 0.0

    def activation_extra_sigma(self) -> float:
        """Extra noise on the provider's forward-pass activation."""
        return 0.0

    def sample_override(self, rng: np.random.Generator, top_k_ids: np.ndarray) -> int | None:
        """Return a token id to force (sampling bug), or None for normal sampling."""
        return None

    def perturb_logits(self, base: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        extra, bias = self.logit_bias_sigma()
        out = base + rng.normal(0.0, self.benign_sigma, base.shape)
        if extra:
            out = out + rng.normal(0.0, extra, base.shape)
        if bias:
            # systematic, position-stable component (drawn once per call here;
            # quantization error is structured, not purely random)
            out = out + rng.normal(0.0, bias, base.shape)
        return out


# ----------------------------------------------------------------------------
# Concrete attacks
# ----------------------------------------------------------------------------
@dataclass
class Quantization(Attack):
    """4-bit / fp8-style quantization: larger logit perturbation."""

    name: str = "quant_4bit"
    extra_sigma: float = 0.18
    bias_sigma: float = 0.06
    act_sigma: float = 0.30

    def logit_bias_sigma(self):
        return self.extra_sigma, self.bias_sigma

    def activation_extra_sigma(self):
        return self.act_sigma


@dataclass
class KVCacheFP8(Quantization):
    name: str = "kv_fp8"
    extra_sigma: float = 0.07
    bias_sigma: float = 0.02
    act_sigma: float = 0.12


@dataclass
class WrongTemperature(Attack):
    name: str = "temp_1.1"
    provider_temp: float = 1.1

    def provider_spec(self, ref: SamplingSpec) -> SamplingSpec:
        return ref.replace(temperature=self.provider_temp)


@dataclass
class WrongSeed(Attack):
    name: str = "seed_43"
    provider_seed: int = 43

    def provider_spec(self, ref: SamplingSpec) -> SamplingSpec:
        return ref.replace(seed=self.provider_seed)


@dataclass
class SamplingBug(Attack):
    """With prob `rate`, ignore logits and pick uniformly from top-k tokens."""

    name: str = "bug_k2"
    rate: float = 0.01
    k: int = 2

    def sample_override(self, rng, top_k_ids):
        if rng.random() < self.rate:
            return int(rng.choice(top_k_ids[: self.k]))
        return None


@dataclass
class AdversarialTemp(Quantization):
    """Quantize, then *lower* the sampling temperature to restore the reference
    mean cross-entropy. Defeats the cross-entropy detector while leaving the
    seed-synchronized Token-DiFR signal intact (paper Fig. 2)."""

    name: str = "adv_quant_temp"
    provider_temp: float = 0.9  # tuned at runtime by experiments/calibrate_adv_temp

    def provider_spec(self, ref: SamplingSpec) -> SamplingSpec:
        return ref.replace(temperature=self.provider_temp)


# Register defaults.
for a in [
    Attack(),
    Quantization(),
    KVCacheFP8(),
    WrongTemperature(),
    WrongSeed(),
    SamplingBug(name="bug_k2", k=2),
    SamplingBug(name="bug_k32", k=32),
    AdversarialTemp(),
]:
    register(a)
