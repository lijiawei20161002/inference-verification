"""ivgym: the Inference Verification Gym.

A standardized environment for the inference-verification game: a cheating
*provider* (an Attack) deviates from a sampling specification, a *verifier*
scores per-token divergence, and the gym reports how reliably the deviation is
caught (detection AUC). Attacks and verifiers are pluggable registries; the
generate -> verify -> calibrate -> evaluate loop and the backend (a real model
on a GPU via HuggingFace transformers) are the fixed infrastructure.

Every detector is one `Verifier` (see `ivgym.verifiers`): a per-token evidence
signal plus a cheap `value` signal that directs where the expensive recompute of
the reference model M is spent. Tier-0 verifiers read only a cheap proxy / the
served outputs; Tier-1 verifiers recompute M. `harness.verify(..., budget=B)` is
the single driver -- `budget=1.0` is a full recompute, `budget<1.0` is
information-directed selective recompute.

Quick start (needs a CUDA host with torch + transformers):
    from ivgym.backends.hf_gpu import HFGPUBackend
    from ivgym.core import SamplingSpec
    from ivgym import attacks, verifiers, harness

    backend = HFGPUBackend(model_name="Qwen/Qwen3-0.6B")
    spec = SamplingSpec()
    honest = harness.verify(backend,
        harness.generate_dataset(backend, attacks.get("honest"), spec, 20, 128,
                                 record_activations=True),
        spec, list(verifiers.all_verifiers().values()))
"""
from . import attacks, harness, metrics, sampling, verifiers  # noqa: F401
from . import spec_decode  # noqa: F401
from .core import SamplingSpec, Sequence, TokenStep, VContext  # noqa: F401

__all__ = [
    "attacks", "harness", "metrics", "sampling", "verifiers", "spec_decode",
    "SamplingSpec", "Sequence", "TokenStep", "VContext",
]
