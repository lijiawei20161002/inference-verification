"""ivgym: the Inference Verification Gym.

A standardized environment for the inference-verification game: a cheating
*provider* (an Attack) deviates from a sampling specification, a *verifier* (a
Defense) scores per-token divergence, and the gym reports how reliably the
deviation is caught (detection AUC). Attacks and defenses are pluggable
registries; the generate -> verify -> calibrate -> evaluate loop and the
backend (a real model on a GPU via HuggingFace transformers) are the fixed
infrastructure.

Quick start (needs a CUDA host with torch + transformers):
    from ivgym.backends.hf_gpu import HFGPUBackend
    from ivgym.core import SamplingSpec
    from ivgym import attacks, defenses, harness

    backend = HFGPUBackend(model_name="Qwen/Qwen3-0.6B")
    spec = SamplingSpec()
    honest = harness.verify(backend,
        harness.generate_dataset(backend, attacks.get("honest"), spec, 20, 128,
                                 record_activations=True),
        spec, list(defenses.all_defenses().values()))
"""
from . import attacks, defenses, harness, io_detectors, metrics, sampling  # noqa: F401
from .core import IOContext, SamplingSpec, Sequence, TokenStep, VerifyContext  # noqa: F401

__all__ = [
    "attacks", "defenses", "harness", "io_detectors", "metrics", "sampling",
    "SamplingSpec", "Sequence", "TokenStep", "VerifyContext", "IOContext",
]
