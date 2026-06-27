"""ivgym: the Inference Verification Gym.

A standardized environment for the inference-verification game: a cheating
*provider* (an Attack) deviates from a sampling specification, a *verifier* (a
Defense) scores per-token divergence, and the gym reports how reliably the
deviation is caught (detection AUC). Attacks and defenses are pluggable
registries; the generate -> verify -> calibrate -> evaluate loop and the
backend (synthetic CPU or a real model on a GPU) are the fixed infrastructure.

Quick start:
    from ivgym.backends.synthetic import SyntheticBackend
    from ivgym.core import SamplingSpec
    from ivgym import attacks, defenses, harness

    backend = SyntheticBackend()
    spec = SamplingSpec()
    honest = harness.verify(backend,
        harness.generate_dataset(backend, attacks.get("honest"), spec, 50, 256),
        spec, list(defenses.all_defenses().values()))
"""
from . import attacks, defenses, harness, io_detectors, metrics, sampling  # noqa: F401
from .core import IOContext, SamplingSpec, Sequence, TokenStep, VerifyContext  # noqa: F401

__all__ = [
    "attacks", "defenses", "harness", "io_detectors", "metrics", "sampling",
    "SamplingSpec", "Sequence", "TokenStep", "VerifyContext", "IOContext",
]
