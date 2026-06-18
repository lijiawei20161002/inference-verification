"""DiFR: a shared harness for inference-verification attacks and defenses.

Quick start:
    from difr.backends.synthetic import SyntheticBackend
    from difr.core import SamplingSpec
    from difr import attacks, defenses, harness

    backend = SyntheticBackend()
    spec = SamplingSpec()
    honest = harness.verify(backend,
        harness.generate_dataset(backend, attacks.get("honest"), spec, 50, 256),
        spec, list(defenses.all_defenses().values()))
"""
from . import attacks, defenses, harness, metrics, sampling  # noqa: F401
from .core import SamplingSpec, Sequence, TokenStep, VerifyContext  # noqa: F401

__all__ = [
    "attacks", "defenses", "harness", "metrics", "sampling",
    "SamplingSpec", "Sequence", "TokenStep", "VerifyContext",
]
