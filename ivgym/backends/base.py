"""Backend protocol.

A backend knows how to (a) produce the *true* logits / activations for a
(prompt, position) under the reference model, and (b) generate provider
sequences given an Attack. Swap the synthetic backend for the vLLM adapter
without touching attacks, defenses, or the harness.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from ..core import SamplingSpec, Sequence


class Backend(Protocol):
    vocab: int
    hidden_dim: int

    def reference_logits(self, prompt_id: int, position: int) -> np.ndarray:
        """Trusted logits under the reference model (verifier side)."""
        ...

    def reference_activation(self, prompt_id: int, position: int) -> np.ndarray:
        """Trusted final-layer hidden state under the reference model."""
        ...

    def generate(
        self,
        prompt_id: int,
        n_tokens: int,
        spec: SamplingSpec,
        attack: "object",          # ivgym.attacks.Attack
        record_activations: bool,
        proj_seed: int,
        proj_dim: int,
    ) -> Sequence:
        """Run provider-side generation under `attack`, returning claimed tokens."""
        ...
