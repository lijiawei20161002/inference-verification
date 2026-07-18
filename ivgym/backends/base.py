"""Backend protocol.

A backend knows how to (a) produce the *true* logits / activations for a
(prompt, position) under the reference model, and (b) generate provider
sequences given an Attack. Swap the HF-GPU backend for the vLLM adapter
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

    # --- optional: support for Tier-0 (black-box) verifiers -----------------
    # These let a Tier-0 verifier (ivgym.verifiers) score from outputs alone,
    # WITHOUT recomputing M. `proxy_logits` returns a *cheap, different* model's
    # logits (never M's forward pass); `served_logits` is the provider's served
    # distribution p (for the accept-rate fingerprint); `prompt_text` / `decode`
    # expose raw I/O. A backend that omits them cannot run those verifiers.

    def proxy_logits(self, prompt_id: int, position: int) -> np.ndarray:
        """Logits from a cheap proxy LM (the cost/accuracy Pareto's cheap end)."""
        ...

    def served_logits(self, prompt_id: int, position: int) -> np.ndarray:
        """The distribution `p` the provider served under (for the accept-rate
        fingerprint). Not a recompute of M -- it is what a provider returning
        logprobs exposes. Backends that omit it cannot run `accept_rate`."""
        ...

    def prompt_text(self, prompt_id: int) -> str | None:
        """Raw prompt text (text backends only; None when unavailable)."""
        ...

    def decode(self, token_ids: list[int]) -> str | None:
        """Decode claimed token ids back to text (text backends only)."""
        ...
