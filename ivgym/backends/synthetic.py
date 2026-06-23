"""Synthetic backend: a deterministic fake LLM, runnable with no GPU.

`reference_logits(prompt, pos)` returns the same peaked logit vector every
time (the "true" model). The provider adds benign + attack noise on top; the
verifier recomputes reference logits with its own independent benign noise.
Same interface as the future vLLM adapter, so attacks/defenses/harness are
identical across backends.
"""
from __future__ import annotations

import numpy as np

from ..attacks import Attack
from ..core import SamplingSpec, Sequence, TokenStep
from ..sampling import (
    filtered_logits,
    gumbel_max_sample,
    gumbel_noise,
    position_seed,
    stable_hash,
)


class SyntheticBackend:
    def __init__(self, vocab: int = 512, hidden_dim: int = 256, peak: float = 2.5,
                 model_seed: int = 0, verifier_sigma: float = 0.02,
                 act_benign_sigma: float = 0.05):
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.peak = peak
        self.model_seed = model_seed
        # The verifier is itself a "correct-but-noisy" deployment.
        self.verifier_sigma = verifier_sigma
        self.act_benign_sigma = act_benign_sigma

    # --- trusted reference computations (verifier side) ---
    def _true_logits(self, prompt_id: int, position: int) -> np.ndarray:
        rng = np.random.default_rng((self.model_seed, prompt_id, position))
        return rng.standard_normal(self.vocab) * self.peak

    def reference_logits(self, prompt_id: int, position: int) -> np.ndarray:
        # Verifier recomputes with its own independent benign noise.
        true = self._true_logits(prompt_id, position)
        nrng = np.random.default_rng((self.model_seed, prompt_id, position, 7))
        return true + nrng.normal(0.0, self.verifier_sigma, self.vocab)

    def _true_activation(self, prompt_id: int, position: int) -> np.ndarray:
        rng = np.random.default_rng((self.model_seed, prompt_id, position, 99))
        return rng.standard_normal(self.hidden_dim)

    def reference_activation(self, prompt_id: int, position: int) -> np.ndarray:
        # Verifier recomputes with its own independent benign noise.
        true = self._true_activation(prompt_id, position)
        nrng = np.random.default_rng((self.model_seed, prompt_id, position, 99, 7))
        return true + nrng.normal(0.0, self.act_benign_sigma, self.hidden_dim)

    # --- provider-side generation ---
    def generate(self, prompt_id: int, n_tokens: int, spec: SamplingSpec,
                 attack: Attack, record_activations: bool = False,
                 proj_seed: int = 123, proj_dim: int = 32) -> Sequence:
        pspec = attack.provider_spec(spec)
        proj = _projection(proj_seed, proj_dim, self.hidden_dim) if record_activations else None
        seq = Sequence(prompt_id=prompt_id, config_name=attack.name)

        for pos in range(n_tokens):
            true = self._true_logits(prompt_id, pos)
            prng = np.random.default_rng((self.model_seed, prompt_id, pos, 11, stable_hash(attack.name)))
            logits = attack.perturb_logits(true, prng)

            gseed = position_seed(pspec.seed, prompt_id, pos)
            g = gumbel_noise(self.vocab, gseed)

            override = None
            filt = filtered_logits(logits, pspec.top_k, pspec.top_p)
            top_ids = np.argsort(filt)[::-1]
            override = attack.sample_override(prng, top_ids)
            if override is not None:
                token = override
            else:
                token = gumbel_max_sample(logits, pspec.temperature, g,
                                          pspec.top_k, pspec.top_p)
                # Seed-aware attacks may swap the claimed token for one inside the
                # verifier's indistinguishable set (default hook is a no-op).
                alt = attack.resample(filt, g, pspec, token, prng)
                if alt is not None:
                    token = int(alt)

            fp = None
            if record_activations:
                # provider's forward pass: true activation + its own benign noise
                # + any attack-induced perturbation.
                act = self._true_activation(prompt_id, pos)
                act = act + prng.normal(0.0, self.act_benign_sigma, self.hidden_dim)
                extra = attack.activation_extra_sigma()
                if extra:
                    act = act + prng.normal(0.0, extra, self.hidden_dim)
                fp = proj @ act

            # verifier should re-verify under the *reference* spec, not pspec
            seq.steps.append(TokenStep(position=pos, claimed_token=token,
                                       sampling=spec, fingerprint=fp))
        return seq


def _projection(seed: int, k: int, d: int) -> np.ndarray:
    """Random orthonormal-rows projection matrix P in R^{k x d} (JL / Activation-DiFR)."""
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((k, d))
    q, _ = np.linalg.qr(m.T)        # d x k, orthonormal columns
    return q[:, :k].T               # k x d
