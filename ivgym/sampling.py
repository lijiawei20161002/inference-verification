"""Seed-synchronized Gumbel-Max sampling (Algorithm 1 in DiFR).

The whole point of Token-DiFR is that provider and verifier draw the *same*
Gumbel noise from a shared per-position seed, so any disagreement in the
sampled token comes purely from differences in the logits.
"""
from __future__ import annotations

import numpy as np

NEG_INF = -1e30


def stable_hash(s: str) -> int:
    """Deterministic 32-bit string hash (Python's built-in hash() is randomized
    per process via PYTHONHASHSEED, which would make runs non-reproducible)."""
    h = 2166136261
    for ch in s.encode():
        h = ((h ^ ch) * 16777619) & 0xFFFF_FFFF
    return h


def position_seed(master_seed: int, prompt_id: int, position: int) -> int:
    """Deterministic per-position seed shared by provider and verifier."""
    # Stable mix; avoids Python hash randomization.
    h = (master_seed * 1_000_003 + prompt_id) * 1_000_003 + position
    return h & 0x7FFF_FFFF


def gumbel_noise(vocab: int, seed: int) -> np.ndarray:
    """Standard Gumbel(0,1) noise vector, reproducible from `seed`.

    Matches vLLM's use of an exponential draw: g = -log(e), e ~ Exp(1) is a
    Gumbel sample up to sign conventions; here we use the canonical
    g = -log(-log(U)).
    """
    rng = np.random.default_rng(seed)
    u = rng.random(vocab)
    return -np.log(-np.log(u))


def apply_top_k(logits: np.ndarray, k: int | None) -> np.ndarray:
    if not k or k >= logits.size:
        return logits
    out = logits.copy()
    kth = np.partition(logits, -k)[-k]
    out[logits < kth] = NEG_INF
    return out


def apply_top_p(logits: np.ndarray, p: float | None) -> np.ndarray:
    if p is None or p >= 1.0:
        return logits
    order = np.argsort(logits)[::-1]
    probs = _softmax(logits[order])
    cum = np.cumsum(probs)
    # Keep tokens up to and including the one that crosses p.
    keep_sorted = cum <= p
    keep_sorted[0] = True  # always keep the top token
    keep = np.zeros_like(logits, dtype=bool)
    keep[order[keep_sorted]] = True
    out = logits.copy()
    out[~keep] = NEG_INF
    return out


def filtered_logits(logits: np.ndarray, top_k: int | None, top_p: float | None) -> np.ndarray:
    return apply_top_p(apply_top_k(logits, top_k), top_p)


def gumbel_max_sample(
    logits: np.ndarray,
    temperature: float,
    gumbel: np.ndarray,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    """Return the sampled token id under Gumbel-Max with the given noise.

    Temperature is applied to the noise (not the logits), matching the DiFR
    parameterization that makes scores comparable across temperatures.
    """
    filt = filtered_logits(logits, top_k, top_p)
    z = filt + temperature * gumbel
    return int(np.argmax(z))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    return x - np.log(np.exp(x).sum())
