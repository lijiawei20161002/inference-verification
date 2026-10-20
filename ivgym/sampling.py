"""Seed-synchronized Gumbel-Max sampling (Algorithm 1 in DiFR).

Token-DiFR assumes provider and verifier use the same Gumbel tensor and sampling
arithmetic. A shared seed alone does not guarantee this across RNGs, dtypes, or
filter conventions. The functions below preserve the legacy NumPy protocol;
changing its conventions requires a new experiment version and calibration.
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

    An exponential draw g = -log(e), e ~ Exp(1), has the same distribution.
    This NumPy RNG does not reproduce vLLM's draws or counter layout. We use
    g = -log(-log(U)).
    """
    rng = np.random.default_rng(seed)
    u = rng.random(vocab)
    return -np.log(-np.log(u))


def apply_top_k(logits: np.ndarray, k: int | None) -> np.ndarray:
    """Retain all logits at or above the kth value, including boundary ties."""
    if not k or k >= logits.size:
        return logits
    out = logits.copy()
    kth = np.partition(logits, -k)[-k]
    out[logits < kth] = NEG_INF
    return out


def apply_top_p(logits: np.ndarray, p: float | None) -> np.ndarray:
    """Apply the legacy nucleus boundary while preserving the input dtype.

    Retain cumulative mass <= p and always the first token. The token crossing p
    is otherwise excluded. Keep this convention for archived-run reproducibility.
    """
    if p is None or p >= 1.0:
        return logits
    # Sort only candidates retained by top-k, including any boundary ties.
    # Other entries are the NEG_INF sentinel and cannot be retained. With no
    # top-k restriction, this is the original full-vocabulary sort.
    cand = np.nonzero(logits > NEG_INF / 2)[0]
    if cand.size == 0:
        return logits
    sub = logits[cand]
    order = np.argsort(sub)[::-1]
    probs = _softmax(sub[order])
    cum = np.cumsum(probs)
    # Legacy convention: exclude the crossing token, except always keep the first.
    keep_sorted = cum <= p
    keep_sorted[0] = True  # always keep the top token
    out = np.full_like(logits, NEG_INF)
    kept = cand[order[keep_sorted]]
    out[kept] = logits[kept]
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


def projection(seed: int, k: int, d: int) -> np.ndarray:
    """Random orthonormal-rows projection matrix P in R^{k x d} (JL / Activation-DiFR).
    Shared by every backend so provider and verifier project activations with the
    same matrix (the Activation-DiFR fingerprint contract)."""
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((k, d))
    q, _ = np.linalg.qr(m.T)        # d x k, orthonormal columns
    return q[:, :k].T               # k x d
