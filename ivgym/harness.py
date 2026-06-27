"""The verification harness: generate -> verify -> calibrate -> evaluate."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .attacks import Attack
from .core import IOContext, SamplingSpec, Sequence, VerifyContext
from .defenses import Defense
from .io_detectors import IODetector
from .metrics import roc_auc, tpr_at_fpr
from .sampling import gumbel_noise, position_seed


@dataclass
class TokenScores:
    """Per-token scores for one config, keyed by defense name."""

    config_name: str
    scores: dict[str, np.ndarray] = field(default_factory=dict)


def generate_dataset(backend, attack: Attack, spec: SamplingSpec, n_prompts: int,
                     n_tokens: int, record_activations: bool = False,
                     proj_seed: int = 123, proj_dim: int = 32) -> list[Sequence]:
    return [
        backend.generate(p, n_tokens, spec, attack, record_activations, proj_seed, proj_dim)
        for p in range(n_prompts)
    ]


def verify(backend, sequences: list[Sequence], spec: SamplingSpec,
           defenses: list[Defense], proj_seed: int = 123, proj_dim: int = 32) -> TokenScores:
    """Run the verifier over provider sequences, scoring each token with each defense."""
    from .backends.synthetic import _projection

    needs_act = any(d.needs_activation for d in defenses)
    proj = _projection(proj_seed, proj_dim, backend.hidden_dim) if needs_act else None
    out = {d.name: [] for d in defenses}
    cfg = sequences[0].config_name if sequences else "?"

    for seq in sequences:
        for step in seq.steps:
            ref_logits = backend.reference_logits(seq.prompt_id, step.position)
            gseed = position_seed(spec.seed, seq.prompt_id, step.position)
            g = gumbel_noise(backend.vocab, gseed)
            ref_fp = None
            if needs_act:
                ref_fp = proj @ backend.reference_activation(seq.prompt_id, step.position)
            ctx = VerifyContext(
                claimed_token=step.claimed_token,
                ref_logits=ref_logits,
                gumbel=g,
                sampling=spec,
                fingerprint=step.fingerprint,
                ref_fingerprint=ref_fp,
            )
            for d in defenses:
                out[d.name].append(d.score(ctx))

    return TokenScores(cfg, {k: np.asarray(v, float) for k, v in out.items()})


def io_context(backend, seq: Sequence, spec: SamplingSpec,
               need_proxy: bool, need_text: bool) -> IOContext:
    """Build the black-box `IOContext` for one provider sequence.

    Crucially this NEVER calls `backend.reference_logits` / `reference_activation`
    -- that would be recomputing M, which is exactly what an I/O detector must
    not do. It may call `backend.proxy_logits` (a *different, cheap* model) and
    `backend.prompt_text` / `backend.decode` (raw I/O, no forward pass)."""
    toks = [s.claimed_token for s in seq.steps]
    proxy = None
    if need_proxy:
        proxy = np.stack([backend.proxy_logits(seq.prompt_id, s.position) for s in seq.steps])
    text = None
    if need_text:
        prompt = backend.prompt_text(seq.prompt_id) if hasattr(backend, "prompt_text") else None
        if prompt is not None:
            # Pack prompt + decoded continuation as "prompt\x00continuation" so a
            # text detector can recover both without widening the dataclass.
            cont = backend.decode(toks) if hasattr(backend, "decode") else ""
            text = f"{prompt}\x00{cont or ''}"
    return IOContext(prompt_id=seq.prompt_id, claimed_tokens=toks, sampling=spec,
                     prompt_text=text, proxy_logits=proxy)


def io_verify(backend, sequences: list[Sequence], spec: SamplingSpec,
              io_detectors: list[IODetector]) -> TokenScores:
    """Black-box analogue of `verify`: score each sequence with each I/O detector
    *without* recomputing M. Each detector returns a per-token score array; they
    are concatenated so the result flows through the SAME winsorize / batch_means
    / evaluate pipeline as `Defense` scores (an `IODetector` is interchangeable
    with a `Defense` from `evaluate`'s point of view -- both expose `.name`)."""
    need_proxy = any(d.needs_proxy for d in io_detectors)
    need_text = any(d.needs_text for d in io_detectors)
    out = {d.name: [] for d in io_detectors}
    cfg = sequences[0].config_name if sequences else "?"

    for seq in sequences:
        ctx = io_context(backend, seq, spec, need_proxy, need_text)
        for d in io_detectors:
            out[d.name].append(np.asarray(d.score_sequence(ctx), float))

    return TokenScores(cfg, {k: (np.concatenate(v) if v else np.array([]))
                             for k, v in out.items()})


def winsorize(scores: np.ndarray, honest_train: np.ndarray, pct: float) -> np.ndarray:
    """Clip scores at a percentile of the honest training split (DiFR feature eng.).
    Infinities/large values are excluded when computing the percentile."""
    finite = honest_train[np.isfinite(honest_train)]
    cap = np.percentile(finite, pct)
    return np.minimum(scores, cap)


def batch_means(scores: np.ndarray, batch_size: int, n_batches: int,
                rng: np.random.Generator) -> np.ndarray:
    """Sample `n_batches` batches of `batch_size` tokens (with replacement across
    batches) and return their mean scores -- the batch-level statistic S."""
    n = len(scores)
    if batch_size > n:
        batch_size = n
    means = np.empty(n_batches)
    for i in range(n_batches):
        idx = rng.choice(n, size=batch_size, replace=False)
        means[i] = scores[idx].mean()
    return means


@dataclass
class EvalResult:
    defense: str
    attack: str
    batch_size: int
    auc: float
    tpr_at_1pct: float


def evaluate(honest: TokenScores, attack: TokenScores, defenses: list[Defense],
             batch_sizes: list[int], n_batches: int = 400, winsor_pct: float | None = 99.9,
             seed: int = 0) -> list[EvalResult]:
    """Compare honest vs attack batch statistics across batch sizes."""
    rng = np.random.default_rng(seed)
    results: list[EvalResult] = []
    for d in defenses:
        h = honest.scores[d.name]
        a = attack.scores[d.name]
        # train/test split at token level
        h_tr, h_te = _split(h, rng)
        a_tr, a_te = _split(a, rng)
        if winsor_pct is not None:
            h_te = winsorize(h_te, h_tr, winsor_pct)
            a_te = winsorize(a_te, h_tr, winsor_pct)
        for b in batch_sizes:
            hb = batch_means(h_te, b, n_batches, rng)
            ab = batch_means(a_te, b, n_batches, rng)
            results.append(EvalResult(
                defense=d.name, attack=attack.config_name, batch_size=b,
                auc=roc_auc(hb, ab), tpr_at_1pct=tpr_at_fpr(hb, ab, 0.01),
            ))
    return results


def _split(x: np.ndarray, rng: np.random.Generator, frac: float = 0.5):
    idx = rng.permutation(len(x))
    cut = int(len(x) * frac)
    return x[idx[:cut]], x[idx[cut:]]
