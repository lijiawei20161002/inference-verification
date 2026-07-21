"""The verification harness: generate -> verify -> calibrate -> evaluate.

`verify` is the SINGLE driver for every verifier -- it replaces the old
`verify` (full white-box recompute) / `io_verify` (black-box, no recompute) /
`verify_selective` (proxy-triaged partial recompute) trio. Which of those three
regimes you get is now just a choice of the `budget` argument and the verifiers
you pass:

  * Tier-0 verifiers (surface / accept-rate / learned) never recompute M, so
    `budget` does not affect them -- that is the old `io_verify`.
  * Tier-1 verifiers (token_difr / cross_entropy / toploc / activation) recompute
    M. At `budget=1.0` every token is audited (old `verify`); at `budget<1.0` the
    driver spends recompute only on the top-`budget` fraction of tokens by a
    cheap, proxy-only `value` signal (old `verify_selective`). The default value
    signal is proxy entropy H(q_t) -- recompute is directed to the positions that
    actually carry evidence about whether M was run.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import verifiers as _verifiers
from .attacks import Attack
from .core import SamplingSpec, Sequence, VContext
from .metrics import partial_auc, roc_auc, tpr_at_fpr
from .sampling import gumbel_noise, position_seed, projection
from .verifiers import Verifier


@dataclass
class TokenScores:
    """Per-token scores for one config, keyed by verifier name. `recompute_ratio`
    is the realized fraction of tokens on which a Tier-1 verifier recomputed M
    (1.0 for a full audit, 0.0 for a purely Tier-0 run)."""

    config_name: str
    scores: dict[str, np.ndarray] = field(default_factory=dict)
    recompute_ratio: float = 1.0


def generate_dataset(backend, attack: Attack, spec: SamplingSpec, n_prompts: int,
                     n_tokens: int, record_activations: bool = False,
                     proj_seed: int = 123, proj_dim: int = 32) -> list[Sequence]:
    return [
        backend.generate(p, n_tokens, spec, attack, record_activations, proj_seed, proj_dim)
        for p in range(n_prompts)
    ]


# ---------------------------------------------------------------------------
# Cheap, proxy-only per-token VALUE: where verification value (and thus expensive
# recompute) should be directed. Reads ONLY backend.proxy_logits (a small, cheap,
# DIFFERENT model), never backend.reference_logits -- so deciding where to spend
# recompute never itself recomputes M. Flat, in the same (seq, step) order the
# driver scores tokens. Generalizes the old `proxy_tie_scores` (== value_fn
# "tie_margin") to any value signal; default is proxy entropy H(q).
# ---------------------------------------------------------------------------
def token_values(backend, sequences: list[Sequence], spec: SamplingSpec,
                 value_fn: str = "entropy") -> np.ndarray:
    if value_fn == "uniform":
        total = sum(len(s.steps) for s in sequences)
        return np.ones(total)
    out = []
    for seq in sequences:
        if not seq.steps:
            continue
        proxy = np.stack([backend.proxy_logits(seq.prompt_id, st.position) for st in seq.steps])
        ctx = VContext(seq.prompt_id, [st.claimed_token for st in seq.steps], spec,
                       proxy_logits=proxy)
        out.append(_verifiers.value_of(value_fn, ctx))
    return np.concatenate(out) if out else np.array([])


def select_triaged(value: np.ndarray, budget: float) -> np.ndarray:
    """Boolean mask of the top-`budget` fraction of tokens by `value` (highest
    audited first). Always audits at least one token."""
    n = len(value)
    if n == 0:
        return np.zeros(0, bool)
    k = int(np.clip(round(budget * n), 1, n))
    mask = np.zeros(n, bool)
    mask[np.argsort(-value, kind="mergesort")[:k]] = True
    return mask


def _seq_text(backend, seq: Sequence) -> str | None:
    """Pack 'prompt\\x00continuation' for a text verifier, or None if unavailable."""
    if not hasattr(backend, "prompt_text"):
        return None
    prompt = backend.prompt_text(seq.prompt_id)
    if prompt is None:
        return None
    toks = [st.claimed_token for st in seq.steps]
    cont = backend.decode(toks) if hasattr(backend, "decode") else ""
    return f"{prompt}\x00{cont or ''}"


def verify(backend, sequences: list[Sequence], spec: SamplingSpec,
           verifiers: list[Verifier], *, budget: float = 1.0, value_fn: str = "entropy",
           values: np.ndarray | None = None, proj_seed: int = 123, proj_dim: int = 32
           ) -> TokenScores:
    """Score every token of every sequence with every verifier.

    `budget` in (0, 1] controls the Tier-1 recompute fraction (ignored by Tier-0
    verifiers). At `budget<1` the driver ranks tokens by the cheap `value_fn`
    signal (proxy entropy by default) and recomputes M only on the top fraction;
    unaudited tokens take each Tier-1 verifier's `neutral` score. Pass a
    precomputed `values` array (from `token_values`) to reuse a triage ranking
    across budgets. Returns a `TokenScores` whose per-verifier arrays are the flat
    per-token scores (concatenated across sequences) plus the realized
    `recompute_ratio`."""
    tier1 = [v for v in verifiers if v.tier == 1]
    need_proxy = any(v.needs_proxy for v in verifiers)
    need_served = any(v.needs_served for v in verifiers)
    need_text = any(v.needs_text for v in verifiers)
    need_act = any(v.needs_activation for v in verifiers)
    proj = projection(proj_seed, proj_dim, backend.hidden_dim) if need_act else None

    # Selective recompute: build the global audit mask from a cheap value signal.
    selective = bool(tier1) and budget < 1.0
    if selective:
        # `token_values` reads the proxy itself; the per-verifier `need_proxy`
        # above governs only whether the *scoring* context needs proxy logits.
        if values is None:
            values = token_values(backend, sequences, spec, value_fn)
        mask_flat = select_triaged(values, budget)
    else:
        mask_flat = None

    out = {v.name: [] for v in verifiers}
    cfg = sequences[0].config_name if sequences else "?"
    audited = total = 0
    i0 = 0
    for seq in sequences:
        steps = seq.steps
        n = len(steps)
        toks = [st.claimed_token for st in steps]
        # audit mask for this sequence's tokens
        if tier1:
            audit = (mask_flat[i0:i0 + n] if selective else np.ones(n, bool))
        else:
            audit = np.zeros(n, bool)
        i0 += n
        audited += int(audit.sum())
        total += n

        # --- Tier-0 fields (cheap) ---
        proxy = np.stack([backend.proxy_logits(seq.prompt_id, st.position)
                          for st in steps]) if (need_proxy and n) else None
        served = np.stack([backend.served_logits(seq.prompt_id, st.position)
                           for st in steps]) if (need_served and n) else None
        text = _seq_text(backend, seq) if need_text else None
        fps = [st.fingerprint for st in steps] if need_act else None

        # --- Tier-1 fields (expensive), audited rows only ---
        ref = gum = ref_fps = None
        if tier1 and n:
            ref = np.zeros((n, backend.vocab))
            gum = np.zeros((n, backend.vocab))
            ref_fps = [None] * n
            for j, st in enumerate(steps):
                if not audit[j]:
                    continue
                ref[j] = backend.reference_logits(seq.prompt_id, st.position)
                gum[j] = gumbel_noise(backend.vocab,
                                      position_seed(spec.seed, seq.prompt_id, st.position))
                if need_act:
                    ref_fps[j] = proj @ backend.reference_activation(seq.prompt_id, st.position)

        ctx = VContext(prompt_id=seq.prompt_id, claimed_tokens=toks, sampling=spec,
                       proxy_logits=proxy, served_logits=served, prompt_text=text,
                       fingerprints=fps, ref_logits=ref, ref_fingerprints=ref_fps,
                       gumbel=gum, audit_mask=audit)
        for v in verifiers:
            out[v.name].append(np.asarray(v.evidence(ctx), float))

    ratio = (audited / total) if (tier1 and total) else 0.0
    return TokenScores(cfg, {k: (np.concatenate(v) if v else np.array([]))
                             for k, v in out.items()}, recompute_ratio=ratio)


def io_contexts(backend, sequences: list[Sequence], spec: SamplingSpec,
                need_proxy: bool = True, need_text: bool = False) -> list[VContext]:
    """Build per-sequence Tier-0 `VContext`s (proxy/text only, NO recompute of M).
    Used to `.fit` a `learned_io` verifier on labeled sequences."""
    ctxs = []
    for seq in sequences:
        steps = seq.steps
        proxy = np.stack([backend.proxy_logits(seq.prompt_id, st.position)
                          for st in steps]) if (need_proxy and steps) else None
        text = _seq_text(backend, seq) if need_text else None
        ctxs.append(VContext(seq.prompt_id, [st.claimed_token for st in steps], spec,
                             proxy_logits=proxy, prompt_text=text))
    return ctxs


def winsorize(scores: np.ndarray, honest_train: np.ndarray, pct: float) -> np.ndarray:
    """Clip scores at a percentile of the honest training split (DiFR feature eng.).
    Infinities/large values are excluded when computing the percentile."""
    finite = honest_train[np.isfinite(honest_train)]
    cap = np.percentile(finite, pct)
    return np.minimum(scores, cap)


def batch_means(scores: np.ndarray, batch_size: int, n_batches: int,
                rng: np.random.Generator) -> np.ndarray:
    """Sample `n_batches` batches of `batch_size` tokens and return their mean
    scores -- the batch-level statistic S."""
    n = len(scores)
    if batch_size > n:
        batch_size = n
    means = np.empty(n_batches)
    for i in range(n_batches):
        idx = rng.choice(n, size=batch_size, replace=False)
        means[i] = scores[idx].mean()
    return means


@dataclass
class EvalConfig:
    """The single, standardized evaluation protocol -- one place that fixes every
    knob the detection numbers depend on, so every experiment scores identically.

    The headline metric is the **standardized partial AUC at FPR <= `max_fpr`**
    (`metrics.partial_auc`): threshold-free separability restricted to the strict
    false-positive regime a verifier actually operates in, on the same 0.5..1.0
    scale as full AUC. The operating-point TPR is calibrated **out of sample** --
    the threshold tau comes from a held-out honest *calibration* split, never the
    honest batches TPR/FPR are then measured on.

    Fields
    ------
    max_fpr        : false-positive budget defining the region (default 0.5%).
    n_batches      : batches drawn per split for the null / attack statistics.
    winsor_pct     : per-token winsorization percentile (honest calib split), or
                     ``None`` to disable. Caps the rare filtered-out token so it
                     cannot dominate a batch mean.
    calib_frac     : fraction of honest tokens reserved for calibration (tau +
                     winsor cap); the rest are the honest eval null.
    seed           : RNG seed -- fixed so a matchup's score is reproducible.
    min_region_pts : soundness floor. Resolving FPR <= `max_fpr` needs about
                     ``n_batches * max_fpr`` honest eval batches above tau; below
                     `min_region_pts` the estimate is too coarse and `evaluate`
                     raises (bump `n_batches`, not the metric).

    Caveat this floor does NOT cover: `n_batches` alone guarantees enough
    resampled *batches* land in the region, not that those batches are close to
    independent draws. `batch_means` resamples without replacement from a FIXED,
    finite token pool, so if `batch_size` is a large fraction of that pool (a
    small `n_prompts`/`n_tokens` token count), the batches overlap heavily and
    the extreme low-FPR tail can look artifically diagonal/noisy regardless of
    `n_batches` -- more independent evidence, i.e. a bigger honest token pool
    (`n_prompts`/`n_tokens`, not a bigger `n_batches`), is what actually
    resolves it. As a rule of thumb keep `batch_size` well under ~10% of the
    honest eval-split token count."""

    max_fpr: float = 0.005
    n_batches: int = 2000
    winsor_pct: float | None = 99.9
    calib_frac: float = 0.5
    seed: int = 0
    min_region_pts: int = 10

    def __post_init__(self) -> None:
        if not 0.0 < self.max_fpr <= 1.0:
            raise ValueError(f"max_fpr must be in (0, 1], got {self.max_fpr}")
        if not 0.0 < self.calib_frac < 1.0:
            raise ValueError(f"calib_frac must be in (0, 1), got {self.calib_frac}")
        region_pts = self.n_batches * self.max_fpr
        if region_pts < self.min_region_pts:
            need = int(np.ceil(self.min_region_pts / self.max_fpr))
            raise ValueError(
                f"n_batches={self.n_batches} resolves only ~{region_pts:.1f} honest "
                f"batches inside FPR<= {self.max_fpr:.3%}; the partial-AUC estimate "
                f"is too coarse. Use n_batches >= {need} (>= {self.min_region_pts} "
                f"points in the region).")


@dataclass
class EvalResult:
    defense: str
    attack: str
    batch_size: int
    auc: float          # HEADLINE: standardized partial AUC at FPR <= max_fpr
    auc_full: float     # threshold-free full-range ROC AUC (context / legacy)
    tpr: float          # out-of-sample TPR at the calibrated FPR = max_fpr point
    max_fpr: float      # the false-positive budget the above are computed at


def evaluate(honest: TokenScores, attack: TokenScores, verifiers: list[Verifier],
             batch_sizes: list[int], config: EvalConfig | None = None, *,
             n_batches: int | None = None, winsor_pct: float | None = -1.0,
             seed: int | None = None) -> list[EvalResult]:
    """Compare honest vs attack batch statistics across batch sizes under the
    standardized `EvalConfig` protocol.

    Pass a `config` to override the protocol wholesale; the legacy `n_batches` /
    `winsor_pct` / `seed` keywords still work and, when given, override the
    corresponding `config` field (so existing call sites keep their behavior
    except for the metric definition). `winsor_pct` defaults to a sentinel so
    that *not passing it* keeps the config value while explicitly passing
    ``winsor_pct=None`` still disables winsorization.

    For each verifier and batch size the honest tokens are split into a
    calibration set (winsor cap + threshold tau) and a disjoint eval null; the
    reported `auc` is the standardized partial AUC at FPR <= `config.max_fpr` of
    the honest-eval vs attack batch statistics, and `tpr` is the catch rate at
    tau = the honest-*calibration* ``(1 - max_fpr)`` quantile."""
    cfg = config or EvalConfig()
    if n_batches is not None:
        cfg = replace(cfg, n_batches=n_batches)
    if winsor_pct != -1.0:
        cfg = replace(cfg, winsor_pct=winsor_pct)
    if seed is not None:
        cfg = replace(cfg, seed=seed)

    rng = np.random.default_rng(cfg.seed)
    results: list[EvalResult] = []
    for d in verifiers:
        h = honest.scores[d.name]
        a = attack.scores[d.name]
        h_cal, h_ev = _split(h, rng, cfg.calib_frac)   # calib (tau/cap) | eval null
        if cfg.winsor_pct is not None:
            h_ev = winsorize(h_ev, h_cal, cfg.winsor_pct)
            a = winsorize(a, h_cal, cfg.winsor_pct)
        for b in batch_sizes:
            cb = batch_means(h_cal, b, cfg.n_batches, rng)   # calibration null -> tau
            hb = batch_means(h_ev, b, cfg.n_batches, rng)    # eval null (negatives)
            ab = batch_means(a, b, cfg.n_batches, rng)       # attack (positives)
            tau = np.quantile(cb, 1.0 - cfg.max_fpr)
            results.append(EvalResult(
                defense=d.name, attack=attack.config_name, batch_size=b,
                auc=partial_auc(hb, ab, cfg.max_fpr),
                auc_full=roc_auc(hb, ab),
                tpr=float(np.mean(ab > tau)),
                max_fpr=cfg.max_fpr,
            ))
    return results


def _split(x: np.ndarray, rng: np.random.Generator, frac: float = 0.5):
    idx = rng.permutation(len(x))
    cut = int(len(x) * frac)
    return x[idx[:cut]], x[idx[cut:]]
