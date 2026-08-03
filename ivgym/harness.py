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

import warnings
from dataclasses import dataclass, field, replace

import numpy as np

from . import verifiers as _verifiers
from .attacks import Attack
from .core import SamplingSpec, Sequence, VContext
from .metrics import partial_auc, roc_auc, tpr_at_fpr
from .sampling import gumbel_noise, position_seed, projection
from .verifiers import Verifier


class RatioCeilingWarning(UserWarning):
    """A detection AUC was measured with the batch too large a fraction of the
    honest token pool. See `EvalConfig.max_pool_ratio` -- the number is a property
    of the particular pool it was drawn from, not of the deviation."""


# Always show it: the artifact is invisible in the resulting table, so a warning
# suppressed after the first occurrence would hide every cell but one.
warnings.simplefilter("always", RatioCeilingWarning)


@dataclass
class TokenScores:
    """Per-token scores for one config, keyed by verifier name.

    Two cost numbers, and the difference between them is the point:

    `recompute_ratio` is the realized fraction of *tokens* a Tier-1 verifier
    scored (1.0 for a full audit, 0.0 for a purely Tier-0 run). It is what the
    selective tier has always reported -- and it is **notional**: recompute is not
    billed per token. `reference_logits` at position `j` needs M run over the
    whole prefix `[prompt + claimed[:j]]`, so the physical unit is prefill tokens,
    and a top-k audit that touches one token in every sequence pays for nearly
    every sequence's prefill.

    `prefill_ratio` is that physical cost: reference-forward input tokens actually
    spent, divided by what a full (`budget=1.0`) audit of the same dataset would
    spend. It is measured, not assumed -- `verify` reads it off the backend's
    counter -- and equals `recompute_ratio` only when the audit happens to be
    prefix-shaped. `None` when the backend cannot report it (eager prefill).

    `audited` is the flat boolean mask of tokens a Tier-1 verifier actually scored;
    everywhere it is False the score array holds the verifier's `neutral`
    placeholder, which is not a measurement. `None` means "every token" -- a full
    audit or a Tier-0-only run -- so every non-selective result is unaffected by
    anything that reads it. `evaluate` needs it to keep the winsorization cap off
    the padding: see the note in `evaluate`.

    `weights` / `baseline` describe the AGGREGATION `evaluate` should use instead of
    a plain batch mean: the batch statistic becomes the mean of
    ``weights * (score - baseline)``. Both are flat per-token arrays in the same
    order as `scores`, and both must be computable WITHOUT recomputing M -- they are
    Tier-0 functions of the cheap proxy (`ivgym.infogain.InfoModel`), so using them
    costs the verifier nothing extra. `None` (the default) is the unweighted mean
    every result in this repo was measured with. See `ivgym/infogain.py` for why the
    matched filter `weights = Delta/v`, `baseline = E[score | honest]` is the
    variance-optimal choice, and `evaluate` for where they enter.
    """

    config_name: str
    scores: dict[str, np.ndarray] = field(default_factory=dict)
    recompute_ratio: float = 1.0
    prefill_ratio: float | None = None
    prefill_tokens: int = 0
    audited: np.ndarray | None = None
    weights: np.ndarray | None = None
    baseline: np.ndarray | None = None


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


def select_triaged(value: np.ndarray, budget: float, tie_seed: int = 20240) -> np.ndarray:
    """Boolean mask of the top-`budget` fraction of tokens by `value` (highest
    audited first). Always audits at least one token.

    Ties are broken by a fixed-seed random permutation, not by array order. That
    matters for the `uniform` value signal, whose whole point is to be the
    equal-cost RANDOM-subsample control: under a stable sort every value ties and
    the mask degenerates to "audit the first k tokens of the first sequences",
    which is a much worse control than random (it concentrates the audit in a
    handful of sequences and starves the rest). The seed is fixed so a matchup's
    score stays reproducible.
    """
    n = len(value)
    if n == 0:
        return np.zeros(0, bool)
    k = int(np.clip(round(budget * n), 1, n))
    mask = np.zeros(n, bool)
    order = np.random.default_rng(tie_seed).permutation(n)
    mask[order[np.argsort(-value[order], kind="stable")[:k]]] = True
    return mask


# ---------------------------------------------------------------------------
# The PHYSICAL cost model for a Tier-1 audit, and a scheduler that respects it.
#
# `select_triaged` above ranks tokens globally and takes the top `budget`
# fraction. That is the right rule if recompute were billed per token. It is not:
# reading M's logits at generated position `j` of a sequence requires a prefill of
# `[prompt + claimed[:j]]`, so the cost of auditing a sequence is set by the
# DEEPEST position audited in it, and every sequence touched at all pays its
# prompt again. A scattered top-k audit therefore costs almost as much as a full
# one, however small the token fraction looks.
#
# `select_prefix_scheduled` is the port of DSpark's hardware-aware prefix
# scheduler (arXiv:2607.05147 sec. "Hardware-Aware Prefix Scheduler"): admission
# is greedy over a globally sorted pool, but the sort key is marginal value per
# unit of marginal COST, and admitting a request commits to a contiguous prefix --
# their causality/early-stopping constraint, which here is not a choice but a
# physical fact about prefills.
# ---------------------------------------------------------------------------
def prefill_cost(mask: np.ndarray, seq_lens: list[int], prompt_lens: list[int]) -> int:
    """Reference-forward input tokens a given audit mask actually costs.

    A sequence with no audited token costs 0. Otherwise it costs
    `prompt_len + deepest_audited_index` -- the minimal prefill that makes that
    row readable (see `HFGPUBackend.prefill_reference`)."""
    total, i0 = 0, 0
    for n, plen in zip(seq_lens, prompt_lens):
        m = mask[i0:i0 + n]
        i0 += n
        if m.any():
            total += plen + int(np.nonzero(m)[0].max())
    return int(total)


def full_prefill_cost(seq_lens: list[int], prompt_lens: list[int]) -> int:
    """What `budget=1.0` costs: every sequence prefilled to its last token."""
    return int(sum(p + max(n - 1, 0) for n, p in zip(seq_lens, prompt_lens)))


def select_prefix_scheduled(value: np.ndarray, seq_lens: list[int],
                            prompt_lens: list[int], cost_budget: float,
                            max_rounds: int | None = None) -> np.ndarray:
    """Audit mask spending at most `cost_budget` x the full prefill cost.

    Greedy admission over (sequence, depth) increments by value density. Admitting
    sequence `i` to depth `d` costs `prompt_len_i + d - 1` and makes ALL of its
    first `d` tokens auditable for free -- so the value of a depth is the *sum* of
    the token values inside it, and the prompt is a startup cost paid once.
    Extending an already-admitted sequence costs one token per extra position, so
    depth is cheap and breadth is expensive; the schedule concentrates the audit
    into few sequences, which is exactly the behaviour a per-token top-k cannot
    express.

    Values are shifted to be non-negative (densities are only meaningful on a
    non-negative scale; `tie_margin` is natively negative).
    """
    n_tot = len(value)
    mask = np.zeros(n_tot, bool)
    if n_tot == 0 or not seq_lens:
        return mask
    v = value - value.min()
    budget = cost_budget * full_prefill_cost(seq_lens, prompt_lens)

    # per-sequence cumulative value: cum[i][d] = sum of the first d token values
    offs, cum = [], []
    i0 = 0
    for n in seq_lens:
        offs.append(i0)
        cum.append(np.concatenate([[0.0], np.cumsum(v[i0:i0 + n])]))
        i0 += n
    depth = [0] * len(seq_lens)
    spent = 0
    rounds = max_rounds if max_rounds is not None else 4 * len(seq_lens) + 8

    for _ in range(rounds):
        best = None                                  # (density, i, d, cost)
        for i, n in enumerate(seq_lens):
            d0 = depth[i]
            if d0 >= n:
                continue
            ds = np.arange(d0 + 1, n + 1)
            if d0 == 0:
                costs = prompt_lens[i] + ds - 1      # prompt startup paid once
                gains = cum[i][ds]
            else:
                costs = ds - d0                      # extend: one token per position
                gains = cum[i][ds] - cum[i][d0]
            dens = np.where(spent + costs <= budget, gains / costs, -np.inf)
            j = int(np.argmax(dens))
            if dens[j] > (-np.inf if best is None else best[0]):
                best = (float(dens[j]), i, int(ds[j]), int(costs[j]))
        if best is None or best[0] <= 0:
            break
        _, i, d, cost = best
        depth[i] = d
        spent += cost

    for i, d in enumerate(depth):
        if d:
            mask[offs[i]:offs[i] + d] = True
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
           values: np.ndarray | None = None, proj_seed: int = 123, proj_dim: int = 32,
           scheduler: str = "topk") -> TokenScores:
    """Score every token of every sequence with every verifier.

    `budget` in (0, 1] controls the Tier-1 recompute fraction (ignored by Tier-0
    verifiers). At `budget<1` the driver ranks tokens by the cheap `value_fn`
    signal (proxy entropy by default) and recomputes M only on the top fraction;
    unaudited tokens take each Tier-1 verifier's `neutral` score. Pass a
    precomputed `values` array (from `token_values`) to reuse a triage ranking
    across budgets.

    `scheduler` picks how the budget is spent:

      * `"topk"`   -- the historical rule: global top-`budget` fraction of tokens
                      by value. `budget` is a TOKEN fraction.
      * `"prefix"` -- `select_prefix_scheduled`: `budget` is a fraction of the real
                      PREFILL cost, and the audit is scheduled as contiguous
                      per-sequence prefixes by value density. Needs the backend to
                      answer `prompt_len`.

    Returns a `TokenScores` carrying both the notional `recompute_ratio` (token
    fraction) and, when the backend exposes `prefill_reference`/`prompt_len`, the
    measured `prefill_ratio` -- the physical cost. See `TokenScores`."""
    tier1 = [v for v in verifiers if v.tier == 1]
    need_proxy = any(v.needs_proxy for v in verifiers)
    need_served = any(v.needs_served for v in verifiers)
    need_text = any(v.needs_text for v in verifiers)
    need_act = any(v.needs_activation for v in verifiers)
    proj = projection(proj_seed, proj_dim, backend.hidden_dim) if need_act else None

    seq_lens = [len(s.steps) for s in sequences]
    can_cost = hasattr(backend, "prompt_len") and hasattr(backend, "prefill_reference")
    prompt_lens = ([backend.prompt_len(s.prompt_id) for s in sequences]
                   if can_cost else [0] * len(sequences))

    # Selective recompute: build the global audit mask from a cheap value signal.
    selective = bool(tier1) and budget < 1.0
    if selective:
        # `token_values` reads the proxy itself; the per-verifier `need_proxy`
        # above governs only whether the *scoring* context needs proxy logits.
        if values is None:
            values = token_values(backend, sequences, spec, value_fn)
        if scheduler == "prefix":
            if not can_cost:
                raise ValueError("scheduler='prefix' needs a backend exposing "
                                 "prompt_len() (see HFGPUBackend)")
            mask_flat = select_prefix_scheduled(values, seq_lens, prompt_lens, budget)
        elif scheduler == "topk":
            mask_flat = select_triaged(values, budget)
        else:
            raise ValueError(f"unknown scheduler {scheduler!r}; use 'topk' or 'prefix'")
    else:
        mask_flat = None

    # Pay for the prefixes this audit needs (lazy backends only; an eager backend
    # already prefilled everything during generation and this is a no-op).
    spent0 = getattr(backend, "prefill_tokens", 0)
    if tier1 and can_cost:
        i0 = 0
        for seq, n in zip(sequences, seq_lens):
            m = mask_flat[i0:i0 + n] if mask_flat is not None else np.ones(n, bool)
            i0 += n
            if n and m.any():
                backend.prefill_reference(seq.prompt_id, int(np.nonzero(m)[0].max()) + 1)

    out = {v.name: [] for v in verifiers}
    cfg = sequences[0].config_name if sequences else "?"
    audited = total = 0
    masks: list[np.ndarray] = []
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
        masks.append(audit)

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
    spent = getattr(backend, "prefill_tokens", 0) - spent0
    denom = full_prefill_cost(seq_lens, prompt_lens) if can_cost else 0
    pratio = (spent / denom) if (tier1 and can_cost and denom) else None
    return TokenScores(cfg, {k: (np.concatenate(v) if v else np.array([]))
                             for k, v in out.items()}, recompute_ratio=ratio,
                       prefill_ratio=pratio, prefill_tokens=int(spent),
                       audited=(np.concatenate(masks) if (selective and masks)
                                else None))


def rescore_at_budget(full: TokenScores, verifiers: list[Verifier], values: np.ndarray,
                      budget: float, *, scheduler: str = "topk",
                      seq_lens: list[int] | None = None,
                      prompt_lens: list[int] | None = None,
                      weights: np.ndarray | None = None,
                      baseline: np.ndarray | None = None) -> TokenScores:
    """Derive a selective-budget `TokenScores` from a FULL-budget one, exactly.

    A Tier-1 per-token score depends only on that token's own
    `(ref_logits, gumbel, claimed_token, spec)` -- never on which other tokens were
    audited (see `Tier1Verifier.evidence`, which writes `score_token` into audited
    rows and `neutral` elsewhere). So a budget-`b` audit's score array is just the
    full-budget array masked to the admitted tokens, with `neutral` elsewhere. No
    recompute, no re-verification, bit-identical to calling
    `verify(..., budget=b, values=values)`.

    That equivalence is what makes a budget SWEEP affordable: verify once at
    budget 1.0, then derive every budget and every value signal in numpy. It is
    asserted against the real driver in
    `tests/test_triage_and_cost.py::test_rescore_matches_verify`.

    It says nothing about COST -- deriving a mask does not pay a prefill. To
    measure what a budget costs, run the real driver against a lazy-prefill
    backend (`exp_prefix_cost_gpu.py`). Pass `seq_lens`/`prompt_lens` to have the
    physical `prefill_ratio` of the derived mask reported here too.

    `weights`/`baseline` set the aggregation the derived result will be evaluated
    under (see `TokenScores`). The weights are **zeroed off the audit mask**, which
    is what the matched-filter derivation requires and what makes an unaudited
    token contribute exactly 0 to the statistic instead of the verifier's `neutral`
    placeholder -- the padding cannot dilute or distort the batch mean at all.
    """
    if scheduler == "prefix":
        if seq_lens is None or prompt_lens is None:
            raise ValueError("scheduler='prefix' needs seq_lens and prompt_lens")
        mask = select_prefix_scheduled(values, seq_lens, prompt_lens, budget)
    elif scheduler == "topk":
        mask = select_triaged(values, budget) if budget < 1.0 else np.ones(len(values), bool)
    else:
        raise ValueError(f"unknown scheduler {scheduler!r}; use 'topk' or 'prefix'")

    scores = {v.name: (np.where(mask, full.scores[v.name], float(v.neutral))
                       if v.tier == 1 else full.scores[v.name])
              for v in verifiers}
    pratio = None
    if seq_lens is not None and prompt_lens is not None:
        denom = full_prefill_cost(seq_lens, prompt_lens)
        pratio = (prefill_cost(mask, seq_lens, prompt_lens) / denom) if denom else None
    return TokenScores(full.config_name, scores, recompute_ratio=float(mask.mean()),
                       prefill_ratio=pratio,
                       audited=(mask if any(v.tier == 1 for v in verifiers) else None),
                       weights=(np.where(mask, np.asarray(weights, float), 0.0)
                                if weights is not None else None),
                       baseline=(np.asarray(baseline, float)
                                 if baseline is not None else None))


def io_contexts(backend, sequences: list[Sequence], spec: SamplingSpec,
                need_proxy: bool = True, need_text: bool = False) -> list[VContext]:
    """Build per-sequence Tier-0 `VContext`s (proxy/text only, NO recompute of M).
    Used to score Tier-0 verifiers (`surface_stat`, `accept_rate`, `llm_judge`)
    on the same sequences a Tier-1 recompute is scored on."""
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
    scores -- the batch-level statistic S.

    A *weighted* statistic is obtained by passing pre-transformed scores
    (`weights * (score - baseline)`, see `evaluate`): the aggregation stays a plain
    mean, so the batch resampling -- and every number in this repo measured through
    it -- is untouched."""
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
    max_pool_ratio : the batch/pool ceiling (default 10%). `batch_means`
                     resamples WITHOUT replacement from a fixed, finite token
                     pool, so as `batch_size` approaches the honest eval split
                     every batch mean converges to the pool mean, the honest
                     variance collapses, and the AUC stops answering "would a
                     fresh batch be flagged?" -- it answers "do these two
                     particular pools have different means?", which is nearly
                     deterministic. Measured on one cell (`token_difr` vs
                     `quant_4bit`, Qwen3-1.7B), the SAME per-token scores read
                     0.977 at a 69% ratio and 0.530 at 1.8%.
    over_ratio     : what `evaluate` does when the ceiling is exceeded --
                     ``"warn"`` (default; emits a `RatioCeilingWarning` naming
                     the pool size that would be needed), ``"raise"``, or
                     ``"allow"``. Experiments that deliberately measure the
                     inflated arm -- `exp_headline_ratio_gpu`,
                     `exp_baseline_headroom_gpu`, `exp_reversal_check_gpu` --
                     must say so by passing ``"allow"``; there is no way to
                     produce an over-ceiling number silently.

    Note the artifact has NO SIGN: a collapsed-variance measurement is pushed
    toward both 1.0 and 0.0, which is why it cannot be spotted from a single
    table and why the ceiling is enforced here rather than left to the reader.
    More independent evidence -- a bigger honest token pool (`n_prompts` x
    `n_tokens`), not a bigger `n_batches` -- is what actually resolves it."""

    max_fpr: float = 0.005
    n_batches: int = 2000
    winsor_pct: float | None = 99.9
    calib_frac: float = 0.5
    seed: int = 0
    min_region_pts: int = 10
    max_pool_ratio: float = 0.10
    over_ratio: str = "warn"

    def __post_init__(self) -> None:
        if not 0.0 < self.max_fpr <= 1.0:
            raise ValueError(f"max_fpr must be in (0, 1], got {self.max_fpr}")
        if not 0.0 < self.calib_frac < 1.0:
            raise ValueError(f"calib_frac must be in (0, 1), got {self.calib_frac}")
        if self.over_ratio not in ("warn", "raise", "allow"):
            raise ValueError(f"over_ratio must be 'warn', 'raise' or 'allow', "
                             f"got {self.over_ratio!r}")
        region_pts = self.n_batches * self.max_fpr
        if region_pts < self.min_region_pts:
            need = int(np.ceil(self.min_region_pts / self.max_fpr))
            raise ValueError(
                f"n_batches={self.n_batches} resolves only ~{region_pts:.1f} honest "
                f"batches inside FPR<= {self.max_fpr:.3%}; the partial-AUC estimate "
                f"is too coarse. Use n_batches >= {need} (>= {self.min_region_pts} "
                f"points in the region).")

    def check_ratio(self, batch_size: int, pool: int, where: str = "") -> float:
        """Ratio of `batch_size` to the honest eval pool, enforcing the ceiling.

        Returns the ratio so the caller can record it on the result -- every AUC
        this repo reports carries the ratio it was measured at, because without
        it a detection AUC is not interpretable."""
        ratio = batch_size / max(pool, 1)
        if ratio > self.max_pool_ratio and self.over_ratio != "allow":
            need = int(np.ceil(batch_size / self.max_pool_ratio))
            msg = (f"batch/pool ratio {ratio:.1%} exceeds the {self.max_pool_ratio:.0%} "
                   f"ceiling{f' ({where})' if where else ''}: batch={batch_size} against "
                   f"an honest eval split of {pool} tokens. Batches resampled without "
                   f"replacement from a pool this small overlap heavily, the honest "
                   f"variance collapses, and the AUC is inflated (or deflated -- the "
                   f"artifact has no sign). Grow the pool to >= {need} eval tokens "
                   f"(~{2 * need} generated tokens per config at calib_frac=0.5), or "
                   f"shrink the batch. To measure the inflated arm on purpose, pass "
                   f"EvalConfig(over_ratio='allow').")
            if self.over_ratio == "raise":
                raise ValueError(msg)
            warnings.warn(msg, RatioCeilingWarning, stacklevel=3)
        return ratio


def _aggregate(x: np.ndarray, ts: TokenScores, sel: np.ndarray | None) -> np.ndarray:
    """Apply `ts`'s aggregation transform to a (possibly subsetted) score array.

    Returns ``weights * (x - baseline)``, restricted to the token positions `sel`
    (``None`` = all of them, in order). With no weights and no baseline it returns
    `x` unchanged, so the default aggregation path is untouched. A length mismatch
    is a programming error worth raising on: silently broadcasting a wrongly-sized
    weight array would produce a plausible number computed from the wrong tokens.
    """
    w, m = ts.weights, ts.baseline
    if w is None and m is None:
        return x
    # A per-token array must cover every token of the config, not just this subset:
    # `sel` indexes into the full pool.
    n_full = len(x) if sel is None else int(np.max(sel)) + 1
    for name, arr in (("weights", w), ("baseline", m)):
        if arr is not None and len(arr) < n_full:
            raise ValueError(
                f"TokenScores.{name} has length {len(arr)} but the score array needs "
                f"at least {n_full} per-token entries; weights/baseline must be flat "
                f"arrays in the same (sequence, step) order as `scores`")
    if m is not None:
        mm = np.asarray(m, float)
        x = x - (mm if sel is None else mm[sel])
    if w is not None:
        ww = np.asarray(w, float)
        x = x * (ww if sel is None else ww[sel])
    return x


@dataclass
class EvalResult:
    defense: str
    attack: str
    batch_size: int
    auc: float          # HEADLINE: standardized partial AUC at FPR <= max_fpr
    auc_full: float     # threshold-free full-range ROC AUC (context / legacy)
    tpr: float          # out-of-sample TPR at the calibrated FPR = max_fpr point
    max_fpr: float      # the false-positive budget the above are computed at
    # The batch/pool ratio the AUC above was measured at, and the honest eval
    # split it was drawn from. Reported on EVERY result because without it a
    # detection AUC is not interpretable (`EvalConfig.max_pool_ratio`).
    pool_ratio: float = float("nan")
    eval_pool: int = 0
    max_pool_ratio: float = EvalConfig.max_pool_ratio

    @property
    def over_ceiling(self) -> bool:
        """True when this number was measured above the batch/pool ceiling and
        should be read as an ordering rather than a level."""
        return self.pool_ratio > self.max_pool_ratio


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
    tau = the honest-*calibration* ``(1 - max_fpr)`` quantile.

    **Aggregation.** The batch statistic is a plain mean of per-token scores unless
    the `TokenScores` carry `weights`/`baseline`, in which case it is the mean of
    ``weights * (score - baseline)`` -- the matched-filter statistic derived in
    `ivgym/infogain.py`. Both arrays are per-token and Tier-0 (functions of the
    cheap proxy), each side supplies its own (the honest and attack runs have
    different claimed tokens, hence different proxy features), and they are applied
    *after* winsorization so the outlier cap is still taken on the raw score scale
    the verifier emits. tau is calibrated on the transformed honest calibration
    split, so the false-positive budget is honoured under either aggregation."""
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
        idx = rng.permutation(len(h))
        cut = int(len(h) * cfg.calib_frac)
        h_cal, h_ev = h[idx[:cut]], h[idx[cut:]]      # calib (tau/cap) | eval null
        if cfg.winsor_pct is not None:
            # The cap is a high quantile of the honest per-token score
            # distribution, there to stop one `delta_max` outlier from carrying a
            # batch. Under a SELECTIVE audit the score array is mostly the
            # verifier's `neutral` placeholder, which is not a score at all, so
            # taking the quantile over the padded array measures the padding: at a
            # 5% budget on `token_difr` -- which is itself exactly 0 wherever the
            # sampled token agrees -- the 99.9th percentile IS 0, `np.minimum`
            # flattens honest and attack alike to a constant, and `partial_auc`
            # returns exactly 0.5. Not "no signal": the signal, deleted by the
            # metric. So the cap comes from the AUDITED honest tokens only.
            # `audited=None` (full audit / Tier-0) means every token, which is
            # what it always was -- no non-selective number moves.
            cal_src = h_cal
            if honest.audited is not None and len(honest.audited) == len(h):
                aud = np.asarray(honest.audited, bool)[idx[:cut]]
                if aud.any():
                    cal_src = h_cal[aud]
            h_ev = winsorize(h_ev, cal_src, cfg.winsor_pct)
            a = winsorize(a, cal_src, cfg.winsor_pct)
        # Weighted (matched-filter) aggregation, if the caller supplied it. A
        # no-op -- bit-identical, not merely equivalent -- when both sides carry the
        # default `weights=baseline=None`, which is every result measured so far.
        h_cal = _aggregate(h_cal, honest, idx[:cut])
        h_ev = _aggregate(h_ev, honest, idx[cut:])
        a = _aggregate(a, attack, None)
        for b in batch_sizes:
            # Enforce (or at minimum record) the batch/pool ceiling BEFORE
            # measuring, so an inflated number can never be produced silently.
            ratio = cfg.check_ratio(b, len(h_ev), f"{d.name} vs {attack.config_name}")
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
                pool_ratio=ratio, eval_pool=len(h_ev),
                max_pool_ratio=cfg.max_pool_ratio,
            ))
    return results


def _split(x: np.ndarray, rng: np.random.Generator, frac: float = 0.5):
    """The honest calib | eval split. `evaluate` inlines this same permutation so
    it can carry the audit mask through the split alongside the scores; kept here
    because it is the definition of the split and is drawn from the same RNG
    stream, i.e. changing one without the other would silently desynchronize them."""
    idx = rng.permutation(len(x))
    cut = int(len(x) * frac)
    return x[idx[:cut]], x[idx[cut:]]
