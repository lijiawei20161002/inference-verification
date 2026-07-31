"""Information-directed verification: rank by INFORMATION, aggregate by matched filter.

Why this module exists
----------------------
`harness.verify(budget<1)` spends its recompute on the top-`budget` fraction of
tokens by a cheap per-token `value` signal, and every value signal in the library so
far -- `entropy`, `tie_margin`, `surprisal`, and the learned `ConfidenceHead` of
`ivgym/triage.py` -- estimates the same thing: **per-token sensitivity**
`Delta(t)`, how much a deviation would move the Tier-1 score at `t`.

Measured, that target does not work. `docs/TRIAGE_AND_AUDIT_COST.md` Part 1 reports
the head losing to the hand-crafted `entropy` it was ported to replace, while
putting a *negative* weight on entropy -- and closes with the observation that
"per-token surrogate sensitivity is not the quantity that maximizes batch-level
detection". This module takes that observation literally: sensitivity is the wrong
target, and the derivation below says what the right one is.

The statistic, and where sensitivity-ranking loses
--------------------------------------------------
Write `x_t` for a token's cheap proxy features, and let

    m(x)  = E[s | x, honest]                     the honest conditional mean
    v(x)  = E[(s - m(x))^2 | x, honest]          its residual second moment
    Delta(x) = E[s | x, deviating] - m(x)        the deviation's mean shift

`v` is deliberately the **residual** second moment given the *cheap features*, not
the within-position variance under nondeterminism. Those are different quantities and
the difference matters: re-scoring an honest `token_difr` row under benign noise moves
nothing at all at most positions (a confidently-sampled token's margin is
deterministically 0 -- measured: zero probe variance at ~98% of positions), yet the
batch statistic still pays a variance, because the *positions* differ from each other
and the verifier only knows `x`. What the statistic divides by is the spread the
verifier cannot predict away, which is exactly the residual above.

`harness.evaluate` draws a batch of `b` tokens uniformly from a pool of `n` and
averages. Consider the general **weighted, centered** statistic
(`TokenScores.weights` / `.baseline`):

    S_w = (1/b) * sum_{i in batch} w(x_t_i) * ( s_t_i - m(x_t_i) )

with `w = 0` at unaudited positions. Then, over both the token draw and the score
noise,

    E[S_w | honest]    = 0
    E[S_w | deviating] = (1/n) * sum_t w_t * Delta_t
    Var[S_w | honest]  = (1/b) * (1/n) * sum_t w_t^2 * v_t

so the batch separation is

    delta(w) = sqrt(b/n) * ( sum_t w_t Delta_t ) / sqrt( sum_t w_t^2 v_t ) .

By Cauchy-Schwarz this is maximized at `w_t = Delta_t / v_t` -- the **matched
filter** -- and its optimum is

    delta^2 = (b/n) * sum_t I_t ,     I_t = Delta_t^2 / v_t                    (*)

`I_t` is the per-token Fisher information, i.e. the squared per-token d-prime.
Three consequences, and they are the whole content of this module:

1. **What to rank by.** `I_t = Delta_t^2 / v_t`, not `Delta_t`. A near-tie position
   is sensitive (large `Delta`) but *also* noisy (large `v`): the sampled token
   there flips under benign nondeterminism alone, which is exactly the honest
   variance the verifier is trying to beat. Ranking on `Delta` buys signal and noise
   together; ranking on `I` buys the ratio. This is the same quantity DiFR-style
   detectors are implicitly betting on, made explicit and *estimable*.
2. **How to aggregate.** With the matched filter the information in (*) is
   **additive over audited tokens**, so the best budget-`k` audit is exactly the top
   `k` by `I_t` -- selection becomes separable, and greedy top-k is optimal rather
   than a heuristic. Under the plain unweighted mean the library uses today it is
   not: the honest *between-position spread* of `m(x_t)` enters the denominator, so
   admitting a high-`Delta` token can lower the batch d-prime, and no per-token
   ranking key can be optimal in general. Centering (`baseline = m`) is what removes
   that term.
3. **What a budget costs, predictably.** `delta^2(A) / delta^2(full)` is the
   *information capture* `sum_{A} I / sum_{all} I`, so an audit that keeps half the
   information detects at `d'/sqrt(2)`. `signal.info_capture` /
   `signal.pauc_of_capture` turn a Tier-0 ranking into a predicted
   detection-vs-budget curve **before any recompute is spent**.

Two bonuses fall out of the weights being zero off the audit mask: the verifier's
`neutral` padding contributes exactly 0 instead of diluting the mean, and the
winsorization pathology that distorted small-budget numbers (same document,
"Incidental fixes") cannot arise, because there is nothing to cap.

Estimating `Delta` and `v` from honest data only
------------------------------------------------
Both are estimated by the same one-time offline **probe** `ivgym/triage.py` already
uses for sensitivity -- honest data, no labeled attack, `ref_logits` read only
during the offline fit:

  * `v_hat(t)`: re-score the honest reference row under repeated *benign-scale*
    logit noise and take the variance. This is the position's nondeterminism, the
    denominator of (*).
  * `Delta_hat(t)`: re-score under benign + a generic zero-mean *deviation* probe
    and take the mean shift. Committed to no particular attack -- only to the
    deviation perturbing logits, which is how quantization, an fp8 KV cache and a
    swapped checkpoint all enter.

`InfoModel` then regresses both onto the nine Tier-0 proxy features of
`triage.feature_matrix` (ridge; `log v` in log space because variances are
multiplicative), so at verification time it reads the cheap proxy only and never
`M`. Absolute scale does not matter: `partial_auc` is invariant to a global positive
rescaling of the statistic, so `I` and `w` are only ever used up to a constant.

What this predicts, so it can fail
----------------------------------
  P1. At equal budget, ranking by `I` beats ranking by sensitivity `Delta`, and the
      matched filter beats the unweighted mean.
  P2. The measured AUC-vs-budget curve is predicted by `signal.pauc_of_capture`
      from the Tier-0 `I` ranking alone.

`experiments/exp_info_directed_gpu.py` tests both on a real model, against the
ORACLE arm of `realized_moments` / `oracle_matched_filter` -- the same `(Delta, v)`
read off labeled honest/attack pairs instead of estimated from the cheap proxy.
That arm is what makes a failure diagnosable rather than merely a null: a P1 that
fails deployably but holds at the oracle indicts the Tier-0 **estimator**, and a P1
that fails at the oracle too indicts the **derivation** -- specifically the
Gaussian/independent-token model of `ivgym/signal.py`, which P2 tests directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import VContext
from .triage import FEATURE_NAMES, feature_matrix

_EPS = 1e-12
# Variance and information floors. `v_hat` can come back as exactly 0 at a position
# where every probe left the score unchanged (a confidently-sampled token whose
# margin no benign perturbation touches), and 1/0 is not the right reading of that:
# such a position is not infinitely informative, it is a position where a *finite*
# probe budget saw nothing. Both floors are relative to the observed scale, set in
# `probe_moments` / `InfoModel.fit`.
_VAR_FLOOR_FRAC = 1e-3


# ---------------------------------------------------------------------------
# 1. The offline probe: honest-only (m, v, Delta) per token.
# ---------------------------------------------------------------------------
def probe_moments(ref_logits: np.ndarray, gumbel: np.ndarray, claimed_tokens: list[int],
                  sampling, verifier, rng: np.random.Generator, *,
                  probe_sigma: float = 0.15, benign_sigma: float = 0.03,
                  n_probe: int = 8) -> dict[str, np.ndarray]:
    """Per-token honest mean/variance and deviation shift, by Monte-Carlo probing.

    For each position, `n_probe` re-scores under benign-scale logit noise give the
    honest mean `m` and variance `v`; `n_probe` re-scores under benign + a
    `probe_sigma` deviation probe give the mean shift `delta`. Returns
    `{"m", "v", "delta", "info"}`, all length `T`, with `info = delta^2 / v`.

    Reads `ref_logits`, so this runs in the ONE-TIME offline fit only -- the same
    trusted-`M`-run amortization `spec_decode.ProxyReference.fit` and
    `triage.surrogate_sensitivity` already assume. Nothing at verification time
    calls it; `InfoModel` is what gets deployed.

    `benign_sigma` is the verifier's own nondeterminism scale (`HFGPUBackend`'s
    `verifier_sigma`), which the verifier knows because it is its own deployment.
    Its absolute value is not critical: `v` enters `info` and the matched-filter
    weights only through ratios across positions, and the batch statistic may be
    rescaled freely without moving any AUC.
    """
    n = len(claimed_tokens)
    m = np.zeros(n)
    v = np.zeros(n)
    delta = np.zeros(n)
    dev_sigma = float(np.hypot(benign_sigma, probe_sigma))
    for i in range(n):
        row, g, tok = ref_logits[i], gumbel[i], claimed_tokens[i]
        honest = np.empty(n_probe)
        dev = np.empty(n_probe)
        for j in range(n_probe):
            honest[j] = verifier.score_token(
                row + rng.normal(0.0, benign_sigma, row.shape), g, tok, sampling)
            dev[j] = verifier.score_token(
                row + rng.normal(0.0, dev_sigma, row.shape), g, tok, sampling)
        m[i] = honest.mean()
        v[i] = honest.var(ddof=1) if n_probe > 1 else 0.0
        delta[i] = dev.mean() - m[i]
    floor = _VAR_FLOOR_FRAC * float(np.mean(v)) + _EPS
    return {"m": m, "v": v, "delta": delta, "info": delta ** 2 / np.maximum(v, floor)}


def realized_moments(honest_scores: np.ndarray, attack_scores: np.ndarray,
                     n_bins: int = 40, key: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Per-token `(m, v, Delta, info)` from labeled honest/attack pairs, by binning.

    A single (honest, attack) pair at a token gives one draw from each distribution,
    which is enough for `Delta` (`triage.paired_effect_size`) but not for `v`. So
    positions are pooled into `n_bins` equal-count bins of `key` (default: the
    honest score itself) and the moments are computed within a bin.

    **This is a diagnostic, not the ceiling arm** -- `oracle_model` is. With the
    default key it conditions on the very quantity whose spread it is measuring, so
    each bin is a narrow slice of the honest score and `v` comes back far below the
    real conditional variance (measured on planted data: 0.04 against a true 1.0).
    Weights `Delta/v` built from it therefore diverge as `n_bins` grows, and a
    "ceiling" that a finer binning can push arbitrarily high bounds nothing. Pass an
    independent `key` (a Tier-0 feature) to get an honest conditional estimate.
    """
    h = np.asarray(honest_scores, float)
    a = np.asarray(attack_scores, float)
    k = h if key is None else np.asarray(key, float)
    order = np.argsort(k, kind="stable")
    m = np.zeros(len(h))
    v = np.zeros(len(h))
    d = np.zeros(len(h))
    for part in np.array_split(order, min(n_bins, max(len(h), 1))):
        if len(part) == 0:
            continue
        m[part] = float(h[part].mean())
        v[part] = float(h[part].var())
        d[part] = float(a[part].mean() - h[part].mean())
    floor = _VAR_FLOOR_FRAC * float(np.mean(v)) + _EPS
    v = np.maximum(v, floor)
    return {"m": m, "v": v, "delta": d, "info": d ** 2 / v}


def realized_info(honest_scores: np.ndarray, attack_scores: np.ndarray,
                  n_bins: int = 40, key: np.ndarray | None = None) -> np.ndarray:
    """ORACLE per-token information `Delta^2/v` -- `realized_moments`'s `info`."""
    return realized_moments(honest_scores, attack_scores, n_bins, key)["info"]


def oracle_matched_filter(honest_scores: np.ndarray, attack_scores: np.ndarray,
                          n_bins: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """`(weights, baseline)` from binned `realized_moments`: `max(Delta,0)/v`, `m`.

    Carries `realized_moments`' bias -- see its docstring, and prefer
    `oracle_model(...).weights/.baseline` for the ceiling arm. Kept because a
    position-level (rather than feature-level) estimate is the right diagnostic for
    "where does the information actually sit", which is a different question from
    "what could a function of the cheap features do".
    """
    mom = realized_moments(honest_scores, attack_scores, n_bins)
    return np.maximum(mom["delta"], 0.0) / mom["v"], mom["m"]


def oracle_model(x: np.ndarray, honest_scores: np.ndarray, attack_scores: np.ndarray,
                 ridge: float = 1e-2) -> "InfoModel":
    """The CEILING arm: the same `InfoModel`, fit on labeled pairs, not on the probe.

    Identical machinery to the deployable model -- three ridge regressions of the
    same nine Tier-0 features onto `(m, log v, Delta)` -- differing only in where
    the targets come from:

      * `m`:     the honest score itself. A regression of `s` on `x` estimates
                 `E[s | x, honest]`, which is the definition of `m(x)`.
      * `v`:     the squared residual around that fit, so the regression estimates
                 the RESIDUAL conditional variance -- the quantity in the module
                 derivation, and the one binning on the score gets wrong.
      * `Delta`: the per-position difference `attack - honest`
                 (`triage.paired_effect_size`'s label), whose conditional mean is
                 `Delta(x)`.

    So this answers exactly one question: **if the two regressions were perfect --
    same features, same statistic, labels for free -- would information-directed
    verification work?** A deployable arm that loses while this one wins indicts the
    honest-only probe as an estimator of `(Delta, v)`; a loss here indicts the
    feature basis or the derivation, neither of which more probing can fix.

    Nine coefficients over the whole eval pool, so the in-sample optimism is small
    and bounded -- unlike a per-position oracle, this cannot be driven up by
    resolving the pool more finely.
    """
    h = np.asarray(honest_scores, float)
    d = np.asarray(attack_scores, float) - h
    ones = np.ones(len(h))
    first = InfoModel(ridge=ridge).fit(x, {"m": h, "v": ones, "delta": d, "info": ones})
    resid2 = (h - first.baseline(x)) ** 2
    floor = _VAR_FLOOR_FRAC * float(resid2.mean()) + _EPS
    v = np.maximum(resid2, floor)
    return InfoModel(ridge=ridge).fit(x, {"m": h, "v": v, "delta": d,
                                          "info": d ** 2 / v})


# ---------------------------------------------------------------------------
# 2. The deployable model: Tier-0 features -> (m, v, Delta) -> (info, weights).
# ---------------------------------------------------------------------------
def _ridge(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """Ridge coefficients on already-standardized `x`, intercept = mean(y)."""
    n, f = x.shape
    a = x.T @ x + lam * n * np.eye(f)
    return np.linalg.solve(a, x.T @ (y - y.mean())), float(y.mean())


@dataclass
class InfoModel:
    """Tier-0 estimator of per-token information and matched-filter weights.

    Three ridge regressions over the nine proxy-only features of
    `triage.feature_matrix`: the honest mean `m` and the deviation shift `Delta`
    directly, and the honest variance `v` in **log space** (variances are
    multiplicative and span orders of magnitude; a linear fit would be dominated by
    the few loudest positions and can predict negatives).

    Fitted on the output of `probe_moments`; deployed against `proxy_logits` only,
    so `info` / `weights` / `baseline` are Tier-0 and cost no recompute. `info` is
    what `harness.verify(value_fn=...)` ranks by; `weights` and `baseline` are what
    `TokenScores` carries into `harness.evaluate`.
    """

    ridge: float = 1e-2
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None
    w_m: np.ndarray | None = None
    b_m: float = 0.0
    w_d: np.ndarray | None = None
    b_d: float = 0.0
    w_v: np.ndarray | None = None
    b_v: float = 0.0
    v_floor: float = _EPS
    fit_report: dict | None = None

    # -- internals ---------------------------------------------------------
    def _std(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, float) - self.mu) / self.sd

    def _pred(self, x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
        return self._std(x) @ w + b

    # -- api ---------------------------------------------------------------
    def fit(self, x: np.ndarray, moments: dict[str, np.ndarray]) -> "InfoModel":
        """Fit on features `x` `[N, F]` and a `probe_moments` payload."""
        if x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"expected {len(FEATURE_NAMES)} features, got {x.shape[1]}")
        m, v, d = (np.asarray(moments[k], float) for k in ("m", "v", "delta"))
        self.mu = x.mean(axis=0)
        self.sd = x.std(axis=0) + 1e-8
        xs = self._std(x)
        self.v_floor = _VAR_FLOOR_FRAC * float(np.mean(v)) + _EPS
        self.w_m, self.b_m = _ridge(xs, m, self.ridge)
        self.w_d, self.b_d = _ridge(xs, d, self.ridge)
        self.w_v, self.b_v = _ridge(xs, np.log(np.maximum(v, self.v_floor)), self.ridge)
        self.fit_report = {
            "n": int(len(x)),
            "r2_m": _r2(m, self.baseline(x)),
            "r2_delta": _r2(d, self.sensitivity(x)),
            "r2_log_v": _r2(np.log(np.maximum(v, self.v_floor)), np.log(self.noise(x))),
            "spearman_info": _spearman(moments["info"], self.info(x)),
            "coef_delta": dict(zip(FEATURE_NAMES, self.w_d.tolist())),
            "coef_log_v": dict(zip(FEATURE_NAMES, self.w_v.tolist())),
        }
        return self

    def baseline(self, x: np.ndarray) -> np.ndarray:
        """Predicted honest mean `m(x)` -- the centering term."""
        return self._pred(x, self.w_m, self.b_m)

    def sensitivity(self, x: np.ndarray) -> np.ndarray:
        """Predicted deviation shift `Delta(x)`, floored at 0.

        A negative predicted shift means "a deviation would, if anything, *lower*
        the evidence here" -- no usable information either way for a one-sided test,
        so the matched filter gives it weight 0 rather than a negative weight that
        would subtract honest noise into the statistic.
        """
        return np.maximum(self._pred(x, self.w_d, self.b_d), 0.0)

    def noise(self, x: np.ndarray) -> np.ndarray:
        """Predicted honest variance `v(x)` (exponentiated log-space fit)."""
        return np.maximum(np.exp(self._pred(x, self.w_v, self.b_v)), self.v_floor)

    def info(self, x: np.ndarray) -> np.ndarray:
        """Predicted per-token information `I = Delta^2 / v` -- the value signal."""
        return self.sensitivity(x) ** 2 / self.noise(x)

    def weights(self, x: np.ndarray) -> np.ndarray:
        """Matched-filter weights `w = Delta / v` (up to a global positive scale)."""
        return self.sensitivity(x) / self.noise(x)

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        """JSON-safe snapshot. Refuses an unfitted model."""
        if self.w_d is None:
            raise RuntimeError("cannot serialize an unfitted InfoModel")
        return {"feature_names": list(FEATURE_NAMES), "ridge": self.ridge,
                "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "w_m": self.w_m.tolist(), "b_m": self.b_m,
                "w_d": self.w_d.tolist(), "b_d": self.b_d,
                "w_v": self.w_v.tolist(), "b_v": self.b_v,
                "v_floor": self.v_floor, "fit_report": self.fit_report}

    @classmethod
    def from_dict(cls, d: dict) -> "InfoModel":
        """Rebuild a fitted model, refusing a different feature layout (the
        coefficients are positional -- see `triage.ConfidenceHead.from_dict`)."""
        names = tuple(d.get("feature_names", ()))
        if names != FEATURE_NAMES:
            raise ValueError(f"model was fit on features {names} but this build uses "
                             f"{FEATURE_NAMES}; refusing to score positionally against "
                             f"a different layout")
        mdl = cls(ridge=float(d.get("ridge", 1e-2)))
        mdl.mu = np.asarray(d["mu"], float)
        mdl.sd = np.asarray(d["sd"], float)
        for k in ("m", "d", "v"):
            setattr(mdl, f"w_{k}", np.asarray(d[f"w_{k}"], float))
            setattr(mdl, f"b_{k}", float(d[f"b_{k}"]))
        mdl.v_floor = float(d.get("v_floor", _EPS))
        mdl.fit_report = d.get("fit_report")
        return mdl


# ---------------------------------------------------------------------------
# 3. Plugging into the harness.
# ---------------------------------------------------------------------------
def info_value_fn(model: InfoModel):
    """Wrap a fitted `InfoModel` as a `verifiers` value function (`VContext -> [T]`).

    Register it with `verifiers.register_value_fn("info_gain", info_value_fn(model))`
    and every existing call site -- `harness.verify(value_fn="info_gain")`,
    `harness.token_values`, `select_prefix_scheduled` -- allocates by information.
    The closure touches `ctx.proxy_logits` only, so the Tier-0 invariant holds.
    """
    def _fn(ctx: VContext) -> np.ndarray:
        return model.info(feature_matrix(ctx))
    return _fn


def matched_filter(model: InfoModel, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(weights, baseline)` for `harness.evaluate`, from a flat feature matrix.

    Assign them to the `TokenScores` of BOTH configs being compared (each from its
    own features -- honest and attack runs claim different tokens) and the batch
    statistic becomes the matched filter of the module derivation. `TokenScores`
    built by `harness.rescore_at_budget(..., weights=...)` additionally zero the
    weights off the audit mask, which is what the derivation assumes.
    """
    return model.weights(features), model.baseline(features)


# ---------------------------------------------------------------------------
# small fit diagnostics (kept here so the experiment prints one number per fit)
# ---------------------------------------------------------------------------
def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, float)
    ss = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - ((y - pred) ** 2).sum() / ss) if ss > 0 else 0.0


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, ties averaged. Used to say whether a Tier-0 estimate orders
    positions the way the probe does -- the only property `select_triaged` uses."""
    ra, rb = _rankdata(a), _rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def _rankdata(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(1, len(x) + 1)
    s = x[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks
