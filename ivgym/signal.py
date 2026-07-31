"""How strong is a verification signal, and what does that predict? The theory side.

Every detection number in this repo is produced the same way: a per-token evidence
score is averaged over a batch of `b` tokens, and the batch statistic `S` is
compared against a threshold calibrated to a false-positive budget
(`harness.evaluate`, `metrics.partial_auc`). That pipeline has a closed form, and
this module is it -- one place for the arithmetic that connects

    per-token effect size  d'   ->   batch separation  delta = d' * sqrt(b)
                                ->   standardized pAUC @ FPR <= alpha

so an experiment can *predict* a detection AUC instead of only measuring one, and a
measured AUC can be checked against what the per-token evidence can support.

The model
---------
Treat the tokens in a batch as independent draws from the honest and deviating
per-token score distributions. Then `S` is a mean of `b` draws, so by the CLT it is
Gaussian under both hypotheses, and after standardizing by the honest per-token sd
the two batch distributions are `N(0, 1/b)` and `N(d', 1/b)` -- i.e. a unit-variance
Gaussian pair separated by

    delta = d' * sqrt(b),        d' = (mean_a - mean_h) / sd_h .

`pauc_of_delta` integrates the resulting ROC over `FPR <= max_fpr` and standardizes
it exactly the way `metrics.partial_auc` does, so a predicted number is on the same
scale as a measured one. `delta_for_pauc` inverts that, and `batch_for_pauc` turns
it into "how many tokens does this deviation need".

What the model does NOT capture (both documented in
`docs/TRIAGE_AND_AUDIT_COST.md`): batches resampled from a finite token pool are not
independent draws -- as the batch approaches the pool size the honest variance
collapses and the *measured* AUC runs above this prediction (§ 3's ratio artifact);
and a heavy-tailed per-token score reaches the Gaussian limit slowly, so at small
`b` the prediction is a guide rather than a guarantee. It is a prediction to be
falsified, which is how `exp_baseline_headroom_gpu` uses it.

Information, and why it is the quantity that allocates a budget
--------------------------------------------------------------
`d'` is a property of a *set* of tokens, not a token, so it cannot rank positions
on its own. The per-token quantity that adds up is the Fisher information

    I(t) = Delta(t)^2 / v(t),     Delta(t) = E[s_t | deviating] - E[s_t | honest],
                                  v(t)     = Var[s_t | honest],

which is the squared per-token d-prime at `t`. Under the *matched-filter* statistic
of `ivgym.infogain` -- weights `w(t) = Delta(t)/v(t)` on centered evidence -- the
batch separation of an audited set `A` out of a pool of `n` is exactly

    d'(A)^2 = (1/n) * sum_{t in A} I(t)

(derived in `ivgym/infogain.py`), so the information is *additive over audited
tokens* and the best budget-`k` audit is the top `k` by `I(t)`. `info_capture` reads
off the fraction of the pool's total information a given audit keeps, and
`pauc_of_capture` turns that fraction into the predicted detection AUC at a budget
-- a compute-accuracy curve computable from the cheap value signal alone, before any
recompute is spent.

This module is pure numpy and has no `ivgym` imports, so it is usable from
`experiments/` plotting code that runs without a GPU or a backend.
"""
from __future__ import annotations

import numpy as np

_SQRT2 = np.sqrt(2.0)


# ---------------------------------------------------------------------------
# Normal distribution helpers (kept scipy-free like the rest of the repo)
# ---------------------------------------------------------------------------
def ndtr(z):
    """Standard normal CDF, from `math.erfc` (scipy-free). Vectorized over an array
    input; a scalar in gives a float out."""
    from math import erfc
    z_arr = np.asarray(z, float)
    flat = np.array([0.5 * erfc(-float(v) / _SQRT2) for v in np.atleast_1d(z_arr).ravel()])
    return float(flat[0]) if z_arr.shape == () else flat.reshape(z_arr.shape)


def ndtri(q: np.ndarray | float):
    """Inverse standard normal CDF (Acklam's rational approximation, |eps| < 1.15e-9).

    Hand-rolled to keep the repo scipy-free. Vectorized over an array input; a
    scalar in gives a float out.
    """
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    q_arr = np.asarray(q, float)
    scalar = q_arr.shape == ()
    x = np.atleast_1d(q_arr).astype(float)
    out = np.empty_like(x)

    lo, hi = x < p_low, x > p_high
    mid = ~(lo | hi)
    if lo.any():
        r = np.sqrt(-2 * np.log(x[lo]))
        out[lo] = ((((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5])
                   / ((((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1))
    if hi.any():
        r = np.sqrt(-2 * np.log(1 - x[hi]))
        out[hi] = -((((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5])
                    / ((((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1))
    if mid.any():
        r = x[mid] - 0.5
        t = r * r
        out[mid] = ((((((a[0]*t+a[1])*t+a[2])*t+a[3])*t+a[4])*t+a[5])*r
                    / (((((b[0]*t+b[1])*t+b[2])*t+b[3])*t+b[4])*t+1))
    return float(out[0]) if scalar else out


# ---------------------------------------------------------------------------
# The Gaussian batch model: d' -> pAUC and back
# ---------------------------------------------------------------------------
def pauc_of_delta(delta: float, max_fpr: float = 0.005, n: int = 20000) -> float:
    """Standardized pAUC @ FPR <= `max_fpr` for two unit-variance Gaussians
    separated by `delta`, by quadrature.

    At false-positive rate `u` the threshold is `tau = ndtri(1-u)` and the
    true-positive rate is `ndtr(delta - tau)`; the partial area is the integral of
    that over `u in (0, max_fpr]`, standardized onto 0.5..1.0 exactly as
    `metrics.partial_auc` does (chance = 0.5, perfect = 1.0).
    """
    if max_fpr <= 0.0:
        return 0.5
    u = np.linspace(max_fpr / n, max_fpr, n)
    tpr = ndtr(delta - ndtri(1.0 - u))
    pauc = float(np.trapezoid(tpr, u))
    return float(0.5 * (1.0 + (pauc - 0.5 * max_fpr ** 2) / (max_fpr - 0.5 * max_fpr ** 2)))


def delta_for_pauc(target: float = 0.90, max_fpr: float = 0.005,
                   hi: float = 40.0, tol: float = 1e-6) -> float:
    """The batch separation `delta` at which `pauc_of_delta` reaches `target`.

    Monotone in `delta`, so a bisection is exact to `tol`. This is the constant
    `batch_for_pauc` (and `experiments/plot_headroom.py`) predict against: at
    `max_fpr = 0.5%` and `target = 0.90` it is **3.767**, which is where that
    literal in earlier versions of this repo came from.
    """
    if not 0.5 < target < 1.0:
        raise ValueError(f"target must be in (0.5, 1), got {target}")
    lo = 0.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if pauc_of_delta(mid, max_fpr) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def batch_for_pauc(d_prime: float, target: float = 0.90,
                   max_fpr: float = 0.005) -> int:
    """Batch size at which `d' * sqrt(b)` reaches the `target` standardized pAUC.

    Returns -1 for a non-positive `d'` (no batch size ever gets there). The number
    it returns is a *prediction*: `exp_baseline_headroom_gpu` measures the same
    quantity by growing the pool alongside the batch.
    """
    if d_prime <= 0:
        return -1
    return int(np.ceil((delta_for_pauc(target, max_fpr) / d_prime) ** 2))


def predicted_pauc(d_prime: float, batch: int, max_fpr: float = 0.005) -> float:
    """Predicted standardized pAUC for `batch` independent tokens at effect size
    `d'`. The forward direction of `batch_for_pauc`."""
    return pauc_of_delta(float(d_prime) * np.sqrt(max(batch, 0)), max_fpr)


def per_token_stats(honest: np.ndarray, attack: np.ndarray,
                    winsor_pct: float | None = 99.9) -> dict:
    """Per-token effect size on exactly the scale `harness.evaluate` scores.

    Winsorized at the honest percentile first (which is what `evaluate` does), then
    `d' = (mean_a - mean_h) / sd_h`. A batch of `b` independent tokens separates by
    `d' * sqrt(b)`, so `d'` -- not the AUC at one batch size -- is the portable
    statement of how strong a deviation's signal is.
    """
    h = np.asarray(honest, float)
    a = np.asarray(attack, float)
    if winsor_pct is not None:
        cap = np.percentile(h[np.isfinite(h)], winsor_pct)
        h, a = np.minimum(h, cap), np.minimum(a, cap)
    return {"d_prime": float((a.mean() - h.mean()) / (h.std() + 1e-12)),
            "honest_mean": float(h.mean()), "attack_mean": float(a.mean()),
            "honest_sd": float(h.std()), "attack_sd": float(a.std())}


# ---------------------------------------------------------------------------
# Information accounting: what a selective audit keeps
# ---------------------------------------------------------------------------
def info_capture(info: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of the pool's total per-token information `sum I(t)` that `mask`
    keeps. 1.0 for a full audit, `budget` for a random one (in expectation), and
    more than `budget` exactly to the extent the ranking is informative."""
    info = np.asarray(info, float)
    total = float(info.sum())
    if total <= 0:
        return 0.0
    return float(info[np.asarray(mask, bool)].sum() / total)


def pauc_of_capture(capture: float, full_d_prime: float, batch: int,
                    max_fpr: float = 0.005) -> float:
    """Predicted pAUC of an audit that captures `capture` of the pool information.

    Under the matched-filter statistic `d'(A)^2 = capture * d'(full)^2`, so the
    audited set's separation is `sqrt(capture)` of the full audit's: a budget that
    keeps half the information detects at `d' / sqrt(2)`, NOT at half the AUC. This
    is the compute-accuracy tradeoff curve, and it is computable from the cheap
    value signal alone -- before any recompute is spent.
    """
    return predicted_pauc(float(full_d_prime) * np.sqrt(max(capture, 0.0)), batch,
                          max_fpr)
