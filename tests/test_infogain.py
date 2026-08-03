"""Tests for information-directed verification (`ivgym.infogain`) and the
signal-strength model (`ivgym.signal`).

Dependency-free (pure numpy), same style as test_smoke.py. Run:
    python tests/test_infogain.py             # or: python -m pytest tests/ -q

The load-bearing ones are the two that pin the derivation in `infogain`'s module
docstring rather than the code's own behaviour:

  * `test_matched_filter_is_the_optimal_weighting` -- `w = Delta/v` beats any other
    weighting, including the unweighted mean and the sensitivity-only `w = Delta`,
    on the exact separation formula the batch statistic realizes.
  * `test_top_k_by_information_is_the_optimal_selection` -- under that weighting the
    separation is additive, so top-k by `I` beats every other size-k subset (checked
    exhaustively on a small pool). This is the property that makes greedy triage
    optimal rather than a heuristic, and it holds for NO other ranking key.

Plus: the weighted-aggregation path through `harness.evaluate` is bit-identical
when unused, weights are zeroed off the audit mask, the probe recovers a planted
(Delta, v) structure, the model stays Tier-0, and `signal`'s Gaussian model round
trips (and still says 3.767).
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import harness, infogain, signal, triage, verifiers
from ivgym.core import SamplingSpec, VContext

SPEC = SamplingSpec()


def _ctx(rng, t=60, v=64):
    """A Tier-0 context: proxy logits + claimed tokens, no reference anything."""
    return VContext(prompt_id=0, claimed_tokens=list(rng.integers(0, v, t)),
                    sampling=SPEC, proxy_logits=rng.standard_normal((t, v)) * 2.0)


def _separation(w: np.ndarray, delta: np.ndarray, v: np.ndarray) -> float:
    """The batch separation a weighting realizes, from `infogain`'s derivation:
    `sum(w*Delta) / sqrt(sum(w^2 * v))` (the `sqrt(b/n)` factor is common to every
    weighting and drops out of any comparison)."""
    num = float((w * delta).sum())
    den = float(np.sqrt((w * w * v).sum()))
    return num / den if den > 0 else 0.0


# ===========================================================================
# The derivation
# ===========================================================================
def test_matched_filter_is_the_optimal_weighting():
    """`w = Delta/v` maximizes the separation; the alternatives the library used
    (unweighted mean, sensitivity-ranking) are strictly worse when `v` varies."""
    rng = np.random.default_rng(0)
    n = 200
    delta = np.abs(rng.normal(0.5, 0.3, n))
    v = np.exp(rng.normal(0.0, 1.2, n))          # heterogeneous honest noise
    best = _separation(delta / v, delta, v)

    # the optimum equals sqrt(sum I) exactly (Cauchy-Schwarz equality case)
    assert abs(best - np.sqrt((delta ** 2 / v).sum())) < 1e-9

    assert best > _separation(np.ones(n), delta, v)         # unweighted mean
    assert best > _separation(delta, delta, v)              # sensitivity weighting
    assert best > _separation(1.0 / v, delta, v)            # inverse-variance only
    for _ in range(200):                                     # and any random weighting
        assert best >= _separation(np.abs(rng.normal(size=n)), delta, v) - 1e-12

    # The two corners that say exactly when the older choices were right:
    #  * homogeneous noise -> the matched filter IS sensitivity weighting, and
    #    ranking by I is ranking by Delta. So sensitivity-based triage is optimal
    #    precisely when v is constant across positions.
    vc = np.full(n, 0.7)
    assert abs(_separation(delta / vc, delta, vc)
               - _separation(delta, delta, vc)) < 1e-9
    assert np.array_equal(np.argsort(-(delta ** 2 / vc)), np.argsort(-delta))
    #  * the unweighted mean is optimal only when Delta/v is constant, i.e. a
    #    position's sensitivity is exactly proportional to its honest noise.
    v_prop = delta / 3.0
    assert abs(_separation(np.ones(n), delta, v_prop)
               - _separation(delta / v_prop, delta, v_prop)) < 1e-9


def test_top_k_by_information_is_the_optimal_selection():
    """Under the matched filter, separation^2 is additive over audited tokens, so
    the best size-k audit is the top k by `I`. Checked against every subset."""
    rng = np.random.default_rng(1)
    n, k = 10, 4
    delta = np.abs(rng.normal(0.5, 0.4, n))
    v = np.exp(rng.normal(0.0, 1.0, n))
    info = delta ** 2 / v

    def sep_of(idx):
        w = np.zeros(n)
        w[list(idx)] = (delta / v)[list(idx)]
        return _separation(w, delta, v)

    top_k = np.argsort(-info)[:k]
    best_subset = max(itertools.combinations(range(n), k), key=sep_of)
    assert set(top_k) == set(best_subset), (sorted(top_k), sorted(best_subset))
    # additivity: the separation of a set is sqrt of its summed information
    assert abs(sep_of(top_k) - np.sqrt(info[top_k].sum())) < 1e-9
    # ...and ranking by sensitivity alone picks a WORSE set than ranking by I
    top_delta = np.argsort(-delta)[:k]
    assert sep_of(top_k) > sep_of(top_delta)
    # `signal.info_capture` is exactly the ratio of squared separations
    mask = np.zeros(n, bool)
    mask[top_k] = True
    cap = signal.info_capture(info, mask)
    assert abs(cap - (sep_of(top_k) / sep_of(range(n))) ** 2) < 1e-9


# ===========================================================================
# The estimator
# ===========================================================================
class _StubVerifier:
    """A scorer with ANALYTIC probe moments, for testing the estimator itself.

    `score = a*z^2 + b*z` at the claimed coordinate `z`, with the pair `(a, b)`
    read out of two reserved coordinates of the row (perturbations of the probe
    scale cannot change `round`). For an honest row with `z = 0`:

        m = a*sigma_b^2 ,  v = 2*a^2*sigma_b^4 + b^2*sigma_b^2 ,
        Delta = a*(sigma_d^2 - sigma_b^2)

    so `b` moves the honest NOISE while leaving the sensitivity `Delta` untouched --
    exactly the confound `Delta`-ranking cannot see and `I = Delta^2/v` can.
    """

    def score_token(self, row, gumbel, claimed, sampling, *a, **kw):
        gain, noise = round(float(row[-1])), round(float(row[-2]))
        z = float(row[claimed])
        return gain * z * z + noise * z


def test_probe_separates_planted_sensitivity_from_planted_noise():
    """`probe_moments` recovers analytic `(m, v, Delta)`, and the information it
    reports ranks by signal-to-noise where `delta` alone is blind."""
    t, v_sz = 200, 8
    ref = np.zeros((t, v_sz))
    loud = np.zeros(t, bool)
    loud[::2] = True                       # same sensitivity, 5x the honest noise
    ref[:, -1] = 1.0                       # a = 1 everywhere
    ref[loud, -2] = 5.0                    # b = 5 on the loud half
    claimed = [0] * t                      # score reads coordinate 0, which is 0.0
    sb, sd = 0.10, 0.30
    mom = infogain.probe_moments(ref, np.zeros((t, v_sz)), claimed, SPEC,
                                 _StubVerifier(), np.random.default_rng(3),
                                 probe_sigma=np.sqrt(sd ** 2 - sb ** 2),
                                 benign_sigma=sb, n_probe=400)
    for key in ("m", "v", "delta", "info"):
        assert mom[key].shape == (t,) and np.isfinite(mom[key]).all(), key
    assert (mom["v"] >= 0).all()

    # analytic values, within Monte-Carlo error at 400 probes
    assert abs(mom["m"].mean() - sb ** 2) < 0.3 * sb ** 2
    assert abs(mom["delta"].mean() - (sd ** 2 - sb ** 2)) < 0.3 * (sd ** 2 - sb ** 2)
    assert abs(mom["v"][~loud].mean() - 2 * sb ** 4) < 1.0 * sb ** 4
    assert abs(mom["v"][loud].mean() - (2 * sb ** 4 + 25 * sb ** 2)) < 0.3 * 25 * sb ** 2

    # The planted structure: sensitivity is FLAT across the two halves, so a
    # Delta-ranking cannot tell them apart, while the information ranking puts
    # every quiet position above every loud one.
    assert abs(mom["delta"][loud].mean() / mom["delta"][~loud].mean() - 1.0) < 0.25
    assert mom["info"][~loud].min() > mom["info"][loud].max()


def test_tie_positions_are_the_noisy_ones_on_real_score_geometry():
    """The empirical seed of this module's thesis, on the real `token_difr` score:
    forcing the top-two logits together raises the honest VARIANCE by orders of
    magnitude -- near-tie-ness (`tie_margin`, the library's quant-tuned value
    signal) is largely a predictor of nondeterminism, and nondeterminism is the
    denominator of the information, not the numerator."""
    rng = np.random.default_rng(2)
    t, v_sz = 120, 48
    ref = rng.standard_normal((t, v_sz)) * 3.0
    tie = np.zeros(t, bool)
    tie[::2] = True
    for i in np.nonzero(tie)[0]:
        order = np.argsort(-ref[i])
        ref[i, order[1]] = ref[i, order[0]] - 0.01
    gum = rng.standard_normal((t, v_sz))
    claimed = [int(np.argmax(ref[i] + gum[i])) for i in range(t)]
    mom = infogain.probe_moments(ref, gum, claimed, SPEC, verifiers.get("token_difr"),
                                 np.random.default_rng(3), probe_sigma=0.3,
                                 benign_sigma=0.05, n_probe=24)
    assert mom["v"][tie].mean() > 100 * (mom["v"][~tie].mean() + 1e-6)
    assert mom["m"][tie].mean() > mom["m"][~tie].mean()
    assert np.isfinite(mom["info"]).all()


def test_info_model_is_tier0_and_serializable():
    """The fitted model reads proxy features only, plugs into the value registry,
    and round-trips through JSON with a feature-layout guard."""
    rng = np.random.default_rng(4)
    ctx = _ctx(rng, t=200)
    x = triage.feature_matrix(ctx)
    n = len(x)
    # a planted structure the ridge can find: sensitivity rises with tie-ness,
    # noise rises with entropy
    tie = x[:, triage.FEATURE_NAMES.index("tie_margin")]
    ent = x[:, triage.FEATURE_NAMES.index("entropy")]
    z = lambda c: (c - c.mean()) / (c.std() + 1e-9)
    # `delta` is planted strictly positive: `sensitivity` floors at 0 (a negative
    # predicted shift carries no one-sided evidence), so a target that goes negative
    # would be measuring the floor rather than the fit.
    mom = {"m": 0.1 * ent, "delta": 3.0 + z(tie), "v": np.exp(0.5 + z(ent))}
    mom["info"] = mom["delta"] ** 2 / mom["v"]
    model = infogain.InfoModel().fit(x, mom)
    assert model.fit_report["r2_delta"] > 0.9 and model.fit_report["r2_log_v"] > 0.9
    assert model.fit_report["spearman_info"] > 0.8
    assert (model.noise(x) > 0).all() and (model.sensitivity(x) >= 0).all()
    assert np.allclose(model.info(x), model.sensitivity(x) ** 2 / model.noise(x))
    assert np.allclose(model.weights(x), model.sensitivity(x) / model.noise(x))

    # Tier-0: scoring a context with no reference fields at all must work, and a
    # context with no proxy must raise (the invariant `feature_matrix` enforces).
    verifiers.register_value_fn("info_gain", infogain.info_value_fn(model))
    got = verifiers.value_of("info_gain", ctx)
    assert got.shape == (n,) and np.allclose(got, model.info(x))
    assert ctx.ref_logits is None and ctx.gumbel is None

    rebuilt = infogain.InfoModel.from_dict(model.to_dict())
    assert np.allclose(rebuilt.info(x), model.info(x))
    bad = model.to_dict()
    bad["feature_names"] = ["entropy"]
    try:
        infogain.InfoModel.from_dict(bad)
        raise AssertionError("expected a ValueError on a different feature layout")
    except ValueError:
        pass


def test_realized_info_is_an_oracle_ceiling():
    """The labeled-pair oracle marks the positions that actually carry signal."""
    rng = np.random.default_rng(5)
    n = 600
    honest = np.abs(rng.normal(0.0, 1.0, n))
    attack = honest.copy()
    hot = honest > np.quantile(honest, 0.8)        # signal only in the loud tail
    attack[hot] += 3.0
    info = infogain.realized_info(honest, attack, n_bins=10)
    assert info[hot].mean() > 5 * info[~hot].mean()


def _planted_moments(seed: int = 11, n: int = 20000):
    """A two-group population with known `(m, v, Delta)`: a quiet half (small mean,
    small variance) and a loud half (large mean, large variance), with the same
    kind of shift in each. Means are far apart so an equal-count binning never
    mixes the two groups."""
    rng = np.random.default_rng(seed)
    quiet = np.arange(n) < n // 2
    m_true = np.where(quiet, 0.0, 10.0)
    v_true = np.where(quiet, 0.01, 1.00)
    d_true = np.where(quiet, 0.05, 0.40)
    h = rng.normal(m_true, np.sqrt(v_true))
    a = rng.normal(m_true + d_true, np.sqrt(v_true))
    return quiet, m_true, v_true, d_true, h, a


def test_realized_moments_understates_v_when_binned_on_the_score_itself():
    """The default key is the honest score, so `realized_moments` conditions on the
    very quantity whose spread it is measuring: each bin is a narrow slice of the
    score and `v` comes back far BELOW the true conditional variance.

    This is the documented bias in `realized_moments`' docstring, and it is why
    that function is a diagnostic and `oracle_model` is the ceiling arm. Pinned as
    a test because the bias is silent -- the returned array looks like a variance,
    it just is not the right one -- and because a `Delta/v` weight built on it
    diverges as `n_bins` grows, which would make an unbounded "ceiling" look like
    a real result.
    """
    quiet, m_true, v_true, d_true, h, a = _planted_moments()
    mom = infogain.realized_moments(h, a, n_bins=20)

    # m and Delta ARE recovered: they are between-bin quantities.
    for side, sel in (("quiet", quiet), ("loud", ~quiet)):
        assert abs(mom["m"][sel].mean() - m_true[sel].mean()) < 0.05, side
        assert abs(mom["delta"][sel].mean() - d_true[sel].mean()) < 0.1, side

    # v is NOT: on the loud half the within-bin spread is an order of magnitude
    # below the planted 1.0, because the bins slice the score finely.
    v_loud = mom["v"][~quiet].mean()
    assert v_loud < 0.2 * v_true[~quiet].mean(), v_loud

    # ... and it keeps shrinking with more bins, so no finite `n_bins` fixes it.
    v_finer = infogain.realized_moments(h, a, n_bins=80)["v"][~quiet].mean()
    assert v_finer < v_loud, (v_loud, v_finer)


def test_realized_moments_recovers_v_given_an_independent_key():
    """The fix the docstring names: bin on a key that is not the score being
    measured. With a key that separates the two groups but carries no information
    about the within-group draw, all three moments come back."""
    quiet, m_true, v_true, d_true, h, a = _planted_moments()
    key = np.where(quiet, 0.0, 1.0)          # group label: independent of the draw
    mom = infogain.realized_moments(h, a, n_bins=2, key=key)
    for side, sel in (("quiet", quiet), ("loud", ~quiet)):
        assert abs(mom["m"][sel].mean() - m_true[sel].mean()) < 0.05, side
        assert abs(mom["v"][sel].mean() - v_true[sel].mean()) < \
            0.1 * v_true[sel].mean() + 0.02, (side, mom["v"][sel].mean())
        assert abs(mom["delta"][sel].mean() - d_true[sel].mean()) < 0.1, side
    # Information is larger where the shift is large relative to the noise: the
    # quiet half has a 5x smaller shift but a 100x smaller variance.
    assert mom["info"][quiet].mean() > mom["info"][~quiet].mean()


def test_oracle_matched_filter_is_a_ceiling_for_the_plain_mean():
    """End-to-end through `evaluate`: weights read off labeled pairs recover the
    same heteroscedastic signal the true weights do, and the plain mean misses.

    This is the arm that makes a deployable negative diagnosable -- if it did not
    beat the mean here, the derivation would be what is wrong, not the estimator.
    """
    rng = np.random.default_rng(12)
    n = 20000
    quiet = np.arange(n) < n // 2
    v = np.where(quiet, 0.04, 4.0)
    m = np.where(quiet, 0.0, 10.0)                 # separated so the bins are clean
    h = rng.normal(m, np.sqrt(v))
    a = rng.normal(m + 0.05, np.sqrt(v))           # same shift in both halves
    td = verifiers.get("token_difr")
    w_o, m_o = infogain.oracle_matched_filter(h, a, n_bins=20)
    assert np.all(w_o >= 0.0)                      # one-sided: no negative weights
    assert w_o[quiet].mean() > 10 * w_o[~quiet].mean()
    assert np.allclose(m_o[quiet].mean(), 0.0, atol=0.05)

    plain = harness.evaluate(harness.TokenScores("honest", {td.name: h}),
                             harness.TokenScores("atk", {td.name: a}), [td], [500],
                             winsor_pct=None)[0]
    oracle = harness.evaluate(
        harness.TokenScores("honest", {td.name: h}, weights=w_o, baseline=m_o),
        harness.TokenScores("atk", {td.name: a}, weights=w_o, baseline=m_o),
        [td], [500], winsor_pct=None)[0]
    assert oracle.auc > plain.auc + 0.2, (oracle.auc, plain.auc)


# ===========================================================================
# The harness path
# ===========================================================================
def test_weighted_aggregation_defaults_to_bit_identical():
    """Every result in this repo was measured with the unweighted mean, so the
    default path must not move by a single bit -- and `weights=1, baseline=0` must
    reproduce it exactly too."""
    rng = np.random.default_rng(6)
    n = 4000
    h = np.abs(rng.normal(0.0, 1.0, n))
    a = np.abs(rng.normal(0.25, 1.0, n))
    td = verifiers.get("token_difr")
    plain = harness.evaluate(harness.TokenScores("honest", {td.name: h}),
                             harness.TokenScores("atk", {td.name: a}), [td], [200])[0]
    ones = harness.evaluate(
        harness.TokenScores("honest", {td.name: h}, weights=np.ones(n),
                            baseline=np.zeros(n)),
        harness.TokenScores("atk", {td.name: a}, weights=np.ones(n),
                            baseline=np.zeros(n)), [td], [200])[0]
    assert plain.auc == ones.auc and plain.tpr == ones.tpr

    # A global positive rescaling of the statistic cannot move a threshold-free
    # metric, which is why `info`/`weights` are only ever used up to a constant.
    scaled = harness.evaluate(
        harness.TokenScores("honest", {td.name: h}, weights=np.full(n, 17.0)),
        harness.TokenScores("atk", {td.name: a}, weights=np.full(n, 17.0)),
        [td], [200])[0]
    assert abs(scaled.auc - plain.auc) < 1e-12

    # A wrongly-sized weight array is a programming error, not a broadcast.
    try:
        harness.evaluate(harness.TokenScores("honest", {td.name: h}, weights=np.ones(7)),
                         harness.TokenScores("atk", {td.name: a}), [td], [200])
        raise AssertionError("expected a ValueError on a short weight array")
    except ValueError:
        pass


def test_matched_filter_recovers_signal_a_plain_mean_misses():
    """End-to-end through `evaluate`: with heteroscedastic honest noise, the
    matched filter detects a deviation the unweighted mean cannot. This is the
    aggregation half of P1 in `infogain`'s docstring, on the real metric."""
    rng = np.random.default_rng(7)
    n = 20000
    quiet = np.zeros(n, bool)
    quiet[: n // 2] = True                       # half the pool is low-noise
    v = np.where(quiet, 0.04, 4.0)
    delta = np.where(quiet, 0.05, 0.05)          # same shift everywhere
    m = np.zeros(n)
    h = rng.normal(m, np.sqrt(v))
    a = rng.normal(m + delta, np.sqrt(v))
    td = verifiers.get("token_difr")
    w = delta / v

    plain = harness.evaluate(harness.TokenScores("honest", {td.name: h}),
                             harness.TokenScores("atk", {td.name: a}), [td], [500],
                             winsor_pct=None)[0]
    mf = harness.evaluate(
        harness.TokenScores("honest", {td.name: h}, weights=w, baseline=m),
        harness.TokenScores("atk", {td.name: a}, weights=w, baseline=m),
        [td], [500], winsor_pct=None)[0]
    assert mf.auc > plain.auc + 0.2, (mf.auc, plain.auc)

    # and the realized separation matches the theory prediction for this pool
    d_prime = float(np.sqrt((delta ** 2 / v).mean()))
    assert abs(mf.auc - signal.predicted_pauc(d_prime, 500)) < 0.1


def test_rescore_zeroes_weights_off_the_audit_mask():
    """An unaudited token must contribute exactly 0 under the matched filter --
    that is what makes the `neutral` padding (and the winsorization pathology it
    caused) harmless by construction."""
    rng = np.random.default_rng(8)
    n = 500
    full = harness.TokenScores("honest", {"token_difr": rng.random(n)})
    values = rng.random(n)
    w = rng.random(n) + 0.5
    out = harness.rescore_at_budget(full, [verifiers.get("token_difr")], values, 0.2,
                                   weights=w, baseline=np.zeros(n))
    assert out.weights is not None and out.audited is not None
    assert np.all(out.weights[~out.audited] == 0.0)
    assert np.allclose(out.weights[out.audited], w[out.audited])


# ===========================================================================
# The signal-strength model
# ===========================================================================
def test_gaussian_batch_model_round_trips():
    """`delta_for_pauc` inverts `pauc_of_delta`, and still returns the 3.767 the
    repo's earlier hard-coded literal came from."""
    d90 = signal.delta_for_pauc(0.90, 0.005)
    assert abs(d90 - 3.767) < 0.001, d90
    assert abs(signal.pauc_of_delta(d90, 0.005) - 0.90) < 1e-4
    for target in (0.60, 0.75, 0.95, 0.99):
        d = signal.delta_for_pauc(target, 0.005)
        assert abs(signal.pauc_of_delta(d, 0.005) - target) < 1e-4, target
    # monotone in delta, and chance at zero separation
    assert abs(signal.pauc_of_delta(0.0) - 0.5) < 1e-6
    grid = [signal.pauc_of_delta(x) for x in (0.5, 1.0, 2.0, 4.0, 8.0)]
    assert all(b > a for a, b in zip(grid, grid[1:]))
    # a batch of b tokens separates by d'*sqrt(b)
    assert signal.batch_for_pauc(d90) == 1
    assert signal.batch_for_pauc(d90 / 10.0) == 100
    assert signal.batch_for_pauc(-1.0) == -1
    # normal helpers agree with known values
    assert abs(signal.ndtri(0.975) - 1.959963985) < 1e-6
    assert abs(signal.ndtr(1.959963985) - 0.975) < 1e-9


def test_capture_curve_predicts_the_budget_tradeoff():
    """Half the information detects at `d'/sqrt(2)`, not at half the AUC -- the
    compute-accuracy curve `pauc_of_capture` states."""
    d_full, batch = 0.20, 400
    assert abs(signal.pauc_of_capture(1.0, d_full, batch)
               - signal.predicted_pauc(d_full, batch)) < 1e-12
    half = signal.pauc_of_capture(0.5, d_full, batch)
    assert abs(half - signal.predicted_pauc(d_full / np.sqrt(2), batch)) < 1e-12
    assert 0.5 < half < signal.predicted_pauc(d_full, batch)
    assert abs(signal.pauc_of_capture(0.0, d_full, batch) - 0.5) < 1e-6
    # a uniformly-informative pool captures exactly its budget; a concentrated one
    # captures more (which is the entire premise of triage)
    flat = np.ones(1000)
    mask = np.zeros(1000, bool)
    mask[:100] = True
    assert abs(signal.info_capture(flat, mask) - 0.1) < 1e-9
    spiky = np.concatenate([np.full(100, 10.0), np.ones(900)])
    assert signal.info_capture(spiky, mask) > 0.5


def test_per_token_stats_matches_the_evaluate_scale():
    """`per_token_stats` winsorizes exactly like `evaluate` before taking d'."""
    rng = np.random.default_rng(9)
    h = np.abs(rng.normal(0, 1, 5000))
    a = h + 0.1
    st = signal.per_token_stats(h, a, winsor_pct=99.9)
    cap = np.percentile(h, 99.9)
    hw, aw = np.minimum(h, cap), np.minimum(a, cap)
    assert abs(st["d_prime"] - (aw.mean() - hw.mean()) / hw.std()) < 1e-9
    raw = signal.per_token_stats(h, a, winsor_pct=None)
    assert abs(raw["d_prime"] - (a.mean() - h.mean()) / h.std()) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
