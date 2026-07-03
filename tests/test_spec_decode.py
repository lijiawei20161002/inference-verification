"""Tests for speculative-decoding trace verification (ivgym.spec_decode).

Dependency-free (pure numpy), same style as test_smoke.py. Run:
    python tests/test_spec_decode.py           # or: python -m pytest tests/ -q

Covers: the vLLM accept rule, the correctness of the honest simulator (its
marginal output equals the target distribution -- the whole point of the residual
correction), that honest traces pass every check, that each cheat is caught by the
right check, and that `draft_as_target` evades all no-recompute checks yet is
caught by the recompute spot-check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import spec_decode as sd
from ivgym.core import SamplingSpec
from ivgym.metrics import roc_auc


def _positions(rng, n, vocab=64, agreement=0.8):
    return sd.synthetic_positions(rng, n, vocab=vocab, agreement=agreement)


def _traces(cheat, n_traces=60, n_pos=200, seed0=0, agreement=0.8):
    out = []
    for t in range(n_traces):
        rng = np.random.default_rng(1_000_003 * seed0 + t)
        pos = _positions(rng, n_pos, agreement=agreement)
        out.append(sd.generate_trace(rng, sd.get_cheat(cheat), pos))
    return out


def _auc(check, honest, cheat):
    oc = lambda t: sd.make_oracle(t) if check.needs_oracle else None
    h = np.array([check.score(t, oc(t)) for t in honest])
    c = np.array([check.score(t, oc(t)) for t in cheat])
    return roc_auc(h, c)


# ---------------------------------------------------------------------------
def test_vllm_accept_rule():
    """accept <=> target_prob(x) >= u * draft_prob(x), verbatim vLLM."""
    p = np.array([0.6, 0.4]); q = np.array([0.3, 0.7])
    x = 0  # p(x)/q(x) = 2.0, so accept for any u <= 2 -> always accept
    assert bool(p[x] >= 0.99 * q[x])
    x = 1  # p(x)/q(x) = 0.571 -> accept iff u <= 0.571
    assert bool(p[x] >= 0.5 * q[x])
    assert not bool(p[x] >= 0.9 * q[x])


def test_honest_marginal_output_equals_target():
    """The honest step's marginal output distribution equals the TARGET p -- the
    speculative-sampling correctness guarantee (Leviathan/Chen). Fix (p, q), draw
    many single drafts x~q, and check the emitted-token histogram matches p."""
    rng = np.random.default_rng(0)
    vocab = 16
    p = sd._softmax(rng.standard_normal(vocab) * 2.0)
    q = sd._softmax(rng.standard_normal(vocab) * 2.0)
    p_lp, q_lp = np.log(p), np.log(q)
    honest = sd.get_cheat("honest")
    n = 200_000
    counts = np.zeros(vocab)
    for _ in range(n):
        x = int(rng.choice(vocab, p=q))
        s = honest.step(rng, p_lp, q_lp, x, 0, 0)
        counts[s.output_token] += 1
    emp = counts / n
    tv = 0.5 * np.abs(emp - p).sum()
    assert tv < 0.02, f"honest spec-decoding output TV from target = {tv:.4f}"


def test_skip_residual_breaks_marginal():
    """Control for the test above: skip_residual (resample from draft on reject)
    does NOT reproduce the target -- its output is biased toward q."""
    rng = np.random.default_rng(1)
    vocab = 16
    p = sd._softmax(rng.standard_normal(vocab) * 2.0)
    q = sd._softmax(rng.standard_normal(vocab) * 2.0)
    p_lp, q_lp = np.log(p), np.log(q)
    cheat = sd.get_cheat("skip_residual")
    n = 200_000
    counts = np.zeros(vocab)
    for _ in range(n):
        x = int(rng.choice(vocab, p=q))
        s = cheat.step(rng, p_lp, q_lp, x, 0, 0)
        counts[s.output_token] += 1
    emp = counts / n
    tv = 0.5 * np.abs(emp - p).sum()
    assert tv > 0.03, f"skip_residual should distort the target, TV={tv:.4f}"


def test_honest_trace_passes_all_checks():
    """An honest trace violates no deterministic check and is not flagged by the
    calibrated verifier (with or without the recompute oracle)."""
    honest = _traces("honest", n_traces=1, seed0=5)[0]
    assert sd.get_check("decision_consistency").score(honest) == 0.0
    assert sd.get_check("residual_support").score(honest) == 0.0
    assert sd.get_check("logprob_normalization").score(honest) < 1e-6
    # oracle-aligned: reported target == recomputed truth -> spot-check TV == 0
    spot = sd.get_check("target_spotcheck")
    assert spot.score(honest, sd.make_oracle(honest)) < 1e-9

    cal = _traces("honest", n_traces=400, seed0=6)
    for use_oracle in (False, True):
        v = sd.TraceVerifier(use_oracle=use_oracle).calibrate(cal, fpr=0.01)
        assert not v.verify(honest).flagged


def test_over_accept_naive_caught_by_decision_consistency():
    """Flipping rejects to accepts while keeping the honest coin makes the reported
    decision contradict p,q,u -- a hard, deterministic catch."""
    honest = _traces("honest", seed0=10)
    cheat = _traces("over_accept_naive", seed0=11)
    assert _auc(sd.get_check("decision_consistency"), honest, cheat) > 0.99
    # every cheat trace has at least one contradictory decision
    assert all(sd.get_check("decision_consistency").score(t) > 0 for t in cheat)


def test_coinfudge_evades_decisions_but_caught_statistically():
    """Recomputing coins to justify over-acceptance defeats decision_consistency,
    but leaves non-uniform coins and an inflated accept rate."""
    honest = _traces("honest", seed0=12)
    cheat = _traces("over_accept_coinfudge", seed0=13)
    # decisions are now internally consistent -> deterministic check blinded
    assert all(sd.get_check("decision_consistency").score(t) == 0.0 for t in cheat)
    # ... but the statistical checks separate it
    assert _auc(sd.get_check("accept_rate_consistency"), honest, cheat) > 0.9
    assert _auc(sd.get_check("coin_uniformity"), honest, cheat) > 0.75


def test_skip_residual_caught_by_residual_checks():
    honest = _traces("honest", seed0=14)
    cheat = _traces("skip_residual", seed0=15)
    assert _auc(sd.get_check("residual_support"), honest, cheat) > 0.9
    assert _auc(sd.get_check("recovered_distribution"), honest, cheat) > 0.9


def test_draft_as_target_evades_selfconsistency_but_caught_by_spotcheck():
    """The core result: relabelling the draft as the target yields a perfectly
    self-consistent trace -- every no-recompute check is at/near chance -- yet the
    recompute spot-check catches it (AUC ~ 1)."""
    honest = _traces("honest", seed0=16)
    cheat = _traces("draft_as_target", seed0=17)
    # The deployed verifier is one-sided (flags only high = anomalous scores), so
    # the relevant quantity is the one-sided AUC. draft_as_target actually makes
    # some checks *lower* than honest (e.g. accept_rate z ~ 0, since all-accept is
    # perfectly expected under p==q) -- a reversed, two-sided signal that is not
    # actionable under our rule and is trivially defeated by a relabel attacker who
    # perturbs p slightly off q. What matters: no no-recompute check flags it.
    for name in ["decision_consistency", "residual_support", "coin_uniformity",
                 "accept_rate_consistency", "recovered_distribution",
                 "logprob_normalization"]:
        auc = _auc(sd.get_check(name), honest, cheat)
        assert auc < 0.75, f"{name} unexpectedly flags draft_as_target ({auc:.3f})"
    assert _auc(sd.get_check("target_spotcheck"), honest, cheat) > 0.99

    cal = _traces("honest", n_traces=400, seed0=18)
    cheap = sd.TraceVerifier(use_oracle=False).calibrate(cal, fpr=0.01)
    full = sd.TraceVerifier(use_oracle=True).calibrate(cal, fpr=0.01)
    t = cheat[0]
    assert not cheap.verify(t).flagged        # invisible without recompute
    assert full.verify(t).flagged             # caught with the spot-check


def test_sampling_bug_caught_without_recompute():
    """SD analogue of bug_k2/bug_k32: a sampler that hijacks the emitted token is a
    *procedure* deviation, so it is caught by trace self-consistency (no recompute):
    an accepted step whose output != drafted token is a hard contradiction."""
    honest = _traces("honest", seed0=30)
    cheat = _traces("sampling_bug", seed0=31)
    assert _auc(sd.get_check("decision_consistency"), honest, cheat) > 0.9
    cal = _traces("honest", n_traces=400, seed0=32)
    cheap = sd.TraceVerifier(use_oracle=False).calibrate(cal, fpr=0.01)
    assert cheap.verify(cheat[0]).flagged           # caught WITHOUT recompute


def test_quant_target_needs_recompute():
    """SD analogue of quant_4bit/kv_fp8: a *forward-pass* deviation corrupts the
    target logits themselves. The rejection sampling on those corrupted logits is
    textbook, so the trace is self-consistent and EVERY no-recompute check is at
    chance -- only target_spotcheck (recompute the true target) catches it."""
    honest = _traces("honest", seed0=33)
    cheat = _traces("quant_target", seed0=34)
    for name in ["decision_consistency", "residual_support", "coin_uniformity",
                 "accept_rate_consistency", "recovered_distribution",
                 "logprob_normalization"]:
        auc = _auc(sd.get_check(name), honest, cheat)
        assert auc < 0.75, f"{name} unexpectedly flags quant_target ({auc:.3f})"
    assert _auc(sd.get_check("target_spotcheck"), honest, cheat) > 0.99
    cal = _traces("honest", n_traces=400, seed0=35)
    cheap = sd.TraceVerifier(use_oracle=False).calibrate(cal, fpr=0.01)
    full = sd.TraceVerifier(use_oracle=True).calibrate(cal, fpr=0.01)
    assert not cheap.verify(cheat[0]).flagged       # invisible without recompute
    assert full.verify(cheat[0]).flagged            # caught by the spot-check


def test_fingerprint_needs_external_anchor_and_scales_with_quant():
    """The offline reference fingerprint (a non-self-consistency, no-per-request-
    recompute detector) is at the floor vs an honest draw and vs *small* quant, but
    separates *large* quant. It needs a trusted honest reference -- the external
    anchor self-consistency lacks."""
    honest_fit = _traces("honest", n_traces=120, seed0=40)
    honest_null = _traces("honest", n_traces=120, seed0=41)
    honest_floor = _traces("honest", n_traces=120, seed0=42)
    fp = sd.ReferenceFingerprint().fit(honest_fit)

    def feat_auc(feature, cheat_traces):
        h = np.array([fp.score(t)[feature] for t in honest_null])
        c = np.array([fp.score(t)[feature] for t in cheat_traces])
        return roc_auc(h, c)

    # floor: honest vs honest ~ chance on both features
    assert max(feat_auc("mean_entropy", honest_floor),
               1 - feat_auc("mean_entropy", honest_floor)) < 0.7

    def quant(sig, seed0):
        out = []
        for t in range(120):
            rng = np.random.default_rng(1_000_003 * seed0 + t)
            pos = _positions(rng, 200)
            out.append(sd.generate_trace(rng, sd.QuantTarget(name=f"q{sig}", sigma=sig), pos))
        return out

    small = quant(0.1, 43)
    large = quant(1.0, 44)
    # small quant hides under honest variance for the no-recompute fingerprint...
    assert feat_auc("mean_entropy", small) < 0.7
    # ...large quant is separable, and the spot-check nails both.
    assert feat_auc("accept_rate", large) > 0.9
    assert _auc(sd.get_check("target_spotcheck"), honest_null, small) > 0.99


def test_accept_rate_fingerprint_survives_temp_retune():
    """The classic evasion (quantize + retune temperature to match honest entropy)
    kills the generic entropy fingerprint but NOT the SD-specific acceptance-rate
    fingerprint -- matching entropy does not restore TV(p_hat, q), which the trusted
    draft anchors. The recompute spot-check catches it regardless."""
    honest_fit = _traces("honest", n_traces=120, seed0=45)
    honest_null = _traces("honest", n_traces=120, seed0=46)
    fp = sd.ReferenceFingerprint().fit(honest_fit)

    def build(cheat, seed0):
        out = []
        for t in range(120):
            rng = np.random.default_rng(1_000_003 * seed0 + t)
            pos = _positions(rng, 200)
            out.append(sd.generate_trace(rng, cheat, pos))
        return out

    def feat_auc(feature, cheat_traces):
        h = np.array([fp.score(t)[feature] for t in honest_null])
        c = np.array([fp.score(t)[feature] for t in cheat_traces])
        return roc_auc(h, c)

    adv = build(sd.AdvQuantTemp(name="adv", sigma=1.0, temp=1.1), 47)
    ent_auc = feat_auc("mean_entropy", adv)
    acc_auc = feat_auc("accept_rate", adv)
    assert ent_auc < 0.8, f"entropy fingerprint should be weakened, got {ent_auc:.3f}"
    assert acc_auc > 0.9, f"accept-rate fingerprint should survive, got {acc_auc:.3f}"
    assert _auc(sd.get_check("target_spotcheck"), honest_null, adv) > 0.99


def test_calibrated_honest_fpr_is_low():
    """The Bonferroni-corrected verifier keeps the honest false-positive rate near
    the target across a held-out null."""
    cal = _traces("honest", n_traces=400, seed0=20)
    null = _traces("honest", n_traces=200, seed0=21)
    v = sd.TraceVerifier(use_oracle=True).calibrate(cal, fpr=0.01)
    fpr = np.mean([v.verify(t).flagged for t in null])
    assert fpr <= 0.06, f"honest FPR too high: {fpr:.3f}"


def test_registries():
    """Cheats and checks are registered and usable as instances (class or instance
    registration, mirroring ivgym.attacks/defenses)."""
    assert "honest" in sd.all_cheats() and "draft_as_target" in sd.all_cheats()
    assert "decision_consistency" in sd.all_checks()
    assert not isinstance(sd.get_cheat("skip_residual"), type)
    assert not isinstance(sd.get_check("coin_uniformity"), type)

    @sd.register_cheat
    class _TmpCheat(sd.CheatStrategy):
        name = "tmp_cheat"

    @sd.register_check
    class _TmpCheck(sd.Check):
        name = "tmp_check"
        def score(self, trace, oracle=None):
            return 0.0

    assert not isinstance(sd.get_cheat("tmp_cheat"), type)
    assert not isinstance(sd.get_check("tmp_check"), type)
    del sd._CHEATS["tmp_cheat"], sd._CHECKS["tmp_check"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
