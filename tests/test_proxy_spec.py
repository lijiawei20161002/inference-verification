"""Tests for the client-side proxy verifier (ivgym.spec_decode).

Dependency-free (pure numpy), same style as test_smoke.py. Run:
    python tests/test_proxy_spec.py           # or: python -m pytest tests/ -q

Covers: the acceptance-rate identity ``accept_rate = 1 − TV``; that an honest
served sample sits at the reference floor; that the draft-anchored acceptance
rate separates a quantized target (and small quant hides under honest variance);
that the temperature-retune evasion blinds the generic entropy fingerprint but
NOT the acceptance rate; that the full recompute baseline nails every
forward-pass cheat; and that the calibrated verifier keeps a low honest FPR.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import harness
from ivgym import spec_decode as sd
from ivgym.core import SamplingSpec
from ivgym.metrics import roc_auc


def _positions(rng, n, vocab=64, agreement=0.8):
    return sd.synthetic_positions(rng, n, vocab=vocab, agreement=agreement)


def _samples(cheat, n=120, n_pos=200, seed0=0, agreement=0.8):
    out = []
    for t in range(n):
        rng = np.random.default_rng(1_000_003 * seed0 + t)
        pos = _positions(rng, n_pos, agreement=agreement)
        out.append(sd.generate_sample(rng, sd.get_cheat(cheat), pos))
    return out


def _quant(sigma, n=120, n_pos=200, seed0=0, temp=1.0):
    out = []
    for t in range(n):
        rng = np.random.default_rng(1_000_003 * seed0 + t)
        pos = _positions(rng, n_pos)
        out.append(sd.generate_sample(rng, sd.QuantTarget(name="q", sigma=sigma, temp=temp), pos))
    return out


def _feat_auc(fp, feature, honest_null, cheat):
    h = np.array([fp.score(s)[feature] for s in honest_null])
    c = np.array([fp.score(s)[feature] for s in cheat])
    return roc_auc(h, c)


def _recompute_auc(honest_null, cheat):
    h = np.array([sd.recompute_divergence(s) for s in honest_null])
    c = np.array([sd.recompute_divergence(s) for s in cheat])
    return roc_auc(h, c)


# ---------------------------------------------------------------------------
def test_accept_rate_identity():
    """accept_rate(p, q) == 1 − TV(p, q) == Σ min(p, q)."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = sd.softmax(rng.standard_normal(32) * 2)
        q = sd.softmax(rng.standard_normal(32) * 2)
        a = sd.accept_rate(p, q)
        assert abs(a - (1.0 - sd.tv(p, q))) < 1e-9
        assert abs(a - np.minimum(p, q).sum()) < 1e-9
        assert 0.0 <= a <= 1.0


def test_per_token_accept_prob():
    p = np.array([0.6, 0.4]); q = np.array([0.3, 0.7])
    assert sd.per_token_accept_prob(p, q, 0) == 1.0          # p/q = 2.0 -> capped at 1
    assert abs(sd.per_token_accept_prob(p, q, 1) - 0.4 / 0.7) < 1e-9


def test_honest_served_target_is_truth():
    """An honest provider serves the true target, so recompute divergence is
    exactly zero and the served token is drawn from the true distribution."""
    honest = _samples("honest", n=1, seed0=5)[0]
    assert sd.recompute_divergence(honest) < 1e-12
    for pos, truth in zip(honest.positions, honest._truth):
        assert np.allclose(pos.target_logprobs, truth)


def test_reference_floor_is_chance():
    """Honest vs an independent honest draw: the acceptance-rate reference sits
    near chance on every feature (read two-sided)."""
    fp = sd.ProxyReference().fit(_samples("honest", seed0=1))
    null = _samples("honest", seed0=2)
    floor = _samples("honest", seed0=3)
    for feat in ("accept_rate", "mean_entropy"):
        auc = _feat_auc(fp, feat, null, floor)
        assert max(auc, 1 - auc) < 0.7, f"{feat} floor not at chance: {auc:.3f}"


def test_accept_rate_separates_large_quant_but_not_small():
    """The draft-anchored acceptance rate separates a strongly quantized target,
    while small quant hides under honest run-to-run variance -- and the full
    recompute nails both."""
    fp = sd.ProxyReference().fit(_samples("honest", seed0=10))
    null = _samples("honest", seed0=11)
    small = _quant(0.1, seed0=12)
    large = _quant(1.0, seed0=13)
    assert _feat_auc(fp, "accept_rate", null, small) < 0.75      # invisible when small
    assert _feat_auc(fp, "accept_rate", null, large) > 0.9       # separable when large
    assert _recompute_auc(null, small) > 0.99                    # recompute sees both
    assert _recompute_auc(null, large) > 0.99


def test_accept_rate_survives_temp_retune():
    """Quantize + retune temperature to match honest entropy: the generic entropy
    fingerprint goes blind, but the acceptance rate does not (matching entropy
    does not restore TV(p̂, q)). The recompute baseline catches it regardless."""
    honest_fit = _samples("honest", seed0=20)
    null = _samples("honest", seed0=21)
    fp = sd.ProxyReference().fit(honest_fit)
    h_ent = np.mean([sd.sequence_features(s)["mean_entropy"] for s in null])

    # tune temperature so the quantized served entropy matches honest
    best_T, best_gap = 1.0, 1e9
    for T in np.linspace(0.6, 2.0, 29):
        tr = _quant(1.0, n=40, seed0=90, temp=float(T))
        gap = abs(np.mean([sd.sequence_features(s)["mean_entropy"] for s in tr]) - h_ent)
        if gap < best_gap:
            best_gap, best_T = gap, float(T)
    adv = _quant(1.0, seed0=22, temp=best_T)

    ent_auc = _feat_auc(fp, "mean_entropy", null, adv)
    acc_auc = _feat_auc(fp, "accept_rate", null, adv)
    assert ent_auc < 0.8, f"entropy fingerprint should be weakened, got {ent_auc:.3f}"
    assert acc_auc > 0.9, f"accept-rate fingerprint should survive, got {acc_auc:.3f}"
    assert _recompute_auc(null, adv) > 0.99


def test_verifier_flags_quant_and_keeps_low_honest_fpr():
    """The calibrated ProxySpecVerifier passes honest samples at ~the target FPR
    and flags a strongly quantized provider."""
    cal = _samples("honest", n=400, seed0=30)
    null = _samples("honest", n=200, seed0=31)
    v = sd.ProxySpecVerifier(feature="accept_rate").calibrate(cal, fpr=0.01)
    fpr = np.mean([v.verify(s).flagged for s in null])
    assert fpr <= 0.06, f"honest FPR too high: {fpr:.3f}"
    quant = _quant(1.0, n=100, seed0=32)
    caught = np.mean([v.verify(s).flagged for s in quant])
    assert caught > 0.8, f"verifier missed strong quant: caught {caught:.2f}"


def test_batched_auc_floor_needs_seed_averaging():
    """Regression for the exp_spec_verifier_cost floor bug: harness.evaluate does
    ONE train/test split, then batch_means concentrates each batch mean around its
    finite subset's sample mean (~1/sqrt(batch)). For two draws of the SAME
    distribution the accidental subset-mean gap is amplified into an AUC far from
    0.5 on any single seed -- so the honest-null 'floor' must be AVERAGED over
    many re-split seeds (the AUC_SEEDS loop) to land at ~0.5. Guards against
    reverting batched_auc to a single hardcoded seed."""
    rng = np.random.default_rng(123)
    x = np.abs(rng.normal(0.3, 0.08, 24 * 128))          # honest-like per-token signal
    perm = np.random.default_rng(0).permutation(len(x))
    half = len(x) // 2
    A, B = x[perm[:half]], x[perm[half:]]                 # two halves, SAME distribution

    def auc_at(seed):
        ts_h = harness.TokenScores("h", {"d": A})
        ts_a = harness.TokenScores("a", {"d": B})
        shim = type("D", (), {"name": "d"})()
        return harness.evaluate(ts_h, ts_a, [shim], [150], seed=seed)[0].auc

    single = auc_at(7)
    averaged = float(np.mean([auc_at(s) for s in range(16)]))
    # a single split is provably biased away from 0.5 (that IS the bug)...
    assert abs(single - 0.5) > 0.08, f"expected single-seed bias, got {single:.3f}"
    # ...but averaging over re-split seeds recovers the true ~0.5 floor.
    assert abs(averaged - 0.5) < 0.05, f"averaged floor not ~0.5: {averaged:.3f}"


def test_positions_from_rows_roundtrip():
    """The real-backend adapter builds Position rows from [T,V] arrays."""
    rng = np.random.default_rng(0)
    T, V = 5, 16
    p = np.stack([sd.log_softmax(rng.standard_normal(V)) for _ in range(T)])
    q = np.stack([sd.log_softmax(rng.standard_normal(V)) for _ in range(T)])
    pos = sd.positions_from_rows(p, q)
    assert len(pos) == T
    assert np.allclose(pos[0].target_logprobs, p[0])
    assert np.allclose(pos[2].proxy_logprobs, q[2])


def test_registries():
    """Cheats register and resolve to instances (class or instance registration,
    mirroring ivgym.attacks/defenses)."""
    assert "honest" in sd.all_cheats() and "quant_target" in sd.all_cheats()
    assert "adv_quant_temp" in sd.all_cheats()
    assert not isinstance(sd.get_cheat("quant_target"), type)

    @sd.register_cheat
    class _TmpCheat(sd.CheatStrategy):
        name = "tmp_cheat"

    assert not isinstance(sd.get_cheat("tmp_cheat"), type)
    del sd._CHEATS["tmp_cheat"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
