"""Tests for the provider-side speculative decoder (ivgym.spec_server).

Dependency-free (pure numpy, no GPU), same style as test_proxy_spec.py. Run:
    python tests/test_spec_server.py          # or: python -m pytest tests/ -q

The load-bearing test is `test_exact_rule_preserves_the_target_distribution`: it
Monte-Carlos the sampler and checks the emitted-token law equals `p` to within
sampling error. Everything the DiFR false-positive experiment claims rests on the
speculative server being *honest*, so that claim is established here, on CPU,
before a GPU is touched -- if the sampler were subtly wrong the false positive
would be a true positive and the whole result would invert.

Also covers: that the `lenient` shortcut genuinely shifts the served distribution
toward the draft (so it is a real deviation, not a cosmetic one); that the round
structure emits `n_accepted + 1` tokens; that the acceptance rate matches
`1 - TV(p, q)`; that seeded uniforms are replayable; and that `spec_probs` agrees
with what `sampling.gumbel_max_sample` draws from, which is what makes the two
arms of the experiment comparable at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import spec_server as ss
from ivgym.core import SamplingSpec
from ivgym.sampling import gumbel_max_sample, gumbel_noise

VOCAB = 32


def _pair(rng, vocab=VOCAB, agreement=0.7, sharpness=1.5):
    """A correlated (target, draft) distribution pair, like a same-family
    draft/target couple: `q` is a blend of `p`'s logits and independent noise."""
    t = rng.standard_normal(vocab) * sharpness
    d = agreement * t + (1.0 - agreement) * rng.standard_normal(vocab) * sharpness
    p = np.exp(t - t.max()); p /= p.sum()
    q = np.exp(d - d.max()); q /= q.sum()
    return p, q


def _emit_empirical(p, q, rule, n=400_000, seed=0):
    """Empirical law of one emitted token under `rule`, by Monte Carlo."""
    rng = np.random.default_rng(seed)
    counts = np.zeros(len(p))
    xs = rng.choice(len(q), size=n, p=q)
    us = rng.random((n, 2))
    for x, (ua, us_) in zip(xs, us):
        tok, _ = ss.emit_token(p, q, int(x), float(ua), float(us_), rule)
        counts[tok] += 1
    return counts / n


# ---------------------------------------------------------------------------
# The honesty certificate
# ---------------------------------------------------------------------------
def test_exact_rule_preserves_the_target_distribution():
    """The exact accept/reject rule emits tokens distributed as `p`, not as `q`.

    This is the claim that makes the false positive a false positive. Checked at
    three draft qualities, including a deliberately bad draft (agreement 0.2)
    where nearly everything is rejected and the residual path carries the mass.
    TV error is compared against the Monte-Carlo floor `~sqrt(V / n)`, not a
    hand-picked constant.
    """
    n = 400_000
    mc_floor = np.sqrt(VOCAB / n)          # ~0.009 at V=32, n=4e5
    for agreement in (0.2, 0.7, 0.95):
        rng = np.random.default_rng(int(agreement * 100))
        p, q = _pair(rng, agreement=agreement)
        emp = _emit_empirical(p, q, ss.AcceptRule("exact"), n=n, seed=7)
        tv_pq = 0.5 * np.abs(p - q).sum()
        tv_err = 0.5 * np.abs(emp - p).sum()
        # exact to MC error, and NOT merely close because p and q are close
        assert tv_err < 3 * mc_floor, (agreement, tv_err, mc_floor)
        assert tv_err < 0.2 * max(tv_pq, 1e-9), (agreement, tv_err, tv_pq)


def test_lenient_rule_shifts_the_served_distribution_toward_the_draft():
    """The `lenient` shortcut is a genuine deviation: its emitted law is measurably
    closer to `q` (and further from `p`) than the exact rule's. Otherwise
    "detecting lenient spec decoding" would be detecting nothing."""
    rng = np.random.default_rng(3)
    p, q = _pair(rng, agreement=0.7)
    exact = _emit_empirical(p, q, ss.AcceptRule("exact"), n=200_000, seed=11)
    lenient = _emit_empirical(p, q, ss.AcceptRule("lenient", threshold=0.3),
                              n=200_000, seed=11)
    tv = lambda a, b: 0.5 * np.abs(a - b).sum()
    assert tv(lenient, p) > 5 * tv(exact, p)
    assert tv(lenient, q) < tv(exact, q)


def test_lenient_rule_accepts_strictly_more_drafts():
    """The economic motive: the cheat is cheaper. At threshold < 1 the lenient rule
    accepts every draft the exact rule could accept and more."""
    rng = np.random.default_rng(5)
    p, q = _pair(rng, agreement=0.7)
    us = np.random.default_rng(6).random(20_000)
    xs = np.random.default_rng(7).choice(len(q), size=20_000, p=q)
    ex = np.mean([ss.AcceptRule("exact").accepts(p, q, int(x), float(u))
                  for x, u in zip(xs, us)])
    le = np.mean([ss.AcceptRule("lenient", threshold=0.3).accepts(p, q, int(x), float(u))
                  for x, u in zip(xs, us)])
    assert le > ex + 0.05, (ex, le)


def test_accept_rate_matches_one_minus_tv():
    """The realized acceptance rate of the exact rule is `sum min(p,q) = 1 - TV`,
    which is the same identity `spec_decode.accept_rate` reads from the client
    side. Provider speedup and client detector are one quantity."""
    rng = np.random.default_rng(9)
    p, q = _pair(rng, agreement=0.6)
    xs = np.random.default_rng(10).choice(len(q), size=200_000, p=q)
    us = np.random.default_rng(11).random(200_000)
    realized = np.mean([ss.AcceptRule("exact").accepts(p, q, int(x), float(u))
                        for x, u in zip(xs, us)])
    assert abs(realized - ss.expected_accept_rate(p, q)) < 0.005


# ---------------------------------------------------------------------------
# Round structure
# ---------------------------------------------------------------------------
def test_round_emits_n_accepted_plus_one():
    """Every round emits exactly `n_accepted + 1` tokens -- the accepted prefix
    plus a correction, or the whole draft plus a free bonus token."""
    rng = np.random.default_rng(13)
    gamma = 4
    for trial in range(200):
        p_rows = np.stack([_pair(rng)[0] for _ in range(gamma + 1)])
        q_rows = np.stack([_pair(rng)[1] for _ in range(gamma)])
        drafted = [int(rng.integers(VOCAB)) for _ in range(gamma)]
        u = rng.random((gamma + 1, 2))
        emitted, n_acc, details = ss.speculative_round(p_rows, q_rows, drafted, u)
        assert len(emitted) == n_acc + 1
        assert len(details) == n_acc + 1
        assert emitted[:n_acc] == drafted[:n_acc]        # accepted prefix is the draft
        assert all(d["accepted"] for d in details[:n_acc])
        assert details[-1]["is_bonus"] == (n_acc == gamma)


def test_bonus_token_comes_from_the_target():
    """A full-accept round's bonus token is drawn from `p_gamma` -- it is the one
    position where the target alone chooses, and it must not be the draft's."""
    rng = np.random.default_rng(17)
    gamma = 2
    p_rows = np.zeros((gamma + 1, VOCAB)) + 1e-9
    p_rows[:, 3] = 1.0
    p_rows /= p_rows.sum(axis=1, keepdims=True)
    q_rows = p_rows[:gamma].copy()                       # q == p => always accept
    u = np.zeros((gamma + 1, 2))
    u[:, 1] = 0.5              # accept uniform 0 (always accept); draw uniform mid-CDF
    emitted, n_acc, details = ss.speculative_round(p_rows, q_rows, [3, 3], u)
    assert n_acc == gamma and emitted[-1] == 3 and details[-1]["is_bonus"]


# ---------------------------------------------------------------------------
# Replayability (what a spec-aware verifier needs)
# ---------------------------------------------------------------------------
def test_seeded_uniforms_are_replayable():
    """A seeded server's decisions are reproducible from the public position seed
    alone -- the precondition for `exp_spec_aware_verifier_gpu`'s fix."""
    a = ss.position_uniforms(12345, 2)
    b = ss.position_uniforms(12345, 2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, ss.position_uniforms(12346, 2))


def test_categorical_from_uniform_matches_the_cdf():
    """Inverse-CDF sampling from one uniform reproduces the intended law (and is
    what makes a single seeded scalar enough to replay a draw)."""
    p = np.array([0.1, 0.5, 0.4])
    assert ss.categorical_from_uniform(p, 0.05) == 0
    assert ss.categorical_from_uniform(p, 0.3) == 1
    assert ss.categorical_from_uniform(p, 0.9) == 2
    us = np.random.default_rng(0).random(60_000)
    emp = np.bincount([ss.categorical_from_uniform(p, float(u)) for u in us],
                      minlength=3) / len(us)
    assert np.max(np.abs(emp - p)) < 0.01


# ---------------------------------------------------------------------------
# The two arms must sample from the SAME distribution
# ---------------------------------------------------------------------------
def test_spec_probs_matches_gumbel_max_sampling():
    """`spec_probs` is the law `sampling.gumbel_max_sample` draws from.

    The experiment compares a Gumbel-Max server against a speculative server; if
    these two disagreed the arms would differ for a reason that has nothing to do
    with speculative decoding. Checked including top-k/top-p filtering, which both
    paths must apply identically.
    """
    rng = np.random.default_rng(21)
    logits = rng.standard_normal(VOCAB) * 2.0
    spec = SamplingSpec(temperature=0.9, top_k=8, top_p=0.95, seed=42)
    p = ss.spec_probs(logits, spec)
    assert abs(p.sum() - 1.0) < 1e-9
    assert (p > 0).sum() <= 8                                  # top-k respected
    counts = np.zeros(VOCAB)
    for i in range(60_000):
        g = gumbel_noise(VOCAB, 900_000 + i)
        counts[gumbel_max_sample(logits, spec.temperature, g, spec.top_k, spec.top_p)] += 1
    emp = counts / counts.sum()
    assert 0.5 * np.abs(emp - p).sum() < 0.01, 0.5 * np.abs(emp - p).sum()


def test_registered_configs():
    """The speculative configs land in the ordinary attack registry, so the harness
    scores them with no special-casing -- `honest_spec` sits in the same table as
    the cheats, which is the finding."""
    from ivgym import attacks
    reg = attacks.all_attacks()
    for name in ("honest_spec", "honest_spec_seeded", "spec_lenient", "spec_topk"):
        assert name in reg, name
        assert getattr(reg[name], "spec_mode", None) is not None
    assert reg["honest_spec"].accept_rule().kind == "exact"
    assert reg["spec_lenient"].accept_rule().kind == "lenient"
    assert reg["honest_spec_seeded"].seeded and not reg["honest_spec"].seeded
    # honest_spec really is honest: no logit perturbation beyond benign noise
    assert reg["honest_spec"].logit_bias_sigma() == (0.0, 0.0)
    assert reg["honest_spec"].activation_extra_sigma() == 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
