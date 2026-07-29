"""Tests for the learned triage head (ivgym.triage) and the prefill cost model
plus prefix scheduler (ivgym.harness).

Dependency-free (pure numpy), same style as test_smoke.py. Run:
    python tests/test_triage_and_cost.py      # or: python -m pytest tests/ -q

Covers, for the head: that features are Tier-0 (proxy only, no ref_logits), that
the head recovers a signal planted in one feature, that STS improves calibration
without reordering within a bucket, and that `head_value_fn` plugs into the
`verifiers` value registry. For the cost model: the prefill-cost identity, that a
scattered top-k audit really costs near a full audit (the accounting gap this
whole thing exists to expose), that the prefix scheduler respects its cost
budget, and that `select_triaged` breaks ties randomly so `uniform` is a genuine
random-subsample control rather than "audit the first k tokens".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import harness, triage, verifiers
from ivgym.core import SamplingSpec, VContext

SPEC = SamplingSpec()


def _ctx(rng, t=40, v=64):
    """A Tier-0 context: proxy logits + claimed tokens, no reference anything."""
    return VContext(prompt_id=0, claimed_tokens=list(rng.integers(0, v, t)),
                    sampling=SPEC, proxy_logits=rng.standard_normal((t, v)) * 2.0)


# ===========================================================================
# The head
# ===========================================================================
def test_features_are_tier0_and_well_formed():
    """`feature_matrix` reads only Tier-0 fields, and every column is finite."""
    rng = np.random.default_rng(0)
    ctx = _ctx(rng)
    x = triage.feature_matrix(ctx)
    assert x.shape == (len(ctx.claimed_tokens), len(triage.FEATURE_NAMES))
    assert np.isfinite(x).all()
    # It must not need -- or silently read -- any Tier-1 field.
    assert ctx.ref_logits is None and ctx.gumbel is None
    bare = VContext(0, ctx.claimed_tokens, SPEC)
    try:
        triage.feature_matrix(bare)
        raise AssertionError("expected a RuntimeError without proxy logits")
    except RuntimeError:
        pass
    # entropy and tie_margin columns must agree with the hand-crafted signals
    # they generalize (same definitions, vectorized).
    i_ent = triage.FEATURE_NAMES.index("entropy")
    i_tie = triage.FEATURE_NAMES.index("tie_margin")
    assert np.allclose(x[:, i_ent], verifiers.value_of("entropy", ctx), atol=1e-9)
    assert np.allclose(x[:, i_tie], verifiers.value_of("tie_margin", ctx), atol=1e-9)


def test_head_recovers_a_planted_signal():
    """With sensitivity driven by one feature, the head learns to rank by it."""
    rng = np.random.default_rng(1)
    xs = [triage.feature_matrix(_ctx(rng)) for _ in range(30)]
    x = np.concatenate(xs)
    i = triage.FEATURE_NAMES.index("surprisal")
    sens = x[:, i] + rng.normal(0, 0.05, len(x))
    head = triage.ConfidenceHead().fit(x, sens)
    # BCE went down, and the driving feature carries the largest coefficient.
    assert head.history[-1] < head.history[0] - 0.05
    w = np.abs(head.w)
    assert int(np.argmax(w)) == i, dict(zip(triage.FEATURE_NAMES, head.w.round(3)))
    # ...and the head's ranking correlates with the true sensitivity.
    p = head.score(x)
    hi, lo = sens > np.quantile(sens, 0.8), sens < np.quantile(sens, 0.2)
    assert p[hi].mean() > p[lo].mean() + 0.3


def test_sts_calibrates_and_preserves_within_bucket_order():
    """STS lowers ECE; being monotone per bucket it cannot reorder within one."""
    rng = np.random.default_rng(2)
    x = np.concatenate([triage.feature_matrix(_ctx(rng)) for _ in range(40)])
    i = triage.FEATURE_NAMES.index("entropy")
    sens = 3.0 * x[:, i] + rng.normal(0, 0.3, len(x))
    head = triage.ConfidenceHead().fit(x, sens)
    pos = x[:, triage.FEATURE_NAMES.index("rel_position")]
    y = (sens > np.quantile(sens, head.label_quantile)).astype(float)

    p_raw = head.score(x, pos)
    _, _, _, ece_raw = triage.reliability(p_raw, y)
    head.calibrate(x, sens, pos)
    p_sts = head.score(x, pos)
    _, _, _, ece_sts = triage.reliability(p_sts, y)
    assert ece_sts <= ece_raw + 1e-9, (ece_raw, ece_sts)
    assert head.calibrator.temperature() > 0

    b = head.calibrator._bucket(pos)
    for k in np.unique(b):
        m = b == k
        if m.sum() > 2:
            o1 = np.argsort(p_raw[m], kind="stable")
            o2 = np.argsort(p_sts[m], kind="stable")
            assert np.array_equal(o1, o2), f"STS reordered bucket {k}"


def test_head_plugs_into_the_value_registry():
    """A fitted head becomes a first-class `value_fn` the driver can name."""
    rng = np.random.default_rng(3)
    x = np.concatenate([triage.feature_matrix(_ctx(rng)) for _ in range(10)])
    head = triage.ConfidenceHead(n_steps=200).fit(x, rng.random(len(x)))
    verifiers.register_value_fn("_test_head", triage.head_value_fn(head))
    assert "_test_head" in verifiers.value_fn_names()
    ctx = _ctx(rng)
    v = verifiers.value_of("_test_head", ctx)
    assert v.shape == (len(ctx.claimed_tokens),)
    assert ((v >= 0) & (v <= 1)).all()      # it is a probability
    del verifiers._VALUE_FNS["_test_head"]


def test_paired_effect_size_marks_the_divergent_tokens():
    """The oracle label is large exactly where the attack score moved."""
    h = np.zeros(100)
    a = np.zeros(100)
    a[[3, 17, 61]] = 8.0
    h += np.random.default_rng(4).normal(0, 1.0, 100)
    e = triage.paired_effect_size(h, a)
    assert e[[3, 17, 61]].min() > np.median(e) * 2


# ===========================================================================
# The cost model + prefix scheduler
# ===========================================================================
def test_prefill_cost_identity():
    """A full audit costs every sequence's prompt plus its generated prefix."""
    n, t, p = 8, 20, 5
    lens, plens = [t] * n, [p] * n
    full = harness.full_prefill_cost(lens, plens)
    assert full == n * (p + t - 1)
    assert harness.prefill_cost(np.ones(n * t, bool), lens, plens) == full
    assert harness.prefill_cost(np.zeros(n * t, bool), lens, plens) == 0
    # One token audited in one sequence still pays that sequence's prompt.
    m = np.zeros(n * t, bool)
    m[0] = True
    assert harness.prefill_cost(m, lens, plens) == p
    # Cost is set by the DEEPEST audited position, not the count.
    m = np.zeros(n * t, bool)
    m[t - 1] = True
    assert harness.prefill_cost(m, lens, plens) == p + t - 1


def test_topk_token_budget_hides_a_near_full_prefill_cost():
    """THE accounting gap: 5% of tokens, scattered, costs most of a full audit."""
    n, t, p = 32, 96, 12
    lens, plens = [t] * n, [p] * n
    full = harness.full_prefill_cost(lens, plens)
    value = np.random.default_rng(5).random(n * t)
    mask = harness.select_triaged(value, 0.05)
    assert abs(mask.mean() - 0.05) < 0.01                 # the reported ratio
    real = harness.prefill_cost(mask, lens, plens) / full  # what it costs
    assert real > 0.7, real                                # ...an order of magnitude more


def test_prefix_scheduler_respects_its_cost_budget():
    """`select_prefix_scheduled` spends at most `budget` x the full prefill cost,
    and buys a comparable token count for it."""
    n, t, p = 32, 96, 12
    lens, plens = [t] * n, [p] * n
    full = harness.full_prefill_cost(lens, plens)
    value = np.random.default_rng(6).random(n * t)
    for b in (0.05, 0.1, 0.25, 0.5):
        mask = harness.select_prefix_scheduled(value, lens, plens, b)
        cost = harness.prefill_cost(mask, lens, plens) / full
        assert cost <= b + 1e-9, (b, cost)
        # the audit is prefix-shaped: contiguous from position 0 in each sequence
        for i in range(n):
            row = mask[i * t:(i + 1) * t]
            if row.any():
                d = int(np.nonzero(row)[0].max()) + 1
                assert row[:d].all() and not row[d:].any()
        # and it beats top-k on tokens-audited-per-unit-cost -- by a factor that
        # grows as the budget shrinks, because that is where top-k's per-sequence
        # prompt overhead dominates (at b=0.5 top-k already pays ~99% of a full
        # audit, so there is little left to win).
        tk = harness.select_triaged(value, b)
        tk_cost = harness.prefill_cost(tk, lens, plens) / full
        gain = (mask.mean() / cost) / (tk.mean() / tk_cost)
        assert gain > (5.0 if b <= 0.1 else 1.5), (b, gain)


def test_prefix_scheduler_prefers_high_value_sequences():
    """Given one sequence with all the value, the schedule goes there first."""
    n, t, p = 6, 20, 4
    lens, plens = [t] * n, [p] * n
    value = np.full(n * t, 0.01)
    value[2 * t:3 * t] = 1.0
    mask = harness.select_prefix_scheduled(value, lens, plens, 0.2)
    assert mask[2 * t:3 * t].any()
    assert mask[2 * t:3 * t].mean() > 0.5


def test_rescore_matches_verify():
    """`rescore_at_budget` must be bit-identical to re-running the driver. This is
    the equivalence the budget sweeps rely on, so it is worth a real driver run
    against a stub backend rather than a hand-rolled mask comparison."""
    from ivgym.core import Sequence, TokenStep

    n, t, v = 5, 12, 32
    rng = np.random.default_rng(7)

    class StubBackend:
        vocab, hidden_dim = v, 8

        def __init__(self):
            self.ref = {p: rng.standard_normal((t, v)) * 3 for p in range(n)}
            self.prox = {p: rng.standard_normal((t, v)) * 3 for p in range(n)}

        def reference_logits(self, p, pos): return self.ref[p][pos]
        def proxy_logits(self, p, pos): return self.prox[p][pos]

    backend = StubBackend()
    seqs = [Sequence(prompt_id=p, config_name="honest",
                     steps=[TokenStep(position=i, claimed_token=int(rng.integers(v)),
                                      sampling=SPEC) for i in range(t)])
            for p in range(n)]
    td = verifiers.get("token_difr")
    full = harness.verify(backend, seqs, SPEC, [td])
    vals = harness.token_values(backend, seqs, SPEC, "entropy")

    for b in (0.1, 0.25, 0.5, 1.0):
        direct = harness.verify(backend, seqs, SPEC, [td], budget=b, values=vals)
        derived = harness.rescore_at_budget(full, [td], vals, b)
        assert np.array_equal(direct.scores["token_difr"], derived.scores["token_difr"]), b
        assert abs(direct.recompute_ratio - derived.recompute_ratio) < 1e-12, b


def test_select_triaged_breaks_ties_randomly():
    """`uniform` must be a random subsample, not 'the first k tokens'. With a
    stable sort over an all-ties array it would degenerate to the latter."""
    n, t = 20, 50
    mask = harness.select_triaged(np.ones(n * t), 0.2)
    assert abs(mask.mean() - 0.2) < 0.01
    touched = [i for i in range(n) if mask[i * t:(i + 1) * t].any()]
    assert len(touched) > n // 2, touched   # spread, not front-loaded
    # ...and it stays reproducible for a fixed seed.
    assert np.array_equal(mask, harness.select_triaged(np.ones(n * t), 0.2))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
