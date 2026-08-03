"""Smoke + sanity tests. Run: .venv/bin/python -m pytest tests/ -q
(or just `.venv/bin/python tests/test_smoke.py` for a dependency-free run).

These cover the backend-agnostic core (sampling RNG, the JL projection, the
metrics, and the registry/plugin contract). Backend behaviour (attack detection
AUCs) is exercised by the GPU experiments (`experiments/exp_gpu.py`,
`experiments/exp_io_detector_gpu.py`), which need a CUDA host and a model
download, so they are not part of this dependency-free suite."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import attacks, verifiers
from ivgym.core import SamplingSpec
from ivgym.metrics import partial_auc, roc_auc, tpr_at_fpr
from ivgym.sampling import gumbel_noise, position_seed, projection


def test_seed_sync_is_deterministic():
    a = gumbel_noise(64, position_seed(42, 3, 5))
    b = gumbel_noise(64, position_seed(42, 3, 5))
    assert np.array_equal(a, b)
    c = gumbel_noise(64, position_seed(42, 3, 6))
    assert not np.array_equal(a, c)


def test_top_id_ordering_is_skipped_only_when_unused():
    """`HFGPUBackend.generate` skips the full-vocabulary argsort that builds
    `top_k_ids` unless the attack actually implements `sample_override`.

    That optimization is only sound because of two facts this test pins, both of
    which a new attack could silently break:

      1. the predicate `type(attack).sample_override is not Attack.sample_override`
         is true for exactly the attacks that override the hook -- including
         subclasses that inherit an override (`bug_k32` from `SamplingBug`); and
      2. the BASE `sample_override` draws nothing from `rng`, so skipping the call
         leaves the generation RNG stream -- and therefore every sampled token and
         every activation noise draw downstream -- bit-identical.

    Fact 2 is the load-bearing one: the same `prng` is used after this hook for
    the activation noise, so a base implementation that consumed a draw would make
    the skip change sampled outputs rather than just their cost."""
    base = attacks.Attack
    overriding = {"bug_k2", "bug_k32"}
    for name, atk in attacks.all_attacks().items():
        pred = type(atk).sample_override is not base.sample_override
        assert pred == (name in overriding), f"{name}: predicate {pred}"

    before = np.random.default_rng(0)
    after = np.random.default_rng(0)
    assert base().sample_override(after, np.arange(8)) is None
    assert np.array_equal(before.random(4), after.random(4))     # no draws consumed


def test_projection_is_seeded_and_orthonormal():
    """The Activation-DiFR projection must be reproducible from its seed (so
    provider and verifier share it) and have orthonormal rows."""
    p = projection(123, 32, 256)
    assert p.shape == (32, 256)
    assert np.array_equal(p, projection(123, 32, 256))            # seeded -> reproducible
    assert not np.array_equal(p, projection(124, 32, 256))        # seed actually matters
    np.testing.assert_allclose(p @ p.T, np.eye(32), atol=1e-9)    # orthonormal rows


def test_register_accepts_class_and_instance():
    """The documented `@register class MyAttack(Attack)` decorator pattern must
    land a usable *instance* in the registry (not the bare class)."""
    @attacks.register
    class _ClassAttack(attacks.Attack):
        name = "tmp_class_attack"

    @verifiers.register
    class _ClassVerifier(verifiers.Verifier):
        name = "tmp_class_verifier"
        def evidence(self, ctx):
            return np.zeros(len(ctx.claimed_tokens))

    atk = attacks.get("tmp_class_attack")
    vf = verifiers.get("tmp_class_verifier")
    assert not isinstance(atk, type) and not isinstance(vf, type)
    # instance methods must be callable (they would fail on a bare class)
    assert atk.provider_spec(SamplingSpec()) == SamplingSpec()
    del attacks._REGISTRY["tmp_class_attack"], verifiers._REGISTRY["tmp_class_verifier"]


def test_plugin_loading_registers_strategies():
    """Loading an external strategy file registers its strategies into the same
    registries the harness and every backend use (the no-edit extension path)."""
    from experiments.run import load_strategies

    root = Path(__file__).resolve().parents[1]
    load_strategies([str(root / "examples" / "custom_strategies.py")])
    assert "logit_spike" in attacks.all_attacks()
    assert "top1_mismatch_toy" in verifiers.all_verifiers()
    # the registered objects are usable instances, not bare classes
    assert not isinstance(attacks.get("logit_spike"), type)
    assert not isinstance(verifiers.get("top1_mismatch_toy"), type)


def test_token_toploc_scores_rank_of_claimed_token():
    """`token_toploc` (built-in, promoted from the examples/ demo) must score 0
    when the claimed token is the verifier's argmax, and a positive rank
    otherwise, capped at `rank_cap`. It is a Tier-1 verifier, so we exercise its
    per-token `score_token` (what the driver calls on audited tokens)."""
    toploc = verifiers.get("token_toploc")
    assert "token_toploc" in verifiers.all_verifiers()
    assert toploc.tier == 1 and toploc.needs_seed is False

    spec = SamplingSpec(temperature=1.0, top_k=None, top_p=None)
    logits = np.array([5.0, 3.0, 1.0, 0.0], dtype=np.float32)

    assert toploc.score_token(logits, None, 0, spec) == 0.0
    # two tokens (idx 0, 1) rank above idx 2
    assert toploc.score_token(logits, None, 2, spec) == 2.0


class _FakeBackend:
    """Minimal backend for the selective-verifier contract: near-tie 'flip'
    positions carry a non-argmax claimed token (so token_difr fires) AND a flat
    (tie-like) proxy; the rest are peaked and honest. Counts reference_logits
    calls so the test can assert recompute is spent only where triage sends it."""

    def __init__(self, n=6, t=20, vocab=16, flip_every=5):
        self.vocab = vocab
        self.hidden_dim = 8
        self.n_ref_calls = 0
        self.flip = {}
        self._ref, self._proxy, self._claim = {}, {}, {}
        rng = np.random.default_rng(0)
        for pid in range(n):
            for pos in range(t):
                ref = rng.normal(0, 3.0, vocab)
                order = np.argsort(-ref)
                is_flip = ((pid * t + pos) % flip_every) == 0
                if is_flip:
                    proxy = rng.normal(0, 0.15, vocab)   # flat -> tie-like -> high tie-ness
                    claim = int(order[3])                # a clearly-worse token -> margin>0
                else:
                    proxy = ref * 2.0                    # peaked -> low tie-ness
                    claim = int(order[0])                # argmax -> margin ~0
                self._ref[(pid, pos)] = ref
                self._proxy[(pid, pos)] = proxy
                self._claim[(pid, pos)] = claim
                self.flip[(pid, pos)] = is_flip
        self.n, self.t = n, t

    def reference_logits(self, pid, pos):
        self.n_ref_calls += 1
        return self._ref[(pid, pos)]

    def proxy_logits(self, pid, pos):
        return self._proxy[(pid, pos)]

    def served_logits(self, pid, pos):
        # The distribution the provider served under; here == the reference (an
        # honest provider). accept_rate compares this against the cheap proxy.
        return self._ref[(pid, pos)]

    def sequences(self):
        from ivgym.core import Sequence, TokenStep
        spec = SamplingSpec(temperature=0.1)
        seqs = []
        for pid in range(self.n):
            s = Sequence(prompt_id=pid, config_name="fake")
            for pos in range(self.t):
                s.steps.append(TokenStep(position=pos, claimed_token=self._claim[(pid, pos)],
                                         sampling=spec))
            seqs.append(s)
        return seqs


def test_selective_recompute_spends_where_value_points():
    """The single driver at `budget<1` must (1) recompute only the budgeted
    fraction, (2) rank the near-tie flip positions above the peaked ones via the
    cheap proxy value signal, and (3) concentrate far more divergence signal than
    a random audit of the same size."""
    from ivgym import harness
    be = _FakeBackend()
    seqs = be.sequences()
    spec = SamplingSpec(temperature=0.1)
    td = verifiers.get("token_difr")
    n_tokens = be.n * be.t
    budget = 0.2                                   # matches flip_every=5 (20% are flips)

    tie = harness.token_values(be, seqs, spec, "tie_margin")
    flip_mask = np.array([be.flip[(seq.prompt_id, st.position)]
                          for seq in seqs for st in seq.steps])
    # (2) proxy tie-ness ranks flip positions above non-flip ones
    assert tie[flip_mask].mean() > tie[~flip_mask].mean()

    # (1) recompute only the budgeted fraction
    be.n_ref_calls = 0
    tri = harness.verify(be, seqs, spec, [td], budget=budget, values=tie)
    assert abs(tri.recompute_ratio - budget) < 1e-6
    assert be.n_ref_calls == int(round(budget * n_tokens))     # NOT n_tokens

    # (3) value-directed audit concentrates divergence vs a random audit, same budget
    rng = np.random.default_rng(1)
    rnd = harness.verify(be, seqs, spec, [td], budget=budget, values=rng.random(n_tokens))
    assert tri.scores["token_difr"].sum() > 3 * rnd.scores["token_difr"].sum()


def test_driver_scores_tier0_verifiers_without_recompute():
    """A Tier-0 run (surface_stat + accept_rate) must (1) never recompute M,
    (2) report recompute_ratio 0.0, and (3) produce one score per token for each
    verifier -- flowing through the SAME TokenScores the Tier-1 path uses."""
    from ivgym import harness
    be = _FakeBackend()
    seqs = be.sequences()
    spec = SamplingSpec(temperature=0.1)
    ss, ar = verifiers.get("surface_stat"), verifiers.get("accept_rate")
    n_tokens = be.n * be.t

    be.n_ref_calls = 0
    ts = harness.verify(be, seqs, spec, [ss, ar])
    assert be.n_ref_calls == 0                          # no recompute of M
    assert ts.recompute_ratio == 0.0
    for name in ("surface_stat", "accept_rate"):
        assert ts.scores[name].shape == (n_tokens,)
    # accept_rate = TV(served, proxy) is >= 0 and non-trivial on the flat-proxy flips
    assert ts.scores["accept_rate"].min() >= 0.0
    assert ts.scores["accept_rate"].max() > 0.0


def test_io_contexts_never_recompute_m():
    """`harness.io_contexts` is the Tier-0 entry point -- it must build a scoreable
    context from the proxy and the served text alone. If it ever touched M the
    whole Tier-0/Tier-1 cost separation would be fiction, so this is pinned."""
    from ivgym import harness
    be = _FakeBackend()
    seqs = be.sequences()
    spec = SamplingSpec(temperature=0.1)

    be.n_ref_calls = 0
    ctxs = harness.io_contexts(be, seqs, spec, need_proxy=True)
    assert be.n_ref_calls == 0                          # never recomputes M
    assert len(ctxs) == len(seqs)
    for ctx, seq in zip(ctxs, seqs):
        assert ctx.proxy_logits is not None
        assert ctx.ref_logits is None                   # no reference distribution
        assert len(ctx.claimed_tokens) == len(seq.steps)


def test_metrics():
    neg = np.array([0.0, 0.1, 0.2, 0.3])
    pos = np.array([0.4, 0.5, 0.6, 0.7])
    assert roc_auc(neg, pos) == 1.0
    assert tpr_at_fpr(neg, pos, 0.25) > 0.5


def test_partial_auc_matches_full_auc_on_perfect_separation():
    """A perfectly-separated pair scores 1.0 at any FPR window (the ROC hugs the
    top-left corner), same as full AUC."""
    rng = np.random.default_rng(0)
    neg = rng.normal(0, 1, 2000)
    pos = rng.normal(6, 1, 2000)      # far enough apart to be separable at FPR<=0.5%
    assert partial_auc(neg, pos, max_fpr=0.005) > 0.999
    assert roc_auc(neg, pos) > 0.999


def test_partial_auc_chance_is_half_regardless_of_max_fpr():
    """McClish standardization must put a random (same-distribution) verifier at
    ~0.5 on the standardized scale, whatever the FPR window -- that is the whole
    point of standardizing (so it's comparable to full AUC's 0.5 chance line)."""
    rng = np.random.default_rng(1)
    neg = rng.normal(0, 1, 20000)
    pos = rng.normal(0, 1, 20000)
    for max_fpr in (0.005, 0.05, 0.5):
        assert abs(partial_auc(neg, pos, max_fpr) - 0.5) < 0.05


def test_partial_auc_raw_is_not_standardized():
    """The raw (non-standardized) variant is mean-TPR-in-region. Under the chance
    diagonal TPR=FPR, that mean over [0, max_fpr] is max_fpr/2, NOT 0.5 -- unlike
    the McClish-standardized score, it does not sit on a fixed chance line, which
    is why `standardized=True` (McClish) is the default headline metric."""
    rng = np.random.default_rng(2)
    neg = rng.normal(0, 1, 20000)
    pos = rng.normal(0, 1, 20000)
    max_fpr = 0.05
    raw = partial_auc(neg, pos, max_fpr, standardized=False)
    assert abs(raw - max_fpr / 2) < 0.01
    std = partial_auc(neg, pos, max_fpr, standardized=True)
    assert abs(std - 0.5) < 0.05


def test_partial_auc_empty_input_is_chance():
    assert partial_auc(np.array([]), np.array([1.0])) == 0.5
    assert partial_auc(np.array([1.0]), np.array([])) == 0.5


def test_eval_config_rejects_undersized_n_batches():
    """The soundness floor: at max_fpr=0.005 you need >= ~min_region_pts/max_fpr
    honest calibration batches to resolve the partial-AUC region at all -- a
    tiny n_batches must raise loudly rather than silently return a noisy number."""
    from ivgym.harness import EvalConfig

    EvalConfig(max_fpr=0.005, n_batches=2000)     # fine
    try:
        EvalConfig(max_fpr=0.005, n_batches=100)  # only ~0.5 expected pts in-region
        assert False, "expected ValueError for undersized n_batches"
    except ValueError:
        pass


def _ratio_fixture(n=4000, shift=0.3, seed=0):
    """Honest/attack score pools with a small real per-token shift, so the ONLY
    thing that moves the AUC between the assertions below is the batch/pool ratio."""
    from ivgym import harness
    rng = np.random.default_rng(seed)
    h = np.abs(rng.normal(0.0, 1.0, n))
    a = np.abs(rng.normal(shift, 1.0, n))
    d = type("V", (), {"name": "s", "tier": 1})()
    return (harness.TokenScores("honest", {"s": h}),
            harness.TokenScores("attack", {"s": a}), [d])


def test_every_result_reports_the_ratio_it_was_measured_at():
    """Deployment rule 6: report the batch/pool ratio next to every AUC. Without
    it a detection AUC is not interpretable, so `evaluate` attaches it always."""
    from ivgym import harness
    hs, as_, defs = _ratio_fixture()
    r = harness.evaluate(hs, as_, defs, [100])[0]
    assert r.eval_pool == 2000                       # half of 4000, calib_frac=0.5
    assert abs(r.pool_ratio - 100 / 2000) < 1e-12
    assert not r.over_ceiling


def test_over_ceiling_batches_cannot_be_measured_silently():
    """The central methodological result, enforced in code: an AUC measured with
    the batch a large fraction of the pool is a property of that pool, not of the
    deviation. It must warn (default), raise on request, and only ever be silent
    when the caller explicitly opts in."""
    import warnings
    from ivgym.harness import EvalConfig, RatioCeilingWarning
    from ivgym import harness
    hs, as_, defs = _ratio_fixture()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = harness.evaluate(hs, as_, defs, [1500])[0]     # 75% ratio
    assert r.over_ceiling and abs(r.pool_ratio - 0.75) < 1e-12
    assert any(issubclass(x.category, RatioCeilingWarning) for x in w)

    try:
        harness.evaluate(hs, as_, defs, [1500], config=EvalConfig(over_ratio="raise"))
        assert False, "expected ValueError above the ratio ceiling"
    except ValueError as e:
        assert "ceiling" in str(e)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        harness.evaluate(hs, as_, defs, [1500], config=EvalConfig(over_ratio="allow"))
    assert not any(issubclass(x.category, RatioCeilingWarning) for x in w)


def test_the_ratio_artifact_is_reproduced_from_identical_scores():
    """The falsification test for the ceiling itself. Holding the per-token scores
    FIXED and growing only the batch, the AUC must climb far above what the
    per-token effect size can support -- which is the whole reason the ceiling is
    a correctness condition and not a style preference."""
    from ivgym.harness import EvalConfig
    from ivgym import harness
    hs, as_, defs = _ratio_fixture()
    cfg = EvalConfig(over_ratio="allow")
    valid = harness.evaluate(hs, as_, defs, [100], config=cfg)[0]     # 5%
    inflated = harness.evaluate(hs, as_, defs, [1500], config=cfg)[0]  # 75%

    h = hs.scores["s"]
    d_prime = (as_.scores["s"].mean() - h.mean()) / h.std()
    from ivgym import signal
    predicted = signal.pauc_of_delta(d_prime * np.sqrt(100), max_fpr=0.005)

    assert abs(valid.auc - predicted) < 0.10, (valid.auc, predicted)   # tracks theory
    assert inflated.auc > valid.auc + 0.3                              # ... and blows past it
    assert inflated.auc > 0.95


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
