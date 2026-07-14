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

from ivgym import attacks, defenses
from ivgym.core import SamplingSpec
from ivgym.metrics import roc_auc, tpr_at_fpr
from ivgym.sampling import gumbel_noise, position_seed, projection


def test_seed_sync_is_deterministic():
    a = gumbel_noise(64, position_seed(42, 3, 5))
    b = gumbel_noise(64, position_seed(42, 3, 5))
    assert np.array_equal(a, b)
    c = gumbel_noise(64, position_seed(42, 3, 6))
    assert not np.array_equal(a, c)


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

    @defenses.register
    class _ClassDefense(defenses.Defense):
        name = "tmp_class_defense"
        needs_seed = False
        def score(self, ctx):
            return 0.0

    atk = attacks.get("tmp_class_attack")
    dfn = defenses.get("tmp_class_defense")
    assert not isinstance(atk, type) and not isinstance(dfn, type)
    # instance methods must be callable (they would fail on a bare class)
    assert atk.provider_spec(SamplingSpec()) == SamplingSpec()
    del attacks._REGISTRY["tmp_class_attack"], defenses._REGISTRY["tmp_class_defense"]


def test_plugin_loading_registers_strategies():
    """Loading an external strategy file registers its strategies into the same
    registries the harness and every backend use (the no-edit extension path)."""
    from experiments.run import load_strategies

    root = Path(__file__).resolve().parents[1]
    load_strategies([str(root / "examples" / "custom_strategies.py")])
    assert "logit_spike" in attacks.all_attacks()
    assert "top1_mismatch_toy" in defenses.all_defenses()
    # the registered objects are usable instances, not bare classes
    assert not isinstance(attacks.get("logit_spike"), type)
    assert not isinstance(defenses.get("top1_mismatch_toy"), type)


def test_token_toploc_scores_rank_of_claimed_token():
    """`token_toploc` (built-in, promoted from the examples/ demo) must score 0
    when the claimed token is the verifier's argmax, and a positive rank
    otherwise, capped at `rank_cap`."""
    from ivgym.core import VerifyContext, SamplingSpec

    toploc = defenses.get("token_toploc")
    assert "token_toploc" in defenses.all_defenses()
    assert toploc.needs_seed is False

    spec = SamplingSpec(temperature=1.0, top_k=None, top_p=None)
    logits = np.array([5.0, 3.0, 1.0, 0.0], dtype=np.float32)
    gumbel = np.zeros_like(logits)

    honest_ctx = VerifyContext(claimed_token=0, ref_logits=logits, gumbel=gumbel, sampling=spec)
    assert toploc.score(honest_ctx) == 0.0

    cheat_ctx = VerifyContext(claimed_token=2, ref_logits=logits, gumbel=gumbel, sampling=spec)
    assert toploc.score(cheat_ctx) == 2.0  # two tokens (idx 0, 1) rank above idx 2


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


def test_verify_selective_spends_recompute_where_triage_points():
    """The cost-aware tier must (1) recompute only the budgeted fraction, (2) rank
    the near-tie flip positions above the peaked ones via the cheap proxy, and (3)
    concentrate far more divergence signal than a random audit of the same size."""
    from ivgym import harness
    be = _FakeBackend()
    seqs = be.sequences()
    spec = SamplingSpec(temperature=0.1)
    td = defenses.get("token_difr")
    n_tokens = be.n * be.t
    budget = 0.2                                   # matches flip_every=5 (20% are flips)

    tie = harness.proxy_tie_scores(be, seqs, spec)
    flip_mask = np.array([be.flip[(seq.prompt_id, st.position)]
                          for seq in seqs for st in seq.steps])
    # (2) proxy tie-ness ranks flip positions above non-flip ones
    assert tie[flip_mask].mean() > tie[~flip_mask].mean()

    # (1) recompute only the budgeted fraction
    be.n_ref_calls = 0
    scores_tri, ratio = harness.verify_selective(be, seqs, spec, td, budget)
    assert abs(ratio - budget) < 1e-6
    assert be.n_ref_calls == int(round(budget * n_tokens))     # NOT n_tokens

    # (3) triage concentrates the divergence vs a random audit of the same budget
    rng = np.random.default_rng(1)
    scores_rnd, _ = harness.verify_selective(be, seqs, spec, td, budget,
                                             triage=rng.random(n_tokens))
    assert scores_tri.scores["token_difr"].sum() > 3 * scores_rnd.scores["token_difr"].sum()


def test_metrics():
    neg = np.array([0.0, 0.1, 0.2, 0.3])
    pos = np.array([0.4, 0.5, 0.6, 0.7])
    assert roc_auc(neg, pos) == 1.0
    assert tpr_at_fpr(neg, pos, 0.25) > 0.5


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
