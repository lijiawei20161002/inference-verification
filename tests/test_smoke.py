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
    assert "topk_overlap" in defenses.all_defenses()
    # the registered objects are usable instances, not bare classes
    assert not isinstance(attacks.get("logit_spike"), type)
    assert not isinstance(defenses.get("topk_overlap"), type)


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
