"""Smoke + sanity tests. Run: .venv/bin/python -m pytest tests/ -q
(or just `.venv/bin/python tests/test_smoke.py` for a dependency-free run)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from difr import attacks, defenses, harness
from difr.backends.synthetic import SyntheticBackend
from difr.core import SamplingSpec
from difr.metrics import roc_auc, tpr_at_fpr
from difr.sampling import gumbel_noise, position_seed


def test_seed_sync_is_deterministic():
    a = gumbel_noise(64, position_seed(42, 3, 5))
    b = gumbel_noise(64, position_seed(42, 3, 5))
    assert np.array_equal(a, b)
    c = gumbel_noise(64, position_seed(42, 3, 6))
    assert not np.array_equal(a, c)


def test_honest_vs_honest_is_chance():
    """Two honest runs should be indistinguishable (AUC ~ 0.5)."""
    be = SyntheticBackend(vocab=256)
    spec = SamplingSpec()
    defs = [defenses.get("token_difr"), defenses.get("cross_entropy")]
    s1 = harness.verify(be, harness.generate_dataset(be, attacks.get("honest"), spec, 40, 128), spec, defs)
    s2 = harness.verify(be, harness.generate_dataset(be, attacks.get("honest"), spec, 40, 128), spec, defs)
    # Null comparison -> AUC ~ 0.5. token_difr is heavy-tailed (mostly zeros),
    # so average several sampling seeds rather than trust one draw.
    for d in defs:
        aucs = [harness.evaluate(s1, s2, [d], [200], n_batches=300, winsor_pct=99.9, seed=es)[0].auc
                for es in range(6)]
        assert 0.40 <= np.mean(aucs) <= 0.60, f"{d.name} mean AUC={np.mean(aucs):.3f}"


def test_quantization_detected_by_token_difr():
    be = SyntheticBackend(vocab=256)
    spec = SamplingSpec()
    d = [defenses.get("token_difr")]
    honest = harness.verify(be, harness.generate_dataset(be, attacks.get("honest"), spec, 40, 128), spec, d)
    quant = harness.verify(be, harness.generate_dataset(be, attacks.get("quant_4bit"), spec, 40, 128), spec, d)
    res = harness.evaluate(honest, quant, d, [500], n_batches=300, winsor_pct=99.9, seed=2)
    assert res[0].auc > 0.95


def test_seed_mismatch_only_caught_by_token_difr():
    spec = SamplingSpec()
    defs = [defenses.get("token_difr"), defenses.get("cross_entropy")]
    # A wrong seed redraws tokens from the SAME distribution, so CE has no real
    # signal -- but any single frozen dataset carries ~1 s.e. of spurious CE gap.
    # Average over independent datasets (different model_seed) to get the true null.
    td, ce = [], []
    for ms in range(5):
        be = SyntheticBackend(vocab=256, model_seed=ms)
        honest = harness.verify(be, harness.generate_dataset(be, attacks.get("honest"), spec, 40, 128), spec, defs)
        seed_atk = harness.verify(be, harness.generate_dataset(be, attacks.get("seed_43"), spec, 40, 128), spec, defs)
        r = {x.defense: x for x in harness.evaluate(honest, seed_atk, defs, [500], n_batches=300, seed=ms)}
        td.append(r["token_difr"].auc)
        ce.append(r["cross_entropy"].auc)
    assert min(td) > 0.95                         # seed-synced metric always catches it
    assert np.mean(ce) < 0.7                       # distribution unchanged -> CE near blind
    assert np.mean(td) > np.mean(ce) + 0.25       # Token-DiFR dominates


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
