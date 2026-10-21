import copy

import numpy as np
import pytest

from experiments.numerical.mechanism_ops import (
    attention, gemm, ordered_sum, sampling_matrix, select_cases, validate_controls,
)


def test_selection_is_balanced_unique_and_frozen():
    events = [dict(prompt_id=i, position=i, historical_event_id=f"old:{i}" if i < 20 else None,
                   control=i >= 20, evidence=f"p{i}.npz", evidence_sha256=str(i)) for i in range(40)]
    cases = select_cases(events, 32)
    assert len(cases) == 64 and len({r["case_id"] for r in cases}) == 64
    for family in ("gemm", "attention"):
        rows = [r for r in cases if r["family"] == family]
        assert sum(r["stratum"] == "historical" for r in rows) == 16
        assert {r["layer"] for r in rows} == {0, 14, 27}
    assert cases == select_cases(list(reversed(events)), 32)
    with pytest.raises(ValueError):
        select_cases(events, 3)


def test_controls_reject_stale_descendants_and_changed_ancestors():
    native = dict(prefix_logits="prefix", logits="native", kv=["kv0", "kv1"])
    alt = dict(prefix_logits="prefix", logits="alt", kv=["kv0", "changed"])
    states = {name: copy.deepcopy(native) for name in ("native", "capture", "a", "sham_a", "reverse")}
    states.update({name: copy.deepcopy(alt) for name in ("b", "sham_b", "forward")})
    assert validate_controls(states, 0)["forward_exact"]
    states["forward"]["kv"] = native["kv"]
    with pytest.raises(ValueError, match="control failed"):
        validate_controls(states, 0)
    states["forward"] = copy.deepcopy(alt)
    with pytest.raises(ValueError, match="ancestor KV"):
        validate_controls(states, 1)


def test_sampler_uses_identical_noise_and_declared_conventions():
    values = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    cells = sampling_matrix(values, values, np.array([0.1, -0.2, 0.4]))
    assert len(cells) == 20
    assert all(not c["token_changed"] and c["support_delta"] == 0 for c in cells)


def test_reduction_order_is_real_not_a_noop_flag():
    torch = pytest.importorskip("torch")
    parts = [torch.tensor([1e20]), torch.tensor([-1e20]), torch.tensor([3.0])]
    assert ordered_sum(parts, "ascending").item() == 3.0
    assert ordered_sum(parts, "descending").item() == 0.0


def test_partitioned_gemm_preserves_the_linear_map():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1, 1, 19)).astype(np.float32)
    weight = rng.normal(size=(7, 19)).astype(np.float32)
    expected = x.astype(np.float64) @ weight.astype(np.float64).T
    for order in ("ascending", "descending"):
        result = gemm(torch.from_numpy(x), torch.from_numpy(weight), chunk=4, order=order)
        np.testing.assert_allclose(result.numpy(), expected, rtol=3e-6, atol=3e-6)


def test_split_attention_matches_dense_reference_with_fully_masked_partitions():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    q = rng.normal(size=(1, 2, 1, 4)).astype(np.float32)
    k = rng.normal(size=(1, 2, 9, 4)).astype(np.float32)
    v = rng.normal(size=(1, 2, 9, 4)).astype(np.float32)
    mask = np.zeros((1, 1, 1, 9), dtype=np.float32)
    mask[..., 3:] = np.finfo(np.float32).min
    scores = q.astype(np.float64) @ k[..., :3, :].astype(np.float64).swapaxes(-1, -2) * 0.5
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    expected = (weights @ v[..., :3, :]).transpose(0, 2, 1, 3)
    for order in ("ascending", "descending"):
        result = attention(*map(torch.from_numpy, (q, k, v, mask)), scale=0.5, chunk=2, order=order)
        np.testing.assert_allclose(result.numpy(), expected, rtol=3e-6, atol=3e-6)
    mask.fill(np.finfo(np.float32).min)
    with pytest.raises(ValueError, match="no allowed keys"):
        attention(*map(torch.from_numpy, (q, k, v, mask)), scale=0.5, chunk=2, order="ascending")
