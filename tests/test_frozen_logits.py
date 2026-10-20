import numpy as np
import pytest

from experiments.numerical.frozen_logits import (
    bf16_bits, widen_bf16, decision, filter_detail, replay_event,
)
from ivgym.sampling import filtered_logits, gumbel_max_sample


def test_bf16_roundtrip_is_bitwise_and_never_silently_rounds():
    bits = np.array([0, 32768, 16256, 49024, 1], dtype=np.uint16)
    np.testing.assert_array_equal(bf16_bits(widen_bf16(bits)), bits)
    with pytest.raises(ValueError, match="exact BF16"):
        bf16_bits(np.array([1.00001], np.float32))
    with pytest.raises(ValueError):
        widen_bf16(np.array([0x7F80], np.uint16))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nucleus_conventions_and_top_k_boundary_ties(dtype):
    logits = np.log(np.array([0.6, 0.3, 0.1], dtype=dtype))
    _, legacy = filter_detail(logits, None, 0.7)
    _, inclusive = filter_detail(logits, None, 0.7, "crossing_inclusive_v1")
    assert np.flatnonzero(legacy["support_mask"]).tolist() == [0]
    assert np.flatnonzero(inclusive["support_mask"]).tolist() == [0, 1]
    _, tied = filter_detail(np.array([3, 2, 2, 1], dtype=dtype), 2, None)
    assert tied["support_mask"].tolist() == [True, True, True, False]


def test_excluded_margin_is_not_reported_as_a_finite_physical_margin():
    d, _ = decision(np.array([4., 2., 0.]), np.zeros(3), 1, None, claimed=2)
    assert d["capped_score"] == 30
    assert d["uncapped_margin"] is None
    assert d["margin_status"] == "excluded_from_support"
    finite, _ = decision(np.array([40., 0.]), np.zeros(2), None, None, claimed=1)
    assert finite["capped_score"] == 30
    assert finite["uncapped_margin"] == 40
    assert finite["claimed_in_support"]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_diagnostic_agrees_with_actual_legacy_sampler(dtype):
    rng = np.random.default_rng(7)
    x = rng.normal(size=120).astype(dtype)
    g = rng.normal(size=120)
    for k in (None, 1, 50, 120):
        for p in (None, 0.1, 0.95, 1.0):
            out, _ = filter_detail(x, k, p)
            np.testing.assert_array_equal(out, filtered_logits(x, k, p))
            d, _ = decision(x, g, k, p)
            assert d["selected_token"] == gumbel_max_sample(x, 1, g, k, p)


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf]])
def test_invalid_logits_fail_closed(values):
    with pytest.raises(ValueError):
        filter_detail(np.array(values, dtype=np.float32), 50, 0.95)


def test_event_replay_validates_actual_noise_and_support(tmp_path):
    x = np.array([3., 2., 1.], np.float32)
    g = np.zeros(3)
    _, p = filter_detail(x, 50, .95)
    _, r = filter_detail(x.astype(np.float64), 50, .95)
    fields = dict(provider_bf16=bf16_bits(x), reference_bf16=bf16_bits(x),
                  provider_gumbel=g, reference_gumbel=g, claimed_token=0)
    for name, d in (("provider", p), ("reference", r)):
        for mask in ("top_k_mask", "support_mask"):
            fields[f"{name}_{mask}"] = d[mask]
    path = tmp_path / "event.npz"
    np.savez(path, **fields)
    result = replay_event(path)
    assert len(result["cells"]) == 30
    assert result["label"] == "no_default_disagreement"
    fields["reference_gumbel"] = np.ones(3)
    np.savez(path, **fields)
    with pytest.raises(ValueError, match="random tensors"):
        replay_event(path)
