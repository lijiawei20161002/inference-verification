import copy
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.numerical.mechanism_diagnostics import (
    native_reference, partition_coverage, sampling_changes, summarize_diagnostics,
)
from experiments.numerical.mechanism_ops import attention, gemm


def test_coverage_counts_only_unmasked_partitions_and_checks_every_query():
    mask = np.full((1, 2, 1, 64), -np.inf, dtype=np.float32)
    mask[..., :18] = 0
    assert partition_coverage(mask, 8)["active_partitions_min"] == 3
    with pytest.raises(ValueError, match="Too few active"):
        partition_coverage(mask, 32)
    mask[:, 1, :, 8:] = -np.inf
    with pytest.raises(ValueError, match="Too few active"):
        partition_coverage(mask, 8)
    with pytest.raises(ValueError, match="Invalid"):
        partition_coverage(mask, 0)


def test_gemm_retains_fp32_effect_that_bf16_cast_removes():
    torch = pytest.importorskip("torch")
    # Both orders round to 1 in BF16, while their FP32 sums differ.
    x = torch.ones((1, 1, 3), dtype=torch.bfloat16)
    w = torch.tensor([[1.0, 2 ** -24, 2 ** -24]], dtype=torch.bfloat16)
    a, da = gemm(x, w, chunk=1, order="ascending", return_details=True)
    b, db = gemm(x, w, chunk=1, order="descending", return_details=True)
    assert torch.equal(a, b)
    assert not torch.equal(da["precast"], db["precast"])
    assert torch.equal(da["partials"], db["partials"])


def test_attention_retains_identical_partials_and_exact_final_cast():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(12)
    q, k, v = [torch.tensor(rng.normal(size=shape), dtype=torch.bfloat16)
               for shape in ((1, 2, 1, 8), (1, 2, 29, 8), (1, 2, 29, 8))]
    mask = torch.zeros((1, 1, 1, 29), dtype=torch.bfloat16)
    mask[..., 23:] = torch.finfo(torch.bfloat16).min
    details = []
    for order in ("ascending", "descending"):
        plain = attention(q, k, v, mask, scale=0.25, chunk=8, order=order)
        actual, d = attention(q, k, v, mask, scale=0.25, chunk=8, order=order, return_details=True)
        assert torch.equal(actual, plain)
        assert torch.equal(actual, d["precast"].to(torch.bfloat16))
        assert torch.isfinite(d["precast"]).all()
        details.append(d)
    for key in ("partial_numerators", "partial_denominators", "partition_maxima"):
        assert torch.equal(details[0][key], details[1][key])


def test_native_attention_bridge_matches_pinned_eager_with_gqa_and_noncontiguous_query():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward, repeat_kv

    rng = np.random.default_rng(13)
    query = torch.tensor(rng.normal(size=(1, 7, 4, 8)), dtype=torch.bfloat16).transpose(1, 2)
    key, value = [torch.tensor(rng.normal(size=(1, 2, 7, 8)), dtype=torch.bfloat16) for _ in range(2)]
    mask = torch.zeros((1, 1, 7, 7), dtype=torch.bfloat16)
    mask.masked_fill_(torch.triu(torch.ones(7, 7, dtype=torch.bool), diagonal=1), torch.finfo(torch.bfloat16).min)
    module = SimpleNamespace(num_key_value_groups=2, training=False)
    expected, _ = eager_attention_forward(module, query, key, value, mask, scaling=0.25)
    actual = native_reference("attention", dict(full_query=query, key=repeat_kv(key, 2),
                              value=repeat_kv(value, 2), full_mask=mask), dict(attention_scale=0.25))
    assert torch.equal(expected, actual)


def test_saved_operands_preserve_noncontiguous_layout(tmp_path):
    torch = pytest.importorskip("torch")
    from experiments.numerical.mechanism_replay import load_tensors, save_tensors, tensor_layouts

    values = {"query": torch.arange(120, dtype=torch.float32).reshape(1, 5, 3, 8).transpose(1, 2).to(torch.bfloat16)}
    save_tensors(tmp_path, "operands", values)
    restored = load_tensors(tmp_path, "operands", "cpu")
    assert torch.equal(values["query"], restored["query"])
    assert tensor_layouts(values) == tensor_layouts(restored)


def test_support_changes_are_not_reported_as_token_changes():
    cells = [dict(dtype="float32", convention="legacy", mode="top_k_top_p", token_changed=False, support_delta=0),
             dict(dtype="float32", convention="legacy", mode="top_k", token_changed=False, support_delta=2)]
    changes = sampling_changes(cells)
    assert changes["any_support_changed"] and not changes["any_token_changed"]
    row = dict(case=dict(family="attention"), sampling=cells,
               native_bridge={s: dict(changed=1) for s in ("a", "b")},
               controls=dict(native_sham_exact=True, native_restore_exact=True),
               diagnostics=dict(native_full_exact=True, fp32_difference=dict(changed=1),
                   selected_logits_changed=True, native_sampling={s: cells for s in ("a", "b")},
                   coverage=dict(active_partitions_min=3, active_partitions_max=3)))
    summary = summarize_diagnostics([row])
    assert summary["any_sampling_support_changed"] == 1
    assert summary["any_sampling_token_changed"] == 0
    changed = copy.deepcopy(row)
    changed["controls"]["native_restore_exact"] = False
    with pytest.raises(ValueError, match="Native bridge control"):
        summarize_diagnostics([changed])
