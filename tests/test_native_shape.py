"""Evidence and intervention contracts for native-shape witnesses."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.numerical.check_native_witness import (
    check_matrix, load_bundle, safe_file, sample, sha,
)
from experiments.numerical.frozen_logits import bf16_bits, decision
from experiments.numerical.mechanism_ops import sampling_matrix
from experiments.numerical.mechanism_replay import tensor_layouts
from experiments.numerical.native_shape import shape_operands, summarize


@pytest.mark.parametrize('position', [0, 2])
def test_gemm_projects_exact_provider_call_rows_and_layout(position):
    torch = pytest.importorskip('torch')
    x = torch.arange(1 * 8 * 6).reshape(1, 8, 6).to(torch.bfloat16)
    w = torch.ones(4, 6, dtype=torch.bfloat16)
    case = dict(position=position, token_index=2 + position)
    length = 3 if position == 0 else 1
    provider = dict(full_x=torch.empty(1, length, 6, dtype=torch.bfloat16), weight=w)
    result = shape_operands('gemm', dict(full_x=x, weight=w), tensor_layouts(provider), case)
    start = 0 if position == 0 else case['token_index']
    assert torch.equal(result['full_x'], x[:, start:case['token_index'] + 1])
    assert tensor_layouts(result) == tensor_layouts(provider)
    assert torch.equal(result['weight'], w)


def attention_inputs():
    torch = pytest.importorskip('torch')
    q = torch.arange(1 * 7 * 2 * 4).reshape(1, 7, 2, 4).transpose(1, 2).to(torch.bfloat16)
    mask = torch.zeros(1, 1, 7, 7, dtype=torch.bfloat16)
    mask.masked_fill_(torch.triu(torch.ones(7, 7, dtype=torch.bool), diagonal=1), torch.finfo(torch.bfloat16).min)
    original = dict(full_query=q, key=q.clone().contiguous(), value=q.clone().contiguous(), full_mask=mask)
    provider = dict(full_query=q[:, :, 3:4].contiguous(), key=q[:, :, :4].contiguous(),
                    value=q[:, :, :4].contiguous(), full_mask=mask[:, :, 3:4, :4].contiguous())
    return original, provider, dict(position=1, token_index=3)


def test_attention_removes_only_masked_suffix_and_preserves_effective_values():
    torch = pytest.importorskip('torch')
    original, provider, case = attention_inputs()
    result = shape_operands('attention', original, tensor_layouts(provider), case)
    assert all(torch.equal(result[k], provider[k]) for k in provider)
    query_only = shape_operands('attention', original, tensor_layouts(provider), case, query_only=True)
    assert query_only['key'].shape[-2] == 7
    assert query_only['full_query'].shape[-2] == 1
    original['full_mask'][..., 3, 6] = 0
    with pytest.raises(ValueError, match='unmasked KV'):
        shape_operands('attention', original, tensor_layouts(provider), case)


def test_wrong_provider_geometry_is_rejected():
    original, provider, case = attention_inputs()
    layouts = tensor_layouts(provider)
    layouts['full_query']['shape'][-2] = 2
    with pytest.raises(ValueError, match='geometry'):
        shape_operands('attention', original, layouts, case)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('convention', ['legacy', 'crossing_inclusive_v1'])
def test_independent_sampler_replays_ties_and_nucleus_boundary(dtype, convention):
    # A top-k tie spans the boundary; the nucleus crossing is also exercised.
    x = np.r_[np.repeat(1., 51), np.linspace(-3, -12, 30)].astype(dtype)
    noise = np.random.default_rng(41).normal(size=x.size)
    for mode, k, p, greedy in [('greedy', None, None, True), ('gumbel', None, None, False),
                              ('top_k', 50, None, False), ('top_p', None, .95, False),
                              ('top_k_top_p', 50, .95, False)]:
        expected, masks = decision(x, noise, k, p, convention, greedy)
        token, support = sample(x, noise, dtype, convention, mode)
        assert token == expected['selected_token']
        np.testing.assert_array_equal(support, masks['support_mask'])


def test_checker_rejects_missing_cells_and_fabricated_token_change():
    x = np.array([1., 2., 3.], dtype=np.float32)
    noise = np.array([.1, .2, .3])
    cells = sampling_matrix(x, x, noise)
    bits = bf16_bits(x)
    assert len(check_matrix(cells, bits, bits, noise)) == 20
    with pytest.raises(ValueError, match='Incomplete sampling'):
        check_matrix(cells[:-1], bits, bits, noise)
    bad = copy.deepcopy(cells)
    bad[0]['token_changed'] = True
    with pytest.raises(ValueError, match='predicate differs'):
        check_matrix(bad, bits, bits, noise)


def test_witness_checks_external_digest_inventory_and_parent_identity(tmp_path):
    required = ['record.json', 'operands.json', 'operands.npz', 'provider_operands.json',
                'provider_operands.npz', 'local.json', 'local.npz', 'logits.npz', 'parent_event.npz']
    for name in required:
        (tmp_path / name).write_bytes(b'evidence')
    inventory = {name: sha(tmp_path / name) for name in required}
    bundle = dict(schema='ivgym-native-shape-witness-v1', artifacts_sha256=inventory,
                  case=dict(evidence_sha256=inventory['parent_event.npz']))
    file = tmp_path / 'bundle.json'
    file.write_text(json.dumps(bundle))
    assert load_bundle(file, sha(file)) == bundle
    with pytest.raises(ValueError, match='manifest digest'):
        load_bundle(file, '0' * 64)
    (tmp_path / 'local.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Artifact digest'):
        load_bundle(file, sha(file))
    inventory.pop('local.npz')
    file.write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match='Incomplete witness'):
        load_bundle(file, sha(file))


def test_witness_paths_cannot_escape_or_use_absolute_names(tmp_path):
    for name in ('../outside', str(tmp_path / 'absolute')):
        with pytest.raises(ValueError, match='escapes'):
            safe_file(tmp_path, name)


def test_unvalidated_or_one_family_only_summary_is_rejected():
    with pytest.raises(ValueError):
        summarize([])
    with pytest.raises(ValueError, match='control'):
        summarize([dict(case=dict(case_id='a', family='gemm'), controls=dict(sham=False))])


@pytest.mark.parametrize('family', ['gemm', 'attention'])
@pytest.mark.parametrize('position', [0, 2])
def test_tiny_model_binds_operands_and_recomputes_descendants(family, position):
    torch = pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from experiments.numerical.native_shape import NativeShapeHook, shape_output
    from experiments.numerical.native_shape_replay import provider_forward
    from experiments.numerical.mechanism_replay import forward
    from experiments.numerical.mechanism_ops import validate_controls
    from experiments.numerical.check_native_witness import intervention, alternate_inputs, operator

    torch.manual_seed(10)
    config = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                        num_attention_heads=2, num_key_value_heads=1, num_hidden_layers=2, head_dim=8)
    config._attn_implementation = 'eager'
    model = Qwen3ForCausalLM(config).to(torch.bfloat16).eval()
    ids = torch.tensor([[1, 5, 9, 2, 7, 3, 6]])
    case = dict(family=family, layer=0, position=position, token_index=2 + position, prompt_length=3)
    recipe = dict(diagnostics=True, attention_scale=model.model.layers[0].self_attn.scaling)
    states = {}
    states['native'], _ = forward(model, ids, case['token_index'], 3)
    with NativeShapeHook(model, case, recipe) as reference:
        states['capture'], _ = forward(model, ids, case['token_index'], 3)
    provider_case = dict(case, token_index=2 if position == 0 else 0)
    provider = NativeShapeHook(model, provider_case, recipe)
    baseline, _ = provider_forward(model, ids, 3, position)
    observed, _ = provider_forward(model, ids, 3, position, provider)
    assert baseline == observed
    layouts = tensor_layouts(provider.operands)
    with torch.inference_mode():
        alt = shape_output(family, reference.operands, recipe, layouts, case)
        independent = operator(family, alternate_inputs(family, reference.operands, provider.operands, case), recipe['attention_scale'])[:, -1:]
        assert torch.equal(alt, independent)
        replacements = dict(a=reference.native, b=alt)
        for mode in ('a', 'b', 'sham_a', 'sham_b', 'forward', 'reverse'):
            with NativeShapeHook(model, case, recipe, reference.operands, replacements, mode, provider_layouts=layouts):
                states[mode], _ = forward(model, ids, case['token_index'], 3)
        assert all(validate_controls(states, 0).values())
        with intervention(model, case, alt, reference.operands, native_output=reference.native):
            checked, _ = forward(model, ids, case['token_index'], 3)
        assert checked == states['b']
