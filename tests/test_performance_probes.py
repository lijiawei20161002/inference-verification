"""Tests of experimental contracts, independent cache correctness and pairing."""
import dataclasses
import pytest
from experiments.exp_performance_probe_gpu import Case, build_plan, layout

def test_truncation_preserves_absolute_positions_and_negative_control():
    assert layout(512, "truncate") == layout(512, "full")
    assert layout(8192, "truncate") == {"effective": 2048, "offset": 6144, "prefix": 0}
    assert layout(8192, "cache_hit") == {"effective": 8192, "offset": 0, "prefix": 8064}
    with pytest.raises(ValueError):
        layout(0, "full")

def test_plan_is_paired_and_has_expected_independent_blocks():
    plans = [build_plan("pilot", blocks, s) for s, blocks in enumerate(([0,1,2],[3,4,5],[6,7]))]
    all_cases = [c for plan in plans for c in plan]
    p1 = [c for c in all_cases if c.suite == "p1"]
    assert len(p1) == 192
    assert len({(c.block,c.length,c.batch) for c in p1}) == 96
    assert len([c for c in all_cases if c.suite == "p0"]) == 160
    assert len([c for c in all_cases if c.suite == "p4"]) == 128
    assert len({c.key for c in all_cases}) == len(all_cases)
    assert plans[0] == build_plan("pilot", [0,1,2], 0)

def test_prefix_reuse_and_truncation_match_independent_tiny_model():
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    from experiments.exp_performance_probe_gpu import allocate_cache, forward_range
    torch.manual_seed(19)
    cfg = tr.Qwen3Config(vocab_size=101, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128)
    model = tr.Qwen3ForCausalLM(cfg).eval()
    ids = torch.randint(0,101,(2,17))
    with torch.inference_mode():
        full = allocate_cache(model,2,24)
        expected = forward_range(model,ids,full,0,17,chunk=5)
        reused = allocate_cache(model,2,24)
        forward_range(model,ids,reused,0,12,chunk=5)
        actual = forward_range(model,ids,reused,12,17,chunk=5)
        torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
        tail = ids[:,-8:]
        shifted = allocate_cache(model,2,16)
        actual = forward_range(model,tail,shifted,0,8,offset=9,chunk=4)
        positions = torch.arange(9,17)[None,:]
        expected = model(input_ids=tail,position_ids=positions,use_cache=False).logits[:,-1,:]
        torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
        assert sum(l.keys.numel()*l.keys.element_size()*2 for l in reused.layers) == 2*2*2*8*24*2*4
