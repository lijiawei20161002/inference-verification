"""Chunked setup must preserve the full-context cache and next decode output."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from experiments.exp_clock_channel_gpu import prefill_cache


@pytest.mark.parametrize("cache_type", ["dynamic", "static"])
def test_chunked_prefill_matches_full_context(cache_type):
    torch.manual_seed(17)
    config = transformers.Qwen3Config(
        vocab_size=97, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64, attention_dropout=0.0,
    )
    model = transformers.Qwen3ForCausalLM(config).eval()
    ids = torch.randint(0, config.vocab_size, (2, 13))

    def make_cache():
        if cache_type == "static":
            return transformers.StaticCache(config=config, max_cache_len=24)
        return transformers.DynamicCache()

    reference, chunked = make_cache(), make_cache()
    calls = []

    def observe(_model, _args, kwargs, output):
        calls.append((kwargs["cache_position"].tolist(), output.logits.shape[1]))

    with torch.inference_mode():
        original = model(input_ids=ids, past_key_values=reference, use_cache=True,
                         logits_to_keep=1)
        expected_token = original.logits.argmax(-1)
        hook = model.register_forward_hook(observe, with_kwargs=True)
        try:
            token = prefill_cache(model, ids, chunked, chunk_tokens=4)
        finally:
            hook.remove()
        assert calls == [([0, 1, 2, 3], 1), ([4, 5, 6, 7], 1),
                         ([8, 9, 10, 11], 1), ([12], 1)]
        assert int(chunked.get_seq_length()) == ids.shape[1]
        torch.testing.assert_close(token, expected_token)
        for expected_layer, actual_layer in zip(reference.layers, chunked.layers):
            torch.testing.assert_close(actual_layer.keys, expected_layer.keys,
                                       atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(actual_layer.values, expected_layer.values,
                                       atol=1e-5, rtol=1e-5)
        expected = model(input_ids=expected_token, past_key_values=reference,
                         use_cache=True).logits
        actual = model(input_ids=token, past_key_values=chunked, use_cache=True).logits
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
