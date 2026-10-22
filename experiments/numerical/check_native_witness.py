"""Independent witness checker: no imports from IVGym or the diagnosis runner.

Run this file directly with a witness index and its externally recorded SHA256.
Requires the pinned model cache, NumPy, PyTorch and Transformers. It reconstructs
native operators and complete descendants, rather than trusting stored verdicts.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from unittest.mock import patch

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_file(root, name):
    path = (root / name).resolve()
    require(not Path(name).is_absolute() and path.is_relative_to(root.resolve()), 'Witness path escapes bundle')
    return path


def load_bundle(file, expected_sha256):
    require(sha(file) == expected_sha256, 'Witness manifest digest differs')
    bundle = read(file)
    require(bundle['schema'] == 'ivgym-native-shape-witness-v1', 'Unsupported witness schema')
    artifacts = bundle['artifacts_sha256']
    required = {'record.json', 'operands.json', 'operands.npz', 'provider_operands.json',
                'provider_operands.npz', 'local.json', 'local.npz', 'logits.npz', 'parent_event.npz'}
    require(required <= set(artifacts), 'Incomplete witness inventory')
    for name, expected in artifacts.items():
        require(sha(safe_file(file.parent, name)) == expected, f'Artifact digest differs: {name}')
    require(artifacts['parent_event.npz'] == bundle['case']['evidence_sha256'], 'Parent event identity differs')
    return bundle


def array_hash(array):
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(json.dumps(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def raw(tensor):
    import torch
    tensor = tensor.detach()
    return (tensor.view(torch.uint16) if tensor.dtype == torch.bfloat16 else tensor).cpu().numpy().copy()


def thash(tensor):
    return array_hash(raw(tensor))


def load_tensors(root, name):
    import torch
    meta = read(root / f'{name}.json')
    values = {}
    with np.load(root / f'{name}.npz', allow_pickle=False) as data:
        require(set(data.files) == set(meta), 'Tensor inventory differs')
        for key, spec in meta.items():
            tensor = torch.from_numpy(data[key].copy())
            if spec['dtype'] == 'torch.bfloat16':
                require(tensor.dtype == torch.uint16, 'Invalid BF16 bit storage')
                tensor = tensor.view(torch.bfloat16)
            require(str(tensor.dtype) == spec['dtype'] and list(tensor.shape) == spec['shape'], 'Tensor metadata differs')
            target = torch.empty_strided(spec['shape'], spec['stride'], dtype=tensor.dtype, device='cuda')
            target.copy_(tensor)
            values[key] = target
    return values


def operator(family, operands, scale):
    import torch
    if family == 'gemm':
        return torch.nn.functional.linear(operands['full_x'], operands['weight'])
    require(family == 'attention', 'Unknown operator family')
    q, k, v = (operands[name] for name in ('full_query', 'key', 'value'))
    scores = (q @ k.transpose(-2, -1)) * scale
    probabilities = torch.softmax(scores + operands['full_mask'][..., :k.shape[-2]], dim=-1,
                                  dtype=torch.float32).to(q.dtype)
    return (probabilities @ v).transpose(1, 2).contiguous()


def alternate_inputs(family, original, provider, case, query_only=False):
    import torch
    end = case['prompt_length'] + case['position']
    start = 0 if case['position'] == 0 else end - 1
    require(end == case['token_index'] + 1, 'Checkpoint index mismatch')
    if family == 'gemm':
        selected = {'full_x': original['full_x'][:, start:end], 'weight': original['weight']}
    else:
        mask = original['full_mask'][:, :, start:end]
        require(bool((mask[..., end:] < -1e20).all()), 'Removed KV includes an allowed key')
        length = original['key'].shape[-2] if query_only else end
        selected = {'full_query': original['full_query'][:, :, start:end],
                    'key': original['key'][:, :, :length], 'value': original['value'][:, :, :length],
                    'full_mask': mask[..., :length]}
    packed = {}
    for name, value in selected.items():
        if query_only and name in ('key', 'value', 'full_mask'):
            packed[name] = value.contiguous()
        else:
            target = provider[name]
            require(target.shape == value.shape and target.dtype == value.dtype, 'Provider shape mismatch')
            packed[name] = torch.empty_strided(target.shape, target.stride(), dtype=target.dtype, device=value.device)
            packed[name].copy_(value)
    return packed


def sample(values, noise, dtype, convention, mode):
    x = values.astype(dtype)
    require(x.ndim == 1 and x.shape == noise.shape and np.isfinite(x).all() and np.isfinite(noise).all(), 'Invalid sampler inputs')
    require(convention in ('legacy', 'crossing_inclusive_v1') and mode in ('greedy', 'gumbel', 'top_k', 'top_p', 'top_k_top_p'), 'Unknown sampling contract')
    filt = x.copy()
    if mode in ('top_k', 'top_k_top_p') and len(x) > 50:
        filt[x < np.partition(x, -50)[-50]] = -1e30
    if mode in ('top_p', 'top_k_top_p'):
        candidates = np.flatnonzero(filt > -5e29)
        ranked = candidates[np.argsort(filt[candidates])[::-1]]
        weights = np.exp(filt[ranked] - filt[ranked].max())
        cumulative = np.cumsum(weights / weights.sum())
        keep = cumulative <= .95
        if convention == 'crossing_inclusive_v1':
            crossing = np.flatnonzero(cumulative > .95)
            if crossing.size:
                keep[crossing[0]] = True
        keep[0] = True
        filt[:] = -1e30
        filt[ranked[keep]] = x[ranked[keep]]
    return int(np.argmax(filt + (0.0 if mode == 'greedy' else 1.0) * noise)), filt > -5e29


def widen(bits):
    require(bits.dtype == np.uint16, 'Expected native BF16 bits')
    return (bits.astype(np.uint32) << 16).view(np.float32)


def check_matrix(cells, left, right, noise):
    expected = {(d, c, m) for d in ('float32', 'float64') for c in ('legacy', 'crossing_inclusive_v1')
                for m in ('greedy', 'gumbel', 'top_k', 'top_p', 'top_k_top_p')}
    keys = [(c['dtype'], c['convention'], c['mode']) for c in cells]
    require(len(keys) == 20 and set(keys) == expected, 'Incomplete sampling matrix')
    results = {}
    for cell in cells:
        key = cell['dtype'], cell['convention'], cell['mode']
        a, am = sample(widen(left), noise, *key)
        b, bm = sample(widen(right), noise, *key)
        delta = int(np.count_nonzero(am != bm))
        require((a, b, a != b, delta) == (cell['a']['selected_token'], cell['b']['selected_token'],
                                        cell['token_changed'], cell['support_delta']), 'Sampling predicate differs')
        results[key] = dict(token_changed=a != b, support_delta=delta)
    return results


def state(out, index):
    logits = raw(out.logits)
    return dict(logits=array_hash(logits), hidden=thash(out.hidden_states[-1]),
                prefix_logits=array_hash(logits[:, :index]),
                kv=[{key: thash(getattr(layer, key)) for key in ('keys', 'values')}
                    for layer in out.past_key_values.layers]), logits[0, index].copy()


@contextmanager
def intervention(model, case, replacement, expected_operands, native_output=None):
    """The checker only installs the saved, independently recomputed row."""
    import torch
    from transformers.models.qwen3 import modeling_qwen3 as qwen
    calls = []
    layer = model.model.layers[case['layer']]
    index = case['token_index']
    if case['family'] == 'gemm':
        def hook(module, args, output):
            require(torch.equal(args[0], expected_operands['full_x']), 'Model GEMM input drift')
            require(torch.equal(module.weight, expected_operands['weight']), 'Model GEMM weight drift')
            if native_output is not None:
                require(torch.equal(output[:, index:index + 1], native_output), 'Native model GEMM output drift')
            calls.append(1)
            if replacement is None:
                return output
            updated = output.clone()
            updated[:, index:index + 1] = replacement
            return updated
        handle = layer.mlp.down_proj.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()
    else:
        original = qwen.eager_attention_forward
        def wrapped(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            output, weights = original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
            if module is not layer.self_attn:
                return output, weights
            actual = dict(full_query=query, key=qwen.repeat_kv(key, module.num_key_value_groups),
                          value=qwen.repeat_kv(value, module.num_key_value_groups), full_mask=attention_mask)
            require(all(torch.equal(actual[k], expected_operands[k]) for k in actual), 'Model attention input drift')
            if native_output is not None:
                require(torch.equal(output[:, index:index + 1], native_output), 'Native model attention output drift')
            calls.append(1)
            if replacement is None:
                return output, weights
            updated = output.clone()
            updated[:, index:index + 1] = replacement
            return updated, weights
        with patch.object(qwen, 'eager_attention_forward', wrapped):
            yield
    require(len(calls) == 1, 'Intervention call count differs')


def check_case(file, expected_sha256, model):
    import torch
    bundle = load_bundle(file, expected_sha256)
    root, case = file.parent, bundle['case']
    record = read(root / 'record.json')
    require(record['case'] == case and record['recipe'] == bundle['recipe'], 'Bundle identity mismatch')
    operands = load_tensors(root, 'operands')
    provider = load_tensors(root, 'provider_operands')
    local = load_tensors(root, 'local')
    with np.load(root / 'logits.npz', allow_pickle=False) as arrays:
        logits = {key: arrays[key].copy() for key in arrays.files}
    with np.load(root / 'parent_event.npz', allow_pickle=False) as event:
        require(np.array_equal(event['provider_gumbel'], event['reference_gumbel'])
                and np.array_equal(logits['gumbel'], event['reference_gumbel']), 'Historical random tensor differs')
        require(np.array_equal(logits['native'], event['reference_bf16'])
                and np.array_equal(logits['provider'], event['provider_bf16']), 'Historical logits differ')
        primary_token, _ = sample(widen(event['provider_bf16']), logits['gumbel'], 'float32', 'legacy', 'top_k_top_p')
        require(primary_token == int(event['claimed_token']), 'Historical claimed token differs')
    family, scale, idx = case['family'], bundle['recipe']['attention_scale'], case['token_index']
    with torch.inference_mode():
        full = operator(family, operands, scale)
        provider_full = operator(family, provider, scale)
        alternate = alternate_inputs(family, operands, provider, case)
        a, b = full[:, idx:idx + 1], operator(family, alternate, scale)[:, -1:]
        query = operator(family, alternate_inputs(family, operands, provider, case, True), scale)[:, -1:]
        for name, actual in dict(a=a, b=b, query_only=query, native_full=full,
                                 provider_full=provider_full, provider=provider_full[:, -1:]).items():
            require(torch.equal(actual, local[name]), f'Independent native operator differs: {name}')
        require(int(np.count_nonzero(raw(a) != raw(b))) == record['local_difference']['changed'], 'Local difference predicate differs')
        equal_inputs = all(torch.equal(value, provider[key]) for key, value in alternate.items())
        require(equal_inputs == record['historical_bridge']['effective_operands_exact'], 'Effective-input predicate differs')
        require(torch.equal(b, local['provider']) == record['historical_bridge']['alternate_provider_output_exact'], 'Native bridge predicate differs')
        ids = torch.from_numpy(logits['input_ids']).to('cuda')
        observed = {}
        for mode, replacement in (('native', None), ('a', a), ('b', b), ('forward', b), ('reverse', a)):
            with ExitStack() as stack:
                if replacement is not None:
                    stack.enter_context(intervention(model, case, replacement, operands, native_output=a))
                out = model(ids, use_cache=True, output_hidden_states=True)
                actual_state, selected = state(out, idx)
            require(actual_state == record['states'][mode], f'Independent descendant state differs: {mode}')
            require(np.array_equal(selected, logits[mode]), f'Selected logit bits differ: {mode}')
            observed[mode] = actual_state
            del out
        for name in ('capture', 'a', 'sham_a', 'reverse'):
            require(record['states'][name] == observed['native'], f'Native/sham control differs: {name}')
        require(record['states']['sham_b'] == observed['b'] == observed['forward'], 'Forward/sham control differs')
        for value in observed.values():
            require(value['prefix_logits'] == observed['native']['prefix_logits']
                    and value['kv'][:case['layer'] + 1] == observed['native']['kv'][:case['layer'] + 1], 'Intervention altered ancestors')
        # Independent provider reconstruction starts from empty KV as well.
        previous, prefix = None, []
        for pos in range(case['position'] + 1):
            plen = case['prompt_length']
            tokens = ids[:, :plen] if pos == 0 else ids[:, plen + pos - 1:plen + pos]
            kwargs = {} if pos == 0 else dict(past_key_values=previous.past_key_values)
            with ExitStack() as stack:
                if pos == case['position']:
                    provider_case = dict(case, token_index=plen - 1 if pos == 0 else 0)
                    stack.enter_context(intervention(model, provider_case, None, provider, native_output=local['provider']))
                previous = model(tokens, use_cache=True, output_hidden_states=True, **kwargs)
            selected = raw(previous.logits[0, -1])
            if pos < case['position']:
                prefix.append(array_hash(selected))
        provider_state = dict(logits=thash(previous.logits), hidden=thash(previous.hidden_states[-1]),
                              prefix_logits=prefix, kv=[{key: thash(getattr(layer, key)) for key in ('keys', 'values')}
                                                       for layer in previous.past_key_values.layers])
        require(provider_state == record['provider_state'] and np.array_equal(selected, logits['provider']), 'Independent provider reconstruction differs')
        del previous
    noise = logits['gumbel']
    matrix = check_matrix(record['sampling'], logits['a'], logits['b'], noise)
    bridge = record['historical_bridge']
    before = check_matrix(bridge['native_to_provider_sampling'], logits['native'], logits['provider'], noise)
    after = check_matrix(bridge['alternate_to_provider_sampling'], logits['b'], logits['provider'], noise)
    key = ('float32', 'legacy', 'top_k_top_p')
    require(matrix[key]['token_changed'] == record['primary_token_changed']
            and matrix[key]['support_delta'] == record['primary_support_delta'], 'Primary sampling summary differs')
    require((before[key]['token_changed'] and not after[key]['token_changed']) == bridge['primary_disagreement_removed'], 'Historical reconciliation predicate differs')
    require((not before[key]['token_changed'] and after[key]['token_changed']) == bridge['primary_disagreement_introduced'], 'Introduced disagreement predicate differs')
    require((not np.array_equal(logits['a'], logits['b'])) == record['selected_logits_changed'], 'Selected-logit predicate differs')
    require(np.array_equal(logits['b'], logits['provider']) == bridge['alternate_provider_logits_exact'], 'Provider-logit equality predicate differs')
    return dict(case_id=case['case_id'], verified=True, primary_token_changed=record['primary_token_changed'],
                local_changed=record['local_difference']['changed'] > 0,
                primary_disagreement_removed=bridge['primary_disagreement_removed'], historical_root_cause='not_established')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Checker output already exists')
    require(sha(args.index) == args.expected_sha256, 'Witness index digest differs')
    entries = read(args.index)
    require(entries and len({e['case_id'] for e in entries}) == len(entries), 'Empty or duplicate witness index')
    bundles = [load_bundle(safe_file(args.index.parent, e['file']), e['sha256']) for e in entries]
    require(all(b['case']['case_id'] == e['case_id'] for b, e in zip(bundles, entries)), 'Index case identities differ')
    first = bundles[0]
    require(all((b['model'], b['revision'], b['environment'], b['model_assets_sha256']) ==
                (first['model'], first['revision'], first['environment'], first['model_assets_sha256']) for b in bundles), 'Mixed witness environments')
    import fcntl
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM
    environment = first['environment']
    require(platform.python_version() == environment['python'], 'Python version differs')
    require(all(importlib.metadata.version(name) == version for name, version in environment['software'].items()), 'Software environment differs')
    device = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    uuid = subprocess.check_output(['nvidia-smi', '-i', device, '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    require(uuid == environment['gpu_uuid'], 'GPU identity differs; transfer is a separate experiment')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip() == environment['driver'], 'Driver differs')
    snapshot = Path(snapshot_download(first['model'], revision=first['revision'], local_files_only=True))
    require(all(sha(snapshot / name) == expected for name, expected in first['model_assets_sha256'].items()), 'Model asset digest differs')
    with ExitStack() as stack:
        for suffix, flag in (('', fcntl.LOCK_SH), (f'-{uuid}', fcntl.LOCK_EX)):
            lock = stack.enter_context(open(f'/tmp/ivgym-research-{os.getuid()}{suffix}.lock', 'a'))
            fcntl.flock(lock.fileno(), flag | fcntl.LOCK_NB)
        require(not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'GPU already in use')
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        require(torch.version.cuda == environment['cuda_runtime'] and torch.cuda.get_device_name(0) == environment['device'], 'CUDA environment differs')
        model = AutoModelForCausalLM.from_pretrained(str(snapshot), torch_dtype=torch.bfloat16,
                    attn_implementation='eager', device_map={'': 0}, local_files_only=True).eval()
        started, results = time.monotonic(), []
        for entry in entries:
            result = check_case(safe_file(args.index.parent, entry['file']), entry['sha256'], model)
            results.append(result)
            print(json.dumps(dict(completed=len(results), total=len(entries), **result)), flush=True)
        args.output.write_text(json.dumps(dict(schema='ivgym-independent-native-witness-review-v1',
                              verified=True, index_sha256=args.expected_sha256, checker_sha256=sha(__file__),
                              cases=len(results), results=results, elapsed_seconds=time.monotonic() - started), indent=2) + '\n')


if __name__ == '__main__':
    main()
