"""Offline audit of the retained partition/bridge run; writes outputs under /tmp.

Run without Python's -O flag: assertions are the audit checks. No GPU execution.
"""
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
from experiments.analyze_real_quantization import digest
from experiments.numerical.dispatch_validation import kernel_launches
from experiments.numerical.frozen_logits import array_hash, widen_bf16
from experiments.numerical.mechanism_ops import sampling_matrix, select_cases, validate_controls
from experiments.numerical.mechanism_diagnostics import partition_coverage, summarize_diagnostics


def read(path):
    return json.loads(path.read_text())


def checked_inventory(base, inventory):
    for name, sha in inventory.items():
        assert digest(base / name) == sha, (base, name)
    return len(inventory)


def tensors(directory, stem):
    meta = read(directory / f'{stem}.json')
    with np.load(directory / f'{stem}.npz', allow_pickle=False) as z:
        assert set(z.files) == set(meta)
        values = {k: z[k].copy() for k in z.files}
    for k, a in values.items():
        assert list(a.shape) == meta[k]['shape']
        assert str(a.dtype) == {'torch.bfloat16': 'uint16', 'torch.float32': 'float32'}[meta[k]['dtype']]
    return values


def metrics(a, b):
    af = widen_bf16(a) if a.dtype == np.uint16 else a
    bf = widen_bf16(b) if b.dtype == np.uint16 else b
    return dict(changed=int(np.count_nonzero(a != b)), elements=int(a.size),
                max_abs=float(np.max(np.abs(af - bf))),
                rms=float(np.sqrt(np.mean((af - bf).astype(np.float64) ** 2))))


def ordered_sum(parts, reverse=False):
    values = parts[::-1] if reverse else parts
    result = values[0].copy()
    for part in values[1:]:
        result = result + part
    return result


if not __debug__:
    raise RuntimeError('Run the reviewer without -O; audit assertions must be enabled')
torch.set_num_threads(1)
run = ROOT / 'runs/numerical-m3-partition-bridge-20261021'
plan, summary = read(run / 'plan.json'), read(run / 'summary.json')
assert plan['round'] == summary['round'] == 'partition_bridge'
assert plan['recipe']['diagnostics'] and plan['recipe']['minimum_active_partitions'] == 3
assert plan['recipe']['attention_chunk'] == 8 and plan['recipe']['gemm_chunk'] == 64
assert read(run / 'status.json')['state'] == 'complete'
assert summary['m3_gate'] == 'pilot_complete_full_mechanism_gate_pending'
assert summary['historical_attribution'] == 'not_established'
assert summary['native_kernel_internals'] == 'not_inferred'
assert plan['discovery_only'] and not plan['injected_noise'] and not plan['m3_full_gate_automatic']
assert plan['cases_per_family'] == summary['cases_per_family'] == 32
assert plan['repeats'] == summary['independent_repeats'] == 3
campaign_hashes = checked_inventory(run, read(run / 'artifacts.json'))
inventory = read(run / 'source_inventory.json')
checked_inventory(run / 'source', inventory)
runtime_sources = {name: sha for name, sha in inventory.items() if name.endswith('.py')}
checked_inventory(ROOT, runtime_sources)
parent = Path(plan['parent_m1'])
assert digest(parent / 'summary.json') == plan['parent_summary_sha256']
assert digest(parent / 'capture/manifest.json') == plan['parent_capture_manifest_sha256']
assert read(parent / 'summary.json')['m1_gate'] == 'passed_for_historical_evaluation_corpus'
parent_artifacts = checked_inventory(parent, read(parent / 'artifacts.json'))
pm = read(parent / 'capture/manifest.json')
parent_capture_artifacts = checked_inventory(parent / 'capture', pm['artifacts_sha256'])
assert plan['cases'] == select_cases(read(parent / 'capture/events.json'), 32)
corpus = {r['prompt_id']: r for r in read(parent / 'capture/corpus.json')}
sequences = {r['prompt_id']: r for r in read(parent / 'capture/sequences.json')}
pf = Path(plan['preflight_from'])
assert read(pf / 'status.json')['state'] == 'complete'
checked_inventory(pf, read(pf / 'artifacts.json'))
assert read(pf / 'summary.json')['saved_operand_replay']['exact']
assert read(pf / 'summary.json')['cases_per_family'] == 2
assert {k: v for k, v in read(pf / 'source_inventory.json').items() if k.endswith('.py')} == runtime_sources
assert read(pf / 'plan.json')['recipe'] == plan['recipe']
assert read(pf / 'plan.json')['parent_m1'] == plan['parent_m1']

workers, base_records, base_arrays, matrices, cases = {}, None, {}, {}, []
native_matrices = {}
comparison = []
names = ['repeat_1', 'repeat_2', 'repeat_3', 'saved_operand_replay']
for worker in names:
    directory = run / worker
    manifest = read(directory / 'manifest.json')
    assert manifest['status'] == 'complete' and manifest['cases'] == 64
    assert manifest['plan_sha256'] == digest(run / 'plan.json')
    assert manifest['model_assets_sha256'] == pm['model_assets_sha256']
    for key in ('device', 'gpu_uuid', 'driver', 'software', 'python', 'cuda_runtime',
                'tf32_matmul', 'tf32_cudnn', 'torch_threads'):
        assert manifest['environment'][key] == pm['environment'][key], key
    count = checked_inventory(directory, manifest['artifacts_sha256'])
    actual_files = {str(p.relative_to(directory)) for p in directory.rglob('*')
                    if p.is_file() and p.name not in ('manifest.json', 'progress.json')}
    assert actual_files == set(manifest['artifacts_sha256'])
    records = read(directory / 'records.json')
    assert hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest() == manifest['comparison_sha256']
    assert [r['case']['case_id'] for r in records] == [c['case_id'] for c in plan['cases']]
    if base_records is None:
        base_records = records
    assert records == base_records
    traces = kernels = 0
    for original, r in zip(plan['cases'], records):
        c = r['case']
        cid, pid = c['case_id'], c['prompt_id']
        d = directory / cid
        assert read(d / 'record.json') == r
        idx = len(corpus[pid]['prompt_tokens']) - 1 + c['position']
        assert c == dict(original, token_index=idx)
        expected_controls = validate_controls(r['states'], c['layer'])
        for mode in ('native_sham', 'native_restore'):
            assert r['states'][mode] == r['states']['native']
            expected_controls[f'{mode}_exact'] = True
        assert r['controls'] == expected_controls
        assert r['recipe'] == dict(plan['recipe'], attention_scale=r['recipe']['attention_scale'])
        operands, local = tensors(d, 'operands'), tensors(d, 'local')
        assert {k: array_hash(a) for k, a in operands.items()} == r['operands']
        assert {k: array_hash(a) for k, a in local.items()} == r['local_hashes']
        assert metrics(local['a'], local['b']) == r['local_difference']
        assert {s: metrics(local['native'], local[s]) for s in ('a', 'b')} == r['native_bridge']
        diag = r['diagnostics']
        assert read(d / 'operands.json') == diag['operand_layouts']
        full_native = tensors(d, 'native_full')
        assert {k: array_hash(a) for k, a in full_native.items()} == diag['native_full_hashes']
        assert np.array_equal(full_native['native'], full_native['replay'])
        assert np.array_equal(full_native['native'][:, idx:idx + 1], local['native'])
        assert diag['native_full_exact'] and diag['partials_exact']
        fp32 = {s: tensors(d, f'fp32_{s}') for s in ('a', 'b')}
        for side, values in fp32.items():
            assert {k: array_hash(a) for k, a in values.items()} == diag['fp32_hashes'][side]
            assert all(a.dtype == np.float32 for a in values.values())
            assert np.isfinite(values['precast']).all()
            cast = torch.from_numpy(values['precast']).to(torch.bfloat16).view(torch.uint16).numpy()
            assert np.array_equal(cast, local[side])
            if c['family'] == 'gemm':
                assert np.array_equal(ordered_sum(values['partials'], side == 'b'), values['precast'])
            else:
                for part, total in (('partial_numerators', 'numerator'), ('partial_denominators', 'denominator')):
                    assert np.array_equal(ordered_sum(values[part], side == 'b'), values[total])
        for key in set(fp32['a']) - {'precast', 'numerator', 'denominator'}:
            assert np.array_equal(fp32['a'][key], fp32['b'][key])
        assert metrics(fp32['a']['precast'], fp32['b']['precast']) == diag['fp32_difference']
        if c['family'] == 'attention':
            mask = widen_bf16(operands['mask'])
            assert partition_coverage(mask, 8, minimum=3) == diag['coverage']
            previous = tensors(d, 'chunk32_control')
            control = diag['chunk32_control']
            assert {k: array_hash(a) for k, a in previous.items()} == control['hashes']
            assert partition_coverage(mask, 32, minimum=1) == control['coverage']
            assert metrics(previous['a'], previous['b']) == control['local_difference']
            assert {s: metrics(previous[s], local[s]) for s in ('a', 'b')} == control['to_chunk8']
        else:
            assert diag['coverage'] is None
        launches = kernel_launches(d / 'native_bridge_trace.json')
        assert dict(kernels=len(launches), launches_sha256=hashlib.sha256(json.dumps(launches).encode()).hexdigest()) == diag['native_dispatch']
        traces += 1
        kernels += len(launches)
        for side, order in (('a', 'ascending'), ('b', 'descending')):
            launches = kernel_launches(d / f'{order}_trace.json')
            signature = dict(kernels=len(launches), launches_sha256=hashlib.sha256(json.dumps(launches).encode()).hexdigest())
            assert signature == r['dispatch'][side]
            traces += 1
            kernels += len(launches)
        with np.load(d / 'logits.npz', allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        hashes = {k: array_hash(a) for k, a in arrays.items()}
        if worker == 'repeat_1':
            base_arrays[cid] = hashes
        assert hashes == base_arrays[cid]
        assert arrays['input_ids'].tolist() == [corpus[pid]['prompt_tokens'] + sequences[pid]['claimed_tokens']]
        event_path = parent / 'capture' / c['evidence']
        assert digest(event_path) == c['evidence_sha256']
        with np.load(event_path, allow_pickle=False) as z:
            assert np.array_equal(arrays['native'], z['reference_bf16'])
            assert np.array_equal(arrays['gumbel'], z['reference_gumbel'])
        with np.load(parent / 'capture' / f'prompt_{pid}/reference_0_state.npz', allow_pickle=False) as z:
            assert r['states']['native']['kv'] == [
                {name: array_hash(z[f'kv_{i}_{name}']) for name in ('keys', 'values')} for i in range(28)]
        for a, b in (('native', 'capture'), ('a', 'sham_a'), ('b', 'sham_b'), ('b', 'forward'), ('a', 'reverse'),
                     ('native', 'native_sham'), ('native', 'native_restore')):
            assert np.array_equal(arrays[a], arrays[b])
        if worker == 'repeat_1':
            matrices[cid] = sampling_matrix(widen_bf16(arrays['a']), widen_bf16(arrays['b']), arrays['gumbel'])
            native_matrices[cid] = {s: sampling_matrix(widen_bf16(arrays['native']), widen_bf16(arrays[s]), arrays['gumbel'])
                                    for s in ('a', 'b')}
        assert matrices[cid] == r['sampling']
        assert native_matrices[cid] == diag['native_sampling']
        assert (not np.array_equal(arrays['a'], arrays['b'])) == diag['selected_logits_changed']
        primary = next(s for s in r['sampling'] if (s['dtype'], s['convention'], s['mode']) == ('float32', 'legacy', 'top_k_top_p'))
        assert primary['token_changed'] == r['primary_token_changed']
        assert primary['support_delta'] == r['primary_support_delta']
        if worker == 'repeat_1':
            detail = dict(case=c, local_difference=r['local_difference'], native_bridge=r['native_bridge'],
                          full_logits_changed=r['states']['a']['logits'] != r['states']['b']['logits'],
                          selected_logits_changed=int(np.count_nonzero(arrays['a'] != arrays['b'])),
                          primary_token_changed=r['primary_token_changed'], primary_support_delta=r['primary_support_delta'],
                          any_sampling_token_changed=any(s['token_changed'] for s in r['sampling']),
                          any_sampling_support_changed=any(s['support_delta'] for s in r['sampling']),
                          local_hashes=r['local_hashes'], dispatch=r['dispatch'],
                          fp32_difference=diag['fp32_difference'],
                          changed_sampling_cells=[cell for cell in r['sampling'] if cell['token_changed'] or cell['support_delta']],
                          native_sampling_changes={s: [cell for cell in cells if cell['token_changed'] or cell['support_delta']]
                                                   for s, cells in diag['native_sampling'].items()})
            if c['family'] == 'attention':
                mask = widen_bf16(operands['mask']) if operands['mask'].dtype == np.uint16 else operands['mask']
                chunk = r['recipe']['attention_chunk']
                detail['allowed_keys'] = int(np.count_nonzero(mask[0, 0, 0] >= -1e20))
                detail['active_partitions'] = sum(bool(np.any(mask[..., i:i + chunk] >= -1e20)) for i in range(0, mask.shape[-1], chunk))
                detail['chunk32_control'] = diag['chunk32_control']
            cases.append(detail)
    workers[worker] = dict(artifacts_checked=count, traces_checked=traces, kernels_checked=kernels,
                           manifest_sha256=digest(directory / 'manifest.json'),
                           peak_cuda_allocated_bytes=manifest['peak_cuda_allocated_bytes'])
    comparison.append(dict(worker=worker, exact=True, cases=64))
    print(f'Verified {worker}: {workers[worker]}', flush=True)

assert comparison == read(run / 'comparisons.json')
assert comparison[:-1] == summary['comparisons'] and comparison[-1] == summary['saved_operand_replay']
families, extra = {}, {}
for family in ('gemm', 'attention'):
    rows = [r for r in cases if r['case']['family'] == family]
    families[family] = dict(cases=len(rows), strata=dict(Counter(r['case']['stratum'] for r in rows)),
                            local_changed=sum(r['local_difference']['changed'] > 0 for r in rows),
                            logits_changed=sum(r['full_logits_changed'] for r in rows),
                            primary_support_changed=sum(r['primary_support_delta'] > 0 for r in rows),
                            primary_token_changed=sum(r['primary_token_changed'] for r in rows),
                            any_sampling_cell_changed=sum(r['any_sampling_token_changed'] for r in rows))
    families[family]['diagnostics'] = summarize_diagnostics([r for r in base_records if r['case']['family'] == family])
    extra[family] = dict(selected_logits_changed_cases=sum(r['selected_logits_changed'] > 0 for r in rows),
                         any_sampling_support_changed_cases=sum(r['any_sampling_support_changed'] for r in rows),
                         native_bridge_changed_cases={s: sum(r['native_bridge'][s]['changed'] > 0 for r in rows) for s in ('a', 'b')},
                         changed_cases=[r['case']['case_id'] for r in rows if r['full_logits_changed']])
    if family == 'attention':
        extra[family]['active_partitions_histogram'] = dict(Counter(r['active_partitions'] for r in rows))
        extra[family]['allowed_keys_range'] = [min(r['allowed_keys'] for r in rows), max(r['allowed_keys'] for r in rows)]
        extra[family]['chunk32_local_changed_cases'] = sum(r['chunk32_control']['local_difference']['changed'] > 0 for r in rows)
        extra[family]['chunk32_to_chunk8_changed_cases'] = {s: sum(r['chunk32_control']['to_chunk8'][s]['changed'] > 0 for r in rows)
                                                         for s in ('a', 'b')}
assert families == summary['families']
assert summary['peak_cuda_allocated_bytes'] == max(w['peak_cuda_allocated_bytes'] for w in workers.values())
report = (run / 'report.md').read_text()
for family, counts in families.items():
    assert f'- {family}: {json.dumps(counts)}' in report
assert 'Cases per family: 32; independent repeats: 3.' in report
assert 'The full M3 mechanism gate remains pending' in report
files = [p for p in run.rglob('*') if p.is_file()]
review = dict(schema='ivgym-m3-partition-bridge-review-v1', reviewed_utc=datetime.now(timezone.utc).isoformat(),
              result='passed', families=families, additional_results=extra, workers=workers,
              campaign_artifact_hashes_checked=campaign_hashes, frozen_source_files_checked=len(inventory),
              current_python_sources_matched=len(runtime_sources), parent_artifact_hashes_checked=parent_artifacts,
              parent_capture_artifact_hashes_checked=parent_capture_artifacts,
              worker_artifact_hashes_checked=sum(w['artifacts_checked'] for w in workers.values()),
              raw_traces_recomputed=sum(w['traces_checked'] for w in workers.values()),
              kernel_launches_recomputed=sum(w['kernels_checked'] for w in workers.values()),
              unique_sampling_cells_recomputed=sum(len(m) for m in matrices.values()),
              native_sampling_cells_recomputed=sum(len(m) for pair in native_matrices.values() for m in pair.values()),
              fp32_case_pairs_checked=256, native_full_output_pairs_checked=256,
              attention_chunk32_pairs_checked=128,
              repeated_selected_arrays_exact=True, cases_checked=256,
              source_sha256={k: v for k, v in runtime_sources.items() if k.startswith('experiments/numerical/') or k.startswith('tests/test_mechanism')},
              summary_sha256=digest(run / 'summary.json'), original_report_sha256=digest(run / 'report.md'),
              run_bytes=sum(p.stat().st_size for p in files), run_files=len(files),
              environment=pm['environment'], model=pm['plan']['model'], revision=pm['plan']['revision'],
              scope='Offline hash, raw local/FP32/native-output tensor, ordered partial-sum, BF16 cast, coverage, chunk-32 control, selected-logit/noise/checkpoint, parent KV, A/B and native sampling, and trace review. Full-model state hashes compared across recorded controls/repeats. No new GPU model or operator execution; attention division and native operators are not independently re-executed.')
Path('/tmp/m3_partition_bridge_review.json').write_text(json.dumps(review, indent=2) + '\n')
Path('/tmp/m3_partition_bridge_cases.json').write_text(json.dumps(cases, indent=2) + '\n')
print(json.dumps(dict(families=families, additional_results=extra, run_bytes=review['run_bytes'], run_files=len(files)), indent=2))
