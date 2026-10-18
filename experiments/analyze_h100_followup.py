"""Validate archived H100 follow-up artifacts and rebuild their summary, without CUDA.

    python -m experiments.analyze_h100_followup --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from experiments.exp_frozen_replay_gpu import DETECTORS, SHAPES, paired_d_prime

DATA = Path(__file__).resolve().parents[1] / 'docs/results/h100_followup_20261018'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timing_summary(data_dir, name, steps, reps):
    manifest = read(data_dir / f'{name}-manifest.json')
    path = data_dir / f'{name}.json'
    if (manifest['status'] != 'complete' or manifest['returncode'] != 0
            or digest(path) != manifest['artifact_sha256']):
        raise ValueError(f'Incomplete or modified timing run: {name}')
    for model, revision in manifest['model_lock'].items():
        if any(manifest[k].get(model) != revision
               for k in ('model_refs_before', 'model_refs_after')):
            raise ValueError(f'Model revision changed: {name}')
    data = read(path)
    contexts = ({256, 512, 2048, 8192, 16384, 32768} if manifest['job'] == 'clock-window'
                else {256, 1024, 4096, 8192, 16384, 32768})
    if ((data['steps'], data['reps']) != (steps, reps) or len(data['cells']) != 6
            or {c['ctx'] for c in data['cells']} != contexts):
        raise ValueError(f'Incomplete timing grid: {name}')
    rows = []
    for c in sorted(data['cells'], key=lambda c: c['ctx']):
        if (c['B'] != 1 or c['mode'] != 'graph'
                or c['model'] != 'Qwen/Qwen3-1.7B'
                or len(c['itl_ms']) != steps * reps or len(c['rep_means']) != reps
                or not all(math.isfinite(x) and x >= 0 for x in c['itl_ms'])
                or not all(math.isfinite(x) and x >= 0 for x in c['rep_means'])
                or not math.isclose(mean(c['itl_ms']), c['mean'], abs_tol=0.000051, rel_tol=0)
                or not math.isclose(mean(c['rep_means']), c['mean'], abs_tol=1e-9, rel_tol=0)):
            raise ValueError(f'Invalid samples or means: {name}/{c["ctx"]}')
        rows.append(dict(context=c['ctx'], mean_ms=c['mean'],
                         repetition_means_ms=c['rep_means']))
    means = {r['context']: r['mean_ms'] for r in rows}
    return dict(cells=rows, gap_ms=means[32768]-means[256],
                elapsed_seconds=manifest['elapsed_seconds'],
                started_utc=manifest['started_utc'], finished_utc=manifest['finished_utc'])


def frozen_summary(data_dir, prefix, n, t):
    capture = read(data_dir / f'{prefix}capture-manifest.json')
    replay = read(data_dir / f'{prefix}replay-manifest.json')
    saved = read(data_dir / f'{prefix}token-summary.json')
    if (capture['status'] != 'complete' or replay['status'] != 'complete'
            or capture['command'] != 'capture' or replay['command'] != 'replay'
            or capture['capture_sha256'] != replay['capture_sha256']
            or saved['capture_sha256'] != capture['capture_sha256']
            or digest(data_dir / f'{prefix}scores.npz') != replay['scores_sha256']
            or digest(data_dir / f'{prefix}token-summary.json') != replay['summary_sha256']):
        raise ValueError('Incomplete or mismatched frozen evidence')
    for record in (capture, replay, saved):
        if (record['n_prompts'], record['n_tokens']) != (n, t):
            raise ValueError('Unexpected frozen evidence dimensions')
    for key in ('source_sha256', 'software', 'cuda_runtime', 'tf32_matmul', 'tf32_cudnn', 'torch_threads'):
        if capture['provenance'][key] != replay['provenance'][key]:
            raise ValueError(f'Capture/replay conditions differ: {key}')
    expected_bootstrap = dict(unit='sequence', draws=400, seed=0, interval=0.90)
    if saved['bootstrap'] != expected_bootstrap:
        raise ValueError('Unexpected bootstrap contract')
    comparisons = {}
    with np.load(data_dir / f'{prefix}scores.npz', allow_pickle=False) as scores:
        expected = {f'{s}/{d}' for s in (*SHAPES, 'rerun') for d in DETECTORS}
        if set(scores.files) != expected:
            raise ValueError('Missing or extra score arrays')
        for key in expected:
            if scores[key].shape != (n, t) or not np.isfinite(scores[key]).all():
                raise ValueError(f'Invalid score array: {key}')
        for d in DETECTORS:
            base = scores[f'chunked_b1/{d}']
            if (not replay['same_shape_exact'][d] or not saved['same_shape_exact'][d]
                    or not np.array_equal(base, scores[f'rerun/{d}'])):
                raise ValueError(f'Failed same-shape control: {d}')
            for shape in SHAPES[1:]:
                row = paired_d_prime(base.ravel(), scores[f'{shape}/{d}'].ravel(), n, t, 400)
                if row != saved['comparisons'][shape][d]:
                    raise ValueError(f'Bootstrap summary does not reproduce: {shape}/{d}')
                comparisons.setdefault(shape, {})[d] = row
    return dict(capture_sha256=capture['capture_sha256'], n_prompts=n, tokens_per_prompt=t,
                same_shape_exact=True, comparisons=comparisons,
                capture_elapsed_seconds=capture['elapsed_seconds'],
                replay_elapsed_seconds=replay['elapsed_seconds'])


def build_summary(data_dir=DATA):
    provenance = read(data_dir / 'provenance.json')
    for name, value in provenance['files'].items():
        if digest(data_dir / name) != value:
            raise ValueError(f'Artifact hash mismatch: {name}')
    timing = {name: timing_summary(data_dir, name, 16 if name.endswith('smoke') else 192,
                                  1 if name.endswith('smoke') else 4)
              for name in provenance['timing_runs']}
    window = {r['context']: r['mean_ms'] for r in timing['clock-window']['cells']}
    stability = {name: max(abs(r['mean_ms'] - window[r['context']])
                          for r in data['cells'] if r['context'] in window)
                 for name, data in timing.items() if name.startswith('clock-stability')}
    token = frozen_summary(data_dir, '', 80, 256)
    smoke = frozen_summary(data_dir, 'smoke-', 4, 16)
    h100 = read(data_dir / 'replay-manifest.json')['provenance']
    a100_capture = read(data_dir.parent / 'a100_followup_20261018/capture-manifest.json')
    a100 = read(data_dir.parent / 'a100_followup_20261018/replay-manifest.json')['provenance']
    for key in ('source_sha256', 'software', 'cuda_runtime', 'tf32_matmul', 'tf32_cudnn', 'torch_threads'):
        if h100[key] != a100[key]:
            raise ValueError(f'H100/A100 scoring conditions differ: {key}')
    for name in provenance['timing_runs']:
        manifest = read(data_dir / f'{name}-manifest.json')
        if (manifest['commit'] != provenance['source_commit']
                or set(manifest['gpu_uuids'].values()) != {h100['gpu_uuid']}):
            raise ValueError(f'Timing source or GPU differs: {name}')
    if token['capture_sha256'] == a100_capture['capture_sha256']:
        raise ValueError('This independent-capture report requires a distinct H100 capture')
    previous_path = data_dir.parent / 'h100_campaign_20261016/token-1.7B.json'
    if digest(previous_path) != provenance['previous_h100_token_sha256']:
        raise ValueError('Previous H100 token artifact hash mismatch')
    previous = read(previous_path)['arms']['clean']['shapes']
    return dict(scope='Independent H100 capture and local timing; no paired A100 replay, attack power, or deployment FPR',
                verified_artifacts=len(provenance['files']), timing=timing,
                max_shared_context_difference_ms=stability,
                token=token, smoke=smoke, paired_with_a100=False,
                previous_h100_aggregate_statistics_exact=(token['comparisons'] == previous))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    summary = build_summary()
    path = DATA / 'summary.json'
    if args.check:
        if read(path) != summary:
            raise ValueError('Saved summary differs from verified artifacts')
        print('Verified hashes, timing coverage/means, frozen pairing, score controls, and bootstrap intervals.')
    else:
        path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        print(path)


if __name__ == '__main__':
    main()
