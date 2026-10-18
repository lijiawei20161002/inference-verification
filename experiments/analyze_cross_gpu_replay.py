"""Rebuild the paired H100/A100 replay result from archived arrays, without CUDA.

    python -m experiments.analyze_cross_gpu_replay --check

The unchanged replay CLI's zero-variance statistic is preserved in comparison.json.
This report marks its standardized effect undefined and also exposes raw changes.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

from experiments.artifact_checks import bootstrap_summary_matches
from experiments.exp_frozen_replay_gpu import DETECTORS, SHAPES, paired_d_prime, validate_capture

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/results/cross_gpu_replay_20261018'
H100 = ROOT / 'docs/results/h100_followup_20261018'
CONDITIONS = ('source_sha256', 'software', 'cuda_runtime', 'tf32_matmul',
              'tf32_cudnn', 'torch_threads')
BOOTSTRAP = dict(unit='sequence', draws=400, seed=0, interval=0.90)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(manifest_path, scores_path, summary_path):
    """Check complete arrays and actual controls, independently of manifest flags."""
    meta, saved = read(manifest_path), read(summary_path)
    if meta['status'] != 'complete' or meta['command'] != 'replay':
        raise ValueError('Incomplete replay')
    if (digest(scores_path) != meta['scores_sha256']
            or digest(summary_path) != meta['summary_sha256']):
        raise ValueError('Replay artifact hash mismatch')
    for key in ('capture_sha256', 'n_prompts', 'n_tokens'):
        if meta[key] != saved[key]:
            raise ValueError(f'Replay summary mismatch: {key}')
    if saved['bootstrap'] != BOOTSTRAP:
        raise ValueError('Unexpected bootstrap contract')
    dims = (meta['n_prompts'], meta['n_tokens'])
    if any(type(x) is not int or x < 1 for x in dims):
        raise ValueError('Invalid replay dimensions')
    expected = {f'{shape}/{detector}' for shape in (*SHAPES, 'rerun') for detector in DETECTORS}
    with np.load(scores_path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise ValueError('Missing or extra score arrays')
        arrays = {key: archive[key].copy() for key in expected}
    for key, array in arrays.items():
        if (array.dtype.kind not in 'fi' or array.shape != dims
                or not np.isfinite(array).all()):
            raise ValueError(f'Invalid score array: {key}')
    for record in (meta, saved):
        if (set(record['same_shape_exact']) != set(DETECTORS)
                or any(v is not True for v in record['same_shape_exact'].values())):
            raise ValueError('Failed or incomplete control flags')
    for detector in DETECTORS:
        if not np.array_equal(arrays[f'chunked_b1/{detector}'], arrays[f'rerun/{detector}']):
            raise ValueError(f'Failed actual same-shape control: {detector}')
    return meta, arrays


def compare_evidence(base_meta, base, alt_meta, alt, saved, bootstrap=400):
    for key in ('capture_sha256', 'n_prompts', 'n_tokens'):
        if base_meta[key] != alt_meta[key]:
            raise ValueError(f'Unpaired evidence: {key}')
    for key in CONDITIONS:
        if base_meta['provenance'][key] != alt_meta['provenance'][key]:
            raise ValueError(f'Replay conditions differ: {key}')
    if (saved['capture_sha256'] != base_meta['capture_sha256']
            or saved['provenance'] != [base_meta['provenance'], alt_meta['provenance']]):
        raise ValueError('Comparison provenance mismatch')
    if set(saved['comparisons']) != set(SHAPES):
        raise ValueError('Missing or extra comparison schedules')
    n, t = base_meta['n_prompts'], base_meta['n_tokens']
    rows = {}
    for shape in SHAPES:
        if set(saved['comparisons'][shape]) != set(DETECTORS):
            raise ValueError('Missing or extra comparison detectors')
        rows[shape] = {}
        for detector in DETECTORS:
            key = f'{shape}/{detector}'
            b, a = base[key].ravel(), alt[key].ravel()
            actual = paired_d_prime(b, a, n, t, bootstrap)
            archived = saved['comparisons'][shape][detector]
            if not bootstrap_summary_matches(actual, archived):
                raise ValueError(f'Bootstrap summary does not reproduce: {key}')
            # Keep verified archived precision stable across NumPy reductions.
            defined = archived['base_sd'] > 1e-12
            cap = float(np.percentile(b, 99.9))
            rows[shape][detector] = dict(
                effect_size_status='defined' if defined else 'undefined_zero_baseline_variance',
                d_prime=archived['d_prime'] if defined else None,
                ci=archived['ci'] if defined else None,
                legacy_statistic=archived,
                raw=dict(baseline_mean=float(b.mean()), alternate_mean=float(a.mean()),
                         mean_difference=float((a-b).mean()),
                         mean_absolute_difference=float(np.abs(a-b).mean()),
                         max_absolute_difference=float(np.abs(a-b).max()),
                         baseline_sd=float(b.std()), alternate_sd=float(a.std()),
                         changed_positions=int(np.count_nonzero(a != b))),
                winsor_cap=cap, alternate_positions_above_cap=int(np.count_nonzero(a > cap)))
    return rows


def build_summary(data_dir=DATA, baseline_dir=H100):
    provenance = read(data_dir / 'provenance.json')
    for name, expected in provenance['files'].items():
        if digest(data_dir / name) != expected:
            raise ValueError(f'Artifact hash mismatch: {name}')
    for name, expected in provenance['baseline_files'].items():
        if digest(baseline_dir / name) != expected:
            raise ValueError(f'Baseline artifact hash mismatch: {name}')
    bmeta, base = load_run(baseline_dir / 'replay-manifest.json', baseline_dir / 'scores.npz',
                           baseline_dir / 'token-summary.json')
    ameta, alt = load_run(data_dir / 'a100-manifest.json', data_dir / 'a100-scores.npz',
                          data_dir / 'a100-summary.json')
    smeta, _ = load_run(data_dir / 'smoke-manifest.json', data_dir / 'smoke-scores.npz',
                        data_dir / 'smoke-summary.json')
    bundle = baseline_dir / 'h100-replay-transfer.tar.gz'
    with tarfile.open(bundle) as tf:
        capture_bytes = tf.extractfile('h100-replay-transfer/capture/capture.json.gz').read()
    if hashlib.sha256(capture_bytes).hexdigest() != bmeta['capture_sha256']:
        raise ValueError('Frozen capture hash mismatch')
    capture = validate_capture(json.loads(gzip.decompress(capture_bytes)))
    if (capture['n_prompts'], capture['n_tokens']) != (80, 256):
        raise ValueError('Unexpected full capture dimensions')
    if (bmeta['n_prompts'], bmeta['n_tokens']) != (80, 256):
        raise ValueError('Unexpected baseline replay dimensions')
    for key in CONDITIONS:
        if (capture['provenance'][key] != bmeta['provenance'][key]
                or smeta['provenance'][key] != ameta['provenance'][key]):
            raise ValueError(f'Capture or smoke conditions differ: {key}')
    smoke_path = data_dir / 'smoke-capture.json.gz'
    if digest(smoke_path) != smeta['capture_sha256']:
        raise ValueError('Smoke capture hash mismatch')
    with gzip.open(smoke_path, 'rt') as f:
        smoke = validate_capture(json.load(f))
    if (smoke['n_prompts'], smoke['n_tokens'], smeta['n_prompts'], smeta['n_tokens']) != (4, 16, 4, 16):
        raise ValueError('Unexpected smoke dimensions')
    if smoke['derived_from_capture_sha256'] != bmeta['capture_sha256']:
        raise ValueError('Smoke parent capture mismatch')
    for key in ('model', 'revision', 'dtype', 'sampling', 'projection', 'provenance'):
        if smoke[key] != capture[key]:
            raise ValueError(f'Smoke capture contract differs: {key}')
    for small, full in zip(smoke['sequences'], capture['sequences']):
        for key in ('prompt_id', 'prompt_tokens', 'claimed_tokens', 'fingerprints'):
            expected = full[key][:16] if key in ('claimed_tokens', 'fingerprints') else full[key]
            if small[key] != expected:
                raise ValueError(f'Smoke evidence differs from frozen capture: {key}')
    queue = read(data_dir / 'queue.json')
    if (queue['status'] != 'complete'
            or [j['name'] for j in queue['jobs']] != ['smoke-replay', 'a100-replay', 'compare']
            or any(j['status'] != 'complete' or j['returncode'] != 0 for j in queue['jobs'])):
        raise ValueError('Incomplete replay queue')
    comparisons = compare_evidence(bmeta, base, ameta, alt, read(data_dir / 'comparison.json'))
    return dict(
        scope='Paired honest H100 capture replayed on A100; hardware and driver both differ; no attack power or deployment FPR',
        direction='baseline H100, alternate A100; positive means higher scores on A100',
        primary_schedule='chunked_b1',
        capture_sha256=bmeta['capture_sha256'], model=capture['model'], revision=capture['revision'],
        n_prompts=80, tokens_per_prompt=256, bootstrap=BOOTSTRAP,
        same_shape_exact={'h100': True, 'a100': True, 'smoke': True},
        baseline={k: bmeta['provenance'][k] for k in ('device', 'driver', 'source_commit')},
        alternate={k: ameta['provenance'][k] for k in ('device', 'driver', 'source_commit')},
        replay_elapsed_seconds=ameta['elapsed_seconds'], comparisons=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    summary = build_summary()
    path = DATA / 'summary.json'
    if args.check:
        saved = read(path)
        # Recomputed raw descriptive means also tolerate float64 roundoff.
        if not summaries_match(saved, summary):
            raise ValueError('Saved paired summary differs from verified artifacts')
        print('Verified capture, matched conditions, raw controls, all 16 paired intervals, and zero-variance handling.')
    else:
        path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        print(path)


def summaries_match(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(summaries_match(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(summaries_match(x, y) for x, y in zip(a, b))
    if type(a) is float and type(b) is float:
        return bool(np.isclose(a, b, rtol=1e-12, atol=1e-14))
    return type(a) is type(b) and a == b


if __name__ == '__main__':
    main()
