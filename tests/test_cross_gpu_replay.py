"""Integrity and interpretation of the paired cross-GPU evidence."""
import copy
import json

import numpy as np
import pytest

from experiments.analyze_cross_gpu_replay import (
    BOOTSTRAP, DATA, DETECTORS, SHAPES, build_summary, compare_evidence,
    digest, load_run, paired_d_prime, summaries_match,
)


def evidence(offset=0.0):
    arrays = {f'{s}/{d}': np.arange(20, dtype=float).reshape(4, 5) + offset
              for s in (*SHAPES, 'rerun') for d in DETECTORS}
    meta = dict(status='complete', command='replay', capture_sha256='a' * 64,
                n_prompts=4, n_tokens=5, same_shape_exact={d: True for d in DETECTORS},
                provenance=dict(source_sha256={'code.py': 'b' * 64}, software={'numpy': '2.1.2'},
                                cuda_runtime='12.8', tf32_matmul=False, tf32_cudnn=True,
                                torch_threads=1))
    return meta, arrays


def comparison(bm, b, am, a):
    return dict(capture_sha256=bm['capture_sha256'], provenance=[bm['provenance'], am['provenance']],
                comparisons={s: {d: paired_d_prime(b[f'{s}/{d}'].ravel(), a[f'{s}/{d}'].ravel(),
                                                   4, 5, 20)
                                 for d in DETECTORS} for s in SHAPES})


def write_run(tmp_path, meta, arrays):
    scores = tmp_path / 'scores.npz'
    np.savez_compressed(scores, **arrays)
    saved = {k: meta[k] for k in ('capture_sha256', 'n_prompts', 'n_tokens', 'same_shape_exact')}
    saved['bootstrap'] = BOOTSTRAP
    summary = tmp_path / 'summary.json'
    summary.write_text(json.dumps(saved))
    meta = dict(meta, scores_sha256=digest(scores), summary_sha256=digest(summary))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(meta))
    return manifest, scores, summary


def test_zero_variance_does_not_masquerade_as_no_difference():
    bm, b = evidence()
    am, a = evidence(0.5)
    key = 'sequential/activation_difr'
    b[key] = np.zeros((4, 5))
    a[key] = np.ones((4, 5))
    rows = compare_evidence(bm, b, am, a, comparison(bm, b, am, a), bootstrap=20)
    row = rows['sequential']['activation_difr']
    assert row['legacy_statistic']['d_prime'] == 0.0  # The zero cap hides the change.
    assert row['effect_size_status'] == 'undefined_zero_baseline_variance'
    assert row['d_prime'] is None and row['ci'] is None
    assert row['raw']['mean_difference'] == 1.0
    assert row['raw']['changed_positions'] == 20
    assert row['alternate_positions_above_cap'] == 20
    assert rows['chunked_b1']['token_difr']['d_prime'] > 0


@pytest.mark.parametrize('error', ['capture', 'software', 'ci', 'missing_detector'])
def test_unpaired_or_falsified_comparison_rejected(error):
    bm, b = evidence()
    am, a = evidence(0.5)
    saved = copy.deepcopy(comparison(bm, b, am, a))
    if error == 'capture':
        am['capture_sha256'] = 'c' * 64
    elif error == 'software':
        am['provenance']['software']['numpy'] = 'different'
    elif error == 'ci':
        saved['comparisons']['chunked_b1']['token_difr']['ci'][0] += 1e-5
    else:
        del saved['comparisons']['chunked_b1']['token_difr']
    with pytest.raises(ValueError):
        compare_evidence(bm, b, am, a, saved, bootstrap=20)


@pytest.mark.parametrize('error', ['rerun', 'nonfinite', 'missing_array', 'wrong_shape', 'missing_flag'])
def test_bad_arrays_rejected_even_with_consistent_hashes(tmp_path, error):
    meta, arrays = evidence()
    if error == 'rerun':
        arrays['rerun/token_difr'][0, 0] += 1
    elif error == 'nonfinite':
        arrays['sequential/cross_entropy'][0, 0] = np.nan
    elif error == 'missing_array':
        del arrays['sequential/token_toploc']
    elif error == 'wrong_shape':
        arrays['chunked_b4/token_difr'] = np.ones((2, 10))
    else:
        del meta['same_shape_exact']['activation_difr']
    with pytest.raises(ValueError):
        load_run(*write_run(tmp_path, meta, arrays))


def test_valid_array_archive_loads(tmp_path):
    meta, arrays = evidence()
    restored, scores = load_run(*write_run(tmp_path, meta, arrays))
    assert restored['n_prompts'] == 4
    np.testing.assert_array_equal(scores['chunked_b4/token_difr'], arrays['chunked_b4/token_difr'])


def test_archived_paired_result_reproduces():
    assert summaries_match(json.loads((DATA / 'summary.json').read_text()), build_summary())
