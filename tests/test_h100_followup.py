"""Scientific artifact checks reject internally inconsistent exported evidence."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from experiments.analyze_h100_followup import DATA, build_summary, frozen_summary, timing_summary


def test_h100_followup_summary_reproduces_from_archived_evidence():
    assert build_summary() == json.loads((DATA / 'summary.json').read_text())


def copy_timing(tmp_path):
    for name in ('clock-window.json', 'clock-window-manifest.json'):
        shutil.copyfile(DATA / name, tmp_path / name)
    return tmp_path / 'clock-window.json', tmp_path / 'clock-window-manifest.json'


def test_timing_artifact_edit_is_detected(tmp_path):
    path, _ = copy_timing(tmp_path)
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='modified timing'):
        timing_summary(tmp_path, 'clock-window', 192, 4)


def test_timing_missing_cell_is_rejected_even_with_updated_hash(tmp_path):
    path, meta_path = copy_timing(tmp_path)
    data = json.loads(path.read_text())
    data['cells'].pop()
    path.write_text(json.dumps(data))
    meta = json.loads(meta_path.read_text())
    meta['artifact_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='Incomplete timing grid'):
        timing_summary(tmp_path, 'clock-window', 192, 4)


def test_rerun_scores_checked_in_addition_to_manifest_flag(tmp_path):
    for name in ('smoke-capture-manifest.json', 'smoke-replay-manifest.json',
                 'smoke-token-summary.json', 'smoke-scores.npz'):
        shutil.copyfile(DATA / name, tmp_path / name)
    path = tmp_path / 'smoke-scores.npz'
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    arrays['rerun/token_difr'][0, 0] += 1.0
    np.savez_compressed(path, **arrays)
    meta_path = tmp_path / 'smoke-replay-manifest.json'
    meta = json.loads(meta_path.read_text())
    meta['scores_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='Failed same-shape control'):
        frozen_summary(tmp_path, 'smoke-', 4, 16)
