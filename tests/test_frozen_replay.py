"""Capture integrity and pairing checks without model downloads or a GPU."""
import copy
import gzip
import json

import numpy as np
import pytest

from experiments.exp_frozen_replay_gpu import (
    DETECTORS, SHAPES, compare_runs, load_capture, restore_sequence, sha256,
    validate_capture, write_json,
)
from ivgym.core import SamplingSpec


def fixture_capture():
    return dict(schema="ivgym-frozen-capture-v1", n_prompts=2, n_tokens=3,
                vocab=100, revision="a" * 40, dtype="bfloat16", injected_noise=False,
                projection={"seed": 123, "dimension": 32},
                generation_shape="sequential_eager", sampling=vars(SamplingSpec()),
                sequences=[dict(prompt_id=i, prompt_tokens=[10, 11],
                                claimed_tokens=[12, 13, 14],
                                fingerprints=np.full((3, 32), i + 0.125).tolist())
                           for i in (2, 7)])


def test_roundtrip_preserves_ids_seeds_and_fingerprints(tmp_path):
    data = fixture_capture()
    path = tmp_path / "capture.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump(data, f)
    restored = load_capture(path)
    spec = SamplingSpec(**restored["sampling"])
    seq = restore_sequence(restored["sequences"][1], spec)
    assert seq.prompt_id == 7  # IDs must not be replaced by array indices.
    assert [s.claimed_token for s in seq.steps] == [12, 13, 14]
    assert [s.position for s in seq.steps] == [0, 1, 2]
    assert all(s.sampling == spec for s in seq.steps)
    np.testing.assert_array_equal([s.fingerprint for s in seq.steps],
                                  data["sequences"][1]["fingerprints"])


@pytest.mark.parametrize("error", ["duplicate", "short", "out_of_vocab", "nonfinite", "projection"])
def test_invalid_capture_rejected(error):
    data = fixture_capture()
    if error == "duplicate":
        data["sequences"][1]["prompt_id"] = 2
    elif error == "short":
        data["sequences"][0]["claimed_tokens"].pop()
    elif error == "out_of_vocab":
        data["sequences"][0]["prompt_tokens"][0] = 100
    elif error == "nonfinite":
        data["sequences"][0]["fingerprints"][0][0] = float("nan")
    else:
        data["projection"]["seed"] = 999
    with pytest.raises(ValueError):
        validate_capture(data)


def make_replay(path, offset=0.0):
    path.mkdir()
    scores = {f"{s}/{d}": np.arange(12, dtype=float).reshape(3, 4) + offset
              for s in SHAPES for d in DETECTORS}
    np.savez_compressed(path / "scores.npz", **scores)
    meta = dict(status="complete", command="replay", capture_sha256="a" * 64,
                n_prompts=3, n_tokens=4, scores_sha256=sha256(path / "scores.npz"),
                same_shape_exact={d: True for d in DETECTORS},
                provenance=dict(source_sha256={"code.py": "b" * 64}, software={},
                                cuda_runtime="12.8", tf32_matmul=False,
                                tf32_cudnn=True, torch_threads=1))
    write_json(path / "manifest.json", meta)
    return meta


def test_compare_zero_control_and_positive_shift(tmp_path):
    base, other = tmp_path / "base", tmp_path / "other"
    make_replay(base)
    make_replay(other, offset=0.5)
    same = compare_runs(base, base, bootstrap=20)
    shifted = compare_runs(base, other, bootstrap=20)
    for shape in SHAPES:
        for d in DETECTORS:
            assert same["comparisons"][shape][d]["d_prime"] == 0.0
            assert shifted["comparisons"][shape][d]["d_prime"] > 0


@pytest.mark.parametrize("error", ["capture", "software", "source", "incomplete", "control", "hash"])
def test_unpaired_or_unverified_runs_rejected(tmp_path, error):
    base, other = tmp_path / "base", tmp_path / "other"
    make_replay(base)
    meta = copy.deepcopy(make_replay(other))
    if error == "capture":
        meta["capture_sha256"] = "c" * 64
    elif error == "software":
        meta["provenance"]["software"] = {"torch": "different"}
    elif error == "source":
        meta["provenance"]["source_sha256"] = {}
    elif error == "incomplete":
        meta["status"] = "running"
    elif error == "control":
        meta["same_shape_exact"][DETECTORS[0]] = False
    else:
        meta["scores_sha256"] = "d" * 64
    write_json(other / "manifest.json", meta)
    with pytest.raises(ValueError):
        compare_runs(base, other, bootstrap=20)
