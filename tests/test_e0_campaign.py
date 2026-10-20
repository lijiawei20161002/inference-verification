import json

import pytest

from experiments.numerical.run_e0_campaign import compare_workers, kernel_sequence, status


def make_worker(path, token=1, state="complete"):
    path.mkdir()
    manifest = dict(status=state, artifacts_sha256={}, source_sha256={"x": "a"},
                    model_assets_sha256={"weights": "b"},
                    plan=dict(prompt_ids=[32], tokens=1, archive_manifest_sha256="c"))
    (path / "manifest.json").write_text(json.dumps(manifest))
    (path / "sequences.json").write_text(json.dumps([
        dict(prompt_id=32, claimed_tokens=[token], hashes={"logits": "d"})]))


def test_output_equivalence_rejects_changed_tokens(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    make_worker(a)
    make_worker(b, token=2)
    result = compare_workers(a, b)
    assert result["exact"] is False
    assert result["failed_prompt_ids"] == [32]


def test_incomplete_worker_is_not_accepted(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    make_worker(a)
    make_worker(b, state="failed")
    with pytest.raises(ValueError, match="Incomplete"):
        compare_workers(a, b)


def test_dispatch_requires_actual_cuda_kernels_and_preserves_launch_shapes(tmp_path):
    p = tmp_path / "prompt_32"
    p.mkdir()
    f = p / "dispatch.json"
    f.write_text(json.dumps({"traceEvents": [{"cat": "cpu_op", "name": "aten::mm"}]}))
    with pytest.raises(ValueError, match="GPU kernels"):
        kernel_sequence(tmp_path)
    f.write_text(json.dumps({"traceEvents": [
        {"cat": "kernel", "name": "gemm", "ts": 2, "args": {"grid": [2], "block": [64]}},
        {"cat": "kernel", "name": "gemv", "ts": 1, "args": {"grid": [1], "block": [32]}}
    ]}))
    assert kernel_sequence(tmp_path) == [["gemv", [1], [32]], ["gemm", [2], [64]]]


def test_status_is_atomically_replaced_and_failure_is_explicit(tmp_path):
    p = tmp_path / "status.json"
    status(p, state="running", stage="capture")
    status(p, state="failed", stage="capture", error="test failure")
    assert json.loads(p.read_text())["state"] == "failed"
    assert not p.with_suffix(".tmp").exists()
