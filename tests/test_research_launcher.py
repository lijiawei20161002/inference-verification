"""Launcher isolation and provenance checks; fake experiments, no GPU required."""
import argparse
import json
from pathlib import Path
import subprocess

import pytest

from scripts import run_research_job as launcher


def arguments(**overrides):
    values = dict(node="test-node", gpu="0", job="clock-window", size="smoke",
                  model_lock=None, dry_run=False)
    values.update(overrides)
    return argparse.Namespace(**values)


def window_data():
    return {"cells": [{"ctx": ctx, "itl_ms": [1.0, 2.0]}
                      for ctx in (256, 512, 2048, 8192, 16384, 32768)]}


@pytest.fixture
def source_repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "experiments").mkdir(parents=True)
    (root / "experiments/__init__.py").touch()
    (root / "docs/results").mkdir(parents=True)
    (root / ".gitignore").write_text("runs/\n__pycache__/\n")
    old = root / "docs/results/slope_verifier_window.json"
    old.write_text('{"historical": true}\n')
    experiment = root / "experiments/exp_slope_verifier_gpu.py"
    experiment.write_text(
        "from pathlib import Path\nimport json\n"
        f"Path('docs/results/slope_verifier_window.json').write_text({json.dumps(json.dumps(window_data()))})\n"
    )
    for command in (
        ["git", "init", "-q"],
        ["git", "add", "."],
        ["git", "-c", "user.name=Launcher Test", "-c", "user.email=test@example.invalid",
         "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"],
    ):
        subprocess.run(command, cwd=root, check=True, capture_output=True)
    monkeypatch.setattr(launcher, "ROOT", root)
    monkeypatch.setattr(launcher.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(launcher, "record_command", lambda command: {"returncode": 0, "output": "fixture"})
    monkeypatch.setattr(launcher, "cached_revisions", lambda: {})
    monkeypatch.setattr(launcher, "gpu_uuid", lambda selector: f"GPU-fixture-{selector}")
    original_run = subprocess.run

    def run_without_cuda(command, **kwargs):
        if len(command) > 2 and command[1] == "-c" and command[2].startswith("import torch;"):
            return subprocess.CompletedProcess(command, 0)
        return original_run(command, **kwargs)

    monkeypatch.setattr(launcher.subprocess, "run", run_without_cuda)
    return root, experiment, old


def commit_fixture(root):
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Launcher Test", "-c", "user.email=test@example.invalid",
                    "-c", "commit.gpgsign=false", "commit", "-qm", "update"], cwd=root, check=True)


def test_run_preserves_history_and_isolates_repeats(source_repo):
    root, _, old = source_repo
    before = old.read_bytes()
    assert launcher.run(arguments()) == 0
    assert launcher.run(arguments()) == 0
    assert old.read_bytes() == before
    manifests = list(root.glob("runs/campaign/test-node/*/manifest.json"))
    assert len(manifests) == 2
    for path in manifests:
        meta = json.loads(path.read_text())
        assert meta["status"] == "complete"
        assert len(meta["artifact_sha256"]) == 64
        assert json.loads((path.parent / "results/slope_verifier_window.json").read_text()) == window_data()
        assert (path.parent / "source/experiments/exp_slope_verifier_gpu.py").is_file()


def test_missing_output_cannot_reuse_committed_result(source_repo):
    root, experiment, old = source_repo
    experiment.write_text("# Deliberately produce no new result.\n")
    commit_fixture(root)
    with pytest.raises(FileNotFoundError):
        launcher.run(arguments())
    meta_path = next(root.glob("runs/campaign/test-node/*/manifest.json"))
    assert json.loads(meta_path.read_text())["status"] == "failed"
    assert old.exists()
    assert not (meta_path.parent / "results/slope_verifier_window.json").exists()


def test_process_failure_is_recorded(source_repo):
    root, experiment, _ = source_repo
    experiment.write_text("raise RuntimeError('simulated failed measurement')\n")
    commit_fixture(root)
    with pytest.raises(RuntimeError, match="Experiment exited"):
        launcher.run(arguments())
    meta_path = next(root.glob("runs/campaign/test-node/*/manifest.json"))
    assert json.loads(meta_path.read_text())["returncode"] != 0
    assert "simulated failed measurement" in (meta_path.parent / "run.log").read_text()


def test_dirty_checkout_is_rejected(source_repo):
    root, experiment, _ = source_repo
    experiment.write_text("# uncommitted\n")
    with pytest.raises(RuntimeError, match="Commit or move"):
        launcher.run(arguments())
    assert not (root / "runs").exists()


def test_host_lock_prevents_concurrent_launchers(tmp_path):
    path = tmp_path / "host.lock"
    with launcher.host_lock(path):
        with pytest.raises(RuntimeError, match="host lock"):
            with launcher.host_lock(path):
                pass
    with launcher.host_lock(path):
        pass


def test_token_host_sharing_excludes_timing_and_same_gpu(tmp_path):
    host, gpu0, gpu1 = (tmp_path / name for name in ("host.lock", "GPU-0.lock", "GPU-1.lock"))
    with launcher.host_lock(host, shared=True), launcher.host_lock(gpu0):
        with launcher.host_lock(host, shared=True), launcher.host_lock(gpu1):
            with pytest.raises(RuntimeError, match="host lock"):
                with launcher.host_lock(host):
                    pass
            with pytest.raises(RuntimeError, match="GPU lock"):
                with launcher.host_lock(gpu0):
                    pass
    with launcher.host_lock(host):
        with pytest.raises(RuntimeError, match="host lock"):
            with launcher.host_lock(host, shared=True):
                pass


def test_missing_and_invalid_timing_measurements_are_rejected():
    data = window_data()
    data["cells"].pop()
    with pytest.raises(ValueError, match="Incomplete context"):
        launcher.validate_result("clock-window", data)
    data = window_data()
    data["cells"][0]["itl_ms"] = [float("nan")]
    with pytest.raises(ValueError, match="finite"):
        launcher.validate_result("clock-window", data)


def test_model_lock_requires_matching_immutable_offline_cache(tmp_path, monkeypatch):
    pinned = {m: "a" * 40 for m in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B")}
    path = tmp_path / "models.json"
    path.write_text(json.dumps(pinned))
    monkeypatch.setattr(launcher, "cached_revisions", lambda: dict(pinned))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    with pytest.raises(ValueError, match="HF_HUB_OFFLINE"):
        launcher.check_model_lock(path, "clock-window")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert launcher.check_model_lock(path, "clock-window") == pinned
    monkeypatch.setattr(launcher, "cached_revisions", lambda: {m: "b" * 40 for m in pinned})
    with pytest.raises(ValueError, match="differs"):
        launcher.check_model_lock(path, "clock-window")


def test_dry_run_needs_no_gpu_or_output_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    assert launcher.run(arguments(dry_run=True, size="baseline")) == 0
    assert json.loads(capsys.readouterr().out)["job"] == "clock-window"
    assert list(tmp_path.iterdir()) == []


def test_token_model_selection_and_interference_arguments(capsys):
    assert launcher.run(arguments(job="token-shape", model="Qwen/Qwen3-4B", gpu="1", dry_run=True)) == 0
    assert json.loads(capsys.readouterr().out)["settings"]["IVGYM_M"] == "Qwen/Qwen3-4B"
    with pytest.raises(ValueError, match="token-shape only"):
        launcher.run(arguments(model="Qwen/Qwen3-4B", dry_run=True))
    with pytest.raises(ValueError, match="requires --load-gpu"):
        launcher.run(arguments(job="clock-interference", dry_run=True))
    with pytest.raises(ValueError, match="different GPUs"):
        launcher.run(arguments(job="clock-interference", load_gpu="0", dry_run=True))
    assert launcher.run(arguments(job="clock-interference", load_gpu="1", dry_run=True)) == 0
    assert json.loads(capsys.readouterr().out)["settings"]["IVGYM_INTERFERENCE_GPU"] == "1"


def test_selected_token_model_is_locked(tmp_path, monkeypatch):
    path = tmp_path / "models.json"
    pinned = {"Qwen/Qwen3-4B": "a" * 40}
    path.write_text(json.dumps(pinned))
    monkeypatch.setattr(launcher, "cached_revisions", lambda: dict(pinned))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert launcher.check_model_lock(path, "token-shape", "Qwen/Qwen3-4B") == pinned
