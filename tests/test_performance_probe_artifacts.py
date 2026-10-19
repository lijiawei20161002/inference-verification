"""Offline provenance and completeness checks for the archived L40S mechanism pilot."""
import hashlib
import json
import math
from pathlib import Path

from experiments.analyze_performance_probes import analyze, read_rows

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/results/performance_probes_l40s_20261019"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_archive_hashes_cover_every_artifact():
    listed = {}
    for line in (ARCHIVE / "SHA256SUMS").read_text().splitlines():
        expected, filename = line.split("  ", 1)
        assert filename not in listed
        listed[filename] = expected
        assert digest(ARCHIVE / filename) == expected, filename
    actual = {str(p.relative_to(ARCHIVE)) for p in ARCHIVE.rglob("*")
              if p.is_file() and p.name != "SHA256SUMS"}
    assert set(listed) == actual


def test_raw_measurements_reproduce_summary_and_controls():
    summary = analyze(ARCHIVE)
    assert summary == json.loads((ARCHIVE / "summary.json").read_text())
    assert summary == json.loads((ARCHIVE / "recomputed_summary.json").read_text())
    assert summary["status_counts"] == {"complete": 524}
    assert summary["suite_counts"] == {"p0": 160, "p1": 192, "p1_control": 32, "p4": 128, "free": 12}
    assert summary["checks"] == {"negative_control_pairs": 48, "prefix_cache_pairs": 32}
    assert len(summary["http"]) == 42
    queue = json.loads((ARCHIVE / "queue.json").read_text())
    assert queue["status"] == "complete"
    assert len(queue["jobs"]) == 8
    assert all(j["status"] == "complete" and j["returncode"] == 0 for j in queue["jobs"])
    for manifest_path in ARCHIVE.glob("session_*/manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        assert manifest["source_status"] == ""
        assert manifest["source_commit"] == queue["source_commit"]
        for filename, expected in manifest["source_sha256"].items():
            assert digest(ARCHIVE / "source" / filename) == expected
        assert digest(ARCHIVE / "model_config.json") == manifest["model_config_sha256"]
        assert digest(ARCHIVE / "source/experiments/data/specdec_longtext.txt") == manifest["fixture_sha256"]


def test_sdpa_control_coverage_numerics_and_dispatch():
    root = ARCHIVE / "sdpa_representation"
    manifest = json.loads((root / "manifest.json").read_text())
    assert digest(ARCHIVE / "source/experiments/exp_sdpa_representation_gpu.py") == manifest["source_sha256"]
    assert manifest["planned_error_tolerance"] == {"atol": 0.005, "rtol": 0.02}
    rows = read_rows(root / "measurements.jsonl")
    arms = {"native_gqa", "repeat_no_mask", "repeat_all_true_mask"}
    expected = {(block, batch, length, arm) for block in range(8) for batch in (1, 4)
                for length in (512, 8192) for arm in arms}
    assert len(rows) == len(expected) == 96
    assert {(r["block"], r["batch"], r["length"], r["arm"]) for r in rows} == expected
    for r in rows:
        assert r["status"] == "complete"
        assert len(r["ms_per_call"]) == 5
        assert all(math.isfinite(t) and t > 0 for t in r["ms_per_call"])
        # This archived run's absolute error is below atol even without the relative term.
        assert 0 <= r["max_abs_error_vs_fp32"] < manifest["planned_error_tolerance"]["atol"]
    traces = {r["arm"]: " ".join(r["kernels"]) for r in rows if "kernels" in r}
    assert set(traces) == arms
    assert "flash_fwd" in traces["native_gqa"]
    assert "direct_copy" not in traces["native_gqa"]
    assert "flash_fwd" in traces["repeat_no_mask"] and "direct_copy" in traces["repeat_no_mask"]
    assert "fmha_cutlass" in traces["repeat_all_true_mask"]
    failed = ARCHIVE / "sdpa_failed_capture_check"
    assert read_rows(failed / "measurements.jsonl") == []
    assert json.loads((failed / "failure.json").read_text())["measurement_rows"] == 0
