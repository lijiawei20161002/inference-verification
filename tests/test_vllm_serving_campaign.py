"""Checks for missing coverage, modified provenance, and stream-window denominators."""
import json
from pathlib import Path

import pytest

from experiments.analyze_vllm_serving_campaign import (
    analyze_campaign, arrival_windows, validate_queue, validate_run,
)

ARCHIVE = Path(__file__).resolve().parents[1] / "docs/results/vllm_serving_l40s_20261019"


def test_late_window_uses_95_intervals_and_preserves_coalesced_arrivals():
    # Two tokens share the first event; all later events arrive every 2 ms.
    frames = [{"received_ns_since_send": 10_000_000,
               "data": {"choices": [{"token_ids": [0, 1]}]}}]
    frames += [{"received_ns_since_send": (10 + 2 * (i - 1)) * 1_000_000,
                "data": {"choices": [{"token_ids": [i]}]}} for i in range(2, 128)]
    assert arrival_windows({"frames": frames}) == {
        "early_ms_per_token": 62 / 32, "late_ms_per_token": 2,
    }
    with pytest.raises(ValueError, match="exactly 128"):
        arrival_windows({"frames": frames[:-1]})


@pytest.mark.parametrize("change", ("missing", "duplicate", "dirty"))
def test_queue_rejects_incomplete_or_ambiguous_campaign(change):
    queue = json.loads((ARCHIVE / "pilot/queue.json").read_text())
    if change == "missing":
        queue["jobs"].pop()
    elif change == "duplicate":
        queue["jobs"][-1] = queue["jobs"][0]
    else:
        queue["source_status"] = " M experiments/exp_vllm_serving.py"
    with pytest.raises(ValueError):
        validate_queue(queue)


def test_modified_source_snapshot_is_rejected_before_analysis(tmp_path):
    queue = json.loads((ARCHIVE / "pilot/queue.json").read_text())
    (tmp_path / "queue.json").write_text(json.dumps(queue))
    first_source = next(iter(queue["source_sha256"]))
    path = tmp_path / "source" / first_source
    path.parent.mkdir(parents=True)
    path.write_text("modified source\n")
    with pytest.raises(ValueError, match="Source mismatch"):
        validate_run(tmp_path)


def test_archived_campaign_recomputes_from_raw_streams():
    actual = analyze_campaign(ARCHIVE)
    expected = json.loads((ARCHIVE / "summary.json").read_text())
    assert actual == expected
    assert sum(p["original_summary"]["measured_requests"] for p in actual["phases"].values()) == 1200
    assert sum(p["primer_requests"] for p in actual["phases"].values()) == 120
    assert sum(p["warmup_requests"] for p in actual["phases"].values()) == 90
