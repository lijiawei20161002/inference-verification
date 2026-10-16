"""Independent calibration, paired condition scheduling and load coverage."""
import copy

import pytest

from experiments.exp_clock_interference_gpu import CONTEXTS, analyze, schedule
from scripts.run_research_job import validate_result


def example():
    blocks = schedule(2, 4)
    cells = []
    for block in blocks:
        block["load_iterations"] = 16 if block["condition"] == "load" else 0
        for ctx in block["context_order"]:
            # Identical slopes under load, with an additive constant that cancels.
            latency = 2 + ctx / 10000 + (3 if block["condition"] == "load" else 0)
            cell = {k: block[k] for k in ("partition", "replicate", "condition")}
            cells.append(dict(cell, ctx=ctx, itl_ms=[latency, latency]))
    return dict(calibration_reps=2, test_reps=4, cells=cells, blocks=blocks,
                load_worker_stopped=True, analysis=analyze(cells))


def test_schedule_pairs_context_order_and_counterbalances_conditions():
    blocks = schedule(2, 4)
    assert all(b["condition"] == "idle" for b in blocks[:2])
    assert all(sorted(b["context_order"]) == CONTEXTS for b in blocks)
    for i in range(4):
        a, b = blocks[2 + 2 * i:4 + 2 * i]
        assert a["context_order"] == b["context_order"]
        assert a["condition"] != b["condition"]
        assert a["condition"] == ("idle" if i % 2 == 0 else "load")


def test_threshold_does_not_use_test_data_and_counts_cycles():
    data = example()
    threshold = data["analysis"]["threshold_ms"]
    for cell in data["cells"]:
        if cell["partition"] == "test" and cell["condition"] == "load":
            cell["itl_ms"] = [2.0, 2.0]
    result = analyze(data["cells"])
    assert result["threshold_ms"] == threshold
    honest = result["results"]["load"]["32768"]
    assert honest["cycles"] == honest["alarms"] == 4


def test_interference_coverage_and_load_are_validated():
    data = example()
    assert validate_result("clock-interference", data)["coverage"] == "complete"
    bad = copy.deepcopy(data)
    bad["cells"].pop()
    with pytest.raises(ValueError, match="Incomplete interference"):
        validate_result("clock-interference", bad)
    bad = copy.deepcopy(data)
    next(b for b in bad["blocks"] if b["condition"] == "load")["load_iterations"] = 0
    with pytest.raises(ValueError, match="load did not run"):
        validate_result("clock-interference", bad)
