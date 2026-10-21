import copy
import json

import pytest

from experiments.numerical.dispatch_validation import (
    compare_dispatch, compare_parent, dispatch_signature, expected_calls, kernel_launches,
)
from experiments.numerical.run_dispatch_campaign import compare_capture, full_gate


def make_trace(path, grid=None):
    path.write_text(json.dumps(dict(traceEvents=[
        dict(cat="kernel", name="gemm", ts=1, args=dict(grid=grid or [2, 1, 1], block=[64, 1, 1]))])))


def make_worker(path):
    path.mkdir()
    (path / "corpus.json").write_text(json.dumps([dict(prompt_id=32, prompt_tokens=[1, 2, 3])]))
    rows = [dict(prompt_id=32, historical=[dict(position=2)], capped_positions=[2, 3])]
    directory = path / "prompt_32"
    directory.mkdir()
    calls = []
    for _, role, step in sorted(expected_calls(rows, 4)):
        name = f"dispatch_{role}_{step}.json"
        length = 7 if role == "reference" else 3 if step == 0 else 1
        calls.append(dict(role=role, step=step, file=name,
                          inputs=[dict(dtype="torch.int64", shape=[1, length], stride=[length, 1], device="cuda:0")]))
        make_trace(directory / name)
    index = dict(prompt_id=32, selected_positions=[0, 2, 3], calls=calls)
    (directory / "dispatch_index.json").write_text(json.dumps(index))
    return rows, directory, index


def test_all_events_controls_and_reference_are_required(tmp_path):
    rows, directory, index = make_worker(tmp_path / "worker")
    signatures = dispatch_signature(directory.parent, rows, 4)
    assert len(signatures) == 4
    assert compare_dispatch(signatures, signatures)["exact"]
    index["calls"].pop()
    (directory / "dispatch_index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="Incomplete dispatch"):
        dispatch_signature(directory.parent, rows, 4)


@pytest.mark.parametrize("change", ["duplicate", "shape", "selection", "missing_trace"])
def test_invalid_coverage_and_shapes_fail_closed(tmp_path, change):
    rows, directory, index = make_worker(tmp_path / "worker")
    if change == "duplicate":
        index["calls"].append(index["calls"][0])
    elif change == "shape":
        index["calls"][0]["inputs"][0]["shape"] = [1, 999]
    elif change == "selection":
        index["selected_positions"] = [0, 2]
    else:
        (directory / index["calls"][0]["file"]).unlink()
    (directory / "dispatch_index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError):
        dispatch_signature(directory.parent, rows, 4)


def test_kernel_changes_are_reported_and_launch_metadata_is_required(tmp_path):
    rows, directory, _ = make_worker(tmp_path / "worker")
    before = dispatch_signature(directory.parent, rows, 4)
    make_trace(directory / "dispatch_provider_2.json", grid=[3, 1, 1])
    after = dispatch_signature(directory.parent, rows, 4)
    result = compare_dispatch(before, after)
    assert not result["exact"]
    assert result["failed_calls"] == [dict(prompt_id=32, role="provider", step=2)]
    file = directory / "dispatch_provider_2.json"
    file.write_text(json.dumps(dict(traceEvents=[dict(cat="kernel", name="gemm", ts=1, args={})])))
    with pytest.raises(ValueError, match="launch dimensions"):
        kernel_launches(file)
    file.write_text(json.dumps(dict(traceEvents=[])))
    with pytest.raises(ValueError, match="no GPU kernels"):
        kernel_launches(file)


def test_missing_or_duplicate_signature_is_not_accepted(tmp_path):
    rows, directory, _ = make_worker(tmp_path / "worker")
    signatures = dispatch_signature(directory.parent, rows, 4)
    for other in ([], signatures[:-1], signatures + signatures[:1]):
        with pytest.raises(ValueError):
            compare_dispatch(signatures, other)


def test_capture_state_cannot_change_under_profiling():
    base = dict(artifacts_sha256={"prompt_32/event_2.npz": "a", "prompt_32/provider_2_state.npz": "b"})
    changed = copy.deepcopy(base)
    changed["artifacts_sha256"]["prompt_32/provider_2_state.npz"] = "c"
    assert compare_capture(base, base)["exact"]
    with pytest.raises(ValueError, match="state differs"):
        compare_capture(base, changed)


def test_subset_cannot_pass_the_full_m1_gate():
    assert not full_gate(dict(limit=1, expected_full_evaluation_sequences=64), 1)
    assert not full_gate(dict(limit=64, expected_full_evaluation_sequences=64), 63)
    assert full_gate(dict(limit=64, expected_full_evaluation_sequences=64), 64)


def test_parent_allows_diagnostic_changes_but_rejects_environment_and_production_drift():
    old = dict(model_assets_sha256={"weights": "a"},
               source_sha256={"ivgym/sampling.py": "b", "experiments/numerical/capture_bf16.py": "c"},
               plan={k: "x" for k in ("model", "revision", "tokens", "sampling", "archive_manifest_sha256")},
               environment={k: "x" for k in ("device", "gpu_uuid", "driver", "software", "cuda_runtime",
                            "python", "tf32_matmul", "tf32_cudnn", "torch_threads")})
    rows = [dict(prompt_id=32, claimed_tokens=[1], hashes={"logits": "a"})]
    new = copy.deepcopy(old)
    new["source_sha256"]["experiments/numerical/capture_bf16.py"] = "updated"
    assert compare_parent(old, rows, new, rows)["exact"]
    for section, field in (("source_sha256", "ivgym/sampling.py"), ("environment", "driver")):
        changed = copy.deepcopy(new)
        changed[section][field] = "changed"
        with pytest.raises(ValueError):
            compare_parent(old, rows, changed, rows)
