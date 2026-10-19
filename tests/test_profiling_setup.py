"""Require GPU evidence and count CPU iterations separately from GPU projections."""
import gzip
import json

import pytest

from experiments.profile_vllm_serving import torch_trace_summary


def write_trace(root, events):
    (root / "traces").mkdir()
    with gzip.open(root / "traces/worker.pt.trace.json.gz", "wt") as handle:
        json.dump({"traceEvents": events}, handle)


def test_cpu_only_trace_is_not_a_successful_gpu_capture(tmp_path):
    write_trace(tmp_path, [{"cat": "user_annotation", "name": "execute_context_0(0)_generation_4(4)"}])
    with pytest.raises(RuntimeError, match="CUDA kernels"):
        torch_trace_summary(tmp_path)


def test_cuda_trace_without_iteration_annotations_is_rejected(tmp_path):
    write_trace(tmp_path, [{"cat": "kernel", "name": "gemm", "dur": 12}])
    with pytest.raises(RuntimeError, match="scheduler iteration"):
        torch_trace_summary(tmp_path)


def test_projected_gpu_ranges_do_not_double_count_iterations(tmp_path):
    name = "execute_context_1(2047)_generation_1(1)"
    write_trace(tmp_path, [{"cat": "kernel", "name": "gemm", "dur": 12},
                           {"cat": "user_annotation", "name": name},
                           {"cat": "gpu_user_annotation", "name": name}])
    result = torch_trace_summary(tmp_path)["traces"][0]
    assert result["kernel_events"] == 1
    assert result["iteration_ranges"] == 1
    assert result["gpu_iteration_annotations"] == 1
    assert result["iteration_names"] == {name: 1}
