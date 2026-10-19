"""Protect scheduling attribution and stream-window measurement semantics."""
import pytest

from experiments.analyze_scheduling_campaign import (
    attach_kernels, label_fields, request_summary, union_duration, validate_iteration_work,
)


def test_async_kernels_follow_launch_correlation_not_cpu_overlap():
    iterations = [dict(start_ms=0, end_ms=2, thread=11),
                  dict(start_ms=2, end_ms=4, thread=11)]
    runtime = [dict(start_ms=1, thread=11, correlation=(1, 5)),
               dict(start_ms=3, thread=11, correlation=(1, 6)),
               dict(start_ms=1, thread=99, correlation=(2, 5))]
    kernels = [dict(start_ms=3, end_ms=5, correlation=(1, 5)),
               dict(start_ms=4, end_ms=6, correlation=(1, 5)),
               dict(start_ms=7, end_ms=8, correlation=(1, 6)),
               dict(start_ms=1, end_ms=2, correlation=(2, 5))]
    result = attach_kernels(iterations, runtime, kernels)
    assert iterations[0]["kernel_count"] == 2
    assert iterations[0]["kernel_sum_ms"] == 4
    assert iterations[0]["kernel_union_ms"] == 3
    assert iterations[1]["kernel_count"] == 1
    assert result["unattributed_kernels"] == 1


def test_context_and_generation_labels_are_not_request_batch_size():
    mixed = label_fields("execute_context_2(2046)_generation_2(2)")
    assert mixed["kind"] == "mixed"
    assert mixed["context_requests"] == 2
    assert mixed["context_tokens"] == 2046
    with pytest.raises(ValueError):
        label_fields("execute_context_unknown")
    assert union_duration([(1, 4), (2, 3), (8, 9)]) == 4


def test_arrival_windows_preserve_coalesced_token_timestamps():
    frames = [{"received_ns_since_send": i*1_000_000,
               "data": {"choices": [{"token_ids": [i, i]}]}} for i in range(64)]
    result = request_summary([dict(frames=frames, request_id="r", cached_tokens=0,
                                   coalesced_token_count=64, post_first_ms_per_token=0.5,
                                   predicted_token_ids=[i for i in range(64) for _ in range(2)])])
    assert result["group_medians"]["early_ms_per_token"] == 0.5
    assert result["group_medians"]["late_ms_per_token"] == pytest.approx(47/95)
    assert result["coalesced_tokens"] == 64


def test_iteration_work_accounts_for_cached_prefix_and_first_output_token():
    job = dict(length=8192, arm="hit", concurrency=4, tokens=128, scheduler_budget=2048)
    observed = dict(context_tokens=512, generation_tokens=508,
                    timeline=[dict(context_tokens=512, generation_tokens=0)])
    validate_iteration_work(observed, job)


@pytest.mark.parametrize("field,value", [("context_tokens",511), ("generation_tokens",512),
                                        ("timeline",[dict(context_tokens=2048,generation_tokens=1)])])
def test_incomplete_or_over_budget_iteration_evidence_is_rejected(field, value):
    job = dict(length=8192, arm="hit", concurrency=4, tokens=128, scheduler_budget=2048)
    observed = dict(context_tokens=512, generation_tokens=508,
                    timeline=[dict(context_tokens=512, generation_tokens=0)])
    observed[field] = value
    with pytest.raises(ValueError):
        validate_iteration_work(observed, job)
