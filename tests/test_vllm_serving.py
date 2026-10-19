"""Experimental contracts for streaming timestamps, truncation, and cache controls."""
import pytest
from experiments.exp_vllm_serving import build_plan, make_payload, Group, summarize_frames


def test_pilot_coverage_and_request_counts():
    plans=[build_plan(config,session) for session in range(3)
           for config in ("graph_off","graph_on","eager_off")]
    assert sum(len(p) for p in plans)==240
    assert sum(g.concurrency for p in plans for g in p)==600
    assert {g.block for p in plans for g in p}==set(range(6))
    assert build_plan("graph_off",0)==build_plan("graph_off",0)
    assert all(g.arm in ("cold","hit") for g in build_plan("graph_on",0))


def test_truncation_keeps_wire_prompt_and_priming_uses_only_prefix():
    corpus=list(range(100))
    group=Group(1,8192,4,"truncate")
    payload=make_payload(corpus,group,2)
    assert len(payload["prompt"])==8192
    assert payload["truncate_prompt_tokens"]==2048
    assert payload["truncation_side"]=="left"
    cold=make_payload(corpus,Group(1,8192,4,"cold"),2)
    hit=make_payload(corpus,Group(1,8192,4,"hit"),2)
    prime=make_payload(corpus,Group(1,8192,4,"hit"),2,prime=True)
    assert cold==hit
    assert prime["prompt"]==hit["prompt"][:-128]
    assert prime["cache_salt"]==hit["cache_salt"]
    assert prime["max_tokens"]==1


def test_stream_parser_counts_tokens_not_sse_frames_and_rejects_wrong_prompt():
    frames=[
        {"received_ns_since_send":10_000_000,"data":{"choices":[
            {"index":0,"prompt_token_ids":[7,8],"token_ids":[11,12],"finish_reason":None}]}},
        {"received_ns_since_send":30_000_000,"data":{"choices":[
            {"index":0,"token_ids":[13],"finish_reason":"length"}]}},
        {"received_ns_since_send":31_000_000,"data":{"choices":[],"usage":
            {"prompt_tokens":2,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":1}}}},
    ]
    result=summarize_frames(frames,[7,8],3)
    assert result["ttft_ms"]==10
    assert result["content_events"]==2
    assert result["coalesced_token_count"]==1
    assert result["cached_tokens"]==1
    assert result["post_first_ms_per_token"]==20
    with pytest.raises(AssertionError,match="Returned prompt"):
        summarize_frames(frames,[8,9],3)
