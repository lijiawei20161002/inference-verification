"""Validate raw pilot artifacts and produce descriptive paired summaries.

No fitted detector, deployment false-positive bound, or independent-token CI.
All timing summaries retain the number of blocks and process sessions.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def median(xs):
    return statistics.median(xs) if xs else None


def summarize(rows):
    grouped=collections.defaultdict(list)
    for r in rows:
        if r["status"]=="complete":
            key=(r["suite"],r["length"],r["batch"],r["condition"],r["cache"],r["mode"],r["instrument"],r["execution"])
            grouped[key].append(r)
    result=[]
    for key,rs in sorted(grouped.items()):
        suite,length,batch,condition,cache,mode,instrument,execution=key
        result.append({"suite":suite,"length":length,"batch":batch,"condition":condition,
            "cache":cache,"mode":mode,"instrument":instrument,"execution":execution,"n":len(rs),
            "blocks":sorted({r["block"] for r in rs}),"sessions":sorted({r["session"] for r in rs}),
            "prefill_ms_median":median([r["prefill_wall_ms"] for r in rs]),
            "decode_ms_per_step_median":median([r["decode_ms_per_step"] for r in rs]),
            "decode_ms_per_step_min":min(r["decode_ms_per_step"] for r in rs),
            "decode_ms_per_step_max":max(r["decode_ms_per_step"] for r in rs),
            "cuda_step_span_ms_median":median([median(r["decode_cuda_spans_ms"]) for r in rs
                                              if r["decode_cuda_spans_ms"]]),
            "peak_allocated_gib":max(r["peak_allocated_bytes"] for r in rs)/1024**3,
            "logical_cache_gib":median([r["cache_allocated_bytes"]/1024**3 for r in rs]),
            "priming_ms_median":median([r["priming_ms"] for r in rs])})
    return result


def validate_pairs(rows):
    pairs=collections.defaultdict(dict)
    for r in rows:
        if r["status"]=="complete" and r["suite"] in {"p1","p4"}:
            key=(r["suite"],r["block"],r["length"],r["batch"],r["cache"],r["mode"])
            if r["condition"] in pairs[key]:
                raise ValueError(f"Duplicate arm: {key}")
            pairs[key][r["condition"]]=r
    checks={"negative_control_pairs":0,"prefix_cache_pairs":0}
    comparisons=[]
    for key,arms in pairs.items():
        suite,block,length,batch,cache,mode=key
        if suite=="p1" and "full" in arms and "truncate" in arms:
            full,tail=arms["full"],arms["truncate"]
            assert full["prompt_sha256"]==tail["prompt_sha256"]
            if length<=2048:
                assert full["first_logits_sha256"]==tail["first_logits_sha256"],key
                assert full["predicted_token_ids"]==tail["predicted_token_ids"],key
                checks["negative_control_pairs"]+=1
            comparisons.append({"suite":suite,"block":block,"session":full["session"],
                "length":length,"batch":batch,"prefill_full_over_truncated":full["prefill_wall_ms"]/tail["prefill_wall_ms"],
                "decode_full_over_truncated":full["decode_ms_per_step"]/tail["decode_ms_per_step"]})
        if suite=="p4":
            if not all(k in arms for k in ("cache_off","cache_cold","cache_hit")):
                raise ValueError(f"Incomplete cache comparison: {key}")
            off,cold,hit=[arms[k] for k in ("cache_off","cache_cold","cache_hit")]
            assert off["prompt_sha256"]==cold["prompt_sha256"]==hit["prompt_sha256"],key
            assert off["first_logits_sha256"]==cold["first_logits_sha256"]==hit["first_logits_sha256"],key
            assert off["predicted_token_ids"]==cold["predicted_token_ids"]==hit["predicted_token_ids"],key
            checks["prefix_cache_pairs"]+=1
            comparisons.append({"suite":suite,"block":block,"session":off["session"],
                "length":length,"batch":batch,
                "prefill_cold_over_hit":cold["prefill_wall_ms"]/hit["prefill_wall_ms"],
                "decode_cold_over_hit":cold["decode_ms_per_step"]/hit["decode_ms_per_step"]})
    return checks,comparisons


def analyze(root):
    rows=[]
    verified=[]
    for path in sorted(root.glob("session_*/measurements.jsonl")):
        manifest=json.loads((path.parent/"manifest.json").read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest["measurement_sha256"]
        plan=json.loads((path.parent/"plan.json").read_text())
        current=read_rows(path)
        assert len(current)==len(plan),(path,len(current),len(plan))
        for row,planned in zip(current,plan):
            assert all(row[k]==v for k,v in planned.items()),(path,row["case_id"])
            if row["status"]=="complete":
                assert len(row["predicted_token_ids"])==row["batch"]
                assert all(len(seq)==row["outputs"] for seq in row["predicted_token_ids"])
                assert len(row["decode_cuda_spans_ms"])==(row["outputs"]-1 if row["instrument"]=="events" else 0)
                assert row["estimated_kv_bytes"]==row["cache_allocated_bytes"]
        rows+=current
        verified.append({"path":str(path.relative_to(root)),"sha256":manifest["measurement_sha256"],
                         "source_commit":manifest["source_commit"],"source_status":manifest["source_status"]})
    if not rows:
        raise ValueError("No session artifacts")
    checks,pairs=validate_pairs(rows)
    http=[]
    for path in sorted(root.glob("http_*/client.jsonl")):
        client=read_rows(path)
        server={r["request_id"]:r for r in read_rows(path.parent/"server.jsonl")}
        manifest=json.loads((path.parent/"manifest.json").read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest["client_sha256"]
        assert hashlib.sha256((path.parent/"server.jsonl").read_bytes()).hexdigest()==manifest["server_sha256"]
        for r in client:
            trace=server[r["request_id"]]
            decoded=[s for e in r["events"] for s in e["steps"]]
            assert [s["step"] for s in decoded]==list(range(r["outputs"]))
            assert [[s["token_ids"][b] for s in decoded] for b in range(r["batch"])]==trace["predicted_token_ids"]
            times=[e["received_ns_since_send"]/1e6 for e in r["events"]]
            http.append({"session":r["session"],"length":r["length"],"batch":r["batch"],
                "condition":r["condition"],"chunk":r["chunk"],"ttft_ms":r["ttft_ms"],
                "last_event_ms":r["last_event_ms"],"n_events":len(times),
                "event_gap_ms_median":median([b-a for a,b in zip(times,times[1:])]),
                "server_prefill_ms":trace["prefill_wall_ms"],"server_decode_ms":trace["decode_wall_ms"],
                "server_setup_ms":trace["setup_ms"]})
    return {"scope":"descriptive mechanism pilot; no FPR/TPR calibration; blocks clustered in three process sessions",
        "status_counts":dict(collections.Counter(r["status"] for r in rows)),
        "suite_counts":dict(collections.Counter(r["suite"] for r in rows)),
        "verified_files":verified,"checks":checks,"groups":summarize(rows),
        "paired_comparisons":pairs,"http":http}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.root)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ("status_counts","suite_counts","checks")},indent=2))


if __name__=="__main__":
    main()
