"""Offline validation and descriptive summaries of the vLLM serving pilot."""
from __future__ import annotations
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import statistics

from experiments.exp_vllm_serving import Group, build_plan, summarize_frames


def read_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path,"rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def analyze(root):
    queue=json.loads((root/"queue.json").read_text())
    assert queue["status"]=="complete"
    groups, requests = [], []
    source = queue["source_commit"]
    for job in queue["jobs"]:
        assert job["status"]=="complete"
        directory=root/f"session_{job['session']}_{job['config']}"
        manifest=json.loads((directory/"manifest.json").read_text())
        assert manifest["status"]=="complete"
        for filename,expected in manifest["hashes"].items():
            assert hashlib.sha256((directory/filename).read_bytes()).hexdigest()==expected,filename
        plan=json.loads((directory/"plan.json").read_text())
        expected_plan=[vars(g) for g in build_plan(job["config"],job["session"],queue["suite"])]
        assert plan==expected_plan
        gs=read_rows(directory/"groups.jsonl")
        rs=read_rows(directory/"requests.jsonl.gz")
        assert len(gs)==len(plan)
        assert len(rs)==sum(g["concurrency"] for g in plan)
        assert len({r["request_id"] for r in rs})==len(rs)
        by_group=collections.defaultdict(list)
        for row in rs:
            assert row["status"]=="complete"
            payload=row["payload"]
            expected_ids=payload["prompt"]
            if "truncate_prompt_tokens" in payload:
                expected_ids=expected_ids[-payload["truncate_prompt_tokens"]:]
            recomputed=summarize_frames(row["frames"],expected_ids,row["outputs"])
            assert all(row[k]==v for k,v in recomputed.items())
            assert len(payload["prompt"])==row["length"]
            assert row["effective_prompt_tokens"]==(min(row["length"],2048) if row["arm"]=="truncate" else row["length"])
            if job["config"]!="graph_on" or row["arm"]=="cold":
                assert row["cached_tokens"]==0
            if row["arm"]=="hit":
                assert 0<row["cached_tokens"]<=row["length"]-128
            by_group[row["group_id"]].append(row)
        for planned,group in zip(plan,gs):
            assert all(group[k]==v for k,v in planned.items())
            rows=by_group[group["group_id"]]
            assert len(rows)==planned["concurrency"]
            assert group["median_ttft_ms"]==statistics.median(r["ttft_ms"] for r in rows)
            assert group["median_post_first_ms_per_token"]==statistics.median(r["post_first_ms_per_token"] for r in rows)
        groups.extend(gs);requests.extend(rs)
    grouped=collections.defaultdict(list)
    for group in groups:
        grouped[group["config"],group["length"],group["concurrency"],group["arm"]].append(group)
    summary=[]
    for (config,length,concurrency,arm),gs in sorted(grouped.items()):
        summary.append({"config":config,"length":length,"concurrency":concurrency,"arm":arm,
            "groups":len(gs),"blocks":sorted({g["block"] for g in gs}),
            "sessions":sorted({g["session"] for g in gs}),
            "ttft_ms_median":statistics.median(g["median_ttft_ms"] for g in gs),
            "post_first_ms_per_token_median":statistics.median(g["median_post_first_ms_per_token"] for g in gs),
            "post_first_ms_per_token_min":min(g["median_post_first_ms_per_token"] for g in gs),
            "post_first_ms_per_token_max":max(g["median_post_first_ms_per_token"] for g in gs),
            "cached_tokens_min":min(min(g["cached_tokens"]) for g in gs),
            "cached_tokens_max":max(max(g["cached_tokens"]) for g in gs)})
    paired=collections.defaultdict(dict)
    for group in groups:
        paired[group["config"],group["block"],group["length"],group["concurrency"]][group["arm"]]=group
    comparisons=[]
    for key,arms in sorted(paired.items()):
        first,second=("cold","hit") if key[0]=="graph_on" and queue["suite"]=="pilot" else ("full","truncate")
        if first not in arms or second not in arms:
            continue
        a,b=arms[first],arms[second]
        comparisons.append({"config":key[0],"block":key[1],"length":key[2],"concurrency":key[3],
            "first":first,"second":second,
            "ttft_ratio":a["median_ttft_ms"]/b["median_ttft_ms"],
            "post_first_ratio":a["median_post_first_ms_per_token"]/b["median_post_first_ms_per_token"]})
    # Output equality is reported, not assumed: scheduling can change floating-point paths.
    rp={(r["config"],r["block"],r["length"],r["concurrency"],r["arm"],r["row_index"]):r for r in requests}
    checks=collections.defaultdict(lambda:{"pairs":0,"identical_sequences":0,"matching_positions":0,"positions":0})
    for key,a in rp.items():
        config,block,length,concurrency,arm,row=key
        candidates=[]
        if arm=="full" and length<=2048:
            candidates.append(("noop_truncation",(config,block,length,concurrency,"truncate",row)))
        if config=="graph_on" and arm=="cold":
            candidates.append(("cold_vs_hit",(config,block,length,concurrency,"hit",row)))
            candidates.append(("cache_off_vs_cold",("graph_off",block,length,concurrency,"full",row)))
        for name,bkey in candidates:
            if bkey not in rp:
                continue
            b=rp[bkey];x,y=a["predicted_token_ids"],b["predicted_token_ids"]
            assert a["payload"]["prompt"]==b["payload"]["prompt"]
            c=checks[name];c["pairs"]+=1;c["identical_sequences"]+=int(x==y)
            c["matching_positions"]+=sum(i==j for i,j in zip(x,y));c["positions"]+=len(x)
    return {"scope":queue["scope"],"source_commit":source,"suite":queue["suite"],
            "server_runs":len(queue["jobs"]),"measurement_groups":len(groups),
            "measured_requests":len(requests),"request_outputs":sorted({r["outputs"] for r in requests}),
            "groups":summary,"paired_comparisons":comparisons,"output_controls":dict(checks),
            "coalesced_tokens":sum(r["coalesced_token_count"] for r in requests),
            "total_generated_tokens":sum(len(r["predicted_token_ids"]) for r in requests),
            "max_cached_tokens":max(r["cached_tokens"] for r in requests)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.root)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("suite","server_runs","measurement_groups",
                                          "measured_requests","output_controls","coalesced_tokens")},indent=2))


if __name__=="__main__":
    main()
