"""Read-only scheduler diagnostics; CPU ranges, GPU work, and arrivals stay distinct."""
from __future__ import annotations

import argparse
import bisect
import collections
import gzip
import json
from pathlib import Path
import re
import sqlite3
import statistics

from experiments.exp_vllm_serving import Group, make_payload, sha, summarize_frames, write_json
from experiments.profile_vllm_serving import check_capture

LABEL = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")


def label_fields(name):
    match = LABEL.fullmatch(name)
    if not match:
        raise ValueError(f"Unknown iteration label: {name}")
    ctx, ctx_tokens, gen, gen_tokens = map(int, match.groups())
    return dict(context_requests=ctx, context_tokens=ctx_tokens,
                generation_requests=gen, generation_tokens=gen_tokens,
                kind="mixed" if ctx and gen else "context" if ctx else "generation" if gen else "empty")


def union_duration(intervals):
    total = 0.0
    end = float("-inf")
    for a, b in sorted(intervals):
        total += max(0, b - max(end, a))
        end = max(end, b)
    return total


def attach_kernels(iterations, runtime, kernels):
    """Use launch correlation and host thread, never GPU/CPU temporal overlap."""
    by_thread = collections.defaultdict(list)
    for i, event in enumerate(iterations):
        event["index"] = i
        by_thread[event["thread"]].append(event)
    starts = {thread: [e["start_ms"] for e in events] for thread, events in by_thread.items()}
    launches = {}
    for call in runtime:
        events = by_thread.get(call["thread"], [])
        j = bisect.bisect_right(starts.get(call["thread"], []), call["start_ms"]) - 1
        if j >= 0 and call["start_ms"] < events[j]["end_ms"]:
            key = call["correlation"]
            owner = events[j]["index"]
            if key in launches and launches[key] != owner:
                raise ValueError("Ambiguous CUDA runtime correlation")
            launches[key] = owner
    assigned = collections.defaultdict(list)
    for kernel in kernels:
        if kernel["correlation"] in launches:
            assigned[launches[kernel["correlation"]]].append(kernel)
    for i, event in enumerate(iterations):
        group = assigned[i]
        event.update(kernel_count=len(group),
                     kernel_sum_ms=sum(k["end_ms"]-k["start_ms"] for k in group),
                     kernel_union_ms=union_duration([(k["start_ms"], k["end_ms"]) for k in group]),
                     gpu_start_ms=min((k["start_ms"] for k in group), default=None),
                     gpu_end_ms=max((k["end_ms"] for k in group), default=None))
    return dict(kernel_events=len(kernels), attributed_kernels=sum(len(x) for x in assigned.values()),
                unattributed_kernels=len(kernels)-sum(len(x) for x in assigned.values()),
                attribution="Kernel -> CUDA runtime correlation -> containing CPU iteration on launch thread")


def nsys_iterations(database):
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as conn:
        iterations = [dict(start_ms=a/1e6, end_ms=b/1e6, thread=tid, name=name, **label_fields(name))
                      for a, b, tid, name in conn.execute(
                          "SELECT n.start,n.end,n.globalTid,COALESCE(n.text,s.value) FROM NVTX_EVENTS n "
                          "LEFT JOIN StringIds s ON n.textId=s.id "
                          "WHERE COALESCE(n.text,s.value) LIKE 'execute_context_%' ORDER BY n.start")]
        # Nsight packs the process identity in the upper bits of globalTid.
        runtime = [dict(start_ms=a/1e6, thread=tid, correlation=(tid & ~0xffffff, corr))
                   for a, tid, corr in conn.execute(
                       "SELECT start,globalTid,correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME")]
        kernels = [dict(start_ms=a/1e6, end_ms=b/1e6, correlation=(pid, corr))
                   for a, b, pid, corr in conn.execute(
                       "SELECT start,end,globalPid,correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL")]
    return iterations, attach_kernels(iterations, runtime, kernels)


def torch_iterations(path):
    with gzip.open(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    iterations = sorted([dict(start_ms=e["ts"]/1000, end_ms=(e["ts"]+e["dur"])/1000,
                              thread=e["tid"], name=e["name"], **label_fields(e["name"]))
                         for e in events if e.get("cat") == "user_annotation"
                         and e.get("name", "").startswith("execute_context_")], key=lambda e:e["start_ms"])
    runtime = [dict(start_ms=e["ts"]/1000, thread=e["tid"], correlation=e["args"]["correlation"])
               for e in events if e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {})]
    kernels = [dict(start_ms=e["ts"]/1000, end_ms=(e["ts"]+e["dur"])/1000,
                    correlation=e["args"]["correlation"]) for e in events if e.get("cat") == "kernel"]
    return iterations, attach_kernels(iterations, runtime, kernels)


def iteration_summary(iterations, attribution):
    if not iterations:
        raise ValueError("No CPU iteration ranges")
    origin = iterations[0]["start_ms"]
    for e in iterations:
        for key in ("start_ms", "end_ms", "gpu_start_ms", "gpu_end_ms"):
            if e[key] is not None:
                e[key] -= origin
    gen = [e for e in iterations if e["generation_requests"]]
    ctx = [e for e in iterations if e["context_requests"]]
    first_gen = gen[0]["index"] if gen else None
    last_ctx = ctx[-1]["index"] if ctx else None
    return dict(**attribution, iterations=len(iterations),
                counts=dict(collections.Counter(e["kind"] for e in iterations)),
                context_tokens=sum(e["context_tokens"] for e in iterations),
                generation_tokens=sum(e["generation_tokens"] for e in iterations),
                first_generation_iteration=first_gen, last_context_iteration=last_ctx,
                context_iterations_after_generation=sum(e["index"] >= first_gen for e in ctx) if gen else 0,
                mixed_per_iteration_kernel_union_ms_sum=sum(e["kernel_union_ms"] for e in iterations if e["kind"]=="mixed"),
                timeline=iterations)


def request_summary(rows):
    summaries = []
    for row in rows:
        arrivals = []
        for frame in row["frames"]:
            for choice in frame["data"].get("choices", []):
                arrivals += [frame["received_ns_since_send"] / 1e6] * len(choice.get("token_ids") or [])
        if len(arrivals) != 128:
            raise ValueError("This frozen analysis requires 128 output tokens")
        summaries.append(dict(request_id=row["request_id"], ttft_ms=arrivals[0],
                              early_ms_per_token=(arrivals[32]-arrivals[0])/32,
                              late_ms_per_token=(arrivals[127]-arrivals[32])/95,
                              post_first_ms_per_token=row["post_first_ms_per_token"],
                              output_token_ids=row["predicted_token_ids"],
                              cached_tokens=row["cached_tokens"],
                              coalesced_token_count=row["coalesced_token_count"]))
    keys = ("ttft_ms", "early_ms_per_token", "late_ms_per_token", "post_first_ms_per_token")
    return dict(group_medians={key:statistics.median(r[key] for r in summaries) for key in keys},
                coalesced_tokens=sum(r["coalesced_token_count"] for r in summaries), requests=summaries)


def validate_iteration_work(scheduling, job):
    effective = min(job["length"], 2048) if job["arm"] == "truncate" else job["length"]
    uncached = 128 if job["arm"] == "hit" else effective
    if scheduling["context_tokens"] != job["concurrency"] * uncached:
        raise ValueError("Iteration annotations do not account for all uncached prompt tokens")
    if scheduling["generation_tokens"] != job["concurrency"] * (job["tokens"] - 1):
        raise ValueError("Iteration annotations do not account for all generation tokens")
    if any(e["context_tokens"] + e["generation_tokens"] > job["scheduler_budget"]
           for e in scheduling["timeline"]):
        raise ValueError("Iteration exceeds the configured scheduler token budget")


def analyze(root, allow_partial=False):
    plan = json.loads((root / "plan.json").read_text())
    for name, expected in plan["source_sha256"].items():
        if sha(root / "source" / name) != expected:
            raise ValueError(f"Campaign source mismatch: {name}")
    runs, missing = [], []
    for job in plan["jobs"]:
        path = root / job["name"]
        if not (path / "manifest.json").exists() or json.loads((path / "manifest.json").read_text())["status"] != "complete":
            missing.append(job["name"])
            continue
        verification = check_capture(path)
        manifest = json.loads((path / "manifest.json").read_text())
        for k in ("config", "profiler", "scheduler_budget"):
            if job[k] != manifest[k]:
                raise ValueError(f"Plan mismatch: {job['name']} {k}")
        for k, manifest_key in (("block", "block"), ("length", "length"), ("tokens", "outputs"),
                                ("concurrency", "concurrency"), ("arm", "arm")):
            if job[k] != manifest["group"][manifest_key]:
                raise ValueError(f"Group plan mismatch: {job['name']} {k}")
        for name, expected in manifest["source_sha256"].items():
            if expected != plan["source_sha256"][name]:
                raise ValueError(f"Run source differs from frozen plan: {name}")
        log = (path / "server.log").read_text()
        budget_confirmed = (f"max_num_batched_tokens={job['scheduler_budget']}" in log or
                            f"'max_num_batched_tokens': {job['scheduler_budget']}" in log)
        if not budget_confirmed or "Using AttentionBackendEnum.FLASH_ATTN backend" not in log:
            raise ValueError("Runtime log does not confirm scheduler budget/attention backend")
        execution = "enforce_eager=True" if job["config"] == "eager_off" else "CUDAGraphMode.FULL_AND_PIECEWISE"
        if execution not in log:
            raise ValueError("Runtime log does not confirm the intended execution mode")
        with gzip.open(path / "requests.jsonl.gz", "rt") as handle:
            rows = [json.loads(line) for line in handle]
        expected_cache = job["length"]-128 if job["arm"]=="hit" else 0
        if any(r["cached_tokens"] != expected_cache for r in rows):
            raise ValueError("Unexpected cache state")
        corpus = json.loads((path / "corpus_token_ids.json").read_text())
        for filename, group, prime in (
            ("warmup.jsonl.gz", Group(99, job["length"], job["concurrency"], "full", 8), False),
            ("primers.jsonl.gz", Group(job["block"], job["length"], job["concurrency"], job["arm"], 128), True),
        ):
            if prime and job["arm"] != "hit":
                if (path / filename).exists():
                    raise ValueError("Unexpected cache primer")
                continue
            with gzip.open(path / filename, "rt") as handle:
                auxiliary = [json.loads(line) for line in handle]
            if len(auxiliary) != job["concurrency"]:
                raise ValueError("Auxiliary request coverage mismatch")
            for i, row in enumerate(auxiliary):
                payload = make_payload(corpus, group, i, prime=prime)
                if payload != row["payload"]:
                    raise ValueError("Auxiliary payload differs from saved fixture")
                actual = summarize_frames(row["frames"], payload["prompt"], payload["max_tokens"])
                if any(row[k] != value for k, value in actual.items()):
                    raise ValueError("Auxiliary raw stream disagrees with summary")
        run = dict(**job, verification=verification["status"], arrivals=request_summary(rows),
                   semaphore_shutdown_warning="leaked semaphore" in log,
                   started_utc=manifest["started_utc"], finished_utc=manifest["finished_utc"])
        if job["profiler"] == "nsys":
            reports = manifest["validation"]["reports"]
            if len(reports) != 1:
                raise ValueError("Expected one Nsight capture per group")
            run["scheduling"] = iteration_summary(*nsys_iterations(path / reports[0]["database"]))
        elif job["profiler"] == "torch":
            traces = [t for t in verification["traces"] if t["kernel_events"] and t["iteration_ranges"]]
            if len(traces) != 1:
                raise ValueError("Expected one GPU worker trace")
            run["scheduling"] = iteration_summary(*torch_iterations(path / traces[0]["path"]))
        if "scheduling" in run:
            validate_iteration_work(run["scheduling"], job)
        runs.append(run)
    if missing and not allow_partial:
        raise ValueError(f"Incomplete campaign: {len(missing)} missing captures")
    return dict(status="partial" if missing else "complete", planned_runs=len(plan["jobs"]),
                verified_runs=len(runs), missing=missing,
                requests=sum(len(r["arrivals"]["requests"]) for r in runs),
                generated_tokens=sum(len(r["arrivals"]["requests"])*128 for r in runs), runs=runs,
                limitations=["Two overlapping offset blocks of one corpus; descriptive, not independent calibration.",
                             "Keep Nsight, PyTorch, and unprofiled arrivals separate.",
                             "Labels describe context/generation requests, not per-request GPU causality.",
                             "CPU range duration is not GPU execution time; kernels assigned through launch correlation.",
                             "Client arrivals and trace clocks are not aligned."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = analyze(args.root, args.allow_partial)
    write_json(args.output, result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("runs", "missing")}, indent=2))


if __name__ == "__main__":
    main()
