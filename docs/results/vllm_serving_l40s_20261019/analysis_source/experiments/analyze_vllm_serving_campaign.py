"""Validate two matched vLLM pilots and summarize temporal and arrival-window controls.

The delayed repeat reuses prompts, seeds, and order. Blocks are paired technical
replicates, not independent deployments. The late window was selected after
inspecting the first pilot and is an exploratory diagnostic, not a detector.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics

from experiments.analyze_vllm_serving import analyze, read_rows
from experiments.exp_vllm_serving import (
    CONFIGS, SOURCE_FILES, Group, build_plan, make_payload, server_command,
    summarize_frames,
)

PHASES = ("pilot", "delayed_repeat")
METRICS = ("ttft_ms", "post_first_ms_per_token", "early_ms_per_token", "late_ms_per_token")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def span(values):
    values = list(values)
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def arrival_windows(row):
    """Receive-time intervals between token indices 0/32/127, not GPU timers."""
    arrivals = []
    for frame in row["frames"]:
        for choice in frame["data"].get("choices", []):
            arrivals.extend([frame["received_ns_since_send"] / 1e6]
                            * len(choice.get("token_ids") or []))
    require(len(arrivals) == 128, "Window diagnostic requires exactly 128 returned tokens")
    require(all(a <= b for a, b in zip(arrivals, arrivals[1:])), "Nonmonotonic arrivals")
    return {"early_ms_per_token": (arrivals[32] - arrivals[0]) / 32,
            "late_ms_per_token": (arrivals[127] - arrivals[32]) / 95}


def validate_queue(queue):
    require(queue["status"] == "complete" and queue["suite"] == "pilot", "Incomplete pilot")
    expected = [(s, c) for s in range(3) for c in (CONFIGS[s:] + CONFIGS[:s])]
    require([(j["session"], j["config"]) for j in queue["jobs"]] == expected,
            "Missing, duplicated, or reordered server runs")
    require(all(j["status"] == "complete" for j in queue["jobs"]), "Incomplete server run")
    require(queue["source_status"] == "", "Formal measurement source was not clean")
    require(set(queue["source_sha256"]) == set(SOURCE_FILES), "Incomplete source coverage")


def validate_run(root):
    queue = json.loads((root / "queue.json").read_text())
    validate_queue(queue)
    for name, expected in queue["source_sha256"].items():
        require(digest(root / "source" / name) == expected, f"Source mismatch: {name}")
    base = analyze(root)  # Recompute original summaries, hashes, stream and cache contracts.
    require((base["server_runs"], base["measurement_groups"], base["measured_requests"],
             base["total_generated_tokens"]) == (9, 240, 600, 76800), "Incomplete coverage")
    corpus = json.loads((root / "corpus_token_ids.json").read_text())
    all_groups, all_requests, gpu_memory = [], {}, []
    primer_count = warmup_count = 0
    for job in queue["jobs"]:
        session, config = job["session"], job["config"]
        directory = root / f"session_{session}_{config}"
        manifest = json.loads((directory / "manifest.json").read_text())
        require(manifest["config"] == config and manifest["session"] == session,
                "Manifest identity mismatch")
        require(manifest["server_returncode_after_shutdown"] == 0, "Server shutdown failed")
        required_files = {"plan.json", "server.log", "requests.jsonl.gz", "warmup.jsonl.gz",
                          "primers.jsonl.gz", "metrics.jsonl.gz", "groups.jsonl"}
        require(set(manifest["hashes"]) == required_files, "Incomplete artifact hash coverage")
        cmd = manifest["command"]
        require(cmd == server_command(cmd[0], cmd[cmd.index("--model") + 1],
                                      int(cmd[cmd.index("--port") + 1]), config),
                "Server command differs from frozen protocol")
        log = (directory / "server.log").read_text()
        require("Using AttentionBackendEnum.FLASH_ATTN backend." in log, "Backend unconfirmed")
        if config == "eager_off":
            require("CUDAGraphMode.NONE" in log and "disabling torch.compile and CUDAGraphs" in log,
                    "Eager configuration unconfirmed")
        else:
            require("CUDAGraphMode.FULL_AND_PIECEWISE" in log and "Graph capturing finished" in log,
                    "Graph configuration unconfirmed")
        plan = build_plan(config, session)
        rows = read_rows(directory / "requests.jsonl.gz")
        expected_rows = [(g, i) for g in plan for i in range(g.concurrency)]
        grouped = collections.defaultdict(list)
        for row, (group, row_index) in zip(rows, expected_rows):
            require(all(row[k] == v for k, v in vars(group).items()), "Request group fields differ")
            require((row["config"], row["session"], row["group_id"], row["row_index"])
                    == (config, session, group.key, row_index), "Request identity mismatch")
            require(row["request_id"] == f"{group.key}_r{row_index}", "Request ID mismatch")
            require(row["payload"] == make_payload(corpus, group, row_index), "Fixture/payload mismatch")
            require(row["send_completed_ns"] >= row["send_start_ns"], "Invalid send timestamps")
            if group.arm == "hit":
                require(row["cached_tokens"] == group.length - 128, "Incomplete expected prefix hit")
            derived = {**row, **arrival_windows(row)}
            grouped[group.key].append(derived)
            key = (config, group.block, group.length, group.concurrency, group.arm, row_index)
            require(key not in all_requests, "Duplicate request identity")
            all_requests[key] = derived
        gs = read_rows(directory / "groups.jsonl")
        for group, saved in zip(plan, gs):
            rs = grouped[group.key]
            require(saved["request_ids"] == [r["request_id"] for r in rs], "Group request IDs differ")
            all_groups.append({"config": config, "session": session, **vars(group),
                               **{metric: statistics.median(r[metric] for r in rs) for metric in METRICS}})
            for key in ("gpu_before", "gpu_after"):
                snapshot = saved[key]
                require(snapshot["returncode"] == 0, "GPU snapshot failed")
                gpu_memory.append(int(snapshot["stdout"].splitlines()[1].split(",")[3].split()[0]))
        metrics = read_rows(directory / "metrics.jsonl.gz")
        require([m["group_id"] for m in metrics] == [g.key for g in plan], "Missing metrics group")
        warmup = [(g, i, False) for g in (Group(99, 512, 1, "full", 16),
                                         Group(99, 8192, 4, "full", 16)) for i in range(g.concurrency)]
        primes = [(g, i, True) for g in plan if g.arm == "hit" for i in range(g.concurrency)]
        for filename, expected in (("warmup.jsonl.gz", warmup), ("primers.jsonl.gz", primes)):
            aux = read_rows(directory / filename)
            require(len(aux) == len(expected), f"Auxiliary request coverage: {filename}")
            for row, (group, index, prime) in zip(aux, expected):
                payload = make_payload(corpus, group, index, prime)
                require(row["status"] == "complete" and row["payload"] == payload,
                        f"Auxiliary request mismatch: {filename}")
                recomputed = summarize_frames(row["frames"], payload["prompt"], payload["max_tokens"])
                require(all(row[k] == v for k, v in recomputed.items()), "Auxiliary stream mismatch")
        primer_count += len(primes)
        warmup_count += len(warmup)
    return {"queue": queue, "original_summary": base, "groups": all_groups,
            "requests": all_requests, "primer_requests": primer_count,
            "warmup_requests": warmup_count, "max_sampled_gpu_memory_mib": max(gpu_memory)}


def summarize_phase(run):
    cells, pairs = collections.defaultdict(list), collections.defaultdict(dict)
    for g in run["groups"]:
        cells[g["config"], g["length"], g["concurrency"], g["arm"]].append(g)
        pairs[g["config"], g["block"], g["length"], g["concurrency"]][g["arm"]] = g
    summaries = [{"config": k[0], "length": k[1], "concurrency": k[2], "arm": k[3],
                  "blocks": len(gs), **{metric: span(g[metric] for g in gs) for metric in METRICS}}
                 for k, gs in sorted(cells.items())]
    contrasts = collections.defaultdict(list)
    for (config, block, length, concurrency), arms in pairs.items():
        first, second = ("cold", "hit") if config == "graph_on" else ("full", "truncate")
        contrasts[config, length, concurrency].append(
            {metric: arms[first][metric] / arms[second][metric] for metric in METRICS})
    ratios = [{"config": k[0], "length": k[1], "concurrency": k[2], "paired_blocks": len(rs),
               **{metric: span(r[metric] for r in rs) for metric in METRICS}}
              for k, rs in sorted(contrasts.items())]
    controls = collections.defaultdict(lambda: {"pairs": 0, "identical_sequences": 0})
    for key, a in run["requests"].items():
        config, block, length, concurrency, arm, row = key
        candidates = []
        if arm == "full" and length <= 2048:
            candidates.append(("noop_truncation", (config, block, length, concurrency, "truncate", row)))
        if config == "graph_on" and arm == "cold":
            candidates.append(("cold_vs_hit", (config, block, length, concurrency, "hit", row)))
            candidates.append(("cache_off_vs_cold", ("graph_off", block, length, concurrency, "full", row)))
        for name, other in candidates:
            b = run["requests"][other]
            counts = controls[name, config, concurrency]
            counts["pairs"] += 1
            counts["identical_sequences"] += int(a["predicted_token_ids"] == b["predicted_token_ids"])
    return {"original_summary": run["original_summary"], "cells": summaries, "contrasts": ratios,
            "output_controls_by_concurrency": [
                {"control": k[0], "config": k[1], "concurrency": k[2], **v}
                for k, v in sorted(controls.items())],
            "primer_requests": run["primer_requests"], "warmup_requests": run["warmup_requests"],
            "max_sampled_gpu_memory_mib": run["max_sampled_gpu_memory_mib"],
            "started_utc": run["queue"]["started_utc"], "finished_utc": run["queue"]["finished_utc"]}


def analyze_campaign(root):
    runs = {phase: validate_run(root / phase) for phase in PHASES}
    first, second = (runs[p] for p in PHASES)
    for key in ("source_commit", "source_sha256", "fixture_sha256", "model_revision", "model_config_sha256"):
        require(first["queue"][key] == second["queue"][key], f"Replication mismatch: {key}")
    gpu_identity = [tuple(x.strip() for x in r["queue"]["gpu_before"]["stdout"].splitlines()[1].split(",")[:3])
                    for r in (first, second)]
    require(gpu_identity[0] == gpu_identity[1], "Replication GPU/driver mismatch")
    for name in ("corpus_token_ids.json", "server_packages.json"):
        require((root / PHASES[0] / name).read_bytes() == (root / PHASES[1] / name).read_bytes(),
                f"Replication mismatch: {name}")
    require(digest(root / "model_config.json") == first["queue"]["model_config_sha256"],
            "Archived model configuration mismatch")
    require(digest(root / "specdec_longtext.txt") == first["queue"]["fixture_sha256"],
            "Archived public fixture mismatch")
    fields = ("config", "block", "length", "concurrency", "arm")
    indexed = [{tuple(g[k] for k in fields): g for g in r["groups"]} for r in (first, second)]
    require(indexed[0].keys() == indexed[1].keys(), "Replication groups differ")
    shifts = collections.defaultdict(list)
    for key, a in indexed[0].items():
        b = indexed[1][key]
        shifts[key[0], key[2], key[3], key[4]].append({m: b[m] / a[m] for m in METRICS})
    temporal = [{"config": k[0], "length": k[1], "concurrency": k[2], "arm": k[3],
                 "paired_blocks": len(rs), **{m: span(r[m] for r in rs) for m in METRICS}}
                for k, rs in sorted(shifts.items())]
    equality = collections.defaultdict(lambda: {"pairs": 0, "identical_sequences": 0,
                                               "matching_positions": 0, "positions": 0})
    require(first["requests"].keys() == second["requests"].keys(), "Replication requests differ")
    for key, a in first["requests"].items():
        b = second["requests"][key]
        require(a["payload"] == b["payload"], "Replication payloads differ")
        x, y = a["predicted_token_ids"], b["predicted_token_ids"]
        e = equality[key[0]]
        e["pairs"] += 1
        e["identical_sequences"] += int(x == y)
        e["matching_positions"] += sum(i == j for i, j in zip(x, y))
        e["positions"] += len(x)
    return {"scope": "same-host, same-input temporal replication; no deployment FPR/TPR",
            "window_note": "Exploratory after pilot inspection: indices 0..32 (32 intervals), "
                           "32..127 (95 intervals), based on SSE token arrival timestamps; not GPU step timing.",
            "source_commit": first["queue"]["source_commit"],
            "phases": {p: summarize_phase(runs[p]) for p in PHASES},
            "repeat_over_pilot": temporal, "repeat_token_agreement": dict(equality)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_campaign(args.root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"validated_phases": list(result["phases"]),
                      "repeat_token_agreement": result["repeat_token_agreement"]}, indent=2))


if __name__ == "__main__":
    main()
