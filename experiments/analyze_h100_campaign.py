"""Rebuild the October 16 H100 campaign summary from preserved measurements.

Standard library only; no GPU, model weights, or network required.
Run from the repository root:
    python experiments/analyze_h100_campaign.py
    python experiments/analyze_h100_campaign.py --check

The archived result JSON is never rewritten. This checks its hashes and grid
coverage, and independently recomputes the exploratory interference alarms.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/results/h100_campaign_20261016"


def build_summary():
    provenance = json.loads((DATA / "provenance.json").read_text())
    data, active = {}, []
    for run in provenance["runs"]:
        path = DATA / run["artifact"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != run["artifact_sha256"]:
            raise ValueError(f"Artifact hash mismatch: {path}")
        data[run["name"]] = json.loads(raw)
        if not run["superseded"]:
            if run["status"] != "complete" or run["returncode"] != 0:
                raise ValueError(f"Incomplete active run: {run['name']}")
            active.append(run)
    expected_tasks = {"token-1.7B", "token-4B", "clock-window", "clock-stability",
                      "clock-floor", "clock-specdec", "clock-batching", "clock-interference"}
    if {r["name"] for r in active} != expected_tasks or len(active) != 8:
        raise ValueError("Unexpected campaign task set")

    grids = {
        "clock-floor": set(itertools.product(
            ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-1.7B-NF4"],
            ["graph", "eager"], [1], [256, 1024, 4096, 8192, 16384, 32768])),
        "clock-batching": set(itertools.product(
            ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-1.7B-NF4"],
            ["graph", "eager"], [1, 2, 4, 8, 16, 32, 64], [1024, 8192])),
    }
    for name, expected in grids.items():
        cells = data[name]["cells"]
        actual = {(c["label"], c["mode"], c["B"], c["ctx"]) for c in cells}
        if actual != expected or len(cells) != len(expected):
            raise ValueError(f"Incomplete or duplicated grid: {name}")

    tokens = []
    for name in ("token-1.7B", "token-4B"):
        d = data[name]
        for detector, control in d["arms"]["clean"]["rerun_control"].items():
            if control["d_prime"] != 0 or control["frac_tokens_changed"] != 0:
                raise ValueError(f"Nonzero rerun control: {name}/{detector}")
        for shape, detectors in d["arms"]["clean"]["shapes"].items():
            for detector, row in detectors.items():
                tokens.append(dict(model=d["M"], shape=shape, detector=detector,
                                   d_prime=row["d_prime"], ci90=row["ci"],
                                   fraction_scores_changed=row["frac_tokens_changed"]))

    window = {c["ctx"]: c["mean"] for c in data["clock-window"]["cells"]}
    stability = {c["ctx"]: c["mean"] for c in data["clock-stability"]["cells"]}
    floor = {(c["label"], c["mode"], c["ctx"]): c["mean"]
             for c in data["clock-floor"]["cells"]}
    floor_rows = [dict(model=label, mode=mode, short_ms=floor[label, mode, 256],
                       long_ms=floor[label, mode, 32768],
                       difference_ms=floor[label, mode, 32768] - floor[label, mode, 256])
                  for label in ("Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-1.7B-NF4")
                  for mode in ("graph", "eager")]
    batching = [dict(model=c["label"], mode=c["mode"], batch=c["B"], context=c["ctx"],
                     mean_ms=c["mean"]) for c in data["clock-batching"]["cells"]]

    interference = data["clock-interference"]
    groups = {}
    for cell in interference["cells"]:
        key = (cell["partition"], cell["replicate"], cell["condition"])
        group = groups.setdefault(key, {})
        if cell["ctx"] in group:
            raise ValueError("Duplicated interference cell")
        group[cell["ctx"]] = mean(cell["itl_ms"])
    expected_groups = ({("calibration", i, "idle") for i in range(16)} |
                       {("test", i, condition) for i in range(16) for condition in ("idle", "load")})
    if set(groups) != expected_groups or any(set(g) != {256, 512, 2048, 8192, 16384, 32768}
                                            for g in groups.values()):
        raise ValueError("Incomplete interference design")
    # With n=16 and method='lower', the 0.5th percentile is the sample minimum.
    threshold = min(g[32768] - g[256] for (part, _, _), g in groups.items() if part == "calibration")
    if abs(threshold - interference["analysis"]["threshold_ms"]) > 1e-9:
        raise ValueError("Interference threshold does not reproduce")
    interference_rows = []
    for condition in ("idle", "load"):
        tests = [g for (part, _, cond), g in groups.items() if part == "test" and cond == condition]
        for ctx in (512, 2048, 8192, 16384, 32768):
            differences = [g[ctx] - g[256] for g in tests]
            alarms = sum(x < threshold for x in differences)
            old = interference["analysis"]["results"][condition][str(ctx)]
            if alarms != old["alarms"] or abs(mean(differences) - old["mean_difference_ms"]) > 1e-9:
                raise ValueError("Interference results do not reproduce")
            interference_rows.append(dict(condition=condition, context=ctx, alarms=alarms,
                                          cycles=len(tests), mean_difference_ms=mean(differences)))
    if not interference["load_worker_stopped"] or any(
        (b["load_iterations"] <= 0 if b["condition"] == "load" else b["load_iterations"] != 0)
        for b in interference["blocks"]
    ):
        raise ValueError("Interference load evidence is incomplete")

    spec = {k: v for k, v in data["clock-specdec"]["specdec"].items() if k != "gaps_ms"}
    spec["total_emitted_tokens"] = sum((i + 1) * n for i, n in enumerate(spec["emitted_hist"]))
    if abs(1 - spec["blocks"] / spec["total_emitted_tokens"] - spec["frac_gaps_zero"]) > 1e-12:
        raise ValueError("Constructed speculative gaps do not reproduce")
    return {
        "campaign_status": provenance["campaign_status"],
        "finished_utc": provenance["finished_utc"], "completed_tasks": len(active),
        "verified_artifacts": len(provenance["runs"]),
        "successful_task_wall_seconds_sum": sum(r["elapsed_seconds"] for r in active),
        "tasks": [{k: r[k] for k in ("name", "status", "commit", "elapsed_seconds", "validation", "artifact")}
                  for r in active],
        "token_shape": tokens,
        "window": [dict(context=c, mean_ms=v, difference_from_256_ms=v-window[256])
                   for c, v in sorted(window.items())],
        "stability": {"full_context_gap_ms": stability[32768] - stability[256],
                      "max_absolute_mean_change_ms_at_shared_contexts":
                          max(abs(window[c] - stability[c]) for c in window.keys() & stability.keys())},
        "floor": floor_rows, "batching": batching,
        "interference": {"threshold_ms": threshold, "rows": interference_rows,
                         "zero_of_16_one_sided_95pct_binomial_upper": 1 - 0.05 ** (1 / 16),
                         "sixteen_of_16_one_sided_95pct_binomial_lower": 0.05 ** (1 / 16),
                         "binomial_note": "Illustrative IID bound only; repeated text/cycles do not establish deployment FPR."},
        "specdec": spec,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check the saved summary without writing")
    args = parser.parse_args()
    result = build_summary()
    output = DATA / "summary.json"
    if args.check:
        if json.loads(output.read_text()) != result:
            raise SystemExit("Saved summary differs; rebuild with this script")
        print("Verified 9 artifact hashes, 8 completed tasks, full timing grids, and recomputed interference results.")
    else:
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
