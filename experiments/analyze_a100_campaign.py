"""Rebuild the 17 October 2026 A100 summary without a GPU or dependencies.

Run from the repository root:
    python3 experiments/analyze_a100_campaign.py
    python3 experiments/analyze_a100_campaign.py --check

Original measurements are never rewritten. Token bootstrap intervals are
preserved; the archived artifacts do not contain per-sequence token scores.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import mean

DATA = Path(__file__).resolve().parents[1] / "docs/results/a100_campaign_20261017"


def build_summary():
    provenance = json.loads((DATA / "provenance.json").read_text())
    data = {}
    for run in provenance["runs"]:
        raw = (DATA / run["artifact"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != run["artifact_sha256"]:
            raise ValueError(f"Artifact hash mismatch: {run['artifact']}")
        if run["status"] != "complete" or run["returncode"] != 0:
            raise ValueError(f"Incomplete measurement: {run['name']}")
        if run["model_refs_before"] != run["model_refs_after"] or any(
            run["model_refs_before"].get(model) != revision
            or provenance["models"].get(model) != revision
            for model, revision in run["model_lock"].items()
        ):
            raise ValueError(f"Model revision changed: {run['name']}")
        data[run["name"]] = json.loads(raw)
    for evidence in provenance["queue_evidence"]:
        if hashlib.sha256((DATA / evidence["artifact"]).read_bytes()).hexdigest() != evidence["sha256"]:
            raise ValueError(f"Queue evidence hash mismatch: {evidence['artifact']}")
    queue = json.loads((DATA / "queue-20261017-v2.json").read_text())
    active = [r for r in provenance["runs"] if r["kind"] == "baseline"]
    expected = {"clock-window", "clock-stability", "token-shape", "clock-floor"}
    if (len(active) != 4 or {r["name"] for r in active} != expected
            or len(data) != 6 or len(provenance["runs"]) != 6
            or set(data) != expected | {"clock-window-smoke", "token-shape-smoke"}
            or queue["status"] != "complete" or provenance["campaign_status"] != "complete"
            or queue["current"] is not None or queue["child_pid"] is not None
            or len(queue["completed"]) != 4
            or {r["job"] for r in queue["completed"]} != expected):
        raise ValueError("Unexpected campaign status or task set")
    for run in active:
        entry = next(e for e in queue["completed"] if e["job"] == run["job"])
        if entry["sha256"] != run["artifact_sha256"] or run["commit"] != provenance["commit"]:
            raise ValueError(f"Queue/manifest mismatch: {run['name']}")

    grids = {
        "clock-window": {(1, c, "graph") for c in [256, 512, 2048, 8192, 16384, 32768]},
        "clock-stability": {(1, c, "graph") for c in [256, 1024, 4096, 8192, 16384, 32768]},
        "clock-floor": set(itertools.product(
            ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-1.7B-NF4"],
            [1], [256, 1024, 4096, 8192, 16384, 32768], ["graph", "eager"])),
    }
    for name, grid in grids.items():
        d = data[name]
        fields = ("label", "B", "ctx", "mode") if name == "clock-floor" else ("B", "ctx", "mode")
        if len(d["cells"]) != len(grid) or {tuple(c[k] for k in fields) for c in d["cells"]} != grid:
            raise ValueError(f"Incomplete or duplicate timing grid: {name}")
        for cell in d["cells"]:
            # itl_ms is rounded to four decimal places in the experiment.
            if (len(cell["itl_ms"]) != d["steps"] * d["reps"]
                    or len(cell["rep_means"]) != d["reps"]
                    or not math.isclose(mean(cell["itl_ms"]), cell["mean"], rel_tol=0, abs_tol=0.000051)
                    or not math.isclose(mean(cell["rep_means"]), cell["mean"], rel_tol=0, abs_tol=1e-9)):
                raise ValueError(f"Timing samples do not reproduce: {name}/{cell['label']}")

    token = data["token-shape"]
    if (token["n_prompts"], token["tokens"], token["n_boot"]) != (80, 256, 400) or set(token["arms"]) != {"clean"}:
        raise ValueError("Unexpected token experiment configuration")
    clean = token["arms"]["clean"]
    if set(clean["shapes"]) != {"chunked_b4", "chunked_b1_sdpa", "sequential"}:
        raise ValueError("Incomplete token shape comparisons")
    if any(r["d_prime"] != 0 or r["frac_tokens_changed"] != 0 for r in clean["rerun_control"].values()):
        raise ValueError("Nonzero same-shape rerun control")
    window = {c["ctx"]: c["mean"] for c in data["clock-window"]["cells"]}
    stability = {c["ctx"]: c["mean"] for c in data["clock-stability"]["cells"]}
    shared = sorted(window.keys() & stability.keys())
    floor = {(c["label"], c["mode"], c["ctx"]): c["mean"] for c in data["clock-floor"]["cells"]}
    return {
        "campaign_status": provenance["campaign_status"],
        "started_utc": provenance["started_utc"], "finished_utc": provenance["finished_utc"],
        "campaign_wall_seconds": (datetime.fromisoformat(provenance["finished_utc"])
                                  - datetime.fromisoformat(provenance["started_utc"])).total_seconds(),
        "completed_baseline_tasks": len(active), "verified_measurement_artifacts": len(data),
        "successful_task_wall_seconds_sum": sum(r["elapsed_seconds"] for r in active),
        "tasks": [{k: r[k] for k in ("name", "status", "commit", "elapsed_seconds", "validation", "artifact")}
                  for r in active],
        "token_shape": [dict(shape=shape, detector=detector, d_prime=row["d_prime"],
                             ci90=row["ci"], fraction_scores_changed=row["frac_tokens_changed"])
                        for shape, detectors in clean["shapes"].items() for detector, row in detectors.items()],
        "window": [dict(context=c, mean_ms=v, difference_from_256_ms=v-window[256])
                   for c, v in sorted(window.items())],
        "stability": {"full_context_gap_ms": stability[32768]-stability[256],
                      "shared_contexts": shared,
                      "max_absolute_mean_change_ms_at_shared_contexts": max(abs(window[c]-stability[c]) for c in shared)},
        "floor": [dict(model=label, mode=mode, short_ms=floor[label, mode, 256],
                       long_ms=floor[label, mode, 32768],
                       difference_ms=floor[label, mode, 32768]-floor[label, mode, 256])
                  for label in ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-1.7B-NF4"]
                  for mode in ["graph", "eager"]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the saved summary without writing")
    args = parser.parse_args()
    summary = build_summary()
    path = DATA / "summary.json"
    if args.check:
        if json.loads(path.read_text()) != summary:
            raise SystemExit("Saved summary differs; rebuild with this script")
        print("Verified 6 result hashes, 4 queue files, 4 completed baselines, timing grids/means, and rerun controls.")
    else:
        path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(path)


if __name__ == "__main__":
    main()
