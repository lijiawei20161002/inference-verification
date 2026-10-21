"""Detached M1 follow-up: selected-forward dispatch at actual event shapes.

Six fresh processes run the complete requested trajectories: unprofiled baseline
and capture, then two independent pairs of profiled baseline and capture.
Raw results and failures stay in a new directory. No M3 intervention is run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from experiments.analyze_real_quantization import digest, write_json
from experiments.numerical.dispatch_validation import (
    compare_dispatch, compare_parent, dispatch_signature, expected_calls,
)
from experiments.numerical.run_e0_campaign import freeze_source, now, status, verify_worker

ROOT = Path(__file__).resolve().parents[2]
WORKERS = (
    ("baseline", "baseline", False), ("capture", "capture", False),
    ("profile_baseline_1", "baseline", True), ("profile_capture_1", "capture", True),
    ("profile_baseline_2", "baseline", True), ("profile_capture_2", "capture", True),
)
FOCUSED_TESTS = ["tests/test_dispatch_validation.py", "tests/test_e0_campaign.py",
                 "tests/test_frozen_logits.py", "tests/test_numerical_diagnostics.py",
                 "tests/test_real_quantization.py"]


def launch(args):
    output, parent = args.output.resolve(), args.parent_e0.resolve()
    parent_state = json.loads((parent / "status.json").read_text())
    if parent_state["state"] != "complete":
        raise ValueError("Parent E0 is incomplete")
    if output.exists():
        raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free < 20 * 1024**3:
        raise RuntimeError("Need 20 GiB free for captures and selected-forward traces")
    output.mkdir(parents=True, exist_ok=False)
    source = freeze_source(output)
    plan = dict(schema="ivgym-m1-dispatch-plan-v1", parent_e0=str(parent),
                parent_summary_sha256=digest(parent / "summary.json"), limit=args.limit,
                tokens=128, repeats=2, expected_full_evaluation_sequences=64,
                workers=[w[0] for w in WORKERS],
                profiling_scope="selected provider forwards plus complete reference forward",
                m3_interventions=False)
    write_json(output / "plan.json", plan)
    environment = dict(os.environ)
    environment.update(HF_HOME=args.hf_home, HF_HUB_OFFLINE="1", HF_HUB_ENABLE_HF_TRANSFER="0",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       PYTHONUNBUFFERED="1")
    command = [sys.executable, "-u", "-m", "experiments.numerical.run_dispatch_campaign",
               "--run", "--output", str(output), "--parent-e0", str(parent),
               "--hf-home", args.hf_home, "--limit", str(args.limit)]
    status(output / "status.json", state="starting", stage="cpu_checks")
    with (output / "supervisor.log").open("xb") as log:
        proc = subprocess.Popen(command, cwd=source, env=environment, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, close_fds=True)
    write_json(output / "launch.json", dict(pid=proc.pid, command=command, source=str(source),
               launched_utc=now(), separate_os_session=True, stdin="DEVNULL", log="supervisor.log"))
    (output / "supervisor.pid").write_text(f"{proc.pid}\n")
    print(json.dumps(dict(pid=proc.pid, output=str(output), status=str(output / "status.json"))))


def compare_capture(reference, candidate):
    """Check actual saved RNG, supports and state, beyond full-pool output hashes."""
    def state_hashes(manifest):
        return {name: sha for name, sha in manifest["artifacts_sha256"].items()
                if name.endswith("_state.npz") or "/event_" in name and name.endswith(".npz")}
    expected, actual = state_hashes(reference), state_hashes(candidate)
    if not expected or expected != actual:
        raise ValueError("Captured event/KV state differs under profiling")
    return dict(exact=True, event_and_state_artifacts=len(expected))


def full_gate(plan, sequence_count):
    return plan["limit"] == plan["expected_full_evaluation_sequences"] == sequence_count


def run(args):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    output = args.output.resolve()
    plan = json.loads((output / "plan.json").read_text())
    parent = Path(plan["parent_e0"])
    started, child, stage = time.monotonic(), None, "cpu_checks"
    checked, signatures, comparisons, captures, dispatch = {}, {}, [], [], []

    def execute(name, command):
        nonlocal stage, child
        stage = name
        log = output / f"{name}.log"
        status(output / "status.json", state="running", stage=stage,
               supervisor_pid=os.getpid(), log=str(log), elapsed_seconds=time.monotonic() - started)
        print(f"{now()} START {name}", flush=True)
        with log.open("xb") as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, close_fds=True)
            status(output / "status.json", state="running", stage=stage,
                   supervisor_pid=os.getpid(), worker_pid=child.pid, log=str(log),
                   elapsed_seconds=time.monotonic() - started)
            code = child.wait()
        child = None
        if code:
            raise RuntimeError(f"Stage {name} exited {code}; see {log}")
        print(f"{now()} DONE {name}", flush=True)

    try:
        inventory = json.loads((output / "source_inventory.json").read_text())
        if any(digest(ROOT / name) != sha for name, sha in inventory.items()):
            raise RuntimeError("Frozen source integrity failure")
        if digest(parent / "summary.json") != plan["parent_summary_sha256"]:
            raise RuntimeError("Parent E0 summary changed")
        execute("cpu_checks", [sys.executable, "-m", "pytest", "-q", *FOCUSED_TESTS])
        parent_manifest, parent_rows = verify_worker(parent / "baseline_1")
        continuity = None
        for name, phase, profile in WORKERS:
            cmd = [sys.executable, "-u", "-m", "experiments.numerical.capture_bf16",
                   "--phase", phase, "--output", str(output / name),
                   "--limit", str(plan["limit"]), "--tokens", str(plan["tokens"])]
            if name != "baseline":
                cmd += ["--selection-from", str(output / "baseline")]
            if profile:
                cmd += ["--profile-events"]
            execute(name, cmd)
            stage = f"verify_{name}"
            status(output / "status.json", state="running", stage=stage,
                   supervisor_pid=os.getpid(), elapsed_seconds=time.monotonic() - started)
            manifest, rows = verify_worker(output / name)
            if (len(rows) != plan["limit"] or [r["prompt_id"] for r in rows] != manifest["plan"]["prompt_ids"]
                    or len({r["prompt_id"] for r in rows}) != len(rows)
                    or any(len(r["claimed_tokens"]) != plan["tokens"] for r in rows)):
                raise ValueError("Incomplete worker trajectories")
            checked[name] = manifest, rows
            if name == "baseline":
                continuity = compare_parent(parent_manifest, parent_rows, manifest, rows)
                write_json(output / "parent_comparison.json", continuity)
            else:
                bm, base_rows = checked["baseline"]
                for key in ("source_sha256", "model_assets_sha256", "environment"):
                    if manifest[key] != bm[key]:
                        raise ValueError(f"Worker {key} changed: {name}")
                comparison = dict(left="baseline", right=name, sequences=len(rows), exact=rows == base_rows)
                comparisons.append(comparison)
                write_json(output / "output_comparisons.json", comparisons)
                if not comparison["exact"]:
                    raise ValueError(f"Profiling/capture changed outputs: {name}")
            if phase == "capture" and profile:
                captures.append(dict(worker=name, **compare_capture(checked["capture"][0], manifest)))
                write_json(output / "capture_comparisons.json", captures)
            if profile:
                signatures[name] = dispatch_signature(output / name, rows, plan["tokens"])
                write_json(output / f"{name}_dispatch.json", signatures[name])
                pairs = []
                if name == "profile_capture_1":
                    pairs = [("profile_baseline_1", name)]
                elif name == "profile_baseline_2":
                    pairs = [("profile_baseline_1", name)]
                elif name == "profile_capture_2":
                    pairs = [("profile_baseline_2", name), ("profile_capture_1", name)]
                for left, right in pairs:
                    result = dict(left=left, right=right,
                                  **compare_dispatch(signatures[left], signatures[right]))
                    dispatch.append(result)
                    write_json(output / "dispatch_comparisons.json", dispatch)
                    if not result["exact"]:
                        raise ValueError(f"Kernel dispatch differs: {left} vs {right}")
            print(f"{now()} VERIFIED {name}", flush=True)
        bm, rows = checked["baseline"]
        calls = expected_calls(rows, plan["tokens"])
        gate = full_gate(plan, len(rows))
        summary = dict(schema="ivgym-m1-dispatch-summary-v1", finished_utc=now(),
                       parent_comparison=continuity, sequences=len(rows), positions=len(rows)*plan["tokens"],
                       selected_provider_calls=sum(role == "provider" for _, role, _ in calls),
                       reference_calls=sum(role == "reference" for _, role, _ in calls),
                       calls_per_profiled_worker=len(calls), profiled_workers=4,
                       historical_events=bm["summary"]["historical_events"],
                       historical_reproduced=bm["summary"]["historical_reproduced"],
                       output_comparisons=comparisons, capture_comparisons=captures,
                       dispatch_comparisons=dispatch,
                       m1_gate="passed_for_historical_evaluation_corpus" if gate else "subset_only_full_gate_pending",
                       m3_status="not_started", elapsed_seconds=time.monotonic()-started,
                       scope="Kernel names/grid/block at selected complete forwards; no mechanism attribution.",
                       discovery_only=True)
        write_json(output / "summary.json", summary)
        report = ["# M1 event-shape dispatch validation", "", f"Finished: {summary['finished_utc']}", "",
                  f"- Sequences: {len(rows)}; positions: {summary['positions']}.",
                  f"- Historical events reproduced: {summary['historical_reproduced']}/{summary['historical_events']}.",
                  f"- Selected provider forwards: {summary['selected_provider_calls']}; full reference forwards: {summary['reference_calls']}.",
                  "- Two independent profiled baseline/capture pairs matched kernel names, grid and block.",
                  "- Full trajectories matched the unprofiled controls and parent E0 outputs.",
                  "- Captured raw event arrays and KV states matched with and without profiling.",
                  f"- M1 gate: {summary['m1_gate']}.", "- M3: not started.", "",
                  "This is historical discovery data on the pinned single-GPU HF execution path.",
                  "Profiling covers complete selected forwards, after regenerating the original prefix/KV state.",
                  "It does not prove equality on untraced calls or identify a reduction mechanism, fault or attack.", ""]
        (output / "report.md").write_text("\n".join(report))
        paths = [output / name for name in ("summary.json", "report.md", "plan.json", "source_inventory.json",
                 "parent_comparison.json", "output_comparisons.json", "capture_comparisons.json", "dispatch_comparisons.json")]
        paths += list(output.glob("profile_*_dispatch.json"))
        write_json(output / "artifacts.json", {str(p.relative_to(output)): digest(p) for p in paths})
        status(output / "status.json", state="complete", stage="complete", supervisor_pid=os.getpid(),
               elapsed_seconds=time.monotonic()-started, m1_gate=summary["m1_gate"],
               summary=str(output / "summary.json"), report=str(output / "report.md"))
    except BaseException as exc:
        status(output / "status.json", state="failed", stage=stage, supervisor_pid=os.getpid(),
               worker_pid=child.pid if child else None, elapsed_seconds=time.monotonic()-started,
               error=f"{type(exc).__name__}: {exc}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-e0", type=Path, required=True)
    parser.add_argument("--hf-home", default="/workspace/.cache/huggingface")
    parser.add_argument("--limit", type=int, default=64)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--detach", action="store_true")
    modes.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 64:
        parser.error("Limit must be between 1 and 64")
    launch(args) if args.detach else run(args)


if __name__ == "__main__":
    main()
