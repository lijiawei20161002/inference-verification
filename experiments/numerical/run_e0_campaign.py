"""Detached, fail-closed E0 campaign. Every worker is a fresh Python process.

--detach freezes the source and starts a new OS session with file-backed logs
and no terminal or chat dependency. It never shuts down or restarts the host.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from experiments.analyze_real_quantization import digest, write_json
from experiments.numerical.frozen_logits import replay_event

ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone.utc).isoformat()


def status(path, **values):
    tmp = path.with_suffix(".tmp")
    write_json(tmp, dict(updated_utc=now(), **values))
    tmp.replace(path)


def verify_worker(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError(f"Incomplete worker: {directory}")
    for name, sha in manifest["artifacts_sha256"].items():
        if digest(directory / name) != sha:
            raise ValueError(f"Evidence hash mismatch: {directory / name}")
    return manifest, json.loads((directory / "sequences.json").read_text())


def compare_workers(left, right):
    lm, a = verify_worker(left)
    rm, b = verify_worker(right)
    for key in ("source_sha256", "model_assets_sha256"):
        if lm[key] != rm[key]:
            raise ValueError(f"Worker {key} differs")
    for key in ("prompt_ids", "tokens", "archive_manifest_sha256"):
        if lm["plan"][key] != rm["plan"][key]:
            raise ValueError(f"Worker plan {key} differs")
    if len(a) != len(b):
        raise ValueError("Incomplete comparison")
    failures = []
    for x, y in zip(a, b):
        if x["prompt_id"] != y["prompt_id"]:
            raise ValueError("Misaligned prompts")
        if x["claimed_tokens"] != y["claimed_tokens"] or x["hashes"] != y["hashes"]:
            failures.append(x["prompt_id"])
    return dict(left=left.name, right=right.name, sequences=len(a),
                exact=not failures, failed_prompt_ids=failures)


def kernel_sequence(directory):
    traces = sorted(directory.glob("prompt_*/dispatch.json"))
    if not traces:
        raise ValueError("No dispatch traces")
    events = []
    for file in traces:
        trace = json.loads(file.read_text())
        kernels = sorted((e for e in trace["traceEvents"] if e.get("cat") == "kernel"),
                         key=lambda e: e["ts"])
        if not kernels:
            raise ValueError("Profiler did not record GPU kernels")
        events.extend([e["name"], e.get("args", {}).get("grid"),
                       e.get("args", {}).get("block")] for e in kernels)
    return events


def freeze_source(directory):
    dest = directory / "source"
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True
    ).splitlines()
    inventory = {}
    for name in names:
        p = Path(name)
        if not (p.suffix == ".py" or name in ("pyproject.toml", "planning/NEXT_EXPERIMENTS.md")
                or name.startswith("docs/results/real_quantization_l40s_20261018/pilot/")):
            continue
        target = dest / p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / p, target)
        inventory[name] = digest(target)
    write_json(directory / "source_inventory.json", inventory)
    return dest


def launch(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(output).free < 20 * 1024**3:
        raise RuntimeError("Need at least 20 GiB free for evidence and traces")
    source = freeze_source(output)
    environment = dict(os.environ)
    environment.update(HF_HOME=args.hf_home, HF_HUB_OFFLINE="1", HF_HUB_ENABLE_HF_TRANSFER="0",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       PYTHONUNBUFFERED="1")
    command = [sys.executable, "-u", "-m", "experiments.numerical.run_e0_campaign",
               "--output", str(output), "--hf-home", args.hf_home, "--run"]
    status(output / "status.json", state="starting", stage="preflight")
    with (output / "supervisor.log").open("xb") as log:
        proc = subprocess.Popen(command, cwd=source, env=environment,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, close_fds=True)
    write_json(output / "launch.json", dict(pid=proc.pid, command=command,
                source=str(source), launched_utc=now(), separate_os_session=True,
                stdin="DEVNULL", log="supervisor.log"))
    (output / "supervisor.pid").write_text(f"{proc.pid}\n")
    print(json.dumps(dict(pid=proc.pid, output=str(output), status=str(output / "status.json"))))


def analyze(output, comparisons, dispatch):
    baseline = output / "baseline_1"
    capture = output / "capture_1"
    bm, rows = verify_worker(baseline)
    cm, _ = verify_worker(capture)
    events = json.loads((capture / "events.json").read_text())
    replay_rows, by_position = [], {}
    with (output / "replay.jsonl").open("x") as stream:
        for event in events:
            result = replay_event(capture / event["evidence"])
            result.update(event)
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()
            replay_rows.append(result)
            by_position[event["prompt_id"], event["position"]] = result
    historical = []
    for row in rows:
        for item in row["historical"]:
            e = by_position[row["prompt_id"], item["position"]]
            historical.append(dict(**item, prompt_id=row["prompt_id"],
                label=e["label"] if item["reproduced"] else "non_reproduced",
                current_event_id=e["event_id"], evidence=e["evidence"]))
    summary = dict(schema="ivgym-e0-campaign-summary-v1", finished_utc=now(),
        discovery_only=True, baseline=bm["summary"], capture=cm["summary"],
        comparisons=comparisons, dispatch_control=dispatch,
        historical_labels=dict(Counter(e["label"] for e in historical)),
        historical_events=historical,
        current_capped_labels=dict(Counter(e["label"] for e in replay_rows if e["current_cap"])),
        current_capped_prompt_count=len({e["prompt_id"] for e in replay_rows if e["current_cap"]}),
        current_cap_events=[dict(event_id=e["event_id"], prompt_id=e["prompt_id"],
                                position=e["position"], label=e["label"])
                            for e in replay_rows if e["current_cap"]],
        captured_controls=sum(e["control"] for e in events), matrix_cells_per_event=30,
        m1_status="full-pool output/RNG/support equivalence checked; kernel dispatch checked only in smoke subset",
        m2_status="frozen-logit discovery matrix complete",
        m3_gate="not passed: extend dispatch validation to event shapes before mechanism attribution",
        interpretation="Residual is not a hardware fault or a causal witness. Historical nonmatches remain non-reproduced.")
    write_json(output / "summary.json", summary)
    lines = [
        "# BF16 E0 discovery campaign", "",
        f"Finished: {summary['finished_utc']}", "",
        f"- Evaluation sequences: {bm['summary']['sequences']}; positions: {bm['summary']['positions']}.",
        f"- Historical capped events reproduced: {bm['summary']['historical_reproduced']}"
        f" / {bm['summary']['historical_events']}.",
        f"- Current capped positions: {bm['summary']['capped_positions']}.",
        f"- Historical labels: {json.dumps(summary['historical_labels'])}.",
        f"- Current capped labels: {json.dumps(summary['current_capped_labels'])}.",
        "- Baseline and capture each ran twice in independent Python processes.",
        "- Full logits, hidden states, fingerprints, scores and claimed tokens matched across the controls.",
        "- Native BF16 bits, actual provider/reference random tensors, sampler masks, and KV states are retained.",
        "- The replay matrix has 3 arithmetic settings × 2 nucleus conventions × 5 sampling modes.",
        "", "The original arithmetic condition uses float32 provider filtering and float64",
        "reference filtering. Explicit float32 and float64 conditions match arithmetic",
        "on both sides, on exactly the same saved logits and Gumbel vectors.", "",
        "Kernel-name/grid/block equality was checked only on the four-token smoke",
        "sequence. Full event-shape kernel tracing is still required before M1 can",
        "be declared fully complete or any M3 mechanism attribution begins.",
        "Module input/output shapes and strides are retained for all captured states.", "",
        "This is discovery data from the historical corpus, not a fresh confirmatory",
        "evaluation. Residual disagreements are not proof of a fault or attack.", "",
        "See summary.json for every historical event; replay.jsonl for all factorial",
        "cells; capture_1/prompt_*/ for independently loadable raw evidence.", "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    inventory = {str(p.relative_to(output)): digest(p) for p in (
        output / "summary.json", output / "replay.jsonl", output / "report.md",
        output / "comparisons.json", output / "dispatch_control.json", output / "source_inventory.json")}
    write_json(output / "artifacts.json", inventory)


def run(args):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    output = args.output.resolve()
    started, comparisons = time.monotonic(), []
    stage = "cpu_checks"
    child = None

    def execute(name, command):
        nonlocal stage, child
        stage = name
        log_path = output / f"{name}.log"
        status(output / "status.json", state="running", stage=name, supervisor_pid=os.getpid(),
               log=str(log_path), elapsed_seconds=round(time.monotonic() - started, 1))
        print(f"{now()} START {name}", flush=True)
        with log_path.open("xb") as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, close_fds=True)
            status(output / "status.json", state="running", stage=name, supervisor_pid=os.getpid(),
                   worker_pid=child.pid, log=str(log_path))
            code = child.wait()
        child = None
        if code:
            raise RuntimeError(f"Stage {name} exited {code}; see {log_path}")
        print(f"{now()} DONE {name}", flush=True)

    def worker(name, phase, selection=None, smoke=False, profile=False):
        cmd = [sys.executable, "-u", "-m", "experiments.numerical.capture_bf16",
               "--phase", phase, "--output", str(output / name)]
        if selection:
            cmd += ["--selection-from", str(output / selection)]
        if smoke:
            cmd += ["--limit", "1", "--tokens", "4"]
        if profile:
            cmd += ["--profile"]
        execute(name, cmd)

    def check(left, right):
        value = compare_workers(output / left, output / right)
        comparisons.append(value)
        write_json(output / "comparisons.json", comparisons)
        if not value["exact"]:
            raise RuntimeError(f"Output equivalence failed: {left} vs {right}")

    try:
        # Prove the frozen files still match the launch snapshot.
        inventory = json.loads((output / "source_inventory.json").read_text())
        if any(digest(ROOT / name) != sha for name, sha in inventory.items()):
            raise RuntimeError("Frozen source integrity failure")
        execute("cpu_checks", [sys.executable, "-m", "pytest", "-q",
            "tests/test_frozen_logits.py", "tests/test_numerical_diagnostics.py",
            "tests/test_real_quantization.py", "tests/test_e0_campaign.py"])
        worker("smoke_baseline", "baseline", smoke=True)
        worker("smoke_capture", "capture", "smoke_baseline", smoke=True)
        check("smoke_baseline", "smoke_capture")
        worker("profile_baseline", "baseline", smoke=True, profile=True)
        worker("profile_capture", "capture", "smoke_baseline", smoke=True, profile=True)
        check("smoke_baseline", "profile_baseline")
        check("smoke_baseline", "profile_capture")
        a, b = kernel_sequence(output / "profile_baseline"), kernel_sequence(output / "profile_capture")
        dispatch = dict(scope="first evaluation prompt, four generated tokens and reference prefill",
                        baseline_kernels=len(a), capture_kernels=len(b), exact=a == b,
                        baseline_sha256=hashlib.sha256(json.dumps(a).encode()).hexdigest(),
                        capture_sha256=hashlib.sha256(json.dumps(b).encode()).hexdigest())
        write_json(output / "dispatch_control.json", dispatch)
        if not dispatch["exact"]:
            raise RuntimeError("Capture changed profiled GPU kernel sequence")
        worker("baseline_1", "baseline")
        worker("baseline_2", "baseline")
        check("baseline_1", "baseline_2")
        worker("capture_1", "capture", "baseline_1")
        check("baseline_1", "capture_1")
        worker("capture_2", "capture", "baseline_1")
        check("baseline_1", "capture_2")
        check("capture_1", "capture_2")
        status(output / "status.json", state="running", stage="frozen_logit_replay",
               supervisor_pid=os.getpid())
        stage = "frozen_logit_replay"
        analyze(output, comparisons, dispatch)
        status(output / "status.json", state="complete", stage="complete",
               supervisor_pid=os.getpid(), elapsed_seconds=time.monotonic() - started,
               summary=str(output / "summary.json"), report=str(output / "report.md"))
    except BaseException as exc:
        status(output / "status.json", state="failed", stage=stage,
               supervisor_pid=os.getpid(), error=f"{type(exc).__name__}: {exc}",
               worker_pid=child.pid if child else None, elapsed_seconds=time.monotonic() - started)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-home", default="/workspace/.cache/huggingface")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--detach", action="store_true")
    modes.add_argument("--run", action="store_true")
    args = parser.parse_args()
    launch(args) if args.detach else run(args)


if __name__ == "__main__":
    main()
