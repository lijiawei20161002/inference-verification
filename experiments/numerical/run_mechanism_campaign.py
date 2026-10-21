"""Detached M3 pilot with fixed cases, technical repeats and saved-operand replay.

The pilot tests explicit FP32 reduction recipes on historical discovery inputs.
It does not infer undocumented native kernel internals or automatically pass M3.
"""
from __future__ import annotations

import argparse
from collections import Counter
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
from experiments.numerical.mechanism_ops import select_cases
from experiments.numerical.mechanism_diagnostics import summarize_diagnostics
from experiments.numerical.mechanism_replay import read, verify_artifacts
from experiments.numerical.run_e0_campaign import freeze_source, now, status

ROOT = Path(__file__).resolve().parents[2]
TESTS = ["tests/test_mechanism_ops.py", "tests/test_mechanism_campaign.py", "tests/test_mechanism_diagnostics.py",
         "tests/test_dispatch_validation.py", "tests/test_e0_campaign.py",
         "tests/test_frozen_logits.py", "tests/test_numerical_diagnostics.py",
         "tests/test_real_quantization.py"]


def compare_records(expected, actual, case_ids):
    for records in (expected, actual):
        ids = [r["case"]["case_id"] for r in records]
        if ids != case_ids or len(set(ids)) != len(ids):
            raise ValueError("Incomplete or reordered mechanism cases")
    if expected != actual:
        failures = [a["case"]["case_id"] for a, b in zip(expected, actual) if a != b]
        raise ValueError(f"Independent mechanism replay differs: {failures}")
    return dict(exact=True, cases=len(actual))


def launch(args):
    output, parent = args.output.resolve(), args.parent_m1.resolve()
    if output.exists():
        raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free < 20 * 1024**3:
        raise RuntimeError("Need 20 GiB free before starting the pilot")
    state, summary = read(parent / "status.json"), read(parent / "summary.json")
    if state["state"] != "complete" or summary["m1_gate"] != "passed_for_historical_evaluation_corpus":
        raise ValueError("Parent M1 gate has not passed")
    for name, sha in read(parent / "artifacts.json").items():
        if digest(parent / name) != sha:
            raise ValueError("Parent M1 artifact changed")
    pm = read(parent / "capture/manifest.json")
    if pm["status"] != "complete":
        raise ValueError("Parent capture is incomplete")
    events_path = parent / "capture/events.json"
    if digest(events_path) != pm["artifacts_sha256"]["events.json"]:
        raise ValueError("Parent event ledger changed")
    cases = select_cases(read(events_path), args.cases_per_family)
    recipe = dict(gemm_chunk=64, attention_chunk=8 if args.partition_bridge else 32,
                  accumulation="float32", a="ascending", b="descending", final_cast="captured BF16 dtype")
    if args.partition_bridge:
        recipe.update(diagnostics=True, minimum_active_partitions=3)
        corpus = {r["prompt_id"]: r for r in read(parent / "capture/corpus.json")}
        for case in cases:
            allowed_keys = len(corpus[case["prompt_id"]]["prompt_tokens"]) + case["position"]
            if case["family"] == "attention" and (allowed_keys + 7) // 8 < 3:
                raise ValueError("Selected attention case has fewer than three active partitions")
    if args.cases_per_family > 2:
        if args.repeats != 3 or args.preflight_from is None:
            raise ValueError("Full pilot requires three repeats and a completed preflight")
        pf = args.preflight_from.resolve()
        if read(pf / "status.json")["state"] != "complete":
            raise ValueError("Preflight is incomplete")
        ps = read(pf / "summary.json")
        if ps["saved_operand_replay"]["exact"] is not True or ps["cases_per_family"] != 2:
            raise ValueError("Preflight did not validate both mechanisms")
        preflight_plan = read(pf / "plan.json")
        if preflight_plan["recipe"] != recipe or preflight_plan["parent_m1"] != str(parent):
            raise ValueError("Preflight used a different recipe or parent")
        pi = read(pf / "source_inventory.json")
        for name, sha in pi.items():
            if name.endswith(".py") and digest(ROOT / name) != sha:
                raise ValueError(f"Code changed after preflight: {name}")
    output.mkdir(parents=True, exist_ok=False)
    source = freeze_source(output)
    plan = dict(schema="ivgym-m3-pilot-plan-v1", created_utc=now(), parent_m1=str(parent),
                parent_summary_sha256=digest(parent / "summary.json"),
                parent_capture_manifest_sha256=digest(parent / "capture/manifest.json"),
                cases=cases, cases_per_family=args.cases_per_family, repeats=args.repeats,
                recipe=recipe, round="partition_bridge" if args.partition_bridge else "original_pilot",
                selection="Equally spaced historical/control positions, half each, layers 0/14/27; frozen before outcomes",
                discovery_only=True, injected_noise=False,
                primary_sampling="Matched float32, legacy top-k 50 then top-p 0.95, actual saved Gumbel",
                checkpoint="Full original reference token IDs, empty KV at start of every model replay",
                intervention="One operation output row; recompute all descendants and KV from token checkpoint",
                mechanism_scope="Explicit FP32 K-partition/softmax-merge reduction order, not undocumented native kernel attribution",
                max_seconds=args.max_seconds, minimum_free_gib=12,
                preflight_from=str(args.preflight_from.resolve()) if args.preflight_from else None,
                m3_full_gate_automatic=False)
    write_json(output / "plan.json", plan)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(HF_HOME=args.hf_home, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       HF_HUB_ENABLE_HF_TRANSFER="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    command = [sys.executable, "-u", "-m", "experiments.numerical.run_mechanism_campaign",
               "--run", "--output", str(output)]
    status(output / "status.json", state="starting", stage="cpu_checks")
    with (output / "supervisor.log").open("xb") as log:
        proc = subprocess.Popen(command, cwd=source, env=environment, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    (output / "supervisor.pid").write_text(f"{proc.pid}\n")
    write_json(output / "launch.json", dict(pid=proc.pid, command=command, source=str(source),
               launched_utc=now(), separate_os_session=True, stdin="DEVNULL", log="supervisor.log"))
    print(json.dumps(dict(pid=proc.pid, output=str(output), status=str(output / "status.json"))))


def run(args):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    def stop(signum, frame):
        raise RuntimeError(f"Supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    output, started = args.output.resolve(), time.monotonic()
    plan = read(output / "plan.json")
    child, stage = None, "source_checks"

    def execute(name, command):
        nonlocal child, stage
        stage = name
        print(f"{now()} START {name}", flush=True)
        with (output / f"{name}.log").open("xb") as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     close_fds=True)
            while True:
                progress = output / name / "progress.json"
                detail = read(progress) if progress.is_file() else None
                elapsed = time.monotonic() - started
                status(output / "status.json", state="running", stage=name, supervisor_pid=os.getpid(),
                       worker_pid=child.pid, elapsed_seconds=elapsed, progress=detail,
                       log=str(output / f"{name}.log"))
                if elapsed > plan["max_seconds"]:
                    raise TimeoutError("Campaign exceeded the frozen runtime budget")
                if shutil.disk_usage(output).free < plan["minimum_free_gib"] * 1024**3:
                    raise RuntimeError("Campaign reached the free-space reserve")
                try:
                    code = child.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    continue
        child = None
        if code:
            raise RuntimeError(f"Stage {name} exited {code}; inspect {name}.log")
        print(f"{now()} DONE {name}", flush=True)

    try:
        for name, sha in read(output / "source_inventory.json").items():
            if digest(ROOT / name) != sha:
                raise ValueError("Frozen source integrity failure")
        parent = Path(plan["parent_m1"])
        if digest(parent / "capture/manifest.json") != plan["parent_capture_manifest_sha256"]:
            raise ValueError("Parent capture manifest changed")
        execute("cpu_checks", [sys.executable, "-m", "pytest", "-q", *TESTS])
        expected, comparisons = None, []
        case_ids = [r["case_id"] for r in plan["cases"]]
        for index in range(plan["repeats"] + 1):
            name = f"repeat_{index + 1}" if index < plan["repeats"] else "saved_operand_replay"
            command = [sys.executable, "-u", "-m", "experiments.numerical.mechanism_replay",
                       "--plan", str(output / "plan.json"), "--output", str(output / name)]
            if index == plan["repeats"]:
                command += ["--replay-from", str(output / "repeat_1")]
            execute(name, command)
            stage = f"verify_{name}"
            status(output / "status.json", state="running", stage=stage, supervisor_pid=os.getpid(),
                   elapsed_seconds=time.monotonic() - started)
            manifest = verify_artifacts(output / name)
            actual = read(output / name / "records.json")
            if manifest["cases"] != len(case_ids):
                raise ValueError("Worker case count differs")
            actual_hash = hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
            if actual_hash != manifest["comparison_sha256"]:
                raise ValueError("Worker comparison digest differs")
            if expected is None:
                expected = actual
            comparisons.append(dict(worker=name, **compare_records(expected, actual, case_ids)))
            write_json(output / "comparisons.json", comparisons)
            print(f"{now()} VERIFIED {name}", flush=True)
        families = {}
        for family in ("gemm", "attention"):
            rows = [r for r in expected if r["case"]["family"] == family]
            families[family] = dict(cases=len(rows), strata=dict(Counter(r["case"]["stratum"] for r in rows)),
                                    local_changed=sum(r["local_difference"]["changed"] > 0 for r in rows),
                                    logits_changed=sum(r["states"]["a"]["logits"] != r["states"]["b"]["logits"] for r in rows),
                                    primary_support_changed=sum(r["primary_support_delta"] > 0 for r in rows),
                                    primary_token_changed=sum(r["primary_token_changed"] for r in rows),
                                    any_sampling_cell_changed=sum(any(c["token_changed"] for c in r["sampling"]) for r in rows))
            if plan["recipe"].get("diagnostics"):
                families[family]["diagnostics"] = summarize_diagnostics(rows)
        summary = dict(schema="ivgym-m3-pilot-summary-v1", finished_utc=now(), discovery_only=True,
                       cases_per_family=plan["cases_per_family"], families=families,
                       independent_repeats=plan["repeats"], comparisons=comparisons[:-1],
                       saved_operand_replay=comparisons[-1],
                       m3_gate="pilot_complete_full_mechanism_gate_pending",
                       historical_attribution="not_established", native_kernel_internals="not_inferred",
                       elapsed_seconds=time.monotonic() - started,
                       scope=plan["mechanism_scope"], round=plan.get("round", "original_pilot"), peak_cuda_allocated_bytes=max(
                           read(output / c["worker"] / "manifest.json")["peak_cuda_allocated_bytes"] for c in comparisons))
        write_json(output / "summary.json", summary)
        lines = ["# M3 controlled reduction-order pilot", "", f"Finished: {summary['finished_utc']}", "",
                 f"- Cases per family: {plan['cases_per_family']}; independent repeats: {plan['repeats']}.",
                 "- Native outputs and KV states matched the parent M1 reference path.",
                 "- Capture, both shams, forward/reverse swaps and ancestor checks passed.",
                 "- Fresh saved-operand replay reproduced all recorded outputs and controls.", ""]
        lines += [f"- {family}: {json.dumps(result)}" for family, result in families.items()]
        if plan["recipe"].get("diagnostics"):
            lines += ["", "Partition/bridge follow-up: attention chunks of 8 with at least 3 active partitions per case.",
                      "FP32 partials and pre-cast outputs are retained; same-input chunk-32 controls are saved.",
                      "Full-shape native operator outputs, native sham and restore controls matched exactly.",
                      "Native-to-recipe sampling comparisons are distinct from the A/B reduction-order comparisons.",
                      "The legacy any_sampling_cell_changed field counts token changes only; diagnostic support counts are explicit."]
        lines += ["", "These are explicit diagnostic FP32 reduction recipes on historical inputs.",
                  "They are not evidence of native cuBLAS split-K or a native attention merge policy.",
                  "No historical mechanism attribution, deployment rate, fault or attack is established.",
                  "Every model replay started from the full token checkpoint with empty KV and recomputed all descendants.",
                  "The full M3 mechanism gate remains pending; zero propagated effects are valid negative results.", ""]
        (output / "report.md").write_text("\n".join(lines))
        write_json(output / "artifacts.json", {name: digest(output / name) for name in
                   ("plan.json", "source_inventory.json", "summary.json", "comparisons.json", "report.md")})
        status(output / "status.json", state="complete", stage="complete", supervisor_pid=os.getpid(),
               elapsed_seconds=time.monotonic() - started, summary=str(output / "summary.json"),
               report=str(output / "report.md"), m3_gate=summary["m3_gate"])
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        status(output / "status.json", state="failed", stage=stage, supervisor_pid=os.getpid(),
               elapsed_seconds=time.monotonic() - started, error=f"{type(exc).__name__}: {exc}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-m1", type=Path)
    parser.add_argument("--hf-home", default="/workspace/.cache/huggingface")
    parser.add_argument("--cases-per-family", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--preflight-from", type=Path)
    parser.add_argument("--max-seconds", type=int, default=14400)
    parser.add_argument("--partition-bridge", action="store_true",
                        help="Require attention coverage, save FP32 partials, and validate full-shape native bridges")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--detach", action="store_true")
    modes.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.detach and (args.parent_m1 is None or args.repeats not in (2, 3) or args.max_seconds < 60):
        parser.error("Launch needs --parent-m1, 2 or 3 repeats, and at least a 60-second budget")
    launch(args) if args.detach else run(args)


if __name__ == "__main__":
    main()
