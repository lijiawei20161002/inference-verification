"""Capture a small, separate vLLM diagnostic using PyTorch Profiler or Nsight Systems.

Reuses the pinned serving pilot's requests and server settings. Profiled timings
are diagnostic only and must not be pooled with unprofiled experiment results.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import gzip
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time

from experiments.exp_vllm_serving import (
    CONFIGS, DEFAULT_MODEL, ROOT, Group, command, gpu_snapshot, local_api,
    request_group, server_command, sha, utc, write_json,
)
from scripts.run_research_job import host_lock

DEFAULT_PYTHON = Path("/workspace/.venvs/ivgym-vllm-0.22.0/bin/python")
DEFAULT_NSYS = Path("/workspace/.tools/bin/nsys")


def profiler_options(kind, trace_dir):
    if kind == "torch":
        return {"profiler": "torch", "torch_profiler_dir": str(trace_dir.resolve()),
                "torch_profiler_with_stack": False, "torch_profiler_record_shapes": False,
                "torch_profiler_with_memory": False, "torch_profiler_use_gzip": True,
                "ignore_frontend": False}
    if kind == "nsys":
        return {"profiler": "cuda"}
    raise ValueError(kind)


def nsys_command(binary, prefix, launch):
    return [str(binary), "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none",
            "--trace-fork-before-exec=true", "--cuda-graph-trace=node",
            "--capture-range=cudaProfilerApi", "--capture-range-end=repeat", "--wait=all",
            "--force-overwrite=false", "--output=" + str(prefix), *launch]


def profile_api(port, action):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    try:
        connection.request("POST", "/" + action)
        response = connection.getresponse()
        body = response.read().decode()
        if response.status != 200:
            raise RuntimeError(f"{action}: HTTP {response.status}: {body}")
    finally:
        connection.close()


def save_rows(path, rows):
    with gzip.open(path, "xt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def stop_owned_server(process, pidfile):
    """Stop the owned API server first, allowing nsys to finish its report."""
    forced = False
    if process.poll() is None:
        try:
            if pidfile.exists():
                os.kill(int(pidfile.read_text()), signal.SIGTERM)
            else:
                os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            forced = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
    return {"returncode": process.returncode, "forced_shutdown": forced}


def torch_trace_summary(root):
    traces = []
    for path in sorted((root / "traces").glob("*.pt.trace.json.gz")):
        with gzip.open(path, "rt") as handle:
            events = json.load(handle)["traceEvents"]
        categories = collections.Counter(e.get("cat", "") for e in events)
        iterations = [e for e in events if e.get("name", "").startswith("execute_context_")]
        kernels = [e for e in events if e.get("cat") == "kernel"]
        traces.append({"path": str(path.relative_to(root)), "categories": dict(categories),
                       "kernel_events": len(kernels), "iteration_ranges": len(iterations),
                       "iteration_names": dict(collections.Counter(e["name"] for e in iterations)),
                       "kernel_duration_us_sum": sum(e.get("dur", 0) for e in kernels)})
    if not traces or not any(t["kernel_events"] > 0 and t["iteration_ranges"] > 0 for t in traces):
        raise RuntimeError("No worker trace with CUDA kernels and scheduler iteration annotations")
    return {"traces": traces}


def nsys_trace_summary(root, binary):
    reports = sorted((root / "traces").glob("*.nsys-rep"))
    if not reports:
        raise RuntimeError("Nsight Systems did not produce an .nsys-rep")
    results = []
    for report in reports:
        database = report.with_suffix(".sqlite")
        with report.with_suffix(".export.log").open("w") as handle:
            subprocess.run([str(binary), "export", "--type=sqlite", "--output=" + str(database),
                            str(report)], stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=300)
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            counts = {}
            for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_RUNTIME", "NVTX_EVENTS"):
                counts[table] = connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] if table in tables else 0
            if not counts["CUPTI_ACTIVITY_KIND_KERNEL"] or not counts["NVTX_EVENTS"]:
                raise RuntimeError(f"No CUDA kernels or NVTX ranges in {database}: {counts}")
            columns = {r[1] for r in connection.execute("PRAGMA table_info(NVTX_EVENTS)")}
            if "textId" in columns and "StringIds" in tables:
                names = connection.execute(
                    "SELECT COALESCE(n.text, s.value), COUNT(*) FROM NVTX_EVENTS n "
                    "LEFT JOIN StringIds s ON n.textId = s.id "
                    "WHERE COALESCE(n.text, s.value) LIKE 'execute_context_%' GROUP BY 1"
                ).fetchall()
            else:
                names = connection.execute(
                    "SELECT text, COUNT(*) FROM NVTX_EVENTS WHERE text LIKE 'execute_context_%' GROUP BY text"
                ).fetchall()
            if not names:
                raise RuntimeError("Nsight trace lacks vLLM iteration annotations")
        with report.with_suffix(".stats.txt").open("w") as handle:
            subprocess.run([str(binary), "stats", "--report=cuda_gpu_kern_sum,nvtx_sum", str(database)],
                           stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=300)
        results.append({"report": str(report.relative_to(root)), "database": str(database.relative_to(root)),
                        "table_rows": counts, "iteration_names": dict(names)})
    return {"reports": results}


def run(args):
    if args.length < 129 or args.length + args.tokens > 16512 or args.tokens < 2:
        raise ValueError("Require length >=129, tokens >=2 and length + tokens <=16512")
    if args.concurrency < 1 or args.concurrency > 8:
        raise ValueError("Require 1..8 concurrent requests")
    if (args.config == "graph_on") != (args.arm in ("cold", "hit")):
        raise ValueError("Use cold/hit with graph_on, or full/truncate with graph_off/eager_off")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "traces").mkdir()
    group = Group(0, args.length, args.concurrency, args.arm, args.tokens)
    manifest = {"status": "preparing", "started_utc": utc(), "profiler": args.profiler,
                "config": args.config, "group": dataclasses.asdict(group),
                "scope": "profiling configuration smoke; not an unprofiled timing experiment",
                "source_commit": command(["git", "rev-parse", "HEAD"])["stdout"].strip(),
                "source_status": command(["git", "status", "--porcelain"])["stdout"],
                "gpu_before": gpu_snapshot(), "source_sha256": {}}
    for name in ("experiments/profile_vllm_serving.py", "experiments/exp_vllm_serving.py",
                 "scripts/run_research_job.py", "scripts/install_nsight_systems.py"):
        snapshot = root / "source" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((ROOT / name).read_bytes())
        manifest["source_sha256"][name] = sha(snapshot)
    write_json(root / "manifest.json", manifest)
    with host_lock(Path(tempfile.gettempdir()) / f"ivgym-research-{os.getuid()}.lock"):
        process = None
        try:
            active = command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
            if active["returncode"] or active["stdout"].strip():
                raise RuntimeError("GPU is not idle; refusing to overlap research workloads")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
            corpus = tokenizer((ROOT / "experiments/data/specdec_longtext.txt").read_text(),
                               add_special_tokens=False).input_ids
            write_json(root / "corpus_token_ids.json", corpus)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            options = profiler_options(args.profiler, root / "traces")
            cmd = server_command(args.server_python, args.model_path, port, args.config)
            cmd += ["--profiler-config", json.dumps(options)]
            pidfile = root / "server.pid"
            launch = [str(args.server_python), "-m", "experiments.profile_vllm_serving",
                      "_exec_server", str(pidfile), *cmd]
            overrides = dict(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                             HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1",
                             DO_NOT_TRACK="1", HF_HOME="/workspace/.cache/huggingface",
                             VLLM_CACHE_ROOT="/workspace/.cache/vllm-round2",
                             TORCHINDUCTOR_CACHE_DIR="/workspace/.cache/vllm-round2/inductor",
                             TRITON_CACHE_DIR="/workspace/.cache/vllm-round2/triton", MAX_JOBS="8",
                             VLLM_SERVER_DEV_MODE="1", VLLM_WORKER_MULTIPROC_METHOD="spawn")
            if args.profiler == "nsys":
                manifest["nsys_version"] = command([str(args.nsys), "--version"])
                manifest["nsys_environment"] = command([str(args.nsys), "status", "--environment"])
                launch = nsys_command(args.nsys, root / "traces/serving", launch)
            manifest.update(command=launch, server_command=cmd, environment=overrides, profiler_config=options)
            write_json(root / "manifest.json", manifest)
            with (root / "server.log").open("x") as logfile:
                process = subprocess.Popen(launch, cwd=ROOT, env=dict(os.environ, **overrides),
                                           stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
                manifest["launcher_pid"] = process.pid
                write_json(root / "manifest.json", manifest)
                deadline = time.monotonic() + 1200
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Server exited {process.returncode}; inspect server.log")
                    try:
                        local_api(port, "/health")
                        break
                    except (OSError, http.client.HTTPException, RuntimeError):
                        if time.monotonic() > deadline:
                            raise TimeoutError("Server readiness timed out")
                        time.sleep(1)
                print("Server ready; warming outside the capture", flush=True)
                warm = request_group(port, corpus, Group(99, args.length, args.concurrency, "full", 8))
                save_rows(root / "warmup.jsonl.gz", warm)
                if args.config == "graph_on":
                    local_api(port, "/reset_prefix_cache", "POST")
                if args.arm == "hit":
                    save_rows(root / "primers.jsonl.gz", request_group(port, corpus, group, prime=True))
                (root / "metrics_before.txt").write_text(local_api(port, "/metrics"))
                profile_api(port, "start_profile")
                manifest.update(status="capturing", capture_started_utc=utc())
                write_json(root / "manifest.json", manifest)
                try:
                    rows = request_group(port, corpus, group)
                    save_rows(root / "requests.jsonl.gz", rows)
                    expected_cached = group.length - 128 if group.arm == "hit" else 0
                    # Cache block alignment rounds a non-aligned prefix down.
                    expected_cached -= expected_cached % 16
                    if any(r["cached_tokens"] != expected_cached for r in rows):
                        raise RuntimeError("Unexpected cache reuse in the profiled requests")
                finally:
                    profile_api(port, "stop_profile")
                manifest["capture_finished_utc"] = utc()
                (root / "metrics_after.txt").write_text(local_api(port, "/metrics"))
                print("Capture stopped; shutting down the server and flushing reports", flush=True)
                manifest["shutdown"] = stop_owned_server(process, pidfile)
            if manifest["shutdown"]["forced_shutdown"] or manifest["shutdown"]["returncode"] != 0:
                raise RuntimeError(f"Unclean profiler shutdown: {manifest['shutdown']}")
            result = torch_trace_summary(root) if args.profiler == "torch" else nsys_trace_summary(root, args.nsys)
            manifest.update(status="complete", validation=result, measured_requests=len(rows),
                            generated_tokens=sum(len(r["predicted_token_ids"]) for r in rows))
            print(json.dumps(result, indent=2), flush=True)
        except BaseException as error:
            manifest.update(status="failed", error=repr(error))
            raise
        finally:
            if process is not None and process.poll() is None:
                manifest["shutdown"] = stop_owned_server(process, root / "server.pid")
            manifest.update(finished_utc=utc(), gpu_after=gpu_snapshot())
            manifest["hashes"] = {str(p.relative_to(root)): sha(p) for p in root.rglob("*")
                                  if p.is_file() and p != root / "manifest.json"}
            write_json(root / "manifest.json", manifest)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_exec_server":
        Path(sys.argv[2]).write_text(str(os.getpid()) + "\n")
        os.execvpe(sys.argv[3], sys.argv[3:], os.environ)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiler", choices=("torch", "nsys"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--nsys", type=Path, default=DEFAULT_NSYS)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config", choices=CONFIGS, default="graph_off")
    parser.add_argument("--arm", choices=("full", "truncate", "cold", "hit"), default="full")
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=32)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
