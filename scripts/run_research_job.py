"""Run an existing research experiment in an isolated, versioned source copy.

This launcher does not implement joint token/timing detection. See
docs/MULTI_MACHINE_EXPERIMENTS.md for the experiment scope and deployment plan.
Only standard-library imports are needed for --dry-run and result inspection.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
JOBS = {
    "token-shape": ("experiments.exp_benign_shape_dprime_gpu", "benign_shape_dprime.json"),
    "clock-window": ("experiments.exp_slope_verifier_gpu", "slope_verifier_window.json"),
    "clock-stability": ("experiments.exp_slope_verifier_gpu", "slope_verifier_stability.json"),
    "clock-floor": ("experiments.exp_clock_channel_gpu", "clock_channel_floor.json"),
    "clock-batching": ("experiments.exp_clock_channel_gpu", "clock_channel_batching.json"),
    "clock-specdec": ("experiments.exp_clock_channel_gpu", "clock_channel_specdec.json"),
    "clock-interference": ("experiments.exp_clock_interference_gpu", "clock_interference.json"),
}
MODELS = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B")


def settings(job, size):
    """Explicitly select arms: never run a legacy script's broad default sweep."""
    small = size == "smoke"
    if job == "token-shape":
        return dict(IVGYM_M="Qwen/Qwen3-1.7B", IVGYM_ARMS="clean",
                    IVGYM_PROMPTS="4" if small else "80",
                    IVGYM_TOKENS="16" if small else "256",
                    IVGYM_BOOT="20" if small else "400", IVGYM_FILLER_BATCH="4")
    if job == "clock-interference":
        return dict(IVGYM_CLOCK_STEPS="16" if small else "192", IVGYM_CLOCK_REPS="1",
                    IVGYM_INTERFERENCE_CALIBRATION="2" if small else "16",
                    IVGYM_INTERFERENCE_TEST="2" if small else "16")
    arm = job.removeprefix("clock-")
    family = "SLOPE" if arm in ("window", "stability") else "CLOCK"
    env = {f"IVGYM_{family}_ARMS": arm, f"IVGYM_{family}_TAG": arm,
           f"IVGYM_{family}_STEPS": "16" if small else "192",
           f"IVGYM_{family}_REPS": "1" if small else "4",
           "IVGYM_SLOPE_KEEP": "1"}
    if family == "CLOCK":
        env["IVGYM_CLOCK_MODES"] = "graph,eager"
    return env


def capture(cmd, cwd=None):
    return subprocess.check_output(cmd, cwd=cwd or ROOT, text=True).strip()


def record_command(cmd):
    try:
        p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=60)
        return dict(returncode=p.returncode, output=p.stdout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return dict(returncode=None, output=str(e))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


@contextmanager
def host_lock(path, shared=False):
    """Token jobs share the host; timing experiments require exclusive access."""
    import fcntl
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another research job holds the host lock or GPU lock: {path}") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def gpu_uuid(selector):
    """Canonicalize numeric indices and UUID aliases for per-device locking."""
    value = capture(["nvidia-smi", f"--id={selector}", "--query-gpu=uuid", "--format=csv,noheader"])
    if not re.fullmatch(r"GPU-[A-Za-z0-9-]+", value):
        raise ValueError(f"Cannot resolve GPU selector {selector}")
    return value


def validate_result(job, data):
    """Check coverage, not statistical validity or the strength of a finding."""
    if job == "token-shape":
        clean = data.get("arms", {}).get("clean", {})
        expected = {"chunked_b4", "chunked_b1_sdpa", "sequential"}
        detectors = {"token_difr", "cross_entropy", "activation_difr", "token_toploc"}
        shapes = clean.get("shapes", {})
        if not expected.issubset(shapes) or any(
            not detectors.issubset(shapes[s]) for s in expected
        ) or not detectors.issubset(clean.get("rerun_control", {})):
            raise ValueError("Missing token-shape comparisons or rerun controls")
        return dict(coverage="complete", comparisons=3)
    if job == "clock-specdec":
        if not data.get("specdec", {}).get("gaps_ms"):
            raise ValueError("Missing speculative-decoding timing artifact")
        return dict(coverage="complete", observation="device blocks with constructed gaps")
    cells = data.get("cells", [])
    if not cells or any(not c.get("itl_ms") for c in cells):
        raise ValueError("Missing timing samples")
    if any(not isinstance(x, (float, int)) or not math.isfinite(x) or x < 0
           for c in cells for x in c["itl_ms"]):
        raise ValueError("Timing samples must be finite and nonnegative")
    if job == "clock-interference":
        contexts = {256, 512, 2048, 8192, 16384, 32768}
        expected = {("calibration", i, "idle", ctx)
                    for i in range(data["calibration_reps"]) for ctx in contexts}
        expected.update(("test", i, condition, ctx) for i in range(data["test_reps"])
                        for condition in ("idle", "load") for ctx in contexts)
        measured = {(c["partition"], c["replicate"], c["condition"], c["ctx"]) for c in cells}
        if measured != expected or len(cells) != len(expected):
            raise ValueError("Incomplete interference measurement grid")
        if not data.get("load_worker_stopped") or not data.get("analysis"):
            raise ValueError("Interference worker or analysis did not finish")
        if any(b["load_iterations"] <= 0 for b in data["blocks"] if b["condition"] == "load"):
            raise ValueError("Interference load did not run during measurement")
        return dict(coverage="complete", cells=len(cells), observation="local interference pilot")
    if job in ("clock-window", "clock-stability"):
        expected = {256, 512, 2048, 8192, 16384, 32768} if job == "clock-window" else {
            256, 1024, 4096, 8192, 16384, 32768}
        measured = {c["ctx"] for c in cells}
        if measured != expected or len(cells) != 6:
            raise ValueError(f"Incomplete context grid: expected {sorted(expected)}, got {sorted(measured)}")
    expected_count = {"clock-window": 6, "clock-stability": 6,
                      "clock-floor": 48, "clock-batching": 84}[job]
    return dict(coverage="complete" if len(cells) == expected_count else "partial",
                cells=len(cells), expected_cells=expected_count)


def cached_revisions():
    """Record resolved Hub refs without copying credentials or model weights."""
    cache = Path(os.environ.get("HF_HUB_CACHE") or
                 Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")
    models = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B")
    refs = {}
    for model in models:
        p = cache / ("models--" + model.replace("/", "--")) / "refs/main"
        refs[model] = p.read_text().strip() if p.is_file() else None
    return refs


def check_model_lock(path, job, model=None):
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("Locked runs require HF_HUB_OFFLINE=1 after model preparation")
    locked = json.loads(Path(path).read_text())
    current = cached_revisions()
    required = {"Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B"}
    if job == "clock-floor":
        required.add("Qwen/Qwen3-4B")
    if job == "token-shape":
        required = {model or "Qwen/Qwen3-1.7B"}
    for model in required:
        revision = locked.get(model)
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"Missing immutable revision in model lock: {model}")
        if current[model] != revision:
            raise ValueError(f"Cached model revision differs from lock: {model}")
    return {m: locked[m] for m in sorted(required)}


def run(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.node):
        raise ValueError("--node must be a simple label, e.g. h100-sxm-01")
    if not re.fullmatch(r"[0-9]+|GPU-[A-Za-z0-9-]+", args.gpu):
        raise ValueError("--gpu must select one full GPU index or GPU UUID")
    model = getattr(args, "model", None)
    load_gpu = getattr(args, "load_gpu", None)
    if model and (args.job != "token-shape" or model not in MODELS):
        raise ValueError("--model selects a supported model for token-shape only")
    if args.job == "clock-interference":
        if not load_gpu or not re.fullmatch(r"[0-9]+|GPU-[A-Za-z0-9-]+", load_gpu):
            raise ValueError("clock-interference requires --load-gpu")
        if load_gpu == args.gpu:
            raise ValueError("Measurement and load require different GPUs")
    elif load_gpu:
        raise ValueError("--load-gpu is only for clock-interference")
    module, artifact = JOBS[args.job]
    env_settings = settings(args.job, args.size)
    if model:
        env_settings["IVGYM_M"] = model
    if load_gpu:
        env_settings["IVGYM_INTERFERENCE_GPU"] = load_gpu
    plan = dict(node=args.node, job=args.job, size=args.size, gpu=args.gpu,
                module=module, settings=env_settings, output_artifact=artifact)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if args.size == "baseline" and not args.model_lock:
        raise ValueError("Baseline runs require --model-lock; see the multi-machine guide")
    model_lock = check_model_lock(args.model_lock, args.job, model) if args.model_lock else None
    # A copied HEAD is reproducible only if it includes the user's intended changes.
    if capture(["git", "status", "--porcelain", "--untracked-files=normal"]):
        raise RuntimeError("Commit or move local changes before running; only committed HEAD is executed.")
    commit = capture(["git", "rev-parse", "HEAD"])
    with ExitStack() as locks:
        lock_root = Path(tempfile.gettempdir())
        locks.enter_context(host_lock(lock_root / f"ivgym-research-{os.getuid()}.lock",
                                      shared=args.job == "token-shape"))
        devices = {selector: gpu_uuid(selector) for selector in (args.gpu, load_gpu) if selector is not None}
        if len(set(devices.values())) != len(devices):
            raise ValueError("Measurement and load require different physical GPUs")
        for uuid in sorted(devices.values()):
            locks.enter_context(host_lock(lock_root / f"ivgym-research-{os.getuid()}-{uuid}.lock"))
        device_env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
        subprocess.run([sys.executable, "-c", "import torch; "
                        "assert torch.cuda.is_available(), 'CUDA GPU required'; "
                        "assert torch.cuda.device_count() == 1, 'Select one GPU'; "
                        "print(torch.cuda.get_device_name(0))"], env=device_env, check=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        dest = ROOT / "runs" / "campaign" / args.node / f"{args.job}-{args.size}-{stamp}"
        source, results = dest / "source", dest / "results"
        source.mkdir(parents=True)
        results.mkdir()
        archive = dest / "source.tar"
        subprocess.run(["git", "archive", "--format=tar", f"--output={archive}", commit],
                       cwd=ROOT, check=True)
        subprocess.run(["tar", "-xf", str(archive), "-C", str(source)], check=True)
        archive.unlink()
        expected = source / "docs/results" / artifact
        expected.unlink(missing_ok=True)  # This is the disposable copy, not the repository.
        env = {k: v for k, v in device_env.items() if not k.startswith("IVGYM_")}
        env.update(env_settings, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
        # Avoid an external checkout overriding modules in the archived source tree.
        env.pop("PYTHONPATH", None)
        meta = dict(**plan, commit=commit, hostname=socket.gethostname(),
                    gpu_uuids=devices, load_gpu=load_gpu,
                    host_access="shared-token" if args.job == "token-shape" else "exclusive-timing",
                    started_utc=stamp, python=sys.version, status="running",
                    observations="local GPU experiments; no joint or network validation",
                    model_lock=model_lock,
                    model_refs_before=cached_revisions(),
                    offline_models=os.environ.get("HF_HUB_OFFLINE") == "1")
        write_json(dest / "manifest.json", meta)
        for filename, command in (
            ("nvidia-before.json", ["nvidia-smi", "-q"]),
            ("packages.json", [sys.executable, "-m", "pip", "list", "--format=json"]),
        ):
            write_json(dest / filename, record_command(command))
        print(f"Run directory: {dest}\nLog: {dest / 'run.log'}", flush=True)
        start = time.monotonic()
        try:
            with (dest / "run.log").open("w") as log:
                p = subprocess.run([sys.executable, "-m", module], cwd=source, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
            meta["returncode"] = p.returncode
            if p.returncode:
                raise RuntimeError(f"Experiment exited {p.returncode}; inspect run.log")
            data = json.loads(expected.read_text())
            if args.model_lock:
                check_model_lock(args.model_lock, args.job, model)
            meta["validation"] = validate_result(args.job, data)
            target = results / artifact
            target.write_bytes(expected.read_bytes())
            meta["artifact_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
            if args.job == "clock-interference":
                load_log = results / "clock_interference_load.log"
                load_log.write_bytes((expected.parent / load_log.name).read_bytes())
                meta["auxiliary_sha256"] = {load_log.name: hashlib.sha256(load_log.read_bytes()).hexdigest()}
            meta["status"] = meta["validation"]["coverage"]
        except BaseException as e:
            meta.update(status="failed", error=f"{type(e).__name__}: {e}")
            raise
        finally:
            meta["elapsed_seconds"] = time.monotonic() - start
            meta["model_refs_after"] = cached_revisions()
            meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(dest / "nvidia-after.json", record_command(["nvidia-smi", "-q"]))
            write_json(dest / "manifest.json", meta)
        print(f"Finished: {meta['status']}. Results: {results}")
        return 0 if meta["status"] == "complete" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", required=True, help="Unique machine label")
    parser.add_argument("--gpu", default="0", help="GPU index or full GPU UUID")
    parser.add_argument("--model", choices=MODELS, help="Model for a complete independent token-shape job")
    parser.add_argument("--load-gpu", help="Separate GPU running the clock-interference workload")
    parser.add_argument("--job", choices=JOBS, required=True)
    parser.add_argument("--size", choices=("smoke", "baseline"), default="smoke")
    parser.add_argument("--model-lock", help="JSON mapping Hub model IDs to frozen revisions")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without CUDA or writes")
    args = parser.parse_args()
    try:
        return run(args)
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
