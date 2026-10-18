"""Run the existing real-weight protocol, preserving commands and process logs."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="1",
           OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1",
           TRANSFORMERS_OFFLINE="1", HF_HUB_ENABLE_HF_TRANSFER="0",
           PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
SETTINGS = {k: ENV[k] for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
            "HF_HUB_ENABLE_HF_TRANSFER", "PYTHONUNBUFFERED", "TOKENIZERS_PARALLELISM")}
QUEUE = {"status": "running", "settings": SETTINGS, "steps": []}


def save():
    (OUT / "queue.json").write_text(json.dumps(QUEUE, indent=2) + "\n")


def run(name, args):
    entry = {"name": name, "argv": args, "cwd": str(ROOT),
             "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
    QUEUE["steps"].append(entry)
    save()
    start = time.monotonic()
    print(f"Starting {name}", flush=True)
    with (OUT / f"{name}.log").open("x") as log:
        proc = subprocess.Popen(args, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
        entry["pid"] = proc.pid
        save()
        code = proc.wait()
    entry.update(returncode=code, elapsed_seconds=time.monotonic() - start,
                 finished_utc=datetime.now(timezone.utc).isoformat(),
                 status="complete" if code == 0 else "failed")
    save()
    print(f"Finished {name}: exit {code}, {entry['elapsed_seconds']:.1f}s", flush=True)
    if code:
        raise RuntimeError(f"{name} failed; see {OUT / (name + '.log')}")


try:
    for name, argv in (("packages.json", [sys.executable, "-m", "pip", "list", "--format=json"]),
                       ("nvidia-before.txt", ["nvidia-smi"])):
        (OUT / name).write_text(subprocess.check_output(argv, text=True, env=ENV))
    QUEUE["source_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run("pip-check", [sys.executable, "-m", "pip", "check"])
    run("tests", [sys.executable, "-m", "pytest", "tests/", "-q"])
    common = [sys.executable, "-m", "experiments.exp_real_quantization_gpu"]
    run("smoke", common + ["--calibration", "2", "--evaluation", "2", "--tokens", "16",
                            "--seed", "202610180", "--bootstrap", "40",
                            "--output", str(OUT / "smoke")])
    run("pilot", common + ["--calibration", "32", "--evaluation", "64", "--tokens", "128",
                            "--seed", "20261018", "--bootstrap", "1000",
                            "--output", str(OUT / "pilot")])
    run("reanalyze", [sys.executable, "-m", "experiments.analyze_real_quantization",
                      str(OUT / "pilot"), "--output", str(OUT / "recomputed-summary.json")])
    original = json.loads((OUT / "pilot/summary.json").read_text())
    recomputed = json.loads((OUT / "recomputed-summary.json").read_text())
    if original != recomputed:
        raise RuntimeError("Offline summary does not reproduce exactly")
    QUEUE["offline_summary_exact"] = True
    QUEUE["status"] = "complete"
except BaseException as exc:
    QUEUE.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    raise
finally:
    (OUT / "nvidia-after.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    save()
