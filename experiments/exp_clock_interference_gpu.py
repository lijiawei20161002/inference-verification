"""Local timing pilot with a controlled BF16 GEMM workload on a second GPU.

Calibration uses idle-only cycles. Test cycles pair idle/load conditions in
counterbalanced order; context order is randomized and shared within each pair.
The unit of analysis is a fresh prefill/decode cycle, never individual tokens.
Repeated fixture text and the small calibration pool do not validate deployment
FPR or network behavior. Effective context length is the deviation proxy.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

CONTEXTS = [256, 512, 2048, 8192, 16384, 32768]
TARGET_FPR = 0.005


def write_json(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def schedule(calibration, tests, seed=20261016):
    rng = np.random.default_rng(seed)
    blocks = []
    for partition, count in (("calibration", calibration), ("test", tests)):
        for replicate in range(count):
            contexts = rng.permutation(CONTEXTS).tolist()
            conditions = ["idle"] if partition == "calibration" else (
                ["idle", "load"] if replicate % 2 == 0 else ["load", "idle"])
            for condition in conditions:
                blocks.append(dict(partition=partition, replicate=replicate,
                                   condition=condition, context_order=contexts))
    return blocks


def analyze(cells):
    groups = {}
    for cell in cells:
        key = (cell["partition"], cell["replicate"], cell["condition"])
        group = groups.setdefault(key, {})
        if cell["ctx"] in group:
            raise ValueError("Duplicate context in a measurement cycle")
        group[cell["ctx"]] = float(np.mean(cell["itl_ms"]))
    if any(set(g) != set(CONTEXTS) for g in groups.values()):
        raise ValueError("Incomplete measurement cycle")
    calibration = [g[32768] - g[256] for (partition, _, condition), g in groups.items()
                   if partition == "calibration" and condition == "idle"]
    if not calibration:
        raise ValueError("Independent idle calibration is required")
    threshold = float(np.quantile(calibration, TARGET_FPR, method="lower"))
    results = {}
    for condition in ("idle", "load"):
        tests = [g for (partition, _, cond), g in groups.items() if partition == "test" and cond == condition]
        if not tests:
            raise ValueError("Both test conditions are required")
        results[condition] = {}
        for ctx in CONTEXTS[1:]:
            values = [g[ctx] - g[256] for g in tests]
            alarms = sum(x < threshold for x in values)
            results[condition][str(ctx)] = dict(
                role="honest" if ctx == 32768 else "effective-context-deviation",
                difference_ms=values, mean_difference_ms=float(np.mean(values)),
                alarms=alarms, cycles=len(values), observed_alarm_rate=alarms / len(values))
    return dict(threshold_ms=threshold, target_fpr=TARGET_FPR,
                calibration_differences_ms=calibration, results=results,
                interpretation="Exploratory fixed-threshold comparison; target FPR is not established. "
                "Repeated text, few calibration cycles, local CUDA-synchronized timings; no network or joint validation.")


def gpu_snapshot():
    result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,utilization.gpu,power.draw,temperature.gpu,clocks.sm,memory.used",
                             "--format=csv"], capture_output=True, text=True, timeout=15)
    return dict(returncode=result.returncode, output=result.stdout, stderr=result.stderr)


def load_worker(directory):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(0)
    assert torch.cuda.device_count() == 1
    # A fixed compute workload; no network, model loading, or disk I/O in the loop.
    a = torch.randn((8192, 8192), dtype=torch.bfloat16, device="cuda")
    b = torch.randn_like(a)
    c = torch.empty_like(a)
    with torch.inference_mode():
        for _ in range(10):
            torch.mm(a, b, out=c)
        torch.cuda.synchronize()
        iterations, last_update = 0, 0.0
        previous = None
        while True:
            command = json.loads((directory / "command.json").read_text())
            state, generation = command["state"], command["generation"]
            if state == "load":
                for _ in range(16):
                    torch.mm(a, b, out=c)
                torch.cuda.synchronize()
                iterations += 16
            elif state == "idle":
                time.sleep(0.02)
            elif state != "stop":
                raise ValueError(f"Unknown load state: {state}")
            now = time.monotonic()
            if generation != previous or now - last_update >= 0.2 or state == "stop":
                write_json(directory / "status.json", dict(state=state, generation=generation,
                           iterations=iterations, monotonic=now, pid=os.getpid()))
                previous, last_update = generation, now
            if state == "stop":
                return


class LoadControl:
    def __init__(self, directory, process):
        self.directory, self.process, self.generation = directory, process, 0

    def set(self, state, timeout=120):
        self.generation += 1
        write_json(self.directory / "command.json", dict(state=state, generation=self.generation))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None and state != "stop":
                raise RuntimeError("Interference load process exited unexpectedly")
            path = self.directory / "status.json"
            if path.exists():
                status = json.loads(path.read_text())
                if status["generation"] == self.generation and status["state"] == state:
                    return status
            if self.process.poll() is not None:
                raise RuntimeError("Interference process exited without acknowledging command")
            time.sleep(0.05)
        raise TimeoutError(f"Load worker did not acknowledge {state}")


def main():
    from experiments import exp_clock_channel_gpu as clock
    load_gpu = os.environ["IVGYM_INTERFERENCE_GPU"]
    calibration = int(os.environ.get("IVGYM_INTERFERENCE_CALIBRATION", "16"))
    tests = int(os.environ.get("IVGYM_INTERFERENCE_TEST", "16"))
    if calibration < 1 or tests < 1:
        raise ValueError("Calibration and test counts must be positive")
    blocks = schedule(calibration, tests)
    output = clock.RES / "clock_interference.json"
    result = dict(calibration_reps=calibration, test_reps=tests, steps=clock.STEPS,
                  seed=20261016, model="Qwen/Qwen3-1.7B", mode="graph",
                  measure_gpu=os.environ["CUDA_VISIBLE_DEVICES"], load_gpu=load_gpu,
                  workload="8192x8192 BF16 GEMM, 16 operations per synchronized loop",
                  idle_condition="Load process resident and paused; no kernels on load GPU",
                  observations="Host wall clock plus CUDA synchronization; effective-context microbenchmark",
                  cells=[], blocks=[], load_worker_stopped=False)
    start = time.monotonic()
    model, cfg, weights = clock.load(result["model"])
    # Warm both GPU and tokenizer outside the measured cycles.
    warmup = clock.itl_samples(model, cfg, 1, 256, 8, 1, "graph")
    if warmup is None:
        raise RuntimeError("Measurement GPU warmup failed")
    with tempfile.TemporaryDirectory(prefix="ivgym-interference-") as name:
        directory = Path(name)
        write_json(directory / "command.json", dict(state="idle", generation=0))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=load_gpu)
        with (clock.RES / "clock_interference_load.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-m", "experiments.exp_clock_interference_gpu",
                                        "--load-control", str(directory)], env=env, stdout=log, stderr=subprocess.STDOUT)
            control = LoadControl(directory, process)
            try:
                control.set("idle")
                for index, block in enumerate(blocks):
                    before = control.set(block["condition"])
                    # Stabilize each condition without timing startup or transitions.
                    time.sleep(1)
                    record = dict(block, started_monotonic=time.monotonic(), gpu_before=gpu_snapshot())
                    clock.log(f"  block {index + 1}/{len(blocks)}: {block['partition']} "
                              f"rep={block['replicate']} {block['condition']} order={block['context_order']}")
                    for ctx in block["context_order"]:
                        if process.poll() is not None:
                            raise RuntimeError("Load worker stopped during measurement")
                        measured = clock.cell(model, cfg, weights, 1, ctx, "graph", block["condition"])
                        if measured is None:
                            raise RuntimeError(f"Missing interference cell: {block}, ctx={ctx}")
                        measured.update({k: block[k] for k in ("partition", "replicate", "condition")})
                        result["cells"].append(measured)
                    record["gpu_after"] = gpu_snapshot()
                    after = control.set("idle")
                    record.update(finished_monotonic=time.monotonic(),
                                  load_iterations=after["iterations"] - before["iterations"])
                    if block["condition"] == "load" and record["load_iterations"] <= 0:
                        raise RuntimeError("No load iterations during measurement block")
                    if block["condition"] == "idle" and record["load_iterations"] != 0:
                        raise RuntimeError("Load leaked into an idle measurement block")
                    result["blocks"].append(record)
                    write_json(output, result)
                control.set("stop")
                process.wait(timeout=15)
                if process.returncode != 0:
                    raise RuntimeError("Load worker failed on shutdown")
                result["load_worker_stopped"] = True
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    result["analysis"] = analyze(result["cells"])
    result["elapsed_s"] = time.monotonic() - start
    write_json(output, result)
    clock.log(f"  wrote {output}: {len(result['cells'])} cells, {result['elapsed_s']:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-control", type=Path)
    args = parser.parse_args()
    if args.load_control:
        load_worker(args.load_control)
    else:
        main()
