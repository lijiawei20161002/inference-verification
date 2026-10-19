"""Serial, resumable scheduler-budget diagnostics; never pool profiler timings."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import random
import subprocess
import sys

from experiments.exp_vllm_serving import ROOT, command, sha, utc, write_json
from experiments.profile_vllm_serving import check_capture
from scripts.run_research_job import host_lock


def build_plan():
    jobs = []

    def add(profiler, budget, block, config, arm, concurrency=4, role="primary"):
        name = f"{profiler}_s{budget}_b{block}_{config}_{arm}_c{concurrency}"
        jobs.append(dict(name=name, profiler=profiler, scheduler_budget=budget,
                         block=block, config=config, arm=arm, concurrency=concurrency,
                         length=8192, tokens=128, role=role))

    for block in (6, 7):
        for budget in (2048, 8192):
            for config, arm in (("graph_off", "full"), ("graph_off", "truncate"),
                                ("graph_on", "cold"), ("graph_on", "hit")):
                add("nsys", budget, block, config, arm)
    for budget in (2048, 8192):
        for arm in ("cold", "hit"):
            add("none", budget, 6, "graph_on", arm, role="unprofiled_control")
    for arm in ("cold", "hit"):
        add("torch", 2048, 6, "graph_on", arm, role="profiler_crosscheck")
        add("nsys", 2048, 6, "graph_on", arm, concurrency=1, role="concurrency_control")
    for arm in ("full", "truncate"):
        add("nsys", 2048, 6, "eager_off", arm, role="execution_control")
    random.Random(2026101903).shuffle(jobs)
    return jobs


def run(root, resume=False):
    sources = ("experiments/profile_vllm_serving.py", "experiments/exp_vllm_serving.py",
               "experiments/run_scheduling_campaign.py", "scripts/run_research_job.py",
               "scripts/install_nsight_systems.py", "experiments/data/specdec_longtext.txt")
    if not resume:
        root.mkdir(parents=True, exist_ok=False)
        for name in sources:
            target = root / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        write_json(root / "plan.json", dict(
            created_utc=utc(), source_commit=command(["git", "rev-parse", "HEAD"])["stdout"].strip(),
            source_sha256={name: sha(ROOT / name) for name in sources}, jobs=build_plan(),
            scope="Exploratory same-host scheduling mechanism study, not calibration.",
            prompt_scope="New offset blocks 6 and 7 from the existing corpus; overlapping windows, not independent corpora.",
            arrival_windows={"early": [0, 32], "late": [32, 127]},
            unit="One request group per fresh server; concurrent requests are not independent replicates.",
            controls="Primary cells have two blocks; auxiliary controls have one. Keep tools separate."))
    plan = json.loads((root / "plan.json").read_text())
    for name, expected in plan["source_sha256"].items():
        if sha(ROOT / name) != expected or sha(root / "source" / name) != expected:
            raise ValueError(f"Source changed since plan was frozen: {name}")
    state = dict(status="running", started_utc=utc(), jobs=[])
    with host_lock(root / "campaign.lock"):
        try:
            for i, job in enumerate(plan["jobs"]):
                destination = root / job["name"]
                state["current"] = job["name"]
                write_json(root / "state.json", state)
                if destination.exists():
                    if not resume:
                        raise ValueError(f"Output already exists: {destination}")
                    check_capture(destination)
                    state["jobs"].append(dict(name=job["name"], status="verified_existing"))
                    continue
                cmd = [sys.executable, "-m", "experiments.profile_vllm_serving",
                       "--output", str(destination)]
                for key in ("profiler", "scheduler_budget", "block", "config", "arm",
                            "concurrency", "length", "tokens"):
                    cmd += ["--" + key.replace("_", "-"), str(job[key])]
                print(f"[{i + 1}/{len(plan['jobs'])}] START {job['name']} {utc()}", flush=True)
                with (root / (job["name"] + ".log")).open("x") as log:
                    subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1",
                                   OPENBLAS_NUM_THREADS="1"), stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
                verification = check_capture(destination)
                state["jobs"].append(dict(name=job["name"], status="verified",
                                          requests=verification["requests"],
                                          generated_tokens=verification["generated_tokens"]))
                print(f"[{i + 1}/{len(plan['jobs'])}] VERIFIED {job['name']} {utc()}", flush=True)
            state["status"] = "complete"
        except BaseException as error:
            state.update(status="failed", error=repr(error))
            raise
        finally:
            state["updated_utc"] = utc()
            write_json(root / "state.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        print(json.dumps(build_plan(), indent=2))
    else:
        run(args.output.resolve(), args.resume)


if __name__ == "__main__":
    main()
