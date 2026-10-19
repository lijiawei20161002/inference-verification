"""Copy completed serving evidence into a new archive without altering local runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from experiments.analyze_vllm_serving_campaign import PHASES, analyze_campaign
from experiments.exp_vllm_serving import DEFAULT_MODEL, ROOT


def archive(source, output):
    # Validate completion before creating the destination; preserve all attempts.
    for phase in PHASES:
        queue = json.loads((source / phase / "queue.json").read_text())
        if queue["status"] != "complete":
            raise ValueError(f"Unfinished phase: {phase}")
    output.mkdir(parents=True, exist_ok=False)
    for name in (*PHASES, "preflight_01", "preflight_02", "preflight_03"):
        shutil.copytree(source / name, output / name)
        shutil.copy2(source / f"{name}.log", output / f"{name}.log")
    shutil.copy2(source / "install.log", output / "install.log")
    shutil.copy2(DEFAULT_MODEL / "config.json", output / "model_config.json")
    shutil.copy2(ROOT / "experiments/data/specdec_longtext.txt", output / "specdec_longtext.txt")
    measurement_commit = json.loads((output / "pilot/queue.json").read_text())["source_commit"]
    for name in ("docs/VLLM_SERVING_EXPERIMENT.md", "scripts/run_research_job.py"):
        target = output / "measurement_source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.check_output(["git", "show", f"{measurement_commit}:{name}"], cwd=ROOT))
    for name in ("analyze_vllm_serving_campaign.py", "archive_vllm_serving.py",
                 "plot_vllm_serving.py", "render_vllm_serving_report.py"):
        target = output / "analysis_source/experiments" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "experiments" / name, target)
    summary = analyze_campaign(output)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"archive": str(output), "phases": list(summary["phases"])}))


def checksum_archive(root):
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}"
             for p in sorted(root.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive(args.source, args.output)
    checksum_archive(args.output)


if __name__ == "__main__":
    main()
