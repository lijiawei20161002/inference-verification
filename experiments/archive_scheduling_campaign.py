"""Archive a completed campaign and render verified, reproducible local results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from experiments.analyze_scheduling_campaign import analyze
from experiments.exp_vllm_serving import DEFAULT_MODEL, ROOT, gpu_snapshot, sha, utc, write_json
from experiments.profile_vllm_serving import DEFAULT_PYTHON
from experiments.report_scheduling_campaign import plots, render
from scripts.prepare_profiler_publication import prepare


def archive(source, output, report):
    state = json.loads((source / "state.json").read_text())
    if state["status"] != "complete":
        raise ValueError("Only archive a completed campaign")
    shutil.copytree(source, output)
    for name in ("experiments/analyze_scheduling_campaign.py", "experiments/report_scheduling_campaign.py",
                 "experiments/archive_scheduling_campaign.py", "tests/test_scheduling_analysis.py",
                 "scripts/prepare_profiler_publication.py", "tests/test_profiler_publication.py"):
        target = output / "analysis_source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    query = ("import importlib.metadata as m,json,sys; "
             "print(json.dumps({'python':sys.version,'packages':{k:m.version(k) for k in "
             "['vllm','torch','transformers','numpy','triton','safetensors']}}))")
    environment = json.loads(subprocess.check_output([str(DEFAULT_PYTHON), "-c", query], text=True))
    environment.update(collected_utc=utc(), gpu_after=gpu_snapshot())
    write_json(output / "environment.json", environment)
    for name in ("config.json", "tokenizer_config.json"):
        shutil.copyfile(DEFAULT_MODEL / name, output / name)
    # Preserve the installed implementation behind the labels used in the report.
    installed = DEFAULT_PYTHON.parent.parent / "lib" / "python3.12" / "site-packages"
    for name in ("vllm/v1/utils.py", "vllm/v1/worker/gpu_worker.py"):
        target = output / "installed_source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(installed / name, target)
    prepare(output, ROOT / "runs/private-profiler-archives" / output.name)
    summary = analyze(output)
    write_json(output / "summary.json", summary)
    figures = output / "figures"
    figures.mkdir()
    plots(summary, figures)
    report.write_text(render(summary, output.name))
    validation = dict(status=summary["status"], verified_runs=summary["verified_runs"],
                      requests=summary["requests"], generated_tokens=summary["generated_tokens"],
                      verified_utc=utc(), report_sha256=sha(report),
                      checks=["measurement and campaign source hashes", "every capture file hash",
                              "plan/configuration/output coverage", "effective execution and attention logs",
                              "raw measured, warm-up, and priming SSE responses", "exact request payloads",
                              "cache hit counts", "CPU annotations plus CUDA kernel evidence",
                              "iteration token accounting and scheduler limits",
                              "clean process shutdown", "CUDA launch correlation attribution"])
    write_json(output / "validation.json", validation)
    (output / "SHA256SUMS").write_text("".join(
        f"{sha(p)}  {p.relative_to(output)}\n" for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "SHA256SUMS"))
    print(json.dumps(validation, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    archive(args.source, args.output, args.report)


if __name__ == "__main__":
    main()
