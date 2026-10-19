"""Render separate scheduling and receive-time diagnostics from verified captures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def fmt(values, digits=2):
    values = list(values)
    mid = statistics.median(values)
    return f"{mid:.{digits}f}" if len(values)==1 else f"{mid:.{digits}f} [{min(values):.{digits}f}, {max(values):.{digits}f}]"


def plots(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    colors = {"context":"#D18B23", "mixed":"#B84A62", "generation":"#287D8E", "empty":"#A0A0A0"}
    lookup = {(r["scheduler_budget"],r["arm"]):r for r in summary["runs"]
              if r["profiler"]=="nsys" and r["config"]=="graph_on" and r["block"]==6 and r["concurrency"]==4}
    fig, axes = plt.subplots(4, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    for ax, (budget, arm) in zip(axes, ((2048,"cold"),(2048,"hit"),(8192,"cold"),(8192,"hit"))):
        run = lookup[budget,arm]
        for event in run["scheduling"]["timeline"]:
            ax.broken_barh([(event["start_ms"], event["end_ms"]-event["start_ms"])],
                           (0,1), facecolors=colors[event["kind"]])
        ax.set_yticks([])
        ax.set_ylabel(f"{budget} / {arm}", rotation=0, ha="right", va="center")
        ax.set_ylim(0,1)
        ax.grid(axis="x", alpha=.15)
    axes[0].set_title("CPU iteration ranges — each capture starts at its first annotation\nNsight, 8K prompt, four concurrent requests, offset block 6", loc="left")
    axes[-1].set_xlabel("Milliseconds since first CPU iteration; not aligned to client arrival clocks")
    axes[0].legend(handles=[Patch(color=v,label=k) for k,v in colors.items()], loc="upper right", ncol=4)
    fig.savefig(output/"cpu_iterations.png", dpi=160)
    fig.savefig(output/"cpu_iterations.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(10,4),constrained_layout=True,sharey=True)
    for ax, metric, title in zip(axes, ("early_ms_per_token","late_ms_per_token"), ("Early: token indices 0–32", "Late: token indices 32–127")):
        for arm,color in (("cold","#B84A62"),("hit","#287D8E")):
            for block,marker in ((6,"o"),(7,"s")):
                points = [next(r for r in summary["runs"] if r["profiler"]=="nsys" and r["config"]=="graph_on" and
                               r["arm"]==arm and r["block"]==block and r["concurrency"]==4 and r["scheduler_budget"]==budget)
                          for budget in (2048,8192)]
                ax.plot([2048,8192], [r["arrivals"]["group_medians"][metric] for r in points],
                        marker=marker, color=color, alpha=.8, label=f"{arm}, block {block}")
        ax.set_title(title)
        ax.set_xticks([2048,8192])
        ax.set_xlabel("Scheduler token budget")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Client receive ms/token (group median)")
    axes[1].legend(frameon=False)
    fig.suptitle("Nsight captures only; two overlapping prompt-offset blocks")
    fig.savefig(output/"arrival_windows.png", dpi=160)
    fig.savefig(output/"arrival_windows.svg")
    plt.close(fig)


def render(summary, artifact_name):
    runs = summary["runs"]
    primary = [r for r in runs if r["role"]=="primary"]
    cold = {budget:[r for r in primary if r['scheduler_budget']==budget and r['arm']=='cold']
            for budget in (2048,8192)}
    finding = (
        "The cache-cold captures directly show generation and unfinished context work in the same iterations. "
        f"At budgets 2,048 and 8,192, mixed-iteration counts are respectively "
        f"**{fmt(r['scheduling']['counts'].get('mixed',0) for r in cold[2048])}** and "
        f"**{fmt(r['scheduling']['counts'].get('mixed',0) for r in cold[8192])}**. "
        f"Early receive intervals are {fmt(r['arrivals']['group_medians']['early_ms_per_token'] for r in cold[2048])} "
        f"versus {fmt(r['arrivals']['group_medians']['early_ms_per_token'] for r in cold[8192])} ms/token; "
        f"late intervals are {fmt(r['arrivals']['group_medians']['late_ms_per_token'] for r in cold[2048])} "
        f"versus {fmt(r['arrivals']['group_medians']['late_ms_per_token'] for r in cold[8192])} ms/token. "
        "These Nsight measurements support a scheduling contribution to the early-stream effect on this host, "
        "with the configuration and sampling limitations below.")
    text = ["# Scheduler-budget diagnostics on L40S", "", "19 October 2026.", "",
            finding, "",
            "This follow-up records context/generation iteration annotations and GPU launch correlations while changing the vLLM scheduler token budget from 2,048 to 8,192. It tests the scheduling explanation for the early-stream effect in the previous serving pilot. These are descriptive mechanism measurements, not a calibrated verifier.", "",
            "## Protocol and validation", "",
            f"- {summary['verified_runs']}/{summary['planned_runs']} fresh-server runs verified; {summary['requests']} measured requests and {summary['generated_tokens']:,} generated tokens. Priming and warm-up requests are excluded.",
            "- Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; BF16 weights/KV; pinned vLLM 0.22.0 and PyTorch 2.11.0+cu130; one L40S; FLASH_ATTN; 12 GiB KV cache; chunked prefill; asynchronous scheduling.",
            "- Primary grid: Nsight × two budgets × two offset blocks (6, 7) × full, truncated, cache-cold, and cache-hit arms. 8,192 input tokens, four concurrent requests, 128 output tokens per request. Truncation retains the final 2,048 tokens and restarts positions.",
            "- Loopback streaming completions use exact token-ID prompts, greedy generation, and ignore-EOS. Receive-time clocks start after sending the request. Cache-hit responses must report exactly 8,064 cached tokens; every other measured response must report zero.",
            "- Auxiliary controls: four unprofiled runs, two PyTorch runs, two single-request runs, and two eager runs. Each auxiliary cell has one offset block. Optimized mode includes compilation and CUDA Graphs; eager disables both.",
            "- `graph_off` and `graph_on` both use optimized execution; their suffix refers to prefix caching off/on. `eager_off` uses eager execution with prefix caching off.",
            "- A seeded shuffled plan and measurement-source snapshots were saved before launch. Each run holds the exclusive host lock, warms the server, captures only the measurement group, shuts down its own server, and validates before the next run.",
            "- Offline validation additionally requires the annotations to account for every uncached prompt token and every post-first generation token, with no iteration exceeding its configured scheduler token budget.",
            "- Early arrivals use indices 0–32 (32 intervals); late arrivals use 32–127 (95 intervals), fixed from the prior study before this campaign. Coalesced tokens retain the same receive timestamp. Client concurrency is not the scheduler batch size.",
            "- New blocks are overlapping windows of the same public corpus, not independent corpora. There are two group observations per primary cell; concurrent requests are not treated as independent replicates. Brackets below are observed ranges, not confidence intervals.",
            "- Lightweight CPU-side inspection and analysis occurred during the campaign; no additional GPU workload was launched. Profiled arrivals include collector overhead and remain separate from unprofiled results.", "",
            "## Primary scheduling and arrival measurements", "",
            "Cells show the median [minimum, maximum] across two block-level group summaries. Mixed iterations have both context requests (no generated token yet) and generation requests in the vLLM annotation.", "",
            "| Budget | Configuration / arm | Mixed iterations | Context iterations after generation begins | Early ms/token | Late ms/token | TTFT ms |",
            "|---:|---|---:|---:|---:|---:|---:|"]
    for budget in (2048,8192):
        for config,arm in (("graph_off","full"),("graph_off","truncate"),("graph_on","cold"),("graph_on","hit")):
            cell=[r for r in primary if r["scheduler_budget"]==budget and r["config"]==config and r["arm"]==arm]
            text.append(f"| {budget} | {config} / {arm} | {fmt(r['scheduling']['counts'].get('mixed',0) for r in cell)} | "
                        f"{fmt(r['scheduling']['context_iterations_after_generation'] for r in cell)} | " +
                        " | ".join(fmt(r['arrivals']['group_medians'][key] for r in cell)
                                   for key in ('early_ms_per_token','late_ms_per_token','ttft_ms')) + " |")
    text += ["", f"![CPU iteration ranges]({artifact_name}/figures/cpu_iterations.png)", "",
             f"![Arrival windows]({artifact_name}/figures/arrival_windows.png)", "",
             "## Paired cold/hit comparisons", "",
             "Ratios pair the same offset block within each tool and budget, then summarize across blocks. Tools are never pooled. Single-block controls describe one group per condition, not a precise overhead estimate.", "",
             "| Tool | Budget | Blocks | Early cold/hit | Late cold/hit | TTFT cold/hit |",
             "|---|---:|---:|---:|---:|---:|"]
    for tool,budget in (("nsys",2048),("nsys",8192),("none",2048),("none",8192),("torch",2048)):
        cell=[r for r in runs if r["profiler"]==tool and r["scheduler_budget"]==budget and r["config"]=="graph_on" and r["concurrency"]==4]
        lookup={(r['block'],r['arm']):r for r in cell}
        blocks=sorted({r['block'] for r in cell})
        text.append(f"| {tool} | {budget} | {len(blocks)} | "+" | ".join(
            fmt(lookup[b,'cold']['arrivals']['group_medians'][key]/lookup[b,'hit']['arrivals']['group_medians'][key] for b in blocks)
            for key in ('early_ms_per_token','late_ms_per_token','ttft_ms'))+" |")
    text += ["", "## Auxiliary profiler, concurrency, and execution controls", "",
             "| Tool | Budget | Configuration / arm | C | Mixed iterations | Early ms/token | Late ms/token |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for r in sorted((r for r in runs if r["role"]!="primary"),key=lambda r:r['name']):
        count=r.get('scheduling',{}).get('counts',{}).get('mixed',0) if 'scheduling' in r else 'not captured'
        text.append(f"| {r['profiler']} | {r['scheduler_budget']} | {r['config']} / {r['arm']} | {r['concurrency']} | {count} | "
                    f"{r['arrivals']['group_medians']['early_ms_per_token']:.2f} | {r['arrivals']['group_medians']['late_ms_per_token']:.2f} |")
    text += ["", "## Output agreement across scheduler budgets", "",
             "Paired Nsight requests use identical inputs at the two budgets. Output disagreement during free generation is recorded rather than treated as a model-identity failure.", "",
             "| Configuration / arm | Identical 128-token sequences / pairs | Matching token positions / total |",
             "|---|---:|---:|"]
    for config,arm in (("graph_off","full"),("graph_off","truncate"),("graph_on","cold"),("graph_on","hit")):
        lookup={(r['scheduler_budget'],r['block']):r for r in primary if r['config']==config and r['arm']==arm}
        identical=matching=pairs=positions=0
        for block in (6,7):
            left=lookup[2048,block]['arrivals']['requests']
            right=lookup[8192,block]['arrivals']['requests']
            for a,b in zip(left,right):
                x,y=a['output_token_ids'],b['output_token_ids']
                identical+=x==y
                matching+=sum(i==j for i,j in zip(x,y))
                pairs+=1
                positions+=len(x)
        text.append(f"| {config} / {arm} | {identical}/{pairs} | {matching}/{positions} |")
    profiled=[r for r in runs if 'scheduling' in r]
    kernels=sum(r['scheduling']['kernel_events'] for r in profiled)
    assigned=sum(r['scheduling']['attributed_kernels'] for r in profiled)
    text += ["", "## Trace interpretation and limits", "",
             f"The traces contain {kernels:,} CUDA kernel events, of which {assigned:,} ({assigned/kernels:.2%}) are attributed through runtime launch correlation to a containing CPU iteration on the launch thread. The remaining {kernels-assigned:,} events are retained as unattributed; they are not assigned by temporal overlap.", "",
             f"All {len(runs)} servers exited with return code zero and without forced termination. "
             f"{sum(r.get('semaphore_shutdown_warning',False) for r in runs)} logs contain a Python resource-tracker semaphore warning at shutdown; the warnings are retained with the raw logs.", "",
             "The iteration timeline shows CPU ranges, not GPU duration. The machine-readable summary separately records kernel duration sums, interval unions, and first/last GPU times for every correlated iteration. CPU ranges do not synchronize asynchronous GPU work. Projected GPU annotations in PyTorch are not counted as extra iterations. Empty scheduler ranges are retained.", "",
             "The context/generation label semantics are preserved in `installed_source/vllm/v1/utils.py` and `installed_source/vllm/v1/worker/gpu_worker.py`. A context request has not yet generated a token; an extended chunked-prefill segment remains in that category.", "",
             "Client receive clocks are not aligned to the profiler clock. These data show group-level scheduling and arrival associations, not which kernel delayed a particular request. Free generation can diverge across schedules, and increasing the token budget also changes the compiler range and execution shapes. This is a serving-configuration intervention, not a scheduler-only hardware microbenchmark.", "",
             "Two overlapping text offsets, one GPU, and one campaign do not establish deployment false-positive rates, cross-hardware transfer, or independent generalization. DRAM, L2, and occupancy counters were not collected. A calibrated detector needs a separately frozen statistic and independent calibration/test inputs.", "",
             "## Reproduction", "",
             "Run from the repository root with the pinned environments and cached model, on an idle GPU, using a new output directory:", "",
             "```sh", ".venv/bin/python -u -m experiments.run_scheduling_campaign --output runs/scheduling-reproduction",
             ".venv/bin/python -m experiments.analyze_scheduling_campaign runs/scheduling-reproduction --output /tmp/scheduling-summary.json",
             "```", "",
             "Offline archive verification and figure/report regeneration:", "",
             "```sh", f"(cd docs/results/{artifact_name} && sha256sum -c SHA256SUMS)",
             f".venv/bin/python -m experiments.analyze_scheduling_campaign docs/results/{artifact_name} --output /tmp/scheduling-summary.json",
             ".venv/bin/python -m experiments.report_scheduling_campaign /tmp/scheduling-summary.json --output /tmp/scheduling-report.md --figures /tmp/scheduling-figures",
             "```", "",
             "The public archive contains the frozen plan, source snapshots, verified summaries, raw responses, logs, PyTorch traces, and sanitized Nsight SQLite exports. Environment metadata was removed from the SQLite exports; proprietary `.nsys-rep` originals remain in ignored local storage. `PUBLICATION.json` records original/published hashes and confirms unchanged kernel, runtime, and NVTX measurement tables. Future profiling/export processes receive an allowlisted environment and use `--discard-environment=true`, but CPU-only smoke tests still found credential strings in raw reports with these measures. Publication therefore always sanitizes exported metadata and keeps original binaries private. Saved measurement sources preserve the original commands.", "",
             "SHA256SUMS covers every published file other than the checksum file itself. No model weights are included.", ""]
    return "\n".join(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--figures",type=Path,required=True)
    args=parser.parse_args()
    summary=json.loads(args.summary.read_text())
    if summary['status']!='complete':
        raise ValueError('Only render a complete verified campaign')
    args.figures.mkdir(parents=True,exist_ok=True)
    plots(summary,args.figures)
    args.output.write_text(render(summary,args.figures.parent.name))


if __name__=="__main__":
    main()
