"""Render the serving report's numerical tables from the validated raw-data summary."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics


def render(summary):
    phases = summary["phases"]

    def cell(phase, config, length, concurrency, arm):
        return next(r for r in phases[phase]["cells"]
                    if (r["config"], r["length"], r["concurrency"], r["arm"])
                    == (config, length, concurrency, arm))

    def contrast(phase, config, length, concurrency):
        return next(r for r in phases[phase]["contrasts"]
                    if (r["config"], r["length"], r["concurrency"]) == (config, length, concurrency))

    def ratio(value):
        return f"{value['median']:.3f} [{value['min']:.3f}, {value['max']:.3f}]"

    def median(phase, config, length, concurrency, arm, metric):
        return cell(phase, config, length, concurrency, arm)[metric]["median"]

    gap = (datetime.fromisoformat(phases['delayed_repeat']['started_utc'].replace('Z', '+00:00'))
           - datetime.fromisoformat(phases['pilot']['finished_utc'].replace('Z', '+00:00'))).total_seconds() / 60
    p1 = contrast('pilot', 'graph_off', 16384, 1)['post_first_ms_per_token']['median']
    p4 = contrast('pilot', 'graph_off', 16384, 4)['post_first_ms_per_token']['median']
    peak = max(p['max_sampled_gpu_memory_mib'] for p in phases.values())
    lines = [
        "# vLLM serving controls and delayed replication on L40S",
        "", "19 October 2026. Follow-up to the [local performance probes](performance_probes_l40s_20261019.md).",
        "", "The context-length timing signal survives optimized vLLM serving on this host, but its size depends on execution mode, concurrency, and the part of the stream being measured. "
        f"In the first pilot, the 16K full/truncated post-first-content ratio is {p1:.2f}× at client concurrency 1 and {p4:.2f}× at concurrency 4. "
        "The tables below retain the delayed repeat separately. Prefix caching changes first-content latency and the early concurrent stream, so neither is a direct measure of retained-context decode cost.",
        "", "These are descriptive mechanism results. No classifier, detection threshold, deployment false-positive rate, or attack-power estimate was fitted.",
        "", "## Protocol and provenance", "",
        "- Qwen3-1.7B, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, BF16 weights and KV; one NVIDIA L40S, driver 595.91.07.",
        "- Pinned vLLM 0.22.0, Transformers 4.57.6, PyTorch 2.11.0+cu130, Python 3.12. Exact package lists and model configuration accompany both runs.",
        "- FLASH_ATTN confirmed in all server logs. Optimized runs confirm FULL_AND_PIECEWISE CUDA Graphs; eager disables both torch.compile and CUDA Graphs. This does not isolate graphs alone.",
        "- Configuration names: `graph_off` and `graph_on` both use compiler/graph optimizations, with prefix caching off/on respectively. `eager_off` uses eager execution with prefix caching off.",
        "- 12 GiB KV budget, 16-token blocks, 2,048-token scheduler budget, maximum eight active sequences. Chunked prefill enabled; one server process group at a time under the host lock.",
        "- Loopback streaming completions with token-ID prompts, greedy generation, 128 output tokens, and ignore-EOS. Each event may carry multiple tokens. Client clocks start after sending the request; connections and prefix priming are excluded.",
        "- Truncation keeps the last 2,048 input tokens and restarts positions at zero. The wire prompt is unchanged. L≤2,048 is a no-op control.",
        "- Each phase: nine fresh server runs, six paired text-offset blocks, 240 groups, 600 measured requests and 76,800 returned tokens; 60 priming and 45 warm-up requests excluded from measured summaries.",
        f"- The delayed repeat began {gap:.2f} minutes after the pilot finished. It reuses the exact prompts, seeds, configuration rotation, and within-run order. It tests temporal repeatability, not generalization to new prompts or deployments.",
        "- Lightweight CPU-side analysis and documentation work continued on the host during the repeat; no additional GPU workload was launched. Host-load variation is part of this temporal comparison, not an independently controlled factor.",
        f"- Both measurement phases used clean source commit `{summary['source_commit']}`. Analysis and archival code was added afterward. Maximum **sampled** GPU memory across group-boundary snapshots was {peak:,} MiB; this is not an instantaneous peak measurement.",
        "", "| Phase | Start UTC | Finish UTC |", "|---|---|---|",
    ]
    for phase, data in phases.items():
        lines.append(f"| {phase} | {data['started_utc']} | {data['finished_utc']} |")
    lines += [
        "", "The [frozen measurement protocol](vllm_serving_l40s_20261019/measurement_source/docs/VLLM_SERVING_EXPERIMENT.md) and per-run plans, commands, environment overrides, package versions, raw SSE frames, GPU snapshots, logs, metrics, and source hashes are archived. "
        "The earlier local study used a different software stack, position policy, and frozen continuation; absolute timings across studies are not a controlled engine-only comparison.",
        "", "## Context response", "",
        "Rates below are medians of group medians. A group contains one or four concurrent requests. Ratios are computed within each matched text block first; brackets show the observed minimum and maximum of six block ratios, not confidence intervals. C is client concurrency, not the internal scheduler batch size.",
        "", "| Mode | L | C | Pilot full / truncated ms per token | Pilot ratio [range] | Repeat ratio [range] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for config in ('graph_off', 'eager_off'):
        for length in (8192, 16384):
            for concurrency in (1, 4):
                a = median('pilot', config, length, concurrency, 'full', 'post_first_ms_per_token')
                b = median('pilot', config, length, concurrency, 'truncate', 'post_first_ms_per_token')
                x, y = [contrast(p, config, length, concurrency)['post_first_ms_per_token'] for p in phases]
                lines.append(f"| {config} | {length} | {concurrency} | {a:.3f} / {b:.3f} | {ratio(x)} | {ratio(y)} |")
    lines += [
        "", "Optimized serving exposes a context response. In eager mode, the concurrency-1 full/truncated ratios remain close to one even at long context. This is a negative result for interpreting a small decode-time difference as evidence that the provider retained less context.",
        "", "![Context response](vllm_serving_l40s_20261019/figures/context_response.png)",
        "", "## Prefix reuse and stream position", "",
        "All off/cold requests return zero cached tokens. Every hit returns exactly L−128 cached tokens: 1,920 at L=2,048 and 8,064 at L=8,192. Priming requests are saved separately. A successful reset response alone is not used as proof of eviction.",
        "", "| L | C | Pilot cold / hit TTFT ms | Pilot TTFT ratio [range] | Repeat TTFT ratio [range] |",
        "|---|---:|---:|---:|---:|",
    ]
    for length in (2048, 8192):
        for concurrency in (1, 4):
            a = median('pilot', 'graph_on', length, concurrency, 'cold', 'ttft_ms')
            b = median('pilot', 'graph_on', length, concurrency, 'hit', 'ttft_ms')
            x, y = [contrast(p, 'graph_on', length, concurrency)['ttft_ms'] for p in phases]
            lines.append(f"| {length} | {concurrency} | {a:.2f} / {b:.2f} | {ratio(x)} | {ratio(y)} |")
    lines += [
        "", "After inspecting the first pilot, we split arrival times at token index 32: early spans indices 0–32 (32 intervals), and late spans 32–127 (95 intervals). This split is exploratory and was not part of the original frozen protocol. It is applied unchanged to the delayed repeat. Coalesced tokens retain their common event timestamp; these are receive-time diagnostics, not GPU-step measurements or a certified steady-state window.",
        "", "| 8K cache control | C | Early cold/hit ratio | Whole post-first ratio | Late cold/hit ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for phase in phases:
        for concurrency in (1, 4):
            c = contrast(phase, 'graph_on', 8192, concurrency)
            lines.append(f"| {phase} | {concurrency} | {ratio(c['early_ms_per_token'])} | {ratio(c['post_first_ms_per_token'])} | {ratio(c['late_ms_per_token'])} |")
    lines += [
        "", "The concurrency-4 cache benefit is concentrated near the beginning of the stream; the late window is much closer to parity. This is consistent with overlapping prefill/scheduling work affecting early arrivals, but no per-step scheduler trace was collected to establish the cause. The full-context/truncation response also becomes smaller in the late concurrent window:",
        "", "| 16K, C=4 | Phase | Whole full/truncated ratio | Late full/truncated ratio |",
        "|---|---|---:|---:|",
    ]
    for config in ('graph_off', 'eager_off'):
        for phase in phases:
            c = contrast(phase, config, 16384, 4)
            lines.append(f"| {config} | {phase} | {ratio(c['post_first_ms_per_token'])} | {ratio(c['late_ms_per_token'])} |")
    lines += [
        "", "![Cache arrival windows](vllm_serving_l40s_20261019/figures/cache_windows.png)",
        "", "## Temporal repeatability and output controls", "",
        "Each cell below first pairs repeat/pilot rates by block. The table then reports the median and range of those cell medians across all lengths, concurrency levels, and arms within each configuration. It does not pool concurrent requests as independent deployments.",
        "", "| Configuration | Cells | Repeat/pilot post-first ratio: median [range of cell medians] |",
        "|---|---:|---:|",
    ]
    for config in ('graph_off', 'graph_on', 'eager_off'):
        rs = [r['post_first_ms_per_token']['median'] for r in summary['repeat_over_pilot'] if r['config'] == config]
        lines.append(f"| {config} | {len(rs)} | {statistics.median(rs):.3f} [{min(rs):.3f}, {max(rs):.3f}] |")
    lines += [
        "", "Greedy output equality is measured, not assumed. Even no-op truncation and legitimate cache controls can diverge during free generation. A small numerical difference at one position can alter the remaining continuation; sequence disagreement is not itself evidence of a different model or malicious inference.",
        "", "| Phase | Control | Identical 128-token sequences / pairs | Matching token positions / positions |",
        "|---|---|---:|---:|",
    ]
    for phase, data in phases.items():
        for name, c in data['original_summary']['output_controls'].items():
            lines.append(f"| {phase} | {name} | {c['identical_sequences']}/{c['pairs']} | {c['matching_positions']}/{c['positions']} |")
    for config, c in summary['repeat_token_agreement'].items():
        lines.append(f"| repeat vs pilot | {config} | {c['identical_sequences']}/{c['pairs']} | {c['matching_positions']}/{c['positions']} |")
    lines += ["", "No-op truncation agreement separates clearly by concurrency:", "",
              "| Phase | C | Identical sequences / no-op pairs (both execution modes) |",
              "|---|---:|---:|"]
    for phase, data in phases.items():
        for concurrency in (1, 4):
            rows = [r for r in data['output_controls_by_concurrency']
                    if r['control'] == 'noop_truncation' and r['concurrency'] == concurrency]
            lines.append(f"| {phase} | {concurrency} | {sum(r['identical_sequences'] for r in rows)}/{sum(r['pairs'] for r in rows)} |")
    coalesced = ' and '.join(str(p['original_summary']['coalesced_tokens']) for p in phases.values())
    lines += [
        "", f"The pilot and repeat contain {coalesced} coalesced tokens respectively. Post-first normalization excludes every token in the first content event, not just one token. Raw frames and token IDs are retained so denominators can be recomputed.",
        "", "## Failures, limits, and next experiment", "",
        "The first preflight failed during server argument parsing. The second reached serving but failed because the prefix-cache reset route was unavailable. The third passed after correcting the arguments and enabling development routes consistently on the loopback-only servers. All three attempts and their source snapshots are retained; none contributes to the 1,200 formal measured requests.",
        "", "The repeat strengthens a same-host timing observation. It does not provide independent corpora, different GPUs, public-network queues, unknown tenant traffic, or deployment calibration. Free generation is not frozen-token replay. No DRAM traffic, L2 hit-rate, occupancy, or hardware-identity claim follows from these data.",
        "", "The next experiment should capture per-step scheduler/prefill diagnostics in a separate profiling run, then freeze an arrival-window statistic before collecting independent prompt and session blocks for calibration and evaluation. Legitimate eager execution, caching, and concurrency belong in the honest-condition envelope. Thresholds should not be fitted to this exploratory pilot and reported as deployment accuracy.",
        "", "## Offline verification and reproduction", "",
        "From the repository root, with the normal project environment (no model or GPU needed for analysis):", "",
        "```sh", "(cd docs/results/vllm_serving_l40s_20261019 && sha256sum -c SHA256SUMS)",
        "python -m experiments.analyze_vllm_serving_campaign \\",
        "  docs/results/vllm_serving_l40s_20261019 --output /tmp/vllm-serving-summary.json",
        "python -m experiments.plot_vllm_serving \\",
        "  docs/results/vllm_serving_l40s_20261019 --output /tmp/vllm-serving-figures",
        "python -m experiments.render_vllm_serving_report \\",
        "  docs/results/vllm_serving_l40s_20261019 --output /tmp/vllm-serving-report.md", "```", "",
        "The campaign validator checks all 18 expected process runs, source snapshots, file hashes, plans, exact fixture payloads, cache counts, output budgets, raw-stream summaries, auxiliary requests, effective backend/graph logs, and matched repetition inputs. The archive includes the frozen measurement helper and analysis sources, model configuration, public fixture, and dependency versions; model weights and credentials are excluded.",
        "", "To repeat GPU measurements, use a separate clean checkout at the measurement commit and the recorded pinned environment/model cache. Each destination must be absent. Run one command at a time on the idle L40S:", "",
        "```sh", f"git checkout {summary['source_commit']}",
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.exp_vllm_serving \\",
        "  --suite pilot --output runs/vllm-reproduction/pilot",
        "# After the first process exits, repeat into a new path and retain its timestamps.",
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.exp_vllm_serving \\",
        "  --suite pilot --output runs/vllm-reproduction/delayed_repeat", "```", "",
    ]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render(json.loads((args.root / 'summary.json').read_text())))


if __name__ == '__main__':
    main()
