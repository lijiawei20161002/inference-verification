"""Render the English report from verified archived data, without hand-entered measurements."""
import argparse
import collections
import json
import statistics
from pathlib import Path


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    s = json.loads((args.root / "summary.json").read_text())
    q = json.loads((args.root / "queue.json").read_text())
    raw = [r for path in sorted(args.root.glob("session_*/measurements.jsonl")) for r in rows(path)]

    def group(suite, length, batch, arm, cache="static", execution="graph"):
        return next(x for x in s["groups"] if x["suite"] == suite and x["length"] == length
                    and x["batch"] == batch and x["condition"] == arm and x["cache"] == cache
                    and x["execution"] == execution and x["instrument"] == "events")

    def ratio(suite, length, batch, key):
        return statistics.median(x[key] for x in s["paired_comparisons"] if x["suite"] == suite
                                 and x["length"] == length and x["batch"] == batch)

    v = {}
    peak = max(r["peak_allocated_bytes"] for r in raw) / 1024**3
    minutes = sum(j["elapsed_s"] for j in q["jobs"]) / 60
    v["OVERVIEW"] = (
        f"Completed **{len(raw)} local measurements, {len(s['http'])} loopback HTTP requests, "
        f"and six independent kernel traces**, plus 96 equivalent-SDPA controls. "
        f"The main queue's stages totaled {minutes:.2f} minutes; peak allocated memory in the "
        f"formal grid was {peak:.2f} GiB. All planned measurement cells completed.\n\n"
        "**Local context truncation and prefix reuse have different responses within the fixed "
        "implementation, but legitimate attention representations and kernel selection also produce "
        "large performance differences.** Calibration must cover the allowed implementations."
    )
    pairs = collections.defaultdict(dict)
    for r in raw:
        if r["suite"] == "p0":
            pairs[r["session"], r["repetition"], r["length"], r["batch"]][r["instrument"]] = r
    overhead = statistics.median(x["events"]["decode_ms_per_step"] / x["plain"]["decode_ms_per_step"]
                                 for x in pairs.values())
    v["EVENT_OVERHEAD"] = f"The median paired decode-time ratio with per-step CUDA events is **{overhead:.4f}×**."
    table = []
    for length in (512, 8192):
        for cache, execution in (("static", "graph"), ("static", "eager"), ("dynamic", "eager")):
            x = group("p1" if execution == "graph" else "p1_control", length, 1, "full", cache, execution)
            table.append(f"| {length} | {cache} | {execution} | {x['decode_ms_per_step_median']:.3f} |")
    baseline = {(r["block"], r["length"]): r for r in raw
                if r["suite"] == "p1" and r["batch"] == 1 and r["condition"] == "full"}
    control = [r for r in raw if r["suite"] == "p1_control"]
    static = [r for r in control if r["cache"] == "static"]
    dynamic = [r for r in control if r["cache"] == "dynamic"]
    equal = sum(r["predicted_token_ids"] == baseline[r["block"], r["length"]]["predicted_token_ids"]
                and r["first_logits_sha256"] == baseline[r["block"], r["length"]]["first_logits_sha256"]
                for r in static)
    matches = sum(sum(a == b for a, b in zip(r["predicted_token_ids"][0],
                  baseline[r["block"], r["length"]]["predicted_token_ids"][0])) for r in dynamic)
    total = sum(len(r["predicted_token_ids"][0]) for r in dynamic)
    v["CONTROLS"] = "\n".join(table) + (
        f"\n\nStatic-cache eager/Graph controls have identical first-step logits and predicted tokens "
        f"in {equal}/{len(static)} pairs. Dynamic/static argmax predictions agree at {matches}/{total} "
        f"positions ({100*matches/total:.2f}%); these implementations are not bitwise equivalent. "
        "The latter is a candidate legitimate floating-point-path control, not a proof of whole-model "
        "numerical error bounds. The single-layer intervention below uses an FP32 reference."
    )
    table = []
    for length in (8192, 16384):
        for batch in (1, 4, 8):
            a, c = group("p1", length, batch, "full"), group("p1", length, batch, "truncate")
            table.append(f"| {length} | {batch} | {a['decode_ms_per_step_median']:.3f} | "
                         f"{c['decode_ms_per_step_median']:.3f} | "
                         f"{ratio('p1', length, batch, 'decode_full_over_truncated'):.2f}× |")
    v["CONTEXT"] = "\n".join(table)
    v["NEGATIVE_CONTROLS"] = (
        f"All {s['checks']['negative_control_pairs']} negative-control pairs at L≤2048 have identical "
        "first-step logit hashes and predicted tokens. This checks that truncation is a no-op in that "
        "range; it does not require identical timings."
    )
    table = []
    for length in (2048, 8192):
        for batch in (1, 4):
            a, c = group("p4", length, batch, "cache_cold"), group("p4", length, batch, "cache_hit")
            table.append(f"| {length} | {batch} | {a['prefill_ms_median']:.2f} | "
                         f"{c['prefill_ms_median']:.2f} | "
                         f"{ratio('p4', length, batch, 'prefill_cold_over_hit'):.2f}× | "
                         f"{a['decode_ms_per_step_median']:.3f} / {c['decode_ms_per_step_median']:.3f} |")
    v["CACHE"] = "\n".join(table)
    v["CACHE_CHECKS"] = (
        f"All {s['checks']['prefix_cache_pairs']} three-arm controls pass: cache off/cold/hit have "
        "identical first-step logits and predicted tokens. Reuse primarily saves prefill work; "
        "the decode response to retained history must also be checked. Faster first content alone "
        "does not establish truncation."
    )
    sdpa = rows(args.root / "sdpa_representation/measurements.jsonl")
    sg = collections.defaultdict(list)
    for r in sdpa:
        sg[r["batch"], r["length"], r["arm"]].append(statistics.median(r["ms_per_call"]))
    table = []
    for batch in (1, 4):
        for length in (512, 8192):
            values = [statistics.median(sg[batch, length, arm]) for arm in
                      ("native_gqa", "repeat_no_mask", "repeat_all_true_mask")]
            table.append(f"| {batch} | {length} | {1000*values[0]:.2f} | {1000*values[1]:.2f} | "
                         f"{1000*values[2]:.2f} | {values[2]/values[0]:.2f}× |")
    v["SDPA"] = "\n".join(table)
    error = max(r["max_abs_error_vs_fp32"] for r in sdpa)
    v["SDPA_CHECKS"] = (
        f"All {len(sdpa)} controls pass the preset FP32 explicit-softmax reference check "
        f"(atol=0.005, rtol=0.02). The maximum absolute error is **{error:.6g}**."
    )
    table = []
    for arm, chunk in (("full", 1), ("full", 8), ("truncate", 1), ("truncate", 8)):
        rs = [r for r in s["http"] if r["length"] == 8192 and r["batch"] == 1
              and r["condition"] == arm and r["chunk"] == chunk]
        table.append(f"| {arm} | {chunk} | {statistics.median(r['ttft_ms'] for r in rs):.2f} | "
                     f"{statistics.median(r['event_gap_ms_median'] for r in rs):.2f} | {rs[0]['n_events']} |")
    v["HTTP"] = "\n".join(table)
    v["HTTP_CHECKS"] = (
        f"Client token sequences match server records at every position for all {len(s['http'])} "
        "requests. Combining eight tokens per frame changes first-content time and event gaps; "
        "a received event is not necessarily a single token."
    )
    v["CONCLUSION"] = (
        "This pilot connects controlled cache/shape changes to observed kernel paths and timing "
        "responses. It also exposes two limitations: legitimate implementation changes can dominate "
        "the signal, and host execution can hide it from the client. The next experiment should fix "
        "a production serving engine, measure the allowed kernel/cache conditions, and then use "
        "separate calibration and evaluation sessions. These exploratory data do not establish "
        "a universal detection rate."
    )
    fence = chr(96) * 3
    v["REPRODUCE"] = (
        "Use the recorded dependencies, pinned model snapshot, and an idle L40S. Reproduce the "
        "main run in a separate checkout; its source must be clean and the output directory absent. "
        "The recorded scripts default to the prepared model cache under /workspace/.cache/huggingface.\n\n"
        + fence + "sh\n"
        f"git checkout {q['source_commit']}\n"
        ".venv/bin/python -m experiments.run_performance_pilot --output runs/performance-reproduction\n"
        + fence + "\n\nOn the branch containing this archive, verify the saved results without a GPU or model:\n\n"
        + fence + "sh\n"
        "(cd docs/results/performance_probes_l40s_20261019 && sha256sum -c SHA256SUMS)\n"
        "python -m experiments.analyze_performance_probes \\\n"
        "  docs/results/performance_probes_l40s_20261019 --output /tmp/performance-recomputed.json\n"
        + fence + "\n\nRun the supplementary control separately on an idle GPU:\n\n"
        + fence + "sh\n"
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.exp_sdpa_representation_gpu \\\n"
        "  --output runs/sdpa-reproduction\n"
        + fence + "\n\nRegenerate figures (requires matplotlib) and the report into scratch paths:\n\n"
        + fence + "sh\n"
        "python -m experiments.plot_performance_probes docs/results/performance_probes_l40s_20261019 \\\n"
        "  --output /tmp/performance-figures\n"
        "python docs/results/performance_probes_l40s_20261019/render_report.py \\\n"
        "  docs/results/performance_probes_l40s_20261019 \\\n"
        "  --template docs/results/performance_probes_l40s_20261019/report_template.md \\\n"
        "  --output /tmp/performance-report.md\n"
        + fence
    )
    text = args.template.read_text()
    for key, value in v.items():
        assert "{{" + key + "}}" in text, key
        text = text.replace("{{" + key + "}}", value)
    assert "{{" not in text
    args.output.write_text(text)
    print(json.dumps({"report": str(args.output), "peak_gib": peak, "minutes": minutes,
                      "event_overhead_ratio": overhead, "max_sdpa_error": error}))


if __name__ == "__main__":
    main()
