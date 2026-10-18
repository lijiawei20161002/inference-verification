"""Archive a completed campaign and derive its report from measured artifacts."""
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path.cwd()
RUN = Path(__file__).resolve().parent
NAME = "real_quantization_l40s_20261018"
DEST = ROOT / "docs/results" / NAME


def read(path):
    return json.loads(path.read_text())


queue = read(RUN / "queue.json")
m = read(RUN / "pilot/manifest.json")
s = read(RUN / "pilot/summary.json")
plan = s["plan"]
assert queue["status"] == m["status"] == "complete"
assert queue["offline_summary_exact"]
assert s == read(RUN / "recomputed-summary.json")
assert m["provenance"]["source_status"] == ""
DEST.mkdir(exist_ok=False)
for name in ("pilot", "smoke"):
    shutil.copytree(RUN / name, DEST / name)
for name in ("queue.json", "packages.json", "nvidia-before.txt", "nvidia-after.txt",
             "pip-check.log", "tests.log", "smoke.log", "pilot.log", "reanalyze.log",
             "run_experiment.py", "archive_results.py", "recomputed-summary.json"):
    shutil.copy2(RUN / name, DEST / name)

detectors = s["detectors"]
prov = m["provenance"]
tested = re.search(r"(\d+) passed", (RUN / "tests.log").read_text()).group(1)
lines = [
    "# Real NF4 and INT8 pilot on L40S, 18 October 2026", "",
    f"[Experiment map](../../experiments/README.md) · [Archived evidence]({NAME}/) ·",
    f"[Numerical summary]({NAME}/pilot/summary.json)", "",
    "Completed the matched real-weight experiment against an independently loaded BF16",
    "reference. Each arm generated 32 calibration and 64 evaluation sequences of",
    "128 output tokens: 36,864 generated positions across BF16, NF4 and INT8.",
    "The 64 held-out sequences per arm, rather than individual tokens, are the",
    "units counted in the rejection-rate table.", "",
    "## Held-out rejection rates", "",
    "Each detector uses its own frozen threshold from the 32 BF16 calibration",
    "sequence means, with nominal alpha 5%. The conformal order statistic is the",
    "32nd of 32 observations (the maximum), giving rank resolution 1/33.",
    "A sequence is rejected only if its raw 128-token mean strictly exceeds the",
    "threshold. Quantized data and evaluation data do not set the thresholds.", "",
    "| Detector | BF16 false positives | NF4 detections | INT8 detections |",
    "|---|---:|---:|---:|",
]
for d, row in detectors.items():
    cells = []
    for arm in ("bf16", "nf4", "int8"):
        v = row["evaluation"][arm]
        lo, hi = v["wilson95"]
        cells.append(f"{v['rejected']}/{v['sequences']} ({100*v['rate']:.2f}%; {100*lo:.2f}–{100*hi:.2f}%)")
    lines.append(f"| `{d}` | " + " | ".join(cells) + " |")
lines += [
    "", "Parentheses show the observed rate and Wilson 95% interval. These intervals",
    "describe held-out counts at the realized calibration threshold; they do not",
    "capture variability across newly collected calibration sets. Four marginal",
    "5% tests do not constitute a jointly calibrated 5% portfolio.", "",
    "`token_difr` requires the original shared RNG contract. `cross_entropy` and",
    "`token_toploc` do not require seed synchronization. `activation_difr` needs",
    "provider-side projected hidden activations and is an additional-visibility",
    "control, not a public-token result.", "",
    "## Effect sizes", "",
    "Effects are evaluation per-token mean shifts divided by BF16 population SD.",
    "The primary descriptive variant caps both arms at the BF16 evaluation 99.9th",
    "percentile; the cap and SD are recomputed in each of 1,000 paired sequence",
    "bootstrap draws (seed 0). Intervals are exploratory 90% intervals, not",
    "simultaneous bounds. Raw effects are included to expose clipping sensitivity.",
    "Pairing is by prompt and sampling seed; generated tokens can differ.", "",
    "| Detector | Arm | Capped d′ [90% interval] | Raw d′ [90% interval] |",
    "|---|---|---:|---:|",
]


def effect_text(v):
    if v["d_prime"] is None:
        return "undefined (zero BF16 variance)"
    interval = v["ci90"]
    ci = f"[{interval[0]:+.5f}, {interval[1]:+.5f}]" if interval else "interval undefined"
    return f"{v['d_prime']:+.5f} {ci}"


for d, row in detectors.items():
    for arm in ("nf4", "int8"):
        v = row["effect_vs_bf16"][arm]
        lines.append(f"| `{d}` | {arm.upper()} | {effect_text(v['winsorized'])} | {effect_text(v['raw'])} |")
lines += [
    "", "No bootstrap token pools, AUC estimates, or extrapolated tokens-per-verdict",
    "claims are used for the rejection rates. The tested decision budget is 128",
    "tokens per fresh sequence.", "",
    "## Logit-difference diagnostics", "",
    "| Provider | Mean absolute logit difference | RMS logit difference | Position MAE Gini | Mean within-sequence Gini |",
    "|---|---:|---:|---:|---:|",
]
for arm, v in s["logit_drift"].items():
    lines.append(f"| {arm.upper()} | {v['mean_absolute_logit_difference']:.6f} | {v['rms_logit_difference']:.6f} | {v['position_mae_gini']:.6f} | {v['mean_within_sequence_position_mae_gini']:.6f} |")
lines += [
    "", "Provider logits come from sequential decoding and reference logits from",
    "batch-1 eager prefill on that provider's returned prefix. The BF16 arm",
    "therefore includes normal forward-shape differences. Gini is computed across",
    "positions of vocabulary-mean absolute logit differences, not across vocabulary",
    "coordinates. There is no matched-Gaussian comparator here; these diagnostics",
    "alone do not establish an advantage for selective verification.", "",
    "## Execution and controls", "",
    f"- Runtime source commit: `{prov['source_commit']}`; clean worktree at execution.",
    f"- Model: `{plan['model']}`, revision `{plan['revision']}`.",
    f"- Hardware: {prov['device']}; driver {prov['driver']}; CUDA runtime {prov['cuda_runtime']}.",
    f"- Python {prov['python']}; " + "; ".join(f"{k} {v}" for k,v in prov['software'].items()) + ".",
    f"- Matmul TF32: {prov['tf32_matmul']}; cuDNN TF32: {prov['tf32_cudnn']}; PyTorch/OMP/OpenBLAS threads: 1.",
    "- Temperature 1, top-k 50, top-p 0.95; fresh per-sequence sampling seeds.",
    "- Eight raw-text prompt templates with randomized numeric fields; no chat template or truncation.",
    "- Corpus seed 20261018; separate RNG streams for calibration and evaluation, unique prompts and seeds.",
    "- Fixed 128-token continuations, including tokens after EOS, as in the research harness.",
    "- NF4 uses double quantization and BF16 compute; INT8 uses threshold 6.0 and the recorded bitsandbytes BF16-to-FP16 input cast.",
    "- NF4 and INT8 each replaced 196 linear layers with packed uint8 and int8 weights, respectively; no CPU offload or synthetic logit noise.",
    "- An independent 2-calibration/2-evaluation × 16-token smoke used corpus seed 202610180; it is excluded from pilot statistics.",
    "- All three arms passed exact same-shape reference-logit, hidden-state and detector-score rerun controls.",
    f"- {tested} tests passed before GPU measurement; two existing dependency/protocol warnings are preserved in the log.",
    "- Offline loading checked artifact hashes, prompt/seed separation, capture alignment and all 33 arrays; recomputed summary matched exactly.", "",
    "| Arm | Generation, s | Reference, s | Reference rerun, s | Scoring/diagnostics, s | Arm elapsed, s | Peak allocated CUDA, GiB |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for arm, v in m["arms"].items():
    t = v["seconds"]
    lines.append(f"| {arm.upper()} | {t['generation']:.2f} | {t['reference']:.2f} | {t['reference_rerun']:.2f} | {t['score_and_diagnostics']:.2f} | {v['elapsed_seconds']:.2f} | {v['peak_cuda_allocated_bytes']/2**30:.3f} |")
lines += [
    "", f"Full pilot process time: {m['elapsed_seconds']/60:.2f} minutes. Per-phase timings",
    "include Python, synchronization and host transfers where applicable. They are",
    "resource records, not pure kernel time or a production serving benchmark.", "",
    "## Scope", "",
    "This is one Qwen3-1.7B checkpoint, one L40S/driver/stack, one corpus seed and",
    "one 128-token budget. Calibration and evaluation share the eight prompt",
    "templates. The pilot does not validate deployment FPR, hardware transfer,",
    "unseen prompt domains, adaptive providers, or a jointly calibrated detector.",
    "It does not supersede the separate Qwen3-0.6B signed-audit NF4 result: model,",
    "budget, evidence and protocol differ. Historical noisy simulated attacks and",
    "the frozen H100/A100 honest replay are not matched arms of this experiment.", "",
    "## Reproduce and verify", "",
    "From the repository root, using the recorded dependencies and pinned local model cache:", "",
    "```sh",
    "CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\",
    "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\",
    "python -m experiments.exp_real_quantization_gpu \\",
    "  --calibration 32 --evaluation 64 --tokens 128 \\",
    "  --seed 20261018 --bootstrap 1000 --output runs/real-quantization-reproduction",
    "```", "",
    "The output directory must be new. Model and source hashes, packed weight",
    "metadata and exact commands are in the manifest and queue. To verify the",
    "archive and regenerate statistics without a GPU:", "",
    "```sh",
    f"(cd docs/results/{NAME} && sha256sum -c SHA256SUMS)",
    "python -m experiments.analyze_real_quantization \\",
    f"  docs/results/{NAME}/pilot \\",
    "  --output /tmp/real-quantization-l40s-recomputed.json",
    "```", "",
    "The analyzer output path must not exist. The saved recomputed summary records",
    "exact agreement in this environment; other NumPy versions can introduce tiny",
    "float64 bootstrap differences. Artifact hash checks and raw rerun equality",
    "remain exact. The archive contains captures, score arrays, manifests,",
    "package inventory, process logs, commands and the smoke; it excludes model weights.",
]
(ROOT / "docs/results" / f"{NAME}.md").write_text("\n".join(lines) + "\n")
inventory = []
for path in sorted(DEST.rglob("*")):
    if path.is_file():
        inventory.append(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.relative_to(DEST).as_posix())
(DEST / "SHA256SUMS").write_text("\n".join(inventory) + "\n")
print("Archived", len(inventory), "files to", DEST)
