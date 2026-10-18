# Real NF4 and INT8 pilot on L40S, 18 October 2026

[Experiment map](../../experiments/README.md) · [Archived evidence](real_quantization_l40s_20261018/) ·
[Numerical summary](real_quantization_l40s_20261018/pilot/summary.json)

Completed the matched real-weight experiment against an independently loaded BF16
reference. Each arm generated 32 calibration and 64 evaluation sequences of
128 output tokens: 36,864 generated positions across BF16, NF4 and INT8.
The 64 held-out sequences per arm, rather than individual tokens, are the
units counted in the rejection-rate table.

At this budget, token-DiFR detected **34/64 NF4 sequences (53.12%)** and
**5/64 INT8 sequences (7.81%)**, with **0/64 BF16 false positives**. Seed-free
token-TOPLOC detected **32/64 NF4 sequences (50.00%)**, with **4/64 BF16 false
positives (6.25%)**, above the nominal 5% target in this sample. Actual NF4 leaves
a measurable token signal here, but detection is incomplete and token-only INT8
detection remains weak. The activation control is stronger and requires extra
provider evidence. Even 0/64 false positives has a 5.66% Wilson upper bound;
this pilot does not establish a deployment false-positive guarantee.

## Held-out rejection rates

Each detector uses its own frozen threshold from the 32 BF16 calibration
sequence means, with nominal alpha 5%. The conformal order statistic is the
32nd of 32 observations (the maximum), giving rank resolution 1/33.
A sequence is rejected only if its raw 128-token mean strictly exceeds the
threshold. Quantized data and evaluation data do not set the thresholds.

| Detector | BF16 false positives | NF4 detections | INT8 detections |
|---|---:|---:|---:|
| `token_difr` | 0/64 (0.00%; 0.00–5.66%) | 34/64 (53.12%; 41.07–64.82%) | 5/64 (7.81%; 3.38–17.02%) |
| `cross_entropy` | 3/64 (4.69%; 1.61–12.90%) | 8/64 (12.50%; 6.47–22.77%) | 6/64 (9.38%; 4.37–18.98%) |
| `token_toploc` | 4/64 (6.25%; 2.46–15.00%) | 32/64 (50.00%; 38.10–61.90%) | 5/64 (7.81%; 3.38–17.02%) |
| `activation_difr` | 0/64 (0.00%; 0.00–5.66%) | 64/64 (100.00%; 94.34–100.00%) | 52/64 (81.25%; 70.03–88.94%) |

Parentheses show the observed rate and Wilson 95% interval. These intervals
describe held-out counts at the realized calibration threshold; they do not
capture variability across newly collected calibration sets. Four marginal
5% tests do not constitute a jointly calibrated 5% portfolio.

`token_difr` requires the original shared RNG contract. `cross_entropy` and
`token_toploc` do not require seed synchronization. `activation_difr` needs
provider-side projected hidden activations and is an additional-visibility
control, not a public-token result.

## Effect sizes

Effects are evaluation per-token mean shifts divided by BF16 population SD.
The primary descriptive variant caps both arms at the BF16 evaluation 99.9th
percentile; the cap and SD are recomputed in each of 1,000 paired sequence
bootstrap draws (seed 0). Intervals are exploratory 90% intervals, not
simultaneous bounds. Raw effects are included to expose clipping sensitivity.
Pairing is by prompt and sampling seed; generated tokens can differ.

| Detector | Arm | Capped d′ [90% interval] | Raw d′ [90% interval] |
|---|---|---:|---:|
| `token_difr` | NF4 | +0.34435 [+0.27621, +0.43242] | +0.34435 [+0.27621, +0.43242] |
| `token_difr` | INT8 | +0.07537 [+0.03451, +0.12258] | +0.07537 [+0.03451, +0.12258] |
| `cross_entropy` | NF4 | +0.07250 [+0.00277, +0.14225] | +0.08002 [+0.00988, +0.15156] |
| `cross_entropy` | INT8 | +0.05650 [+0.01330, +0.10581] | +0.05804 [+0.01456, +0.10704] |
| `token_toploc` | NF4 | +0.25431 [+0.19437, +0.32363] | +0.25431 [+0.19437, +0.32363] |
| `token_toploc` | INT8 | +0.07300 [+0.03565, +0.11522] | +0.07300 [+0.03565, +0.11522] |
| `activation_difr` | NF4 | +9.32543 [+8.01108, +10.47190] | +9.71600 [+8.10795, +12.28333] |
| `activation_difr` | INT8 | +2.66184 [+2.16763, +3.16415] | +2.48383 [+2.01548, +3.16356] |

No bootstrap token pools, AUC estimates, or extrapolated tokens-per-verdict
claims are used for the rejection rates. The tested decision budget is 128
tokens per fresh sequence.

## Logit-difference diagnostics

| Provider | Mean absolute logit difference | RMS logit difference | Position MAE Gini | Mean within-sequence Gini |
|---|---:|---:|---:|---:|
| BF16 | 0.075376 | 0.129846 | 0.347981 | 0.277201 |
| NF4 | 0.699094 | 0.989637 | 0.260957 | 0.198851 |
| INT8 | 0.250318 | 0.392748 | 0.342477 | 0.256572 |

Provider logits come from sequential decoding and reference logits from
batch-1 eager prefill on that provider's returned prefix. The BF16 arm
therefore includes normal forward-shape differences. Gini is computed across
positions of vocabulary-mean absolute logit differences, not across vocabulary
coordinates. There is no matched-Gaussian comparator here; these diagnostics
alone do not establish an advantage for selective verification.

## Execution and controls

- Runtime source commit: `a6e1966dc6843f15e63712f5de6ce8d37efe067f`; clean worktree at execution.
- Model: `Qwen/Qwen3-1.7B`, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- Hardware: NVIDIA L40S; driver 595.91.07; CUDA runtime 12.8.
- Python 3.12.3; torch 2.8.0+cu128; transformers 4.57.6; numpy 2.1.2; accelerate 1.15.0; bitsandbytes 0.50.2.
- Matmul TF32: False; cuDNN TF32: True; PyTorch/OMP/OpenBLAS threads: 1.
- Temperature 1, top-k 50, top-p 0.95; fresh per-sequence sampling seeds.
- Eight raw-text prompt templates with randomized numeric fields; no chat template or truncation.
- Corpus seed 20261018; separate RNG streams for calibration and evaluation, unique prompts and seeds.
- Fixed 128-token continuations, including tokens after EOS, as in the research harness.
- NF4 uses double quantization and BF16 compute; INT8 uses threshold 6.0 and the recorded bitsandbytes BF16-to-FP16 input cast.
- NF4 and INT8 each replaced 196 linear layers with packed uint8 and int8 weights, respectively; no CPU offload or synthetic logit noise.
- An independent 2-calibration/2-evaluation × 16-token smoke used corpus seed 202610180; it is excluded from pilot statistics.
- All three arms passed exact same-shape reference-logit, hidden-state and detector-score rerun controls.
- 216 tests passed before GPU measurement; two existing dependency/protocol warnings are preserved in the log.
- Offline loading checked artifact hashes, prompt/seed separation, capture alignment and all 33 arrays; recomputed summary matched exactly.

| Arm | Generation, s | Reference, s | Reference rerun, s | Scoring/diagnostics, s | Arm elapsed, s | Peak allocated CUDA, GiB |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 344.07 | 8.28 | 8.10 | 133.82 | 498.40 | 6.571 |
| NF4 | 469.15 | 8.35 | 8.22 | 136.25 | 626.80 | 4.623 |
| INT8 | 1361.62 | 8.42 | 8.32 | 178.09 | 1564.87 | 5.261 |

Full pilot process time: 45.10 minutes. Per-phase timings
include Python, synchronization and host transfers where applicable. They are
resource records, not pure kernel time or a production serving benchmark.

## Scope

This is one Qwen3-1.7B checkpoint, one L40S/driver/stack, one corpus seed and
one 128-token budget. Calibration and evaluation share the eight prompt
templates. The pilot does not validate deployment FPR, hardware transfer,
unseen prompt domains, adaptive providers, or a jointly calibrated detector.
It does not supersede the separate Qwen3-0.6B signed-audit NF4 result: model,
budget, evidence and protocol differ. Historical noisy simulated attacks and
the frozen H100/A100 honest replay are not matched arms of this experiment.

Next, repeat independent calibration/evaluation corpora and transfer this same
real-weight protocol to H100/A100. Larger independently calibrated decision
budgets are needed to test whether weak INT8 power improves while controlling
false positives. AWQ/GPTQ and the matched-Gaussian concentration comparison remain
unfinished parts of the broader quantization study.

## Reproduce and verify

From the repository root, using the recorded dependencies and pinned local model cache:

```sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m experiments.exp_real_quantization_gpu \
  --calibration 32 --evaluation 64 --tokens 128 \
  --seed 20261018 --bootstrap 1000 --output runs/real-quantization-reproduction
```

The output directory must be new. Model and source hashes, packed weight
metadata and exact commands are in the manifest and queue. To verify the
archive and regenerate statistics without a GPU:

```sh
(cd docs/results/real_quantization_l40s_20261018 && sha256sum -c SHA256SUMS)
python -m experiments.analyze_real_quantization \
  docs/results/real_quantization_l40s_20261018/pilot \
  --output /tmp/real-quantization-l40s-recomputed.json
```

The analyzer output path must not exist. The saved recomputed summary records
exact agreement in this environment; other NumPy versions can introduce tiny
float64 bootstrap differences. Artifact hash checks and raw rerun equality
remain exact. The archive contains captures, score arrays, manifests,
package inventory, process logs, commands and the smoke; it excludes model weights.
