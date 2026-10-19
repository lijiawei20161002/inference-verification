# Experiment map

[Project home](../README.md) · [Research guide](../docs/RESEARCH_GUIDE.md) ·
[Complete claim-to-artifact index](../docs/RESEARCH_GUIDE.md#which-experiment-backs-which-claim)

Run commands from the repository root. Fresh model experiments require CUDA and
model weights. Saved artifacts can be inspected on a laptop.

For parallel runs on H100, A100, and L40S hosts, follow the
[multi-machine experiment guide](../docs/MULTI_MACHINE_EXPERIMENTS.md). It includes
an isolated launcher for existing token and timing baselines and distinguishes
those measurements from the planned joint-channel evaluation.

The [16 October H100 campaign report](../docs/results/h100_campaign_20261016.md)
documents all eight completed baseline tasks, including the successful batching
retry and controlled second-GPU interference pilot. Rebuild its numerical
summary without a GPU with `python3 experiments/analyze_h100_campaign.py`.

The [17 October A100 campaign report](../docs/results/a100_campaign_20261017.md)
records four completed independent baselines: token shape, context window,
short-term stability and the 48-cell timing floor. Rebuild its summary with
`python3 experiments/analyze_a100_campaign.py`; use `--check` to verify the
archive and saved summary. Paired H100/A100 replay remains a separate experiment.

The [18 October A100 follow-up](../docs/results/a100_followup_20261018.md)
completed frozen capture/replay and clean timing repeats. Its score arrays and
hashes support offline verification with
`python -m experiments.analyze_a100_followup --check`. Paired replay of that A100 capture on H100 remains open.

The [18 October H100 follow-up](../docs/results/h100_followup_20261018.md)
completed a new 80 × 256 frozen capture/replay and immediate/delayed timing
repeats. Verify its archived evidence with
`python -m experiments.analyze_h100_followup --check`. The original A100 capture
was unavailable on that host;
a [portable bundle of the new H100 capture](../docs/results/h100_followup_20261018/h100-replay-transfer.tar.gz)
is tracked in the result archive for replay; extraction instructions
are in the report.

The [18 October paired H100-to-A100 replay](../docs/results/cross_gpu_replay_20261018.md)
now scores that identical H100 capture on A100: 80 × 256 positions, four schedules,
four detectors, and exact rerun controls. Batch-1 eager token intervals include
zero; sequential token-DiFR shifts by +0.04596 d′. Verify all sixteen paired
comparisons with `python -m experiments.analyze_cross_gpu_replay --check`.
The report flags the sequential activation effect as undefined because its H100
baseline variance is zero. This is one honest GPU/driver comparison, not a
deployment false-positive-rate measurement or an attack-power experiment.

The follow-up artifact checkers allow float64 roundoff in recomputed bootstrap
statistics (`rtol=1e-12`, `atol=1e-14`) across NumPy versions and retain the
verified archived precision in their summaries. Artifact hashes, changed-score
fractions, and raw same-shape rerun equality remain exact checks.

The [18 October L40S real-quantization pilot](../docs/results/real_quantization_l40s_20261018.md)
completed BF16, actual NF4 and INT8 generation against an independent BF16
reference: 32 calibration and 64 evaluation sequences per arm, 128 tokens each.
Token-DiFR flagged 34/64 NF4, 5/64 INT8 and 0/64 BF16 sequences. Seed-free
token-TOPLOC flagged 32/64 NF4 and 4/64 BF16 sequences; its observed 6.25% false
positive rate exceeds the nominal 5% target in this sample. The report includes
all four detectors, confidence intervals, exact rerun controls, raw captures and
offline reproduction commands. This is a same-GPU pilot; calibration transfer
and deployment false-positive rates remain open.

The [19 October L40S performance-probe pilot](../docs/results/performance_probes_l40s_20261019.md)
records 524 local measurements, 42 loopback HTTP requests, six kernel traces, and
96 equivalent-SDPA controls. It distinguishes local context truncation from
prefix reuse, identifies legitimate kernel/representation differences, and
shows an eager HTTP path masking the decode signal. These are mechanism results,
not calibrated deployment detection rates. Recompute the saved summary without a GPU:

```sh
python -m experiments.analyze_performance_probes \
  docs/results/performance_probes_l40s_20261019 --output /tmp/performance-recomputed.json
```

The report includes source snapshots, raw records, hashes, plotting commands,
and separate instructions for reproducing the main and SDPA runs.

## Frozen capture and paired GPU replay

`exp_frozen_replay_gpu.py` separates generation from verification. Generate one
honest capture, transfer its exact bytes to each host, and replay those tokens.
The clean BF16 experiment uses the same four detectors and replay schedules as
`exp_benign_shape_dprime_gpu`, with a same-shape rerun control. It saves per-token
score arrays so paired sequence bootstrap intervals can be recomputed later.
Activation-DiFR requires captured internal activations and is an additional
visibility control.

Prepare the pinned model cache and environment using the multi-machine guide.
The default checkpoint is Qwen3-1.7B revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; this script only loads local snapshots.
Use the same source files, software versions, and thread settings on both hosts.
For example, after setting `HF_HOME` to the prepared cache:

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m experiments.exp_frozen_replay_gpu capture \
  --node a100-80-02 --prompts 80 --tokens 256 \
  --output runs/frozen/capture
python -c 'import json; print(json.load(open("runs/frozen/capture/manifest.json"))["capture_sha256"])'
```

Use the printed SHA-256 below, and the same capture file and hash on the H100.
Output directories must not already exist. Smoke runs can use four prompts and
16 tokens in separately named directories.

```sh
python -m experiments.exp_frozen_replay_gpu replay \
  --node a100-80-02 --capture runs/frozen/capture/capture.json.gz \
  --capture-sha256 CAPTURE_SHA256 --output runs/frozen/a100
python -m experiments.exp_frozen_replay_gpu compare \
  --baseline runs/frozen/a100 --alternate runs/frozen/h100 \
  --output runs/frozen/comparison.json
```

The comparison rejects different captures, source hashes, software, CUDA
runtimes, numerical flags, incomplete runs, failed rerun controls, and modified
score artifacts. Positive d′ means higher scores on the alternate host; intervals
are 90% paired sequence bootstraps. GPU and driver metadata are preserved, so
changing both must be interpreted as a combined shift. These honest comparisons
do not measure attack power or deployment false-positive rates. New captures are
written under ignored `runs/`; preserve them and the source files with the run
archive. The reviewed H100 follow-up transfer bundle linked above is included in
Git so it can be replayed on another host.

## First experiment

```sh
python -m pip install -e '.[gpu]'
python -m experiments.exp_gpu
```

This downloads the default Qwen3-0.6B model on first use and reports a small
attack/detector grid, per-token effect sizes, and the batch/pool ratio. Runtime
depends on your hardware. See the
[research quickstart](../docs/RESEARCH_GUIDE.md#quickstart) for interpreting it.

For your own strategies, use [the extension guide](../docs/EXTENDING.md).

## Pick a research question

| Question | Entry points | Saved evidence |
|---|---|---|
| Does the independent API work end to end? | `exp_verification_service_gpu.py` | [Service integration report](../docs/results/verification_service_h100.md) |
| What can tokens reveal? | `exp_gpu.py`, `exp_robustness_gpu.py` | [Robustness report](../docs/results/robustness_report.md) |
| When does a proxy catch model substitution? | `exp_spec_substitution_gpu.py`, `exp_detectability_vs_kl.py` | [Proxy study](../docs/SPEC_DECODING_AND_PROXY_DETECTION.md) |
| Does timing expose context truncation? | `exp_slope_verifier_gpu.py`, `exp_clock_algos_gpu.py` | [Clock measurements](../docs/CLOCK_MEASURED.md) |
| Which hardware mechanisms shape performance probes? | exp_performance_probe_gpu.py, exp_performance_probe_http.py, exp_sdpa_representation_gpu.py | [L40S performance probes](../docs/results/performance_probes_l40s_20261019.md) |
| How much does a verdict cost? | `exp_prefix_cost_gpu.py`, `exp_cost_of_a_verdict_gpu.py` | [Audit-cost study](../docs/TRIAGE_AND_AUDIT_COST.md) |
| Are detection scores inflated by a small pool? | `exp_baseline_headroom_gpu.py`, `exp_headline_ratio_gpu.py`, `exp_pool_scaling_gpu.py` | [Evaluation protocol](../docs/RESEARCH_GUIDE.md#the-evaluation-protocol) |
| Can honest numerical differences trigger a detector? | `exp_specdec_shape_gpu.py`, `exp_benign_shape_dprime_gpu.py` | [Benign variation](../docs/RESEARCH_GUIDE.md#the-honest-null-is-not-one-thing-the-forward-pass-shape-moves-the-logits) |
| What did the signed-artifact pilot detect? | `exp_audit_protocol_gpu.py` | [H100 pilot](../docs/results/audit_protocol_h100.md) |

Read each script's module docstring and configuration before running it. Some
experiments use environment variables and some write into `docs/results/`.
Use a separate checkout for reproductions so the committed evidence remains
available for comparison. The legacy `requirements.txt` collects the broader
research dependencies; it is not needed for the laptop demo or basic API.

## Rebuild the paper's tables and figures

These commands use the committed artifacts; no GPU or model download is needed:

```sh
python -m pip install -e . matplotlib
python paper/make_tables.py
python paper/make_figs.py
```

Compiling `paper/paper.tex` additionally requires a LaTeX installation. Plot
scripts named `plot_*.py` regenerate other figures; inspect their inputs first.

When publishing a new result, retain honest controls and distinguish simulated
logit perturbations from genuinely quantized weights. Contribution requirements
are in [CONTRIBUTING.md](../CONTRIBUTING.md#research-results).
