# Experiment map

[Project home](../README.md) · [Research guide](../docs/RESEARCH_GUIDE.md) ·
[Complete claim-to-artifact index](../docs/RESEARCH_GUIDE.md#which-experiment-backs-which-claim)

Run commands from the repository root. Fresh model experiments require CUDA and
model weights. Saved artifacts can be inspected on a laptop.

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
