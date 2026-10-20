# Roadmap

[Project home](README.md) · [Contributing](CONTRIBUTING.md) ·
[Current implementation plan](planning/NEXT_EXPERIMENTS.md)

This is an ordered list of open work, not a release schedule. The API and
research harness are experimental; more representative calibration is the
main requirement for interpreting deployment results.

## Available now

- Independent token and clock verification API, including queued requests.
- CPU replay of saved clock evidence and real request/response examples.
- CUDA research harness with custom attack and verifier registration.
- CPU numerical diagnostics with a validated, evaluation-only capped-event
  ledger and legacy sampler regression checks.
- Signed-artifact audit, offline verification, and GPU score replay as a
  separate experiment.

## Accessible contributions

- Document installation results on Linux, macOS, and Windows.
- Add annotated API examples that explain inconclusive and unsupported checks.
- Extend artifact consistency checks to the clock claims.
- Make experiment prerequisites and configuration easier to discover.

## Provider integrations

- Add `/chat/completions` support with explicit handling of templates and hidden
  preprocessing; the current live adapter uses raw `/completions`.
- Implement and validate the vLLM backend; the current adapter is a skeleton.
- Evaluate clock observations from public endpoints, including buffering,
  chunked events, jitter, and provider traffic shaping.

## Research priorities

- **Numerical causal diagnosis (current focus):** implement paired BF16
  capture and sampler conformance, then state-consistent interventions and
  reproducible witnesses. The [E0–E5 design](docs/NUMERICAL_CAUSAL_DIAGNOSIS_EXPERIMENTS.md)
  prioritizes explanatory correctness and execution conformance. The
  [offline preparation tools](experiments/numerical/README.md) are available;
  GPU capture, the localizer, and independent witness replay remain planned.
- **Cross-hardware honest controls:** extend the
  [first paired H100-to-A100 replay](docs/results/cross_gpu_replay_20261018.md)
  to fresh prompt/seed blocks and more hardware, then compare against matched
  attack controls.
- **Actual quantization:** extend the
  [matched L40S NF4/INT8 pilot](docs/results/real_quantization_l40s_20261018.md)
  to independent corpora, additional hardware, larger calibrated budgets and
  AWQ/GPTQ. Complete the matched-Gaussian concentration comparison.
- **Representative calibration:** collect fresh held-out blocks across serving
  conditions and measure false positives and detection power. The
  [L40S vLLM serving pilot and temporal repeat](docs/results/vllm_serving_l40s_20261019.md)
  identify eager execution, prefix reuse, concurrency, and stream position as
  controls to include; they do not yet provide a calibrated detector.
- **Adaptive providers:** test attacks designed with knowledge of the verifier.

The [current plan](planning/NEXT_EXPERIMENTS.md) specifies implementation
milestones, controls, and decision gates. The other research opportunities above
do not precede the numerical-diagnosis work queue.
