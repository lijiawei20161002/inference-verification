# Roadmap

[Project home](README.md) · [Contributing](CONTRIBUTING.md) ·
[Detailed experiment designs](planning/NEXT_EXPERIMENTS.md)

This is an ordered list of open work, not a release schedule. The API and
research harness are experimental; more representative calibration is the
main requirement for interpreting deployment results.

## Available now

- Independent token and clock verification API, including queued requests.
- CPU replay of saved clock evidence and real request/response examples.
- CUDA research harness with custom attack and verifier registration.
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

- **Cross-hardware honest controls:** compare benign variation to the attack
  effect sizes used for decisions.
- **Actual quantization:** expand NF4/INT8/AWQ/GPTQ measurements beyond simulated
  perturbations and the small NF4 pilot.
- **Representative calibration:** collect fresh held-out blocks across serving
  conditions and measure false positives and detection power.
- **Adaptive providers:** test attacks designed with knowledge of the verifier.

The [research plan](planning/NEXT_EXPERIMENTS.md) specifies hypotheses, controls,
costs, and falsifying outcomes. Please use it when proposing a replication.
