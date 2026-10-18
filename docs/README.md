# Documentation

[Project home](../README.md) · [Examples](../examples/README.md) · [Contributing](../CONTRIBUTING.md)

IVGym has two main entry points: an independent verification API for checking
inference behavior, and a research harness for comparing attacks and detectors.
The signed-artifact protocol is a separate, optional experiment.

## Start with your goal

| Goal | Guide | Requirements |
|---|---|---|
| See a result on a laptop | [Quickstart](../README.md#try-it-without-a-gpu) | Python + NumPy; replays saved traces |
| Send an HTTP verification request | [Verification service](VERIFICATION_SERVICE.md) | Service extras; captured clock checks need no GPU |
| Check tokens or probe a live endpoint | [Reference setup](VERIFICATION_SERVICE.md#configure-a-reference-model-for-token-and-live-checks) | CUDA reference, model checkpoint, allowed provider URL; calibration for a verdict |
| Add an attack or verifier | [Extending IVGym](EXTENDING.md) | NumPy to inspect strategies; CUDA to run the research sweep |
| Reproduce research | [Experiment map](../experiments/README.md) | Saved artifacts for inspection; CUDA for fresh model measurements |
| Run experiments on multiple GPU hosts | [Multi-machine experiment guide](MULTI_MACHINE_EXPERIMENTS.md) | Token and timing baselines, isolated jobs, hardware controls, and result collection |
| Verify signed artifacts | [Audit implementation](AUDIT_IMPLEMENTATION.md) | Audit extras for offline checks; CUDA for score replay |

## Understand the research

| Topic | Read |
|---|---|
| Results, methodology, costs, and limitations | [Research guide](RESEARCH_GUIDE.md) · [paper](../paper/paper.pdf) · [poster](../paper/mats_poster.pdf) |
| Attacker/verifier game and evaluation pipeline | [Game specification](GAME.md) |
| What timing can detect | [Clock measurements](CLOCK_MEASURED.md) · [clock algorithms](CLOCK_ALGORITHMS.md) |
| Selective recomputation and evaluation pitfalls | [Triage and audit cost](TRIAGE_AND_AUDIT_COST.md) |
| Proxy detection and speculative decoding | [Acceptance-rate fingerprint](ACCEPTANCE_RATE_FINGERPRINT.md) · [speculative decoding study](SPEC_DECODING_AND_PROXY_DETECTION.md) |
| Optional signed receipts | [Protocol RFC](AUDIT_PROTOCOL_RFC.md) · [implementation](AUDIT_IMPLEMENTATION.md) |

## Follow a claim to its evidence

- [Claim → experiment → artifact index](RESEARCH_GUIDE.md#which-experiment-backs-which-claim)
- [18 October paired H100-to-A100 replay: honest cross-GPU score differences](results/cross_gpu_replay_20261018.md)
- [16 October H100 campaign: results, implications, and limitations](results/h100_campaign_20261016.md)
- [17 October A100 campaign: four completed baselines and archived measurements](results/a100_campaign_20261017.md)
- [Independent service H100 report](results/verification_service_h100.md)
- [Actual API requests and responses](../examples/api-samples/README.md)
- [Separate signed-artifact H100 pilot](results/audit_protocol_h100.md)
- [Raw results](results/) and [figures](figures/)

The [roadmap](../ROADMAP.md) identifies contribution opportunities. The
[research plan](../planning/NEXT_EXPERIMENTS.md) gives the hypotheses and controls
needed for the next measurements.
