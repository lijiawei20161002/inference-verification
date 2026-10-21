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
| Prepare the next BF16 diagnosis experiment | [Numerical tools](../experiments/numerical/README.md) · [English next-step plan](../planning/NEXT_EXPERIMENTS.md) | NumPy for archive diagnostics; planned capture runs need a dedicated L40S |
| Run experiments on multiple GPU hosts | [Multi-machine experiment guide](MULTI_MACHINE_EXPERIMENTS.md) | Token and timing baselines, isolated jobs, hardware controls, and result collection |
| Verify signed artifacts | [Audit implementation](AUDIT_IMPLEMENTATION.md) | Audit extras for offline checks; CUDA for score replay |

## Understand the research

| Topic | Read |
|---|---|
| Results, methodology, costs, and limitations | [Research guide](RESEARCH_GUIDE.md) · [paper](../paper/paper.pdf) · [poster](../paper/mats_poster.pdf) |
| Attacker/verifier game and evaluation pipeline | [Game specification](GAME.md) |
| Design numerical causal diagnosis experiments | [English experiment plan](NUMERICAL_CAUSAL_DIAGNOSIS_EXPERIMENTS.md) — BF16 mechanisms, controlled interventions, reproducible witnesses, baselines, and decision gates; proposed, not measured |
| What timing can detect | [Clock measurements](CLOCK_MEASURED.md) · [clock algorithms](CLOCK_ALGORITHMS.md) |
| Design hardware-based performance probes | [Performance probe experiments](PERFORMANCE_PROBE_EXPERIMENTS.md) (中文) — proposed mechanisms, controls, hardware, and falsification criteria |
| Continue performance probes through a production serving engine | [vLLM serving experiment](VLLM_SERVING_EXPERIMENT.md) — pinned configuration, cache controls, and measurement contract |
| Profile the pinned vLLM server | [Profiler setup](PROFILING_SETUP.md) — tested PyTorch and Nsight Systems capture, inspection, and host permission limits |
| Selective recomputation and evaluation pitfalls | [Triage and audit cost](TRIAGE_AND_AUDIT_COST.md) |
| Proxy detection and speculative decoding | [Acceptance-rate fingerprint](ACCEPTANCE_RATE_FINGERPRINT.md) · [speculative decoding study](SPEC_DECODING_AND_PROXY_DETECTION.md) |
| Optional signed receipts | [Protocol RFC](AUDIT_PROTOCOL_RFC.md) · [implementation](AUDIT_IMPLEMENTATION.md) |

## Follow a claim to its evidence

- [Claim → experiment → artifact index](RESEARCH_GUIDE.md#which-experiment-backs-which-claim)
- [21 October L40S M3 follow-up: partition coverage, FP32 intermediates and native operator bridges](results/numerical_m3_partition_bridge_l40s_20261021.md)
- [21 October L40S M3: controlled reduction-order pilot, replay checks and negative-result limits](results/numerical_m3_pilot_l40s_20261021.md)
- [21 October L40S M1: selected-forward dispatch equivalence and unprofiled controls](results/numerical_m1_dispatch_l40s_20261021.md)
- [20 October L40S E0: historical BF16 events and frozen-logit discovery replay](results/numerical_e0_l40s_20261020.md)
- [19 October L40S scheduling diagnostics: token budgets, iteration traces, and arrival windows](results/scheduling_l40s_20261019.md)
- [19 October L40S vLLM serving: optimized/eager controls, prefix caching, and delayed replication](results/vllm_serving_l40s_20261019.md)
- [19 October L40S performance probes: context, prefix reuse, kernel selection, and HTTP limits](results/performance_probes_l40s_20261019.md)
- [18 October L40S real NF4/INT8 pilot: independent calibration and held-out detection](results/real_quantization_l40s_20261018.md)
- [18 October paired H100-to-A100 replay: honest cross-GPU score differences](results/cross_gpu_replay_20261018.md)
- [16 October H100 campaign: results, implications, and limitations](results/h100_campaign_20261016.md)
- [17 October A100 campaign: four completed baselines and archived measurements](results/a100_campaign_20261017.md)
- [Independent service H100 report](results/verification_service_h100.md)
- [Actual API requests and responses](../examples/api-samples/README.md)
- [Separate signed-artifact H100 pilot](results/audit_protocol_h100.md)
- [Raw results](results/) and [figures](figures/)

The [roadmap](../ROADMAP.md) identifies contribution opportunities. The
[current implementation plan](../planning/NEXT_EXPERIMENTS.md) specifies the
next coding tasks, machine, execution order, and acceptance gates. The
[historical verification agenda](../planning/LEGACY_VERIFICATION_EXPERIMENTS.md)
retains earlier hypotheses and dated progress.
