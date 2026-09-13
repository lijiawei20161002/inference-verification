# Authenticated audit protocol: H100 pilot, 2026-09-13

The experimental `audit` / `verify` / `replay` workflow ran successfully on an
NVIDIA H100 80GB HBM3 using a real HTTP provider and a separate BF16 reference
model instance. The sampler deviation was detectable at this budget; real NF4
quantization was not. These results do not validate a deployment operating point.

| Held-out provider condition | Flagged audit blocks | Flag rate | Descriptive exact 95% interval |
|---|---:|---:|---:|
| Honest BF16, temperature 1 | 1 / 20 | 5% | 0.13–24.87% |
| BF16, actual temperature 2 while claiming 1 | 10 / 10 | 100% | 69.15–100% |
| Actual NF4 weights while claiming BF16 | 0 / 10 | 0% | 0–30.85% |

No held-out block abstained. The NF4 result is a failure to demonstrate power,
not evidence that quantized execution is compliant or generally undetectable.
The honest result is too small a sample to establish a rare false-positive rate.
Intervals describe this pilot conditional on one frozen calibration; they assume
independent representative blocks and are not simultaneous deployment guarantees.

The model was `Qwen/Qwen3-0.6B` at immutable revision
`c1899de289a04d12100db370d81485cdf75e47ca`. The run used PyTorch 2.8.0+cu128,
Transformers 4.57.6, eager attention, and BF16 reference arithmetic. NF4 used
bitsandbytes 0.50.2, 4-bit weight modules, double quantization and BF16 computation.
This is actual quantized generation, with no synthetic logit noise.

Three development blocks, 63 independent fresh calibration blocks and 40
held-out blocks ran through the full protocol. Each block scheduled four
requests, capped at 24 output tokens each, and uniformly selected two responses
after provider-signed closure. The input population was newly generated numeric
tasks in four fixed templates, with disjoint prompt bytes and separate RNG
streams across partitions. Repeated templates, one machine and one day limit
generalization. No shuffled or bootstrapped tokens were counted as fresh blocks.

The fixed detector pair was mean unfiltered reference NLL and mean capped log
rank. At `alpha=0.05`, each Bonferroni threshold was 0.025; 63 calibration blocks
gave minimum attainable p-value 1/64 = 0.015625. Development and calibration
artifacts were explicitly inconclusive while the null bundle was provisioned.

| Condition | Mean complete-block wall time | Mean reference CUDA elapsed time | Mean reference-prefill tokens | Mean captured body bytes |
|---|---:|---:|---:|---:|
| Honest BF16 | 2.018 s | 49.8 ms | 93.05 | 10,222 |
| Temperature 2 | 1.973 s | 47.5 ms | 94.30 | 11,122 |
| Real NF4 | 2.597 s | 50.2 ms | 92.30 | 10,206 |

Total experiment wall time, including loading models, calibration and replay,
was 262.0 seconds. GPU elapsed measurements use CUDA events around reference
scoring; they are not utilization, kernel-only time, or a compute bill. Wire
costs count completion request/response bodies, not TLS or all protocol messages.
Dollar costs are unknown and remain null.

The complete repository suite passed **112 tests**, including **45 protocol and
conformance tests**. The suite emitted one expected warning from the existing
batch/pool ceiling test.

All 106 signed block artifacts and the aggregate report signature were verified.
One artifact from each condition was independently replayed with zero score
delta and no decision-boundary disagreement. A further installed-CLI test ran
the provider, auditor and replay in separate processes: the temperature-deviating
provider returned `inconsistent`, offline integrity was `valid`, and replay
again had zero score delta. All **107 saved transcripts** therefore verified,
with **four replay checks** in total. The extra CLI audit is excluded from the
table and the 262-second experiment measurement.

Reproduce using [the implementation guide](../AUDIT_IMPLEMENTATION.md).
The [machine-readable report](audit_protocol_h100.json) includes each block's
scores, p-values, artifact root and costs. Its detached envelope is
[`audit_protocol_h100.sig`](audit_protocol_h100.sig); the recorded experimental
public keys are in [`audit_protocol_h100_trust.json`](audit_protocol_h100_trust.json).
The [evidence archive](audit_protocol_h100_artifacts.tar.gz) contains the complete
transcripts, frozen calibration, file/software manifests, and CLI-test result,
with private signing keys excluded. Public keys included alongside evidence
still require independently trusted provenance.

The tested scope is one H100, one small model, one synthetic prompt domain and
Hugging Face HTTP serving. vLLM, cross-hardware and mixed-routing calibration,
rare-FPR certification, witnessed plans, and the optional exact-relation/privacy
backend remain unimplemented or untested. Required validated operating points
therefore return inconclusive in v1.
