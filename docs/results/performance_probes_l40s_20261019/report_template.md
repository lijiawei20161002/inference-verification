# L40S performance probes: context, prefix reuse, and kernel selection

19 October 2026 · Qwen3-1.7B · Single NVIDIA L40S · **Mechanism pilot; no calibrated remote detector.**

{{OVERVIEW}}

[Experiment design](../PERFORMANCE_PROBE_EXPERIMENTS.md) ·
[Summary and raw evidence](performance_probes_l40s_20261019/summary.json) ·
[Run queue](performance_probes_l40s_20261019/queue.json)

## Configuration and scope

- Main run source: 0d9e5407148db43fa2bfef57d13fd0b3d8c14465. Each subprocess records source hashes and Git status.
- Model: Qwen/Qwen3-1.7B, revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e.
- BF16; PyTorch 2.8.0+cu128; Transformers 4.57.6; NVIDIA driver 595.91.07.
- Eight text-offset blocks across three fresh process sessions, drawn from the repository's public long-text fixture.
  These blocks share a corpus; they are not eight independent deployments or a broad prompt distribution.
  Stability across days remains untested.
- Main timings replay a fixed continuation. Twelve additional controls generate greedily.
  All use a fixed output budget, continuing after EOS; no top-p sampling is used.
- The main 128 output positions comprise one prefill output and 127 decode steps.
  P0 uses 16 outputs, HTTP uses 32, and profiles use five. Different budgets are not pooled.
- The main path uses preallocated StaticCache and CUDA Graph decode.
  Graph capture, cache allocation, and input preparation are outside the local prefill/decode timers
  and have separate recorded costs. These timings are not production request costs.
- CUDA event spans include host submission gaps; they are not pure kernel durations.
  Kernel durations come from separate profiler runs.
- Nsight Compute retries returned ERR_NVGPUCTRPERM. The host sets RmProfilingAdminOnly=1,
  and the container lacks CAP_SYS_ADMIN. DRAM traffic, L2 hit rate, and occupancy were not measured.
  Reported KV bytes describe tensor storage, not measured memory traffic.
  See the [permission log](performance_probes_l40s_20261019/counter_permission.log) and
  [permission environment](performance_probes_l40s_20261019/profiling_environment.json).

## P0: timing instrumentation and implementation controls

Four shapes, each with 20 paired events/plain technical repetitions, distributed across three processes.
{{EVENT_OVERHEAD}}
These repetitions are not independent deployment samples.

The following unprofiled results use batch=1. Static versus dynamic cache also changes the mask,
GQA representation, and SDPA dispatch; this is not an isolated allocation-policy intervention.

| Initial context | KV cache | Execution | Median decode ms/step |
|---|---|---|---:|
{{CONTROLS}}

## P1: context length × actual batch

L ∈ {512,2048,8192,16384}, B ∈ {1,4,8}, eight paired text blocks: 192 measurements.
The truncation condition removes all but the last 2,048 prompt tokens before prefill,
preserves absolute RoPE positions, and appends KV during decode. It does not evict tokens on every step.

| Original context | B | Full ms/step | Truncated ms/step | Median paired full/truncated |
|---|---:|---:|---:|---:|
{{CONTEXT}}

{{NEGATIVE_CONTROLS}}

![Context response](performance_probes_l40s_20261019/figures/response_surface.png)

Shading shows the observed minimum–maximum across eight text blocks, not a confidence interval.
No detector threshold was trained and evaluated on this grid; no deployment FPR/TPR is reported.

## P4: legitimate prefix reuse

For L=2048/8192 and B=1/4, compare cache off, cold, hit, and input truncation: 128 measurements.
The reused prefix contains L−128 tokens; the hit arm processes only the final 128 prompt tokens.
Prefix priming happens outside the request timer, with its time recorded separately.
This is an explicit KV-reuse microbenchmark, not a vLLM APC benchmark. It excludes cache lookup,
eviction, and cross-request routing costs.

| L | B | Cold prefill ms | Hit prefill ms | Median paired cold/hit | Cold/hit decode ms/step |
|---|---:|---:|---:|---:|---:|
{{CACHE}}

{{CACHE_CHECKS}}

![Prefix reuse](performance_probes_l40s_20261019/figures/prefix_cache.png)

## Hardware mechanism and kernel evidence

The device reports compute capability 8.9, 142 SMs, and 96 MB of L2 cache.
The [device inventory](performance_probes_l40s_20261019/device_inventory.json)
was recorded after the experiments; per-case GPU snapshots are in the measurement rows.

For this model, the persistent BF16 KV storage per token per sequence is
2 × 28 layers × 8 KV heads × 128 dimensions × 2 bytes = **114,688 bytes**.
At L=8192, B=1 the prompt KV alone occupies 0.875 GiB; at L=16384, B=8 it occupies 14 GiB.
The preallocated cache also reserves positions for decode. These sizes explain why changing
history length is a meaningful memory intervention, but do not establish how many bytes
actually reach DRAM.

The independent traces show B=1 linear layers primarily using GEMV kernels, while B=4
introduces BF16 WMMA Tensor Core GEMM kernels. Static-cache decode uses memory-efficient
attention with KV expansion/copying; dynamic-cache decode uses Flash Attention and cache concatenation.

In this Transformers version, an explicit mask disables the integration's native-GQA branch.
It expands eight KV heads to sixteen query heads before calling SDPA.
The persistent cache remains at eight heads; the expanded operands are temporary.
Consequently, two complete BF16 implementations can have different copying costs and attention kernels.
See the [pinned integration source](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/integrations/sdpa_attention.py)
and [PyTorch SDPA documentation](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html).

![Kernel breakdown](performance_probes_l40s_20261019/figures/kernel_breakdown.png)

Categories use actual kernel names. The copy category is not exclusively KV copying.
Summed durations are not bandwidth, occupancy, or critical-path percentages.
Each profile contains only four decode steps and is subject to profiler effects.
Its static cache is also 123 positions smaller than the main 128-output experiment.

## Causal control: equivalent SDPA representations

Keep Q/K/V identical and compare native GQA without a mask, explicit KV repetition without a mask,
and explicit repetition with an all-True Boolean mask. A single query can see every key in all three
arms, so their mathematical attention semantics are the same. There are eight random input blocks
per shape; each arm uses five timing groups of 100 graph replays. Repetition/copy costs are included.
Technical repetitions stabilize timing and are not independent input samples.

| B | KV length | Native GQA μs | Repeat/no-mask μs | Repeat/all-True μs | Masked/native ratio |
|---|---:|---:|---:|---:|---:|
{{SDPA}}

{{SDPA_CHECKS}}

The B=4, L=8192 traces confirm that native GQA uses Flash Attention without the explicit KV-copy
kernels; repeated KV without a mask still uses Flash Attention but adds copying; adding the mask
switches to memory-efficient attention. At this shape the two repeated-KV arms have similar total
times, so a mask does not uniformly make attention slower. Representation and dispatch must be
considered together.

These repeated accesses use the same Q/K/V and have cache-warmth effects.
Their absolute times cannot be extrapolated directly to a multi-layer LLM.
This control supports implementation-dependent performance differences; it does not certify
hardware identity, model precision, or floating-point equivalence on all inputs.

The first supplementary attempt failed before any timing rows were written: it checked the captured
graph's output before replaying the graph. The correction adds one replay before validation;
the error tolerance was unchanged. The failed plan, source, and explanation are
[preserved separately](performance_probes_l40s_20261019/sdpa_failed_capture_check/failure.json).
All 96 reported measurements come from the corrected run.

## HTTP: buffering and host overhead change the observable signal

Three fresh server processes, each queried serially by a separate CPU client; C=1.
B is an explicitly selected server batch and is not concurrent-request count.
Transport is loopback NDJSON token-ID frames, not a public text API.
Requests select a fixture; they do not transmit/tokenize the full long prompt.
Raw events retain token IDs, positions, and client monotonic timestamps.

Results for L=8192, B=1; six requests per condition:

| Condition | Tokens/event | First content event ms | Median event gap ms | Events/request |
|---|---:|---:|---:|---:|
{{HTTP}}

![HTTP buffering](performance_probes_l40s_20261019/figures/http_buffering.png)

{{HTTP_CHECKS}}

There is also a negative result: with one token per event, full and truncated contexts both have
roughly 24 ms event gaps in this eager server. The clear local CUDA Graph decode separation is
largely hidden by the eager execution/host path. A local mechanism signal does not automatically
survive the serving stack. Prefix-reuse separation was measured locally, not validated as an HTTP detector.

This experiment does not cover public-network queues, production dynamic batching, cross-region
networks, or adaptive providers. Server logs are diagnostic evidence, not hidden client-detector features.

## Conclusions and remaining work

{{CONCLUSION}}

Weight/KV precision probes, cross-hardware transfer, capacity frontiers, and full adaptive evasion
remain future work in this performance-probe series. These BF16 results are not quantization-detection results.

## Reproduction and verification

{{REPRODUCE}}

The archive retains raw records, plans, manifests, package lists, logs, compressed CUDA traces,
source snapshots, the model configuration, figures, and supplementary measurements.
Model weights and access credentials are not included.
