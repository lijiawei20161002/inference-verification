# vLLM serving performance probes on L40S

19 October 2026. Follow-up to the [local mechanism pilot](results/performance_probes_l40s_20261019.md).
Status: preflight passed; the protocol below is frozen before the formal pilot.

## Question

Does the local context-length signal remain visible through an optimized serving engine,
and can legitimate prefix reuse or execution-mode changes explain the observed response?

The preceding experiment found large implementation-dependent differences and an eager HTTP
path that largely hid the decode signal. This follow-up changes the serving stack on the same
GPU, then varies controlled settings within that stack. It does not establish deployment
false-positive or detection rates.

## Fixed configuration

- One exclusive NVIDIA L40S; one server at a time, with the existing host lock.
- Qwen3-1.7B revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e; BF16 weights and KV.
- Isolated vLLM 0.22.0, Transformers 4.57.6, PyTorch 2.11.0+cu130, Python 3.12.
  The prior stack used PyTorch 2.8/cu128, so across-round differences are a combined software-stack change.
- Explicit FLASH_ATTN backend; maximum model length 16,512; maximum eight active sequences.
- Fixed 12 GiB KV budget, 16-token cache blocks, chunked prefill enabled, 2,048-token scheduler budget.
- Greedy generation, fixed 128 output tokens, ignore EOS. Unlike the first local pilot, this is
  free generation rather than frozen-continuation replay.
- Loopback OpenAI-compatible completions API, token-ID prompts and returned token IDs.
  This exercises the serving engine and HTTP transport but excludes text tokenization and a public network.
- Client clocks start after request submission finishes. Connection establishment is excluded;
  send time, raw SSE frames, and response usage are retained separately.
- A received SSE event may contain multiple tokens. Post-first-content time is normalized by
  the number of tokens arriving after the first content event; it is not a GPU step timer.

The complete resolved package list, model configuration hash, source snapshot, commands,
server logs, GPU snapshots, and compressed raw records accompany each run.

## Conditions and planned coverage

| Server configuration | Intended contrast |
|---|---|
| graph_off | Default compiler/CUDA Graph optimizations, prefix cache disabled |
| graph_on | Same optimized path, automatic prefix caching enabled |
| eager_off | Eager execution, prefix cache disabled |

The eager flag disables compiler/graph optimizations together; it does not isolate CUDA Graphs alone.
The graph labels denote requested configuration. Server logs must confirm the effective backend
and graph configuration; unsupported or silently changed settings are reported.

Three fresh process sessions per configuration use two text-offset blocks each: six paired blocks.
Configuration order rotates across sessions. Measurement order is randomized within each server run.
Blocks share a public text corpus and are not independent deployments.

For graph_off and eager_off:

- Prompt lengths: 512, 2,048, 8,192, 16,384.
- Client concurrency: 1 and 4, released through a client-side barrier.
- Full prompt versus vLLM preprocessing that keeps the last 2,048 prompt tokens.
- The wire prompt stays identical. Truncated positions restart at zero, unlike the absolute-position
  policy in the first local pilot. L≤2,048 is the no-op intervention control.
- Client concurrency C is not asserted to equal the internal scheduler batch B.

For graph_on:

- Prompt lengths: 2,048 and 8,192; client concurrency 1 and 4.
- Explicit cache reset before every group. The reset route requires VLLM_SERVER_DEV_MODE=1;
  this is enabled consistently for all configurations, with the server bound only to loopback.
  A successful HTTP reset is not itself proof of eviction, so returned cached-token counts are checked.
- Cold requests versus hit requests whose first L−128 input tokens are primed separately.
- Priming uses one output token and is outside the measured request; its time is retained.
- Per-request public fixture salts isolate different rows while preserving the paired prefix.
- Returned cached-token counts must be zero for cold/off and positive, bounded by L−128, for hit.
  Metrics and logs provide additional diagnostics; their labels are not client-detector inputs.

Total: **nine server runs, 240 measurement groups, 600 measured requests, 76,800 output tokens**.
Sixty priming requests and 45 warm-up requests are excluded from measured-request summaries.
The preflight uses a separate output directory and a smaller token budget.

## Analysis and falsification

Analyze paired block/group medians, preserve raw denominators, and report observed block ranges.
Do not treat concurrent requests or tokens from the same group as independent deployments.

1. Compare full/truncated post-first-content rates and TTFT at each L/C.
2. Compare cold/hit TTFT and decode response, with actual cached-token counts.
3. Compare optimized/eager curves at matched inputs and client concurrency.
4. Report token-sequence agreement for no-op truncation and off/cold/hit controls.
   Floating-point scheduling differences may change greedy output; equality is measured rather than assumed.

Reject the measurement contract if prompt IDs, output budgets, cache controls, or source coverage
do not match the plan. Preserve failed attempts and errors instead of silently replacing cells.
If optimized serving or legitimate cache/scheduling conditions erase the difference, report that
negative result. No classifier or detection threshold is fit to this pilot.

Hardware performance counters were unavailable in the preceding container run. This serving
experiment does not claim DRAM-traffic, L2-hit-rate, or occupancy measurements.

## Running

Install into a separate environment; preserve the existing HF experiment environment:

~~~sh
uv venv --python python3 /workspace/.venvs/ivgym-vllm-0.22.0
uv pip install --python /workspace/.venvs/ivgym-vllm-0.22.0/bin/python \
  -r experiments/requirements_vllm_serving.txt
~~~

Use the existing prepared model cache. Run the preflight first; each output path must be new:

~~~sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.exp_vllm_serving \
  --suite smoke --output runs/vllm-preflight
~~~

Commit the final source and require a clean working tree before the formal pilot:

~~~sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.exp_vllm_serving \
  --suite pilot --output runs/vllm-serving-pilot
python -m experiments.analyze_vllm_serving runs/vllm-serving-pilot \
  --output runs/vllm-serving-pilot/summary.json
~~~

The harness owns and shuts down its server process groups. It records failures, stops on a failed
configuration, and never runs two GPU server configurations concurrently.

Protocol references:
[vLLM 0.22.0 serving arguments](https://docs.vllm.ai/en/v0.22.0/cli/serve/) and
[pinned completion protocol](https://github.com/vllm-project/vllm/blob/v0.22.0/vllm/entrypoints/openai/completion/protocol.py).
