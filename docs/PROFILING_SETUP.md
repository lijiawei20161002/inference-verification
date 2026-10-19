# PyTorch Profiler and Nsight Systems on the L40S

Configured and smoke-tested on 19 October 2026. Both tools captured the pinned
Qwen3-1.7B vLLM server, including CUDA kernels and prefill/decode iteration labels.
These are separate diagnostic runs, not additional timing-pilot samples.

## Installed environment

| Component | Configuration |
|---|---|
| PyTorch Profiler | Included in the existing PyTorch **2.11.0+cu130** environment |
| Server Python | `/workspace/.venvs/ivgym-vllm-0.22.0/bin/python` |
| Serving engine | vLLM **0.22.0**, Transformers **4.57.6** |
| Nsight Systems CLI | **2026.5.1.161**, extracted under `/workspace/.tools/nsight-systems/2026.5.1` |
| Nsight command | `/workspace/.tools/bin/nsys`; also linked at `/usr/local/bin/nsys` on this machine |
| GPU | NVIDIA L40S, driver **595.91.07** |

The [installer](../scripts/install_nsight_systems.py) downloads the fixed CLI
package from NVIDIA's official Ubuntu 24.04 repository and checks its SHA-256
before extracting it. It does not upgrade drivers, CUDA, PyTorch, or the server
environment. The package hash is
`61829db6392e5c293ada1319df86a97c3356bc810335a1975cb551fcfe3eca08`.

On this machine, `nsys --version` works directly. To reproduce the isolated
installation on another matching Ubuntu x86-64 host:

```sh
.venv/bin/python scripts/install_nsight_systems.py
. scripts/profiling_env.sh
nsys --version
```

The installer creates the workspace launcher; the activation script exposes it
without requiring a system-wide symlink. See the
[NVIDIA installation guide](https://docs.nvidia.com/nsight-systems/InstallationGuide/).

## Capture a small serving workload

Run from the repository root with an idle GPU. Output directories must be new.
Run the two commands sequentially; the harness uses the existing exclusive host
lock and refuses to start if another compute process is already using the GPU.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.profile_vllm_serving \
  --profiler torch --output runs/profile-torch

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.profile_vllm_serving \
  --profiler nsys --output runs/profile-nsys
```

Defaults: optimized compiler/CUDA Graph execution, prefix caching off, 8,192
input tokens, four concurrent requests, and 32 generated tokens per request.
The harness retains the previous serving pilot's model revision, FLASH_ATTN
backend, 12 GiB KV budget, and 2,048-token scheduler budget. It binds the API only
to `127.0.0.1`, warms up before capture, calls `/start_profile`, submits the
requests, calls `/stop_profile`, and shuts down its own server. Under Nsight it
stops the server before waiting for the profiler to flush the report.

Use `--config eager_off` for eager execution; `--arm truncate` for truncation.
For prefix-cache controls, use `--config graph_on --arm cold` or
`--config graph_on --arm hit`. Cache reset and hit priming occur before capture.
Other controls are `--length`, `--concurrency` (1–8), and `--tokens`.
The installation smoke tests cover the default optimized, cache-off case.

The follow-up runner also accepts `--scheduler-budget` (default 2,048) and
`--block` (default 0) to vary the scheduler limit and prompt offset explicitly.
`--profiler none` runs an uninstrumented control with the same server lifecycle,
warm-up, cache checks, and response validation. It does not enable a collector or
call profiling endpoints. For the frozen multi-condition study, use
`python -m experiments.run_scheduling_campaign --output runs/scheduling-new`;
see the [scheduling report](results/scheduling_l40s_20261019.md).

PyTorch configuration enables CPU/CUDA activity recording, compressed Chrome
traces, and worker iteration annotations. Stack capture, tensor-shape recording,
and memory profiling are disabled to limit overhead. Nsight configuration is:

```text
--trace=cuda,nvtx,osrt
--sample=none --cpuctxsw=none
--discard-environment=true
--trace-fork-before-exec=true --cuda-graph-trace=node
--capture-range=cudaProfilerApi --capture-range-end=repeat
```

Future profiling and export processes receive only allowlisted runtime paths,
localization variables, and explicit experiment settings. The launcher also uses
`--discard-environment=true`. CPU-only live smoke tests still found credential
strings in raw reports with this flag, including with an allowlisted launch
environment. These options are not a publication-safety guarantee: original
reports remain private and every public SQLite export must be sanitized. The original smoke
and scheduling sources retain the commands actually used. Their public SQLite
exports have environment metadata removed, with unchanged kernel, runtime, and
NVTX measurement tables; `PUBLICATION.json` records this transformation.
Original `.nsys-rep` files are retained in ignored local storage and are excluded
from the published archives.

The worker multiprocessing method is explicitly `spawn`. The vLLM profiler
backend is `torch` for PyTorch runs and `cuda` for Nsight runs. The two CUPTI
collectors are run separately. These settings follow the
[pinned vLLM profiling interface](https://docs.vllm.ai/en/v0.22.0/contributing/profiling/).
Profiled request timings include instrumentation overhead and must remain
separate from the unprofiled serving campaign.

## Verified artifacts

The [setup archive](results/profiling_setup_l40s_20261019/) preserves both captures,
raw request frames, commands, source snapshots, manifests, environment checks,
and installation provenance. Each tool profiled four requests with 32 output
tokens each after a separate four-request warm-up.

| Tool | CUDA kernel events | CPU/NVTX iteration ranges | Artifacts |
|---|---:|---:|---|
| PyTorch Profiler | 18,278 | 50 CPU ranges | Compressed worker/frontend Chrome traces and operator summary |
| Nsight Systems | 18,135 | 50 matching NVTX ranges | Sanitized SQLite export and kernel/NVTX statistics; original `.nsys-rep` retained locally |

The PyTorch trace additionally contains **48 projected GPU iteration annotations**.
Those are not another 48 engine iterations. The original smoke manifest's
`iteration_ranges=98` counted both CPU and GPU annotations; the revised read-only
checker separates them, and the archive's `verification.json` records that
correction without modifying the captured trace or original manifest.
Nsight recorded 51 total NVTX events, of which 50 have the vLLM iteration label.

For example, `execute_context_1(2047)_generation_1(1)` identifies an annotated
iteration with context work and generation work. This validates that the
instrumentation needed for the next scheduling study is present; it is not yet
a causal attribution of the previous timing results. Kernel counts may differ
between independently profiled runs with asynchronous scheduling.

Both server shutdowns completed without forced termination, and the GPU was
released. Both logs include a Python resource-tracker warning about one leaked
semaphore; the reports were saved and validated successfully.

## Read and verify the captures

PyTorch `.pt.trace.json.gz` files can be decompressed and opened in a Chrome trace
viewer such as Perfetto. The worker trace contains GPU activity; a small frontend
trace alone is not evidence that GPU capture succeeded.

Open locally retained `.nsys-rep` files using the corresponding Nsight Systems
desktop application, or inspect the published sanitized SQLite and CLI summary:

```sh
nsys stats --report=cuda_gpu_kern_sum,nvtx_sum \
  docs/results/profiling_setup_l40s_20261019/nsys_smoke_01/traces/serving.1.sqlite
```

Offline validation requires only the repository's normal Python environment;
it does not need a GPU, model weights, the vLLM environment, or `nsys`:

```sh
.venv/bin/python -m experiments.profile_vllm_serving check \
  docs/results/profiling_setup_l40s_20261019/torch_smoke_01
.venv/bin/python -m experiments.profile_vllm_serving check \
  docs/results/profiling_setup_l40s_20261019/nsys_smoke_01
(cd docs/results/profiling_setup_l40s_20261019 && sha256sum -c SHA256SUMS)
```

The checker verifies hashes, exact request fixtures, stream/output budgets, and
the presence of both CUDA kernels and iteration annotations. Validation rejects
a CPU-only trace. `tests/test_profiling_setup.py` also checks that projected GPU
annotations do not double-count CPU iteration ranges.

## Current host permission limits

`nsys status --environment` reports that `perf_event_open` and CPU sampling are
unavailable, with `kernel.perf_event_paranoid=4`. CPU instruction sampling and
OS context-switch tracing are therefore explicitly disabled. CUDA runtime,
CUDA kernel, NVTX, and OS-runtime API tracing worked in the tested configuration;
PyTorch CPU operator activity also worked.

The NVIDIA driver reports `RmProfilingAdminOnly: 1`; earlier Nsight Compute
hardware-counter collection was permission-blocked. This setup has not changed
host-driver permissions and does not claim DRAM, L2, or occupancy counters.
Enabling those requires the host administrator's configuration; see
[NVIDIA's permission guidance](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters).
