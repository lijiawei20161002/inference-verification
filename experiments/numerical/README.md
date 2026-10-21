# Numerical diagnostics

These tools support the numerical causal diagnosis study. Archive inspection
and frozen-logit replay use Python and NumPy without a GPU. BF16 capture and the
detached E0 campaign require the pinned CUDA/model environment.

[Current implementation plan](../../planning/NEXT_EXPERIMENTS.md) ·
[Full E0–E5 experimental design](../../docs/NUMERICAL_CAUSAL_DIAGNOSIS_EXPERIMENTS.md) ·
[Experiment map](../README.md)

## Code map

| Component | Responsibility | Status |
|---|---|---|
| [score_tails.py](score_tails.py) | Validate an existing archive, summarize evaluation tails, export capped-event identities | Implemented |
| [sampler_contract.py](sampler_contract.py) | Illustrate the preserved NumPy sampling conventions on synthetic inputs | Implemented |
| [Archive loader](../analyze_real_quantization.py) | Check source hashes, captures, dimensions, corpus splits, and reference reruns | Existing, reused |
| [Regression checks](../../tests/test_numerical_diagnostics.py) | Protect event identity, split boundaries, score interpretation, and legacy sampling behavior | Implemented |
| [BF16 capture](capture_bf16.py) | Observe exact logits, actual RNG/support, masks/positions and KV states; preserve the archived runner | M1 passed on the historical evaluation corpus and selected forwards |
| [Frozen-logit replay](frozen_logits.py) | CPU dtype/nucleus/sampling matrix on captured evidence | Implemented; production sampling unchanged |
| [Detached campaign](run_e0_campaign.py) | Frozen source, independent processes, equivalence gates and persistent status/logs | Implemented |
| [Event-shape dispatch follow-up](run_dispatch_campaign.py) | Complete selected-forward traces, independent repeats and unprofiled controls | Validated on L40S; [reviewed report](../../docs/results/numerical_m1_dispatch_l40s_20261021.md) |
| Interventions, localizer, witness checker | Explain residual numerical events with validated witnesses | Planned |

Imports perform no archive I/O. The slides script
[`analyze_score_tails.py`](../../slides/mlsys-paper-plan/analyze_score_tails.py)
delegates to these modules for its derived presentation artifact.

## Run the CPU preparation

From the repository root, in an environment with NumPy installed:

```sh
mkdir -p runs/numerical-e0
python3 -m experiments.numerical.sampler_contract \
  --output runs/numerical-e0/sampler-contract.json
python3 -m experiments.numerical.score_tails \
  docs/results/real_quantization_l40s_20261018/pilot \
  --output runs/numerical-e0/score-tails.json \
  --events runs/numerical-e0/capped-events.jsonl
```

Output files must not already exist. Use a new directory name when repeating
the commands. The archive and existing outputs are never overwritten by these
CLIs. The sampler command also accepts no arguments to print its JSON to stdout.

With pytest installed, run the focused checks:

```sh
python3 -m pytest tests/test_numerical_diagnostics.py tests/test_real_quantization.py -q
```

`experiments` is repository tooling, not part of the installed `ivgym` package;
run these commands from a source checkout.

## Interpret the outputs

`score-tails.json` records the validated source hashes, calibration/evaluation
counts, score caps, and per-arm score concentration. The analyzer reads the
split and sequence length from the plan; it does not assume 96 sequences or
128 tokens. It currently supports only the real-quantization v1 archive and
its token-DiFR cap of 30 and local token-rank cap of 50.

For the 18 October archive, the evaluation-only capped counts are BF16 **49**,
NF4 **252**, and INT8 **94**, producing **395** JSONL event records. Each record
includes arm, corpus/evaluation indices, prompt ID, sampling seed, generated-token
position, and raw score values. Positions are zero-based and exclude the prompt.
The summary's source hashes establish which archive the event identities refer
to; retain it alongside the ledger.

BF16's capped positions contribute about 98.4% of its score mass, but this is
neither a false-positive rate nor proof of support mismatch. A cap can arise
from a filtered-out token or a clipped finite margin. Every ledger event is
therefore labeled `unresolved_from_scores`. The `token_toploc` archive field is
a local rank baseline, not a reproduction of the original TOPLOC system.
Zero total mass is represented as JSON `null` for the mass fraction.

`sampler-contract.json` contains two hand-constructed examples: a top-k boundary
can amplify a small logit change, and the legacy top-p path excludes the token
crossing the probability threshold except when that token is first. The latter
behavior is preserved; only the historical misleading code comment was corrected.
These examples do not explain any particular archived GPU event or prove
cross-engine RNG compatibility.

The [20 October E0 report](../../docs/results/numerical_e0_l40s_20261020.md)
records reacquired paired evidence for 49 historical capped events and 64
controls. All 49 remain disagreements with matching arithmetic under the legacy
sampler; the crossing-inclusive variant leaves 6/49 disagreements. These are
frozen-logit discovery results, not causal attribution or fresh evaluation.

## Detached E0 campaign

Use the pinned HF environment and model revision in the implementation plan.
After the model is cached, launch from the repository root with a **new** output
directory:

```sh
.venv/bin/python -m experiments.numerical.run_e0_campaign \
  --detach --output runs/numerical-e0-new \
  --hf-home /workspace/.cache/huggingface
```

The launcher freezes source files and the discovery archive, then starts a new
OS session with stdin disconnected and stdout/stderr written to
`supervisor.log`. SSH/chat disconnection does not terminate it; the host/container
must remain running. `supervisor.pid`, `launch.json`, and the atomically updated
`status.json` identify the job. Stage-specific logs preserve failures.

The sequence is CPU checks, four-token baseline/capture/profiler controls,
two independent full baseline workers, two independent full capture workers,
then CPU factorial replay. A failed equivalence or evidence-integrity check
stops the campaign. Each full worker uses the 64 archived evaluation prompts
and 128 generated tokens. Historical prefix/token nonmatches remain explicitly
non-reproduced. All current capped positions plus the historical positions and
one preselected control per prompt are captured.

Each event retains native BF16 logit bits, actual provider/reference Gumbel
vectors and support masks, sorted nucleus boundaries, and exact prefix tokens.
State files retain actual attention masks, position/cache IDs and post-forward
KV tensors, with module shapes/strides and lineage metadata. Baseline workers
have no observation hooks. Full returned logits, hidden states, fingerprints,
scores and tokens are hashed for comparison.

The matrix's `original` arithmetic is the **archived sampler** arithmetic:
float32 provider and float64 reference. It does not mean direct BF16 filtering.
The other settings use matching float32 or float64 on both sides. The two nucleus
variants and five sampling modes share identical saved logits and actual noise.
`crossing_inclusive_v1` retains through the first strictly crossing token and
does not change library sampling or recalibrate any detector.

On success, `summary.json`, `replay.jsonl` and `report.md` summarize the discovery
results. The CUDA kernel-name/grid/block control covers only the short smoke
sequence. The separate [21 October dispatch follow-up](../../docs/results/numerical_m1_dispatch_l40s_20261021.md)
passed the event-shape gate for this historical corpus and selected forwards.
A matched-protocol residual is not a hardware fault or a causal witness.

Actual RNG/support arrays are retained for selected events and controls; full
output hashes cover all 8,192 evaluation positions. The committed report and
small summaries do not include the raw tensors. Keep the original `runs/`
directory and its source snapshots for independent replay.

The E0 focused suite passed 51 tests, including the new evidence and campaign
checks:

```sh
python3 -m pytest tests/test_frozen_logits.py tests/test_numerical_diagnostics.py \
  tests/test_real_quantization.py tests/test_e0_campaign.py -q
```

## Detached M1 dispatch follow-up

This campaign regenerates all 64 complete 128-token trajectories in six
fresh processes: unprofiled baseline/capture, followed by two profiled
baseline/capture pairs. It traces each selected provider forward and each full
reference forward, with a fresh profiler per call. The prefix and KV state are
reconstructed naturally before a selected call; the sequence is not shortened
to the event position.

```sh
.venv/bin/python -m experiments.numerical.run_dispatch_campaign \
  --detach --parent-e0 runs/numerical-e0-20261020 \
  --output runs/numerical-m1-dispatch-new \
  --hf-home /workspace/.cache/huggingface
```

Use `--limit 1` with a separate output directory for preflight. A subset cannot
pass the full M1 gate. Full-pool output equality, the recorded parent E0
environment and production source hashes, complete trace coverage and input
shapes, kernel names/grid/block, and captured event/KV artifact equality are
checked. Missing traces, output drift, state drift or dispatch differences stop
the campaign. Raw traces and failures are retained for diagnosis.

`status.json`, `supervisor.pid`, `supervisor.log` and per-worker logs persist
independently of SSH/chat. The host/container must stay running. A successful
full run writes `summary.json`, `report.md`, per-call dispatch signatures and
artifact hashes. Its gate is scoped to the historical evaluation corpus and
selected complete forwards on this HF path; it is not a causal witness or a
guarantee about all kernels, contexts or backends. M3 is not automatically run.

The [reviewed 21 October run](../../docs/results/numerical_m1_dispatch_l40s_20261021.md)
completed in 1,714.96 seconds. All 49 historical events reproduced; each of the
four profiled workers covered 113 provider forwards and 64 reference forwards
(177 traces, 336,299 kernel launches). Both baseline/capture pairs and their
independent repeats matched. Full trajectories matched the unprofiled controls
and parent E0; each profiled capture matched all 290 event/state artifacts from
the unprofiled capture. This passes M1 for the tested scope; M3 has not started.

Run the focused CPU checks, including dispatch coverage and state-drift checks:

```sh
python3 -m pytest tests/test_dispatch_validation.py tests/test_e0_campaign.py \
  tests/test_frozen_logits.py tests/test_numerical_diagnostics.py \
  tests/test_real_quantization.py -q
```

## Cleanup validation — 20 October 2026

The focused suite passed 37 tests. The existing NumPy-core CI command passed
193 tests with two skips in the local environment; the service and audit
modules were excluded, as in that CI job:

```sh
python3 -m pytest tests/ -q \
  --ignore=tests/test_audit_protocol.py \
  --ignore=tests/test_audit_conformance.py \
  --ignore=tests/test_service.py
```

Both CLI outputs and the slides adapter were exercised in a fresh local run
directory. All per-arm summary values and the score archive hash matched the
previous slide diagnostic exactly. The adapter imported without analysis or
file I/O. No fresh GPU measurement was performed.
