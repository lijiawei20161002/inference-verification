# Numerical diagnostics

These tools prepare the numerical causal diagnosis study. They run on saved
evidence with Python and NumPy; they do not load a model or require a GPU.

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
| GPU capture, interventions, localizer, witness checker | Collect and explain new numerical evidence | Planned; no runnable entry points yet |

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

Full paired logits, actual support masks, random tensors, and valid intermediate
state must be reacquired by the planned capture adapter before causal diagnosis.

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
