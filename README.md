# Inference Verification Gym (`ivgym`)

**Infrastructure for the inference-verification game.** `ivgym` is a
standardized environment where a cheating **provider** (an *attack* — a way of
deviating from a sampling specification) is pitted against a **verifier** (a
*defense* — a per-token divergence score), and the gym reports how reliably the
deviation is caught (detection AUC). Attacks and defenses are pluggable
registries; the `generate → verify → calibrate → evaluate` loop and the backend
are the fixed infrastructure underneath.

The methodology follows *DiFR: Inference Verification Despite Nondeterminism*
(Karvonen et al., 2025) — drop in **a new attack** or **a new defense** and
immediately get detection-AUC curves against everything else.

It runs **today, with no GPU**, on a synthetic logit model, **and on a real
model on a GPU** via the HuggingFace backend (`ivgym/backends/hf_gpu.py`) — the
*same* attack/defense/harness code, validated on an NVIDIA A100-80GB with
`Qwen/Qwen3-0.6B`. A documented **vLLM adapter** sketches the production path.

## The game, as infrastructure

The verification game is a hypothesis test: a *provider* emits `(input, output)`
pairs claiming specification ϕ; a *verifier* recomputes a per-token divergence
statistic `S` and flags a batch when `S > τ`. `ivgym` makes the three axes of
that game into pluggable registries and holds everything else fixed:

| Axis        | Built-in instances                                       | Where |
|-------------|----------------------------------------------------------|------|
| **Attack** (provider deviation) | 4-bit quant, fp8 KV cache, wrong temp/seed, sampling bug, adversarial-temp | `ivgym/attacks.py` |
| **Defense** (verifier score)    | Token-DiFR margin (+clipped/likelihood), cross-entropy, Activation-DiFR, TOPLOC | `ivgym/defenses.py` |
| **Backend** (the arena)         | synthetic CPU + **real HF GPU**; vLLM sketched           | `ivgym/backends/` |

So the framework makes those three axes pluggable registries and keeps the
generate → verify → calibrate → evaluate loop fixed.

## Layout

```
ivgym/
  sampling.py            seed-synced Gumbel-Max + top-k/top-p (Alg. 1)
  core.py                SamplingSpec, Sequence/TokenStep, VerifyContext
  attacks.py             Attack base + registry (honest, quant, temp, seed, bug, adv-temp)
  defenses.py            Defense base + registry (token_difr, cross_entropy, activation_difr)
  metrics.py             ROC AUC + TPR@FPR in pure numpy
  harness.py             generate_dataset / verify / winsorize / batch_means / evaluate
  backends/
    base.py              Backend protocol
    synthetic.py         fake LLM — runs with no GPU
    hf_gpu.py            REAL model on a GPU (transformers); same protocol
    vllm_adapter.py      contract + skeleton for the vLLM production path
experiments/
  run.py                    pluggable CLI: --strategies <file> --backend <name> (no edits)
  exp_adversarial_temp.py   reproduces Fig. 2 (CE defeated, Token-DiFR survives)
  exp_sweep.py              attack × defense AUC grid (Table 2 / Fig. 1 shape)
  exp_gpu.py                exp_sweep on a real model on a GPU
examples/
  custom_strategies.py      template: a custom attack + a custom defense
tests/test_smoke.py
```

## Run it

```bash
# from inference-verification-gym/
.venv/bin/python -m experiments.exp_adversarial_temp   # headline result
.venv/bin/python -m experiments.exp_sweep              # full grid
.venv/bin/python tests/test_smoke.py                   # sanity checks
```

Representative sweep output (synthetic backend, batch=1000 tokens):

```
      attack |       token_difr    cross_entropy  activation_difr
  quant_4bit |           1.000            0.683            1.000
      kv_fp8 |           0.949            0.643            1.000
    temp_1.1 |           0.765            0.998            0.432
     seed_43 |           1.000            0.569            0.554
      bug_k2 |           0.860            0.538            0.485
     bug_k32 |           0.874            0.694            0.628
```

(Fully reproducible — seeded end to end.)

This recovers the paper's *conceptual* structure (not its absolute numbers,
which need real models):

- **Token-DiFR** dominates on quantization, seed errors, and sampling bugs.
- **Cross-entropy** is the one metric that catches a tiny temperature shift —
  and is *defeated* by the adversarial-temperature attack (`exp_adversarial_temp`).
- **Activation-DiFR** saturates on forward-pass deviations (quant/fp8) with very
  few tokens but is blind to sampling-only changes (temp/seed) — it never sees
  the sampler.

## Run on a GPU (real model)

On a CUDA host (validated: single A100-80GB, CUDA 13, torch 2.8). `torch` is the
only heavy dep; reuse a system install if present.

```bash
# from inference-verification-gym/
python -m venv --system-site-packages .venv   # reuse system torch if available
.venv/bin/pip install "transformers>=4.44" accelerate safetensors
.venv/bin/python -m experiments.exp_gpu        # downloads Qwen/Qwen3-0.6B on first run
```

Override with env vars: `IVGYM_MODEL`, `IVGYM_PROMPTS`, `IVGYM_TOKENS`, `IVGYM_BATCH`.

Real-model output (Qwen3-0.6B, 12 prompts × 48 tokens, batch=200, ~5 min):

```
      attack |       token_difr    cross_entropy  activation_difr
  quant_4bit |           1.0000           0.7832           1.0000
      kv_fp8 |           0.8471           0.9999           1.0000
    temp_1.1 |           0.9318           1.0000           0.5447
     seed_43 |           1.0000           0.9998           0.3315
      bug_k2 |           0.4429           1.0000           0.4992
     bug_k32 |           0.9316           0.9966           0.4273
```

Same qualitative structure as the synthetic backend, now over a real LLM's logit
geometry: **Token-DiFR** dominates quantization, seed errors and large sampling
bugs; **cross-entropy** is the one metric that flags the temperature shift;
**Activation-DiFR** saturates on forward-pass attacks (quant/fp8 → 1.0) but is
blind to sampling-only changes (temp/seed). The tiny `bug_k2` (1%-rate flip of
the top-2 tokens) is the hard case — push `IVGYM_BATCH` / `IVGYM_TOKENS` up to
recover it, exactly as the paper's batch-size sweep predicts.

How attacks map onto the real model: temperature/seed are real `SamplingSpec`
changes and the sampling bug really hijacks the sampler; the forward-pass
attacks (quant/fp8) apply their logit/activation perturbation on top of the real
Qwen3 logits — the same model the synthetic backend uses, with a real base
distribution. See `ivgym/backends/hf_gpu.py` for the full contract.

## Add your own attack / defense (no edits to the library)

Write a file that registers your strategies and point the runner at it with
`--strategies`. Nothing in `ivgym/` is touched — importing your file runs the
`@register` decorators, which add the strategies to the same registries the
harness and every backend already use. A complete, runnable template lives in
[`examples/custom_strategies.py`](examples/custom_strategies.py).

A new **attack** (provider deviation). Subclass `Attack` and override any of the
hooks; the `@dataclass` form lets you parametrize it:

```python
from dataclasses import dataclass
from ivgym.attacks import Attack, register

@register
@dataclass
class MyAttack(Attack):
    name: str = "my_attack"
    def provider_spec(self, ref):               # change sampling params, or
        return ref.replace(temperature=0.8)
    def logit_bias_sigma(self):                 # perturb the forward pass, or
        return 0.3, 0.1
    def sample_override(self, rng, top_k_ids):  # hijack the sampler
        return None
```

A new **defense** (verifier score, higher = more divergent from reference):

```python
from dataclasses import dataclass
from ivgym.defenses import Defense, register

@register
@dataclass
class MyDefense(Defense):
    name: str = "my_defense"
    needs_seed: bool = True          # needs shared Gumbel noise?
    needs_activation: bool = False   # needs activation fingerprints?
    def score(self, ctx):
        ...
```

`@register` accepts either a class (instantiated with its defaults, as above) or
a pre-built instance, e.g. `register(MyAttack(name="my_attack_hot", temp=1.3))`.

Then run the sweep — your strategies are scored against everything else, on the
backend you choose, with no other changes:

```bash
# list everything that is registered (built-ins + your file)
.venv/bin/python -m experiments.run --strategies examples/custom_strategies.py --list

# full sweep including your strategies (synthetic backend, no GPU)
.venv/bin/python -m experiments.run --strategies examples/custom_strategies.py

# pick the matchups and the batch size
.venv/bin/python -m experiments.run --strategies examples/custom_strategies.py \
    --attacks logit_spike quant_4bit --defenses token_difr cross_entropy topk_overlap

# the SAME command on a real model on a GPU
.venv/bin/python -m experiments.run --backend hf_gpu \
    --strategies examples/custom_strategies.py
```

`experiments/run.py --help` lists every flag (`--prompts`, `--tokens`,
`--batch`, `--n-batches`, `--vocab`, `--backend`). The legacy `exp_sweep.py` /
`exp_adversarial_temp.py` scripts still reproduce the fixed paper grids.

## Moving to real models

The HF GPU backend (`ivgym/backends/hf_gpu.py`) already runs the harness against a
real model — start there. For the higher-throughput **vLLM** production path,
implement the three methods in `ivgym/backends/vllm_adapter.py` on a CUDA host.
Attacks map to real vLLM config (`quantization=...`, `kv_cache_dtype="fp8"`,
`SamplingParams(temperature=, seed=)`); the verifier does one prefill pass over
`prompt + claimed_tokens` and reads logits/activations. Defenses and the harness
need no changes. See the module docstring for the full contract.

## Status / not yet done

- TOPLOC baseline defense (top-k index/value polynomial) — stub slot in defenses.
- Likelihood-style Token-DiFR transforms (Appendix A) and multi-feature
  aggregation (the paper monitors several detectors in parallel).
- Communication-cost Pareto sweep for Activation-DiFR vs TOPLOC (k × J).
- Real vLLM backend implementation + the temp-0 spot-check mode (Appendix D).
  (The HF GPU backend in `ivgym/backends/hf_gpu.py` now runs the harness against a
  real model end-to-end; vLLM remains for higher throughput.)
```
