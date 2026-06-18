# DiFR: shared attack/defense harness for inference verification

A small, backend-agnostic framework for prototyping **inference-verification**
attacks and defenses, based on *DiFR: Inference Verification Despite
Nondeterminism* (Karvonen et al., 2025). The goal is a single harness where you
can drop in **a new attack** (a way a provider deviates from spec) or **a new
defense** (a divergence-from-reference score) and immediately get
detection-AUC curves against everything else.

It runs **today, with no GPU**, on a synthetic logit model, and exposes a
documented **vLLM adapter** so the *same* attack/defense code runs against real
models on a CUDA host.

## Why this design

DiFR is a hypothesis test: a *provider* emits `(input, output)` pairs claiming
specification ϕ; a *verifier* recomputes a per-token divergence statistic `S`
and flags a batch when `S > τ`. Everything in the paper is a choice of:

| Axis        | Paper instances                                          | Here |
|-------------|----------------------------------------------------------|------|
| **Attack**  | 4-bit quant, fp8 KV cache, wrong temp/seed, sampling bug, adversarial-temp | `difr/attacks.py` |
| **Defense** | Token-DiFR margin (+clipped/likelihood), cross-entropy, Activation-DiFR, TOPLOC | `difr/defenses.py` |
| **Backend** | vLLM / HF across GPUs                                    | `difr/backends/` |

So the framework makes those three axes pluggable registries and keeps the
generate → verify → calibrate → evaluate loop fixed.

## Layout

```
difr/
  sampling.py            seed-synced Gumbel-Max + top-k/top-p (Alg. 1)
  core.py                SamplingSpec, Sequence/TokenStep, VerifyContext
  attacks.py             Attack base + registry (honest, quant, temp, seed, bug, adv-temp)
  defenses.py            Defense base + registry (token_difr, cross_entropy, activation_difr)
  metrics.py             ROC AUC + TPR@FPR in pure numpy
  harness.py             generate_dataset / verify / winsorize / batch_means / evaluate
  backends/
    base.py              Backend protocol
    synthetic.py         fake LLM — runs with no GPU
    vllm_adapter.py      contract + skeleton for real models
experiments/
  exp_adversarial_temp.py   reproduces Fig. 2 (CE defeated, Token-DiFR survives)
  exp_sweep.py              attack × defense AUC grid (Table 2 / Fig. 1 shape)
tests/test_smoke.py
```

## Run it

```bash
# from inference-verification/
../.venv/bin/python -m experiments.exp_adversarial_temp   # headline result
../.venv/bin/python -m experiments.exp_sweep              # full grid
../.venv/bin/python tests/test_smoke.py                   # sanity checks
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

## Add a new attack

```python
from difr.attacks import Attack, register

@register
class MyAttack(Attack):
    name = "my_attack"
    def provider_spec(self, ref):           # change sampling params, or
        return ref.replace(temperature=0.8)
    def logit_bias_sigma(self):             # perturb the forward pass, or
        return 0.3, 0.1
    def sample_override(self, rng, top_k_ids):  # hijack the sampler
        return None
```

## Add a new defense

```python
from difr.defenses import Defense, register

@register
class MyDefense(Defense):
    name = "my_defense"
    needs_seed = True          # needs shared Gumbel noise?
    needs_activation = False   # needs activation fingerprints?
    def score(self, ctx):      # higher = more divergent from reference
        ...
```

Both are picked up automatically by `exp_sweep.py` and the harness.

## Moving to real models

Implement the three methods in `difr/backends/vllm_adapter.py` on a CUDA host.
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
```
