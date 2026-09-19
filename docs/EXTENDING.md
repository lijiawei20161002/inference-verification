# Add an attack or verifier

[Project home](../README.md) · [Contributing](../CONTRIBUTING.md) · [Game specification](GAME.md)

The research runner loads Python files that register strategies. Start with
[examples/custom_strategies.py](../examples/custom_strategies.py): it contains
a logit-spike attack and a simple top-1 mismatch verifier. Importing the file
registers both without changes to the library.

## Inspect strategies without a GPU

From the repository root, in your active environment:

```sh
python -m pip install -e .
python -m experiments.run --strategies examples/custom_strategies.py --list
```

The output includes `logit_spike` and `top1_mismatch_toy`. The `--list` path does
not load a model.

## Run a small comparison on CUDA

```sh
python -m pip install -e '.[gpu]'
python -m experiments.run \
  --strategies examples/custom_strategies.py \
  --attacks logit_spike \
  --defenses token_difr top1_mismatch_toy \
  --prompts 12 --tokens 64 --batch 24
```

The default reference is `Qwen/Qwen3-0.6B`, downloaded on first use. This is a
small demonstration, not enough evidence to rank close detectors. Increasing
the batch without growing the evaluation pool can inflate the reported AUC;
read the [evaluation protocol](RESEARCH_GUIDE.md#the-evaluation-protocol).

## Implement a strategy

An attack subclasses `ivgym.attacks.Attack` and uses `@register` from that module.
Override `provider_spec` for sampler changes, `perturb_logits` for simulated
forward-pass deviations, or `sample_override` for a sampler implementation.
Label simulated perturbations explicitly when reporting results.

A recomputing verifier subclasses `ivgym.verifiers.Tier1Verifier`, implements
`score_token`, and uses the verifier module's `@register`. Declare `needs_seed`
and `needs_activation` according to the evidence it actually uses. The supplied
template shows both registration patterns.

Strategies added here enter the **research harness**. Exposing a detector in the
HTTP service also requires an API schema, an evidence contract, and calibration
handling; registration alone does not make it a public service endpoint.

For a new measured result, include an honest control, the model and serving
configuration, token-pool size, batch/pool ratio, and a named artifact. See the
[contributing guide](../CONTRIBUTING.md#research-results).
