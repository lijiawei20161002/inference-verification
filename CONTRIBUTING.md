# Contributing to IVGym

Useful contributions include clearer onboarding, provider adapters, detector
comparisons, and replications on different hardware. Small fixes can go straight
to a pull request. For a new detector or provider integration, describe the
evidence contract and intended test in an issue first.

## Development setup

From a clone of the repository:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[service,audit,test]'
python examples/quickstart.py
python -m pytest tests/ -q
```

The suite runs without model weights or CUDA. It covers the NumPy core, service
boundaries, signed-artifact protocol, and checks on committed research evidence.
GPU measurements are separate experiments, not part of these tests.

The [CI workflow](.github/workflows/tests.yml) also exercises the NumPy-only
installation so the laptop entry point stays lightweight.

## Places to start

| Contribution | Scope | GPU needed? |
|---|---|---|
| Improve a confusing step in the quickstart | Reproduce the problem in a fresh environment and document the working command | No |
| Explain an API response | Add an annotated example with its evidence provenance and calibration status | No, for saved captures |
| Add a clock-artifact consistency check | Extend `tests/test_claims.py` to tie a named clock finding to saved evidence | No |
| Add a research strategy | Copy `examples/custom_strategies.py`; follow [the extension guide](docs/EXTENDING.md) | To evaluate on a model |
| Replicate a benign variation result | Record reference identity, serving shape, controls, and effect sizes | Yes |

The [roadmap](ROADMAP.md) lists larger open problems and links their research
designs. These are proposed tasks, not claims that a feature already exists.

## Pull requests

Explain the problem, resulting behavior, and how you checked it. Keep changes
focused. Add regression coverage when changing runtime behavior, and execute
any new quickstart commands exactly as written. Documentation fixes do not
need artificial tests.

Keep raw credentials, provider keys, and private prompts out of examples and
artifacts. The `runs/` directory is ignored for local outputs.

## Research results

Every detection claim should link to a committed artifact and a reproduction
command. Record:

- Model/revision, hardware, precision, serving backend, and sampling settings.
- Honest control, calibration split, evaluation pool, and batch/pool ratio.
- Whether the attack changes real weights or simulates a logit perturbation.
- Detection statistic, uncertainty, and verification cost.
- Timing units and observation provenance for clock measurements.

Preserve negative results and superseded measurements with an explanation.
Avoid overwriting a result from one attack with a different attack's output.
The existing claim checks in `tests/test_claims.py` illustrate why these details
matter; [the evaluation protocol](docs/RESEARCH_GUIDE.md#the-evaluation-protocol)
explains the statistical constraints.
