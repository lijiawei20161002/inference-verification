# Examples

[Project home](../README.md) · [API guide](../docs/VERIFICATION_SERVICE.md)

Run commands from the repository root after installing the project into your
active Python environment.

| Example | What it demonstrates | Requirements |
|---|---|---|
| [quickstart.py](quickstart.py) | Replays measured full-context and truncated-context timing through the clock verifier | `pip install -e .`; no GPU, keys, or downloads of model weights |
| [verify-clock.json](verify-clock.json) | Sends illustrative paired intervals to the API | `.[service]` and a running verifier; no GPU |
| [verify-tokens.json](verify-tokens.json) | Scores submitted exact token IDs | `.[service,gpu]`, CUDA, configured reference checkpoint |
| [verify-live.json](verify-live.json) | Collects and scores evidence from a live `/completions` provider | Configured reference, allowed provider URL, running target |
| [api-samples/](api-samples/README.md) | Five actual H100 request/response pairs | Nothing to read them; matching operator setup to reproduce live requests |
| [custom_strategies.py](custom_strategies.py) | Registers a custom attack and detector | NumPy to list strategies; CUDA to evaluate them |
| [safe_set_strategies.py](safe_set_strategies.py) | Explores an evasion of seed-synchronized verification | GPU research harness |
| [seed_free_strategies.py](seed_free_strategies.py) | Explores quantization plus temperature retuning against a seed-free detector | GPU research harness |
| [audit-spec.template.json](audit-spec.template.json) | Configures the optional signed-artifact experiment | [Audit setup](../docs/AUDIT_IMPLEMENTATION.md) |

## Laptop demo

```sh
python -m pip install -e .
python examples/quickstart.py
```

The conditions come from a saved experiment. The demo reports scores, without
fitting a calibration or assigning a provider verdict.

## Clock API

Start the service with `ivgym serve` after installing `.[service]`. From another
terminal:

```sh
curl http://127.0.0.1:8000/v1/verify \
  -H 'Content-Type: application/json' \
  --data-binary @examples/verify-clock.json
```

The default service has no trusted calibration, so this returns `inconclusive`.
See [the API guide](../docs/VERIFICATION_SERVICE.md) to configure models,
providers, and calibration. [service.h100.json](service.h100.json) records the
original machine's setup; its checkpoint path must be changed on another host.
