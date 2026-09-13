# Actual API request and response samples

These five request/response pairs were collected over HTTP on the H100. Every
request returned HTTP 200. Response excerpts below omit identity, reference,
design, observation, and cost fields; the linked JSON files contain the complete
unaltered responses. IDs and runtimes change between calls.

The live examples used a demo provider in one process and an independent verifier
in another. The operator loaded the existing
[19-block demo calibration](../../docs/results/verification_service_h100_calibration.json)
into a temporary verifier. This small fixed-prompt calibration demonstrates the
API decisions; it does not establish a deployment error rate. The existing API
on port 8000 was used for submitted text and historical timing traces and was
left unchanged. Temporary demo processes were stopped after collection.

## Send a request

From the repository root, with the verifier running:

```sh
curl http://127.0.0.1:8000/v1/verify \
  -H 'Content-Type: application/json' \
  --data-binary @examples/api-samples/submitted-text.request.json
```

Use the other request files in the same way. The two live files additionally
require a provider on port 8766 and the matching calibration on the verifier.

## 1. Live token check: honest demo provider

The service calls the provider itself and checks eight returned tokens with its own Qwen3-0.6B reference. `no_deviation_detected` means this check did not flag a difference. DiFR needs exact token IDs and the declared shared sampling contract.

Request:

```json
{
  "model": "qwen3-0.6b",
  "target": {
    "base_url": "http://127.0.0.1:8766/v1",
    "model": "test-provider",
    "sampler_contract": "ivgym_numpy_gumbel_v1"
  },
  "verifiers": [
    "token_difr"
  ],
  "sampling": {
    "temperature": 1.0,
    "top_k": null,
    "top_p": 1.0,
    "seed": 42
  },
  "prompts": [
    "Write an imaginative beginning to a story about a lighthouse:"
  ],
  "max_tokens": 8
}
```

Actual response, selected fields:

```json
{
  "outcome": "no_deviation_detected",
  "evidence_provenance": "service_observed",
  "verifiers": {
    "token_difr": {
      "status": "no_deviation_detected",
      "reason": null,
      "score": 0.003284000799214315,
      "sample_count": 8,
      "p_value": 1.0,
      "threshold": 0.05
    }
  }
}
```

[Full request](live-honest.request.json) · [Full response](live-honest.response.json)

## 2. Live token check: changed sampling temperature

The requested temperature remains 1.0. In the test provider only, the model name `temperature-2` selects a test behavior that actually samples at temperature 2.0. The caller does not supply the score or the decision. The independent verifier detects the difference.

Request:

```json
{
  "model": "qwen3-0.6b",
  "target": {
    "base_url": "http://127.0.0.1:8766/v1",
    "model": "temperature-2",
    "sampler_contract": "ivgym_numpy_gumbel_v1"
  },
  "verifiers": [
    "token_difr"
  ],
  "sampling": {
    "temperature": 1.0,
    "top_k": null,
    "top_p": 1.0,
    "seed": 42
  },
  "prompts": [
    "Write an imaginative beginning to a story about a lighthouse:"
  ],
  "max_tokens": 8
}
```

Actual response, selected fields:

```json
{
  "outcome": "inconsistent",
  "evidence_provenance": "service_observed",
  "verifiers": {
    "token_difr": {
      "status": "inconsistent",
      "reason": null,
      "score": 1.1333009744219735,
      "sample_count": 8,
      "p_value": 0.05,
      "threshold": 0.05
    }
  }
}
```

[Full request](live-temperature-2.request.json) · [Full response](live-temperature-2.response.json)

## 3. Check text already captured by the caller

This request supplies a prompt and answer instead of asking the service to contact a provider. The service reconstructs tokens from text, so the evidence is labeled `client_reported`. It calculates scores, but no matching calibration is installed, so it cannot give a calibrated decision.

Request:

```json
{
  "model": "qwen3-0.6b",
  "verifiers": [
    "cross_entropy",
    "token_toploc"
  ],
  "captures": [
    {
      "prompt": "The capital of France is",
      "output": " Paris."
    }
  ]
}
```

Actual response, selected fields:

```json
{
  "outcome": "inconclusive",
  "evidence_provenance": "client_reported",
  "verifiers": {
    "cross_entropy": {
      "status": "inconclusive",
      "reason": "no_applicable_calibration",
      "score": 0.5229521989822388,
      "sample_count": 2,
      "p_value": null,
      "threshold": 0.025
    },
    "token_toploc": {
      "status": "inconclusive",
      "reason": "no_applicable_calibration",
      "score": 0.0,
      "sample_count": 2,
      "p_value": null,
      "threshold": 0.025
    }
  }
}
```

[Full request](submitted-text.request.json) · [Full response](submitted-text.response.json)

## 4–5. Clock checks: full context versus a 512-token window

These requests submit the first 64 paired intervals from the earlier
[clock experiment](../../docs/results/slope_verifier_window.json). They are
historical GPU measurements labeled `device_token` and `client_reported`, not
fresh network arrival measurements. Both use 256-token and 32,768-token contexts.
The full request files contain all intervals; replay them directly:

```sh
curl http://127.0.0.1:8000/v1/verify \
  -H 'Content-Type: application/json' \
  --data-binary @examples/api-samples/clock-honest.request.json

curl http://127.0.0.1:8000/v1/verify \
  -H 'Content-Type: application/json' \
  --data-binary @examples/api-samples/clock-window-512.request.json
```

The clock score is the negative of the average long-minus-short time gap.
A smaller time gap produces a larger anomaly score. Neither trace receives a
calibrated verdict because this API has no matching clock calibration.

### Full-context trace

```json
{
  "outcome": "inconclusive",
  "evidence_provenance": "client_reported",
  "verifiers": {
    "clock_slope": {
      "status": "inconclusive",
      "reason": "no_applicable_calibration",
      "score": -53.631264062499994,
      "sample_count": 64,
      "p_value": null,
      "threshold": 0.05,
      "details": {
        "delta_itl_ms": 53.631264062499994,
        "short_mean_ms": 7.290699999999999,
        "long_mean_ms": 60.921964062499995,
        "unit": "device_token",
        "interpretation": "A smaller long-minus-short gap increases the anomaly score."
      }
    }
  }
}
```

[Full request](clock-honest.request.json) · [Full response](clock-honest.response.json)

### 512-token-window trace

```json
{
  "outcome": "inconclusive",
  "evidence_provenance": "client_reported",
  "verifiers": {
    "clock_slope": {
      "status": "inconclusive",
      "reason": "no_applicable_calibration",
      "score": -0.36642968750000005,
      "sample_count": 64,
      "p_value": null,
      "threshold": 0.05,
      "details": {
        "delta_itl_ms": 0.36642968750000005,
        "short_mean_ms": 7.290699999999999,
        "long_mean_ms": 7.6571296875,
        "unit": "device_token",
        "interpretation": "A smaller long-minus-short gap increases the anomaly score."
      }
    }
  }
}
```

[Full request](clock-window-512.request.json) · [Full response](clock-window-512.response.json)

## Reproduce the calibrated live examples on this H100

Start the separate test provider in one terminal:

```sh
.venv/bin/python -m experiments.service_demo_provider \
  --snapshot /workspace/model-cache/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --port 8766
```

As the independent service operator, create a demo service configuration and
start a separate verifier on port 8001 in another terminal:

```sh
mkdir -p runs/api-samples
.venv/bin/python - <<'PYCONFIG'
import json
from pathlib import Path
config = json.loads(Path('examples/service.h100.json').read_text())
config['calibration_files'] = [str(Path(
    'docs/results/verification_service_h100_calibration.json'
).resolve())]
Path('runs/api-samples/service.json').write_text(json.dumps(config, indent=2))
PYCONFIG
.venv/bin/ivgym serve --config runs/api-samples/service.json --port 8001
```

Post either live request to that verifier:

```sh
curl http://127.0.0.1:8001/v1/verify \
  -H 'Content-Type: application/json' \
  --data-binary @examples/api-samples/live-honest.request.json

curl http://127.0.0.1:8001/v1/verify \
  -H 'Content-Type: application/json' \
  --data-binary @examples/api-samples/live-temperature-2.request.json
```

The caller supplies a target and requested settings. Reference weights and
trusted calibration remain under the independent operator's control.
