# Independent verification service: H100 integration results

The restructured service passed **135 tests**, including **23 API tests**, and
**nine H100 integration cases**. The provider and verification service ran in
separate processes communicating over HTTP. The verifier loaded its own
Qwen3-0.6B reference; the provider supplied neither trusted logits nor calibration.

The integration cases exercised existing real token captures, the original clock
experiment arrays, fresh live token and clock collection, operator-owned
calibration, calibrated decisions, and asynchronous job submission/polling.

| Live API request | TokenDiFR score | p-value | Outcome |
|---|---:|---:|---|
| Honest control | 0.003284 | 1.0 | `no_deviation_detected` |
| Actual temperature 2 while claiming 1 | 1.133301 | 0.05 | `inconsistent` |

The independent operator collected 19 fresh control requests through
`ivgym calibrate`; the public caller could not alter this null. Each calibration
request used the same fixed story prompt, eight output tokens and the original
NumPy Gumbel sampler contract. The single-detector threshold was 0.05. This small
exercise tests the service workflow and detector reuse; it does **not** estimate
a deployment FPR or attack-detection power.

The clock API reproduced the earlier experiment's statistic exactly on 64
paired device intervals: **−53.631264 ms** for `honest_hi` and **−0.366430 ms** for
`window_512`, both compared with `probe_lo` from `slope_verifier_window.json`.
These imported timestamps were correctly labeled `client_reported` and
`device_token`. A fresh live check separately collected token arrivals on the
service's own connection and labeled them `service_observed`; it ran all four
requested detectors: `token_difr`, `cross_entropy`, `token_toploc`, `clock_slope`.

Prior BF16, temperature-override and real NF4 captures also passed through the
API's independent GPU scorer. Requests without an applicable service calibration
returned measured scores and `inconclusive`, rather than reusing the old artifact
pilot's different detector transforms or applying device calibration to wire
measurements. The queued verification completed successfully.

[Machine-readable results](verification_service_h100.json) contain the per-case
responses, reference identity, raw detector scores, design/provenance fields,
observations, costs and job result. These are new service integration results;
the earlier [signed-artifact pilot](audit_protocol_h100.md) is a separate study.

The [integration-test control profile](verification_service_h100_calibration.json)
is included for inspection and p-value reproduction; it is not a deployment
calibration profile.

Reproduce with:

```sh
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m experiments.exp_verification_service_gpu \
  --snapshot /workspace/model-cache/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --out runs/service-h100-reproduction
```

See the [service guide](../VERIFICATION_SERVICE.md) for public API requests and
operator setup. The integration test's localhost provider is test support, not
part of the verifier or a provider deployment requirement. Both processes shared
this H100 for testing; this establishes software/process separation, not a
hardware-isolation or cross-machine validation claim.
