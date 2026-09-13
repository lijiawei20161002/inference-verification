"""Service boundary, original detector parity, live SSE and queue tests."""

import json
import threading
import time
import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from ivgym.core import SamplingSpec, VContext
from ivgym.sampling import gumbel_noise, position_seed
from ivgym.verifiers import CrossEntropy, TokenTOPLOC, TokenDiFR
from ivgym.clock import ClockSlope
from ivgym.service.app import create_app
from ivgym.service.engine import VerificationEngine
from ivgym.service.schemas import Settings, Calibration, VerificationRequest
from ivgym.service.provider import ProviderClient


class FakeModel:
    identity = {
        "repository": "test/ref",
        "revision": "a" * 40,
        "files_sha256": "b" * 64,
    }
    logits = np.array([[4.0, 2.0, 1.0, 0.0], [1.0, 4.0, 2.0, 0.0]])

    def encode(self, text):
        return [0] * len(text.split())

    def prepare(self, capture):
        if capture.prompt_token_ids is not None:
            return capture.prompt_token_ids, capture.output_token_ids, "exact_ids"
        return self.encode(capture.prompt), [0, 1], "retokenized_text"

    def score(self, prepared, capture, sampling, detectors):
        _, output, _ = prepared
        logits = np.resize(self.logits, (len(output), 4))
        g = np.stack(
            [
                gumbel_noise(4, position_seed(sampling.seed, capture.prompt_id, i))
                for i in range(len(output))
            ]
        )
        ctx = VContext(
            capture.prompt_id,
            output,
            SamplingSpec(**sampling.model_dump()),
            ref_logits=logits,
            gumbel=g,
        )
        registry = {v.name: v for v in [CrossEntropy(), TokenTOPLOC(), TokenDiFR()]}
        return {d: registry[d].evidence(ctx).tolist() for d in detectors}, 0.01


class FakeReferences:
    def __init__(self):
        self.calls = 0

    def get(self, name):
        self.calls += 1
        return FakeModel()

    def encode(self, name, text):
        return FakeModel().encode(text)


def clock_request():
    return dict(
        model="test",
        verifiers=["clock_slope"],
        timing_pairs=[
            dict(
                short_context_tokens=16,
                long_context_tokens=1024,
                short_itl_ms=[2.0, 3.0, 4.0],
                long_itl_ms=[6.0, 7.0, 8.0],
                unit="token",
            )
        ],
    )


def token_request():
    return dict(
        model="test",
        verifiers=["token_toploc", "cross_entropy", "token_difr"],
        sampling=dict(temperature=1.0, top_k=None, top_p=1.0, seed=42),
        include_token_scores=True,
        captures=[
            dict(
                prompt_token_ids=[0, 1],
                output_token_ids=[0, 2],
                prompt_id=9,
                sampler_contract="ivgym_numpy_gumbel_v1",
            )
        ],
    )


def client(settings=None, provider=None):
    settings = settings or Settings()
    refs = FakeReferences()
    app = create_app(settings, VerificationEngine(settings, refs, provider))
    return TestClient(app), refs


def test_public_api_and_party_separation():
    c, refs = client()
    with c:
        assert c.get("/health").status_code == 200
        assert c.get("/docs").status_code == 200
        assert "/v1/verify" in c.get("/openapi.json").json()["paths"]
        r = c.post("/v1/verify", json=clock_request())
        assert r.status_code == 200, r.text
        result = r.json()
        assert result["parties"]["verifier"] == "independent third-party service"
        assert result["parties"]["provider"] == "untrusted inference endpoint"
        assert result["evidence_provenance"] == "client_reported"
        assert result["verifiers"]["clock_slope"]["score"] == -4.0
        assert result["costs"]["reference_prefill_tokens"] == 0 and refs.calls == 0
        assert result["outcome"] == "inconclusive"


def test_token_scores_are_original_experiment_detectors():
    req = token_request()
    c, _ = client()
    with c:
        r = c.post("/v1/verify", json=req)
        assert r.status_code == 200, r.text
        result = r.json()
        expected, _ = FakeModel().score(
            ([0, 1], [0, 2], "exact_ids"),
            VerificationRequest(**req).captures[0],
            VerificationRequest(**req).sampling,
            req["verifiers"],
        )
        for name in req["verifiers"]:
            assert result["verifiers"][name]["token_scores"] == expected[name]
            assert result["verifiers"][name]["score"] == np.mean(expected[name])
        assert result["costs"]["reference_prefill_tokens"] == 3
        assert result["reference"] == FakeModel.identity


def test_clock_statistic_matches_prior_experiments():
    lo, hi = [1.0, 2.0, 3.0], [8.0, 9.0, 10.0]
    assert np.array_equal(
        ClockSlope().score_pairs(lo, hi), -(np.array(hi) - np.array(lo))
    )
    with pytest.raises(ValueError):
        ClockSlope().score_pairs([1], [2, 3])
    with pytest.raises(ValueError):
        ClockSlope().score_pairs([-1], [2])


def test_only_service_owned_calibration_can_produce_decision():
    req = clock_request()
    settings = Settings()
    engine = VerificationEngine(settings, FakeReferences())
    design = engine.verify(VerificationRequest(**req))["verifiers"]["clock_slope"][
        "design"
    ]
    settings.calibrations = [
        Calibration(
            id="trusted",
            design=design,
            conditions={"honest": [-4.0] * 39},
            provenance="operator trusted control",
        )
    ]
    with TestClient(create_app(settings, engine)) as c:
        assert (
            c.post("/v1/verify", json=req).json()["outcome"] == "no_deviation_detected"
        )
        req["timing_pairs"][0]["long_itl_ms"] = [2.0, 3.0, 4.0]
        r = c.post("/v1/verify", json=req).json()
        assert r["outcome"] == "inconsistent"
        assert r["verifiers"]["clock_slope"]["p_value"] == 1 / 40
        assert c.get("/v1/calibrations").json()["calibrations"][0]["id"] == "trusted"
        req["calibration"] = {"honest_scores": [999]}
        assert c.post("/v1/verify", json=req).status_code == 422


def test_client_cannot_inject_reference_or_calibration():
    c, _ = client()
    with c:
        for name in [
            "reference_logits",
            "reference_url",
            "honest_scores",
            "calibration",
        ]:
            req = token_request()
            req[name] = [1, 2, 3]
            assert c.post("/v1/verify", json=req).status_code == 422
        req = token_request()
        req["captures"][0]["ref_logits"] = [[1, 2]]
        assert c.post("/v1/verify", json=req).status_code == 422


def test_difr_requires_exact_ids_and_original_rng_contract():
    req = token_request()
    req["captures"][0].pop("sampler_contract")
    c, _ = client()
    with c:
        r = c.post("/v1/verify", json=req).json()
        assert r["verifiers"]["token_difr"]["status"] == "unsupported"
        assert r["verifiers"]["cross_entropy"]["score"] is not None
        req["captures"] = [dict(prompt="hello world", output="hello there")]
        r = c.post("/v1/verify", json=req).json()
        assert r["verifiers"]["token_toploc"]["design"]["input_modes"] == [
            "retokenized_text"
        ]
        assert r["verifiers"]["token_difr"]["status"] == "unsupported"


def test_calibration_units_and_provenance_cannot_be_reused():
    req = clock_request()
    settings = Settings()
    engine = VerificationEngine(settings, FakeReferences())
    design = engine.verify(VerificationRequest(**req))["verifiers"]["clock_slope"][
        "design"
    ]
    wrong = {**design, "provenance": "service_observed"}
    settings.calibrations = [
        Calibration(
            id="live",
            design=wrong,
            conditions={"honest": [-100] * 39},
            provenance="live",
        )
    ]
    assert engine.verify(VerificationRequest(**req))["outcome"] == "inconclusive"
    wrong = {**design, "unit": "sse_chunk"}
    settings.calibrations = [
        Calibration(
            id="chunk",
            design=wrong,
            conditions={"honest": [-100] * 39},
            provenance="chunk",
        )
    ]
    assert engine.verify(VerificationRequest(**req))["outcome"] == "inconclusive"


def test_missing_detector_does_not_turn_into_pass():
    req = clock_request()
    req["verifiers"] = ["clock_slope", "token_toploc"]
    c, _ = client()
    with c:
        r = c.post("/v1/verify", json=req).json()
        assert r["verifiers"]["token_toploc"]["status"] == "unsupported"
        assert r["outcome"] == "inconclusive"
        assert r["verifiers"]["clock_slope"]["threshold"] == 0.025


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["timing_pairs"][0].update(long_itl_ms=[1]),
        lambda r: r["timing_pairs"][0].update(short_itl_ms=[-1, 2, 3]),
        lambda r: r["timing_pairs"][0].update(long_context_tokens=2),
        lambda r: r.update(verifiers=["clock_slope", "clock_slope"]),
        lambda r: r.update(alpha=0),
    ],
)
def test_invalid_evidence_rejected(mutation):
    req = clock_request()
    mutation(req)
    c, _ = client()
    with c:
        assert c.post("/v1/verify", json=req).status_code == 422


def test_resource_and_body_limits():
    c, _ = client(Settings(max_reference_tokens=2, max_request_bytes=1024))
    with c:
        assert c.post("/v1/verify", json=token_request()).status_code == 413
        r = c.post(
            "/v1/verify",
            content="x" * 1025,
            headers={"content-type": "application/json"},
        )
        assert r.status_code == 413


def test_credentials_not_echoed_in_validation_errors():
    c, _ = client()
    with c:
        req = clock_request()
        req["target"] = {
            "base_url": "http://x/v1",
            "model": "test",
            "api_key": "private-provider-secret",
        }
        r = c.post("/v1/verify", json=req)
        assert r.status_code == 422
        assert "private-provider-secret" not in r.text


def test_live_provider_allowlist_checked_before_model_access():
    c, refs = client()
    with c:
        req = dict(
            model="test",
            verifiers=["token_toploc"],
            target={"base_url": "http://169.254.169.254", "model": "x"},
            prompts=["hello"],
        )
        assert c.post("/v1/verify", json=req).status_code == 403
        assert refs.calls == 0


def test_queue_public_polling_capacity_and_health():
    started, release = threading.Event(), threading.Event()

    class Blocking:
        def verify(self, request):
            started.set()
            release.wait(3)
            return {"outcome": "inconclusive"}

    settings = Settings(max_pending_jobs=1)
    with TestClient(create_app(settings, Blocking())) as c:
        try:
            r = c.post("/v1/verifications", json=clock_request())
            assert r.status_code == 202
            key = r.json()["id"]
            assert started.wait(1)
            assert c.get("/health").status_code == 200
            assert c.get("/v1/verifications/" + key).json()["status"] == "running"
            assert c.post("/v1/verify", json=clock_request()).status_code == 429
            assert c.get("/v1/verifications/not-a-job").status_code == 404
        finally:
            release.set()
        for _ in range(100):
            result = c.get("/v1/verifications/" + key).json()
            if result["status"] == "succeeded":
                break
            time.sleep(0.01)
        assert result["result"]["outcome"] == "inconclusive"


def test_failed_job_sanitizes_internal_exception():
    class Broken:
        def verify(self, request):
            raise RuntimeError("secret credential in internal error")

    with TestClient(create_app(Settings(), Broken())) as c:
        r = c.post("/v1/verify", json=clock_request())
        assert r.status_code == 500
        assert "secret credential" not in r.text


def test_live_sse_collected_by_service():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            received.append((self.path, body, self.headers.get("Authorization")))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for token in [0, 1, 2]:
                chunk = dict(
                    prompt_token_ids=[0] * len(body["prompt"].split()),
                    choices=[
                        dict(index=0, text="x", token_ids=[token], finish_reason=None)
                    ],
                )
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.flush()
                time.sleep(0.005)
            self.wfile.write(
                b'data: {"choices":[{"index":0,"text":"","finish_reason":"length"}]}\n\ndata: [DONE]\n\n'
            )
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        settings = Settings(allowed_provider_urls=[url])
        req = dict(
            model="test",
            target={
                "base_url": url,
                "model": "provider-alias",
                "api_key": "caller-key",
            },
            prompts=["one two"],
            clock_probes=dict(short_prompt="one", long_prompt="one two three", pairs=1),
            max_tokens=3,
        )
        c, _ = client(settings)
        with c:
            r = c.post("/v1/verify", json=req)
            assert r.status_code == 200, r.text
            result = r.json()
            assert result["evidence_provenance"] == "service_observed"
            assert result["verifiers"]["clock_slope"]["details"]["unit"] == "token"
            assert len(result["observations"]) == 3
            assert all(x["ttft_ms"] >= 0 for x in result["observations"])
            assert result["reference"] == FakeModel.identity
            assert all(
                path == "/v1/completions"
                and body["model"] == "provider-alias"
                and auth == "Bearer caller-key"
                for path, body, auth in received
            )
            assert "caller-key" not in r.text
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_calibration_is_operator_only_and_collects_fresh_live_blocks(tmp_path):
    from ivgym.service.calibrate import collect, load_settings

    req = VerificationRequest(
        model="test",
        target=dict(base_url="http://control/v1", model="x"),
        prompts=["hello"],
        verifiers=["cross_entropy"],
    )

    class Control:
        calls = 0

        def verify(self, request):
            self.calls += 1
            return {
                "verifiers": {
                    "cross_entropy": dict(
                        score=float(self.calls),
                        design={"model": "test", "detector": "cross_entropy"},
                    )
                }
            }

    control = Control()
    bundle = collect(Settings(), req, 3, "operator-controlled test endpoint", control)
    assert control.calls == 3
    assert bundle["profiles"][0]["conditions"]["trusted_control"] == [1.0, 2.0, 3.0]
    (tmp_path / "cal.json").write_text(json.dumps(bundle))
    (tmp_path / "config.json").write_text(
        json.dumps({"calibration_files": ["cal.json"]})
    )
    settings = load_settings(tmp_path / "config.json")
    assert len(settings.calibrations) == 1
    with pytest.raises(Exception):
        collect(Settings(), VerificationRequest(**clock_request()), 3, "test", control)
    c, _ = client(settings)
    with c:
        assert c.post("/v1/calibrations", json=bundle).status_code == 405


def test_incomplete_clock_preserves_independent_token_results():
    class Collector:
        def collect(self, request, encode):
            return (
                [VerificationRequest(**token_request()).captures[0]],
                [],
                [{"role": "clock_error", "reason": "unpaired_clock_streams"}],
            )

    settings = Settings(allowed_provider_urls=["http://control/v1"])
    c, _ = client(settings, Collector())
    with c:
        r = c.post(
            "/v1/verify",
            json=dict(
                model="test",
                target=dict(base_url="http://control/v1", model="x"),
                prompts=["hello"],
                clock_probes=dict(short_prompt="short", long_prompt="much longer"),
            ),
        )
        assert r.status_code == 200, r.text
        result = r.json()
        assert result["verifiers"]["token_toploc"]["score"] is not None
        assert result["verifiers"]["clock_slope"]["status"] == "unsupported"


def test_ordinary_text_sse_is_not_claimed_as_exact_tokens():
    import httpx

    body = (
        'data: {"choices":[{"index":0,"text":"hello ","finish_reason":null}]}\n\n'
        'data: {"choices":[{"index":0,"text":"world","finish_reason":null}]}\n\n'
        'data: {"choices":[{"index":0,"text":"","finish_reason":"length"}]}\n\n'
        "data: [DONE]\n\n"
    )

    class Factory:
        def __call__(self, **kwargs):
            return httpx.Client(
                transport=httpx.MockTransport(
                    lambda r: httpx.Response(
                        200, headers={"content-type": "text/event-stream"}, text=body
                    )
                ),
                **kwargs,
            )

    from ivgym.service.schemas import Target, Sampling

    transport = ProviderClient(
        Settings(allowed_provider_urls=["http://test/v1"]), Factory()
    )
    capture, obs = transport.completion(
        Target(base_url="http://test/v1", model="x"),
        "a prompt",
        Sampling(),
        8,
        0,
        lambda s: [0, 1],
    )
    assert capture.output == "hello world" and capture.output_token_ids is None
    assert obs["unit"] == "sse_chunk"


def test_multitoken_sse_event_is_chunk_timing():
    import httpx

    body = (
        'data: {"prompt_token_ids":[0],"choices":[{"index":0,"text":"hello","token_ids":[0,1],"finish_reason":null}]}\n\n'
        'data: {"choices":[{"index":0,"text":"world","token_ids":[2],"finish_reason":null}]}\n\n'
        'data: {"choices":[{"index":0,"text":"","finish_reason":"length"}]}\n\n'
        "data: [DONE]\n\n"
    )
    factory = lambda **kwargs: httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, text=body
            )
        ),
        **kwargs,
    )
    from ivgym.service.schemas import Target, Sampling

    transport = ProviderClient(
        Settings(allowed_provider_urls=["http://test/v1"]), factory
    )
    capture, obs = transport.completion(
        Target(base_url="http://test/v1", model="x"),
        "prompt",
        Sampling(),
        8,
        0,
        lambda s: [0],
    )
    assert capture.output_token_ids == [0, 1, 2]
    assert obs["unit"] == "sse_chunk"
