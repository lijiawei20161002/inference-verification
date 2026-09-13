"""Adversarial artifact, selection, statistics and live HTTP contract tests."""
import copy
import math
import threading

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ivgym.audit.crypto import VERSION, Invalid, b64, canonical, digest, loads, public, read, sign, write
from ivgym.audit.contract import DETECTORS, FEATURES, SELECTION, decision, profile, validate_spec
from ivgym.audit.http import HTTPProviderClient, ReceiptMiddleware, make_server
from ivgym.audit.protocol import FILES, AuditRunner, replay, seal, select, verify


def fixture_spec():
    return dict(version=VERSION, mode="receipted",
        model=dict(repository="test/model", revision="a" * 40, files_sha256="b" * 64,
                   tokenizer_sha256="c" * 64, chat_template_sha256="d" * 64, adapters=[], claimed_dtype="bfloat16"),
        generation=dict(temperature=1., top_p=1., top_k=None, max_output_tokens=2,
                        eos_token_ids=[99], context_limit=128, context_policy="reject_overflow", input_mode="raw_token_ids"),
        reference=dict(executor_id="reference", software_sha256="e" * 64, hardware="test",
                       attention="eager", dtype="bfloat16", replay_atol=1e-5),
        calibration=dict(sha256="0" * 64, publisher_id="calibration", require_validated_operating_point=False),
        audit=dict(scheduled_requests=3, audited_responses=2, selection=SELECTION,
                   detectors=DETECTORS, feature_sha256=digest(FEATURES), alpha=.1,
                   alpha_scope="one_audit_only", traffic_profile="test-domain"),
        budget=dict(max_requests=3, max_generated_tokens=6, max_reference_prefill_tokens=100, max_cost_usd=None),
        identities=dict(auditor="auditor", provider="provider", reference="reference"))


def calibration(spec, count=39):
    return dict(version=VERSION, profile_sha256=profile(spec),
                conditions={"test": [dict(block_id=f"block-{i}", scores={d: float(i) for d in DETECTORS})
                                     for i in range(count)]}, power_reference=None)


class FakeReference:
    def __init__(self, score=1.):
        self.value = score

    def score(self, records, spec):
        tokens = sum(len(r["capture"]["payload"]["output_token_ids"]) for r in records)
        prefill = sum(len(r["request"]["prompt_token_ids"]) + len(r["capture"]["payload"]["output_token_ids"]) - 1
                      for r in records)
        return dict(model=spec["model"], profile=spec["reference"],
                    per_token_scores=[{d: self.value for d in DETECTORS} for _ in range(tokens)],
                    scores={d: self.value for d in DETECTORS}, token_count=tokens, prefill_tokens=prefill, gpu_seconds=0.)


@pytest.fixture
def env():
    keys = {role: Ed25519PrivateKey.generate() for role in ("auditor", "reference", "provider", "calibration")}
    return keys, {role: {role: public(key)} for role, key in keys.items()}


def run(tmp_path, env, spec=None, score=1., receipts=True, middleware_cls=ReceiptMiddleware):
    keys, trust = env
    spec = spec or fixture_spec()
    bundle = sign(calibration(spec), keys["calibration"], "calibration", "calibration", "calibration")
    spec["calibration"]["sha256"] = digest(bundle)
    # Bind the actual OS-selected ephemeral port before signing the deployment.
    server = make_server(("127.0.0.1", 0), None)
    endpoint = f"http://127.0.0.1:{server.server_port}"
    server.server_close()
    middleware = middleware_cls(spec, keys["provider"], trust, lambda req: ([4, 5], "text", "length"),
                               endpoint, "test-epoch", receipts=receipts)
    server = make_server(("127.0.0.1", int(endpoint.rsplit(":", 1)[1])), middleware)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        runner = AuditRunner(HTTPProviderClient(endpoint), FakeReference(score), keys["auditor"], keys["reference"], trust)
        verdict = runner.run(spec, bundle, [[1, 2], [2, 3], [3, 4]], tmp_path / "artifact")
        return tmp_path / "artifact", verdict
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def reseal(path, env, mutate, target):
    keys, _ = env
    data = {name: read(path / name) for name in FILES}
    mutate(data)
    seal(target, data, keys["auditor"], "auditor")
    return target


def test_http_roundtrip_and_replay(tmp_path, env):
    path, verdict = run(tmp_path, env)
    assert verdict["outcome"] == "no_deviation_detected"
    assert verify(path, env[1])["integrity"] == "valid"
    assert replay(path, env[1], FakeReference())["replay"] == "agrees"
    assert replay(path, env[1], FakeReference(2.))["replay"] == "disagreement"


def test_real_decision_rejection(tmp_path, env):
    path, verdict = run(tmp_path, env, score=100.)
    assert verdict["outcome"] == "inconsistent"
    assert verify(path, env[1])["integrity"] == "valid"


@pytest.mark.parametrize("name", FILES)
def test_byte_mutation(tmp_path, env, name):
    path, _ = run(tmp_path, env)
    write(path / name, {"mutated": True})
    assert verify(path, env[1])["integrity"] == "invalid"


@pytest.mark.parametrize("case", ["omit", "reorder", "seed", "selection", "verdict", "session", "reference", "version"])
def test_semantic_tamper_even_with_fresh_auditor_signature(tmp_path, env, case):
    path, _ = run(tmp_path, env)
    def mutate(d):
        if case == "omit": d["records.json"].pop()
        if case == "reorder": d["records.json"].reverse()
        if case == "seed": d["selection.json"]["seed"] = b64(b"x" * 32)
        if case == "selection": d["selection.json"]["indices"] = [0, 0]
        if case == "verdict": d["verdict.json"]["outcome"] = "inconsistent"
        if case == "session": d["plan.json"]["payload"]["session"] = "another-session"
        if case == "reference": d["reference.json"]["payload"]["scores"][DETECTORS[0]] += 10
        if case == "version": d["spec.json"]["version"] = "future"
    altered = reseal(path, env, mutate, tmp_path / "altered")
    assert verify(altered, env[1])["integrity"] == "invalid"


def test_cross_session_substitution(tmp_path, env):
    a, _ = run(tmp_path / "a", env)
    b, _ = run(tmp_path / "b", env)
    changed = reseal(a, env, lambda d: d.update({"reference.json": read(b / "reference.json")}), tmp_path / "changed")
    assert verify(changed, env[1])["integrity"] == "invalid"


def test_external_trust_required(tmp_path, env):
    path, _ = run(tmp_path, env)
    assert verify(path, {})["integrity"] == "unverifiable"
    wrong = copy.deepcopy(env[1])
    wrong["provider"]["provider"] = public(Ed25519PrivateKey.generate())
    assert verify(path, wrong)["integrity"] == "invalid"


@pytest.mark.parametrize("mode,receipts,outcome", [("receipted", False, "unsupported"), ("observed", False, "no_deviation_detected")])
def test_no_receipt_fallback(tmp_path, env, mode, receipts, outcome):
    spec = fixture_spec()
    spec["mode"] = mode
    path, verdict = run(tmp_path, env, spec, receipts=receipts)
    assert verdict["outcome"] == outcome
    assert verdict["provider_endorsement"] is False
    assert verify(path, env[1])["integrity"] == "valid"


@pytest.mark.parametrize("budget", ["max_requests", "max_generated_tokens", "max_reference_prefill_tokens", "max_cost_usd"])
def test_budget_exhaustion_is_inconclusive(tmp_path, env, budget):
    spec = fixture_spec()
    spec["budget"][budget] = 0
    path, verdict = run(tmp_path, env, spec)
    assert verdict["outcome"] == "inconclusive"
    assert verify(path, env[1])["integrity"] == "valid"


def test_dropped_final_receipt(tmp_path, env):
    class Drop(ReceiptMiddleware):
        def handle(self, route, body):
            result = super().handle(route, body)
            if route == "completions" and body["sequence"] == 2:
                result["receipt"] = None
            return result
    path, verdict = run(tmp_path, env, middleware_cls=Drop)
    assert verdict["outcome"] == "inconclusive"
    assert "missing_final_receipt" in verdict["reasons"]
    assert verify(path, env[1])["integrity"] == "valid"


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"x":NaN}', '{"x":1e999}', '{"x":9007199254740992}', '"\\ud800"'])
def test_strict_json(raw):
    with pytest.raises((ValueError, UnicodeError)):
        loads(raw)


def test_rank_test_composite_null_ties_and_resolution():
    spec = fixture_spec()
    cal = calibration(spec)
    scores = {d: 100. for d in DETECTORS}
    assert decision(spec, cal, scores)["outcome"] == "inconsistent"
    cal["conditions"]["benign_shift"] = [dict(block_id=f"shift-{i}", scores=scores) for i in range(39)]
    result = decision(spec, cal, scores)
    assert result["outcome"] == "no_deviation_detected"
    assert all(p == 1 for p in result["p_values"].values())
    spec["audit"]["alpha"] = 1e-4
    cal["profile_sha256"] = profile(spec)
    assert "insufficient_calibration_resolution" in decision(spec, cal, scores)["reasons"]


def test_unvalidated_and_unsupported_profiles():
    spec = fixture_spec()
    cal = calibration(spec)
    spec["calibration"]["require_validated_operating_point"] = True
    assert decision(spec, cal, {d: 1. for d in DETECTORS})["outcome"] == "inconclusive"
    spec["generation"]["top_p"] = .9
    with pytest.raises(Invalid): validate_spec(spec)


def test_unbiased_selection_without_replacement():
    for count in (1, 3, 10, 257):
        values = select(b"x" * 32, "root", "session", count, count)
        assert sorted(values) == list(range(count))
        assert values == select(b"x" * 32, "root", "session", count, count)


def test_canonical_json_vector():
    assert canonical({"z": 1., "a": "é", "n": 1e-7}) == b'{"a":"\xc3\xa9","n":1e-7,"z":1}'
