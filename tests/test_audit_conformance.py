"""Wire vectors and protocol edge cases, independent of any model download."""
import copy
from pathlib import Path

import pytest
pytest.importorskip("rfc8785")
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ivgym.audit.crypto import Invalid, authenticate, canonical, digest, loads, public, read, sign, signing_bytes
from ivgym.audit.contract import DETECTORS, decision, profile
from ivgym.audit.protocol import commitment, select, verify
from ivgym.audit.http import ReceiptMiddleware
from test_audit_protocol import env, fixture_spec, calibration, run


def test_frozen_wire_vectors():
    v = read(Path(__file__).parent / "fixtures" / "audit-v1-vectors.json")
    key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(v["test_private_seed_hex"]))
    assert public(key) == v["public_key_base64"]
    env = v["envelope"]
    unsigned = {k: val for k, val in env.items() if k != "signature"}
    assert signing_bytes(unsigned).hex() == v["signing_bytes_hex"]
    assert sign(env["payload"], key, env["key_id"], env["role"], env["kind"]) == env
    assert authenticate(env, {env["role"]: {env["key_id"]: public(key)}}, env["role"], env["kind"]) == env["payload"]
    s = v["selection"]
    seed = bytes.fromhex(s["seed_hex"])
    assert commitment(s["session"], seed) == s["commitment"]
    assert select(seed, s["root"], s["session"], s["count"], s["sample"]) == s["indices"]


@pytest.mark.parametrize("index", [0, 1])
def test_missing_receipt_inside_chain_is_inconclusive(tmp_path, env, index):
    class Drop(ReceiptMiddleware):
        def handle(self, route, body):
            result = super().handle(route, body)
            if route == "completions" and body["sequence"] == index:
                result["receipt"] = None
            return result
    path, verdict = run(tmp_path, env, middleware_cls=Drop)
    assert verdict["outcome"] == "inconclusive"
    assert not verdict["provider_endorsement"]
    assert verify(path, env[1])["integrity"] == "valid"


def test_provider_rejects_replayed_plan_and_requests(tmp_path, env):
    path, _ = run(tmp_path, env)
    spec, plan, deployment = read(path / "spec.json"), read(path / "plan.json"), read(path / "deployment.json")
    m = ReceiptMiddleware(spec, env[0]["provider"], env[1], lambda req: ([4, 5], "text", "length"),
                          deployment["payload"]["origin"], "test-epoch")
    m.deployment = deployment
    m.handle("start", plan)
    with pytest.raises(Invalid): m.handle("start", plan)
    reqs = plan["payload"]["requests"]
    with pytest.raises(Invalid): m.handle("completions", reqs[1])
    m.handle("completions", reqs[0])
    with pytest.raises(Invalid): m.handle("completions", reqs[0])


def test_calibration_provisioning_has_no_fake_null():
    spec = fixture_spec()
    c = dict(version=spec["version"], profile_sha256=profile(spec), conditions={}, status="provisioning")
    result = decision(spec, c, {d: 1. for d in DETECTORS})
    assert result["outcome"] == "inconclusive"
    assert result["p_values"] == {}
    assert "calibration_provisioning" in result["reasons"]


def test_calibration_wrong_profile_and_duplicate_blocks():
    spec = fixture_spec()
    c = calibration(spec)
    c["profile_sha256"] = "f" * 64
    assert decision(spec, c, {d: 1. for d in DETECTORS})["outcome"] == "inconclusive"
    c["conditions"]["test"].append(c["conditions"]["test"][0])
    with pytest.raises(Invalid): decision(spec, c, {d: 1. for d in DETECTORS})


def test_role_domain_separation():
    key = Ed25519PrivateKey.generate()
    s = sign({"x": 1}, key, "key", "provider", "receipt")
    trust = {r: {"key": public(key)} for r in ("provider", "reference")}
    with pytest.raises(Invalid): authenticate(s, trust, "reference", "scores")


def test_unmanifested_files_are_rejected(tmp_path, env):
    path, _ = run(tmp_path, env)
    (path / "extra.json").write_text("{}")
    assert verify(path, env[1])["integrity"] == "invalid"
