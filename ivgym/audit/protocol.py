"""Collection, commitment/selection, immutable evidence and offline verification."""
from __future__ import annotations

import hashlib
import hmac
import math
import secrets
import time
from pathlib import Path
from typing import Protocol

from .crypto import (VERSION, Invalid, Untrusted, authenticate, b64, canonical, digest,
                     framed, read, require, sign, unb64, write)
from .contract import (DETECTORS, decision, fields, integer, sha, token_ids, validate_spec,
                       valid_scores)


class ProviderClient(Protocol):
    def deployment(self) -> dict | None: ...
    def start(self, plan: dict) -> None: ...
    def complete(self, request: dict) -> dict: ...
    def close(self, closure: dict, records: list[dict]) -> dict | None: ...


class ReferenceExecutor(Protocol):
    def score(self, records: list[dict], spec: dict) -> dict: ...


def commitment(session, seed):
    require(len(seed) == 32, "selection seed must be 256 bits")
    return hashlib.sha256(framed(b"ivgym-selection-v1", session.encode(), seed)).hexdigest()


def select(seed, root, session, count, sample):
    """Versioned HMAC KDF/stream + unbiased partial Fisher-Yates, ordered draw."""
    require(len(seed) == 32 and 0 <= sample <= count, "invalid selection input")
    key = hmac.digest(seed, framed(b"ivgym-selection-kdf-v1", root.encode(), session.encode()), "sha256")
    counter = 0
    pool = list(range(count))
    for i in range(sample):
        n = count - i
        limit = 2**256 - (2**256 % n)
        while True:
            x = int.from_bytes(hmac.digest(key, counter.to_bytes(8, "big"), "sha256"), "big")
            counter += 1
            if x < limit:
                break
        j = i + x % n
        pool[i], pool[j] = pool[j], pool[i]
    return pool[:sample]


def output_payload(request, deployment, tokens, text, finish, previous):
    return dict(session=request["session"], nonce=request["nonce"], sequence=request["sequence"],
                request_sha256=digest(request), spec_sha256=request["spec_sha256"],
                deployment_sha256=digest(deployment), previous_receipt_sha256=previous,
                prompt_token_ids=request["prompt_token_ids"], output_token_ids=tokens,
                text_utf8=b64(text.encode("utf-8")), finish_reason=finish)


def validate_output(payload, request, s):
    fields(payload, "session nonce sequence request_sha256 spec_sha256 deployment_sha256 previous_receipt_sha256 prompt_token_ids output_token_ids text_utf8 finish_reason")
    for k in ("session", "nonce", "sequence", "spec_sha256", "prompt_token_ids"):
        require(payload[k] == request[k], f"response {k} mismatch")
    require(payload["request_sha256"] == digest(request), "response request binding mismatch")
    token_ids(payload["output_token_ids"])
    require(len(payload["output_token_ids"]) <= s["generation"]["max_output_tokens"], "output cap exceeded")
    require(payload["finish_reason"] in ("length", "eos"), "unsupported finish reason")
    eos = s["generation"]["eos_token_ids"]
    ids = payload["output_token_ids"]
    require(not any(t in eos for t in ids[:-1]), "tokens after EOS")
    if payload["finish_reason"] == "eos":
        require(ids[-1] in eos, "EOS finish without EOS token")
    else:
        require(len(ids) == s["generation"]["max_output_tokens"] and ids[-1] not in eos,
                "incomplete length finish")
    unb64(payload["text_utf8"]).decode("utf-8")


FILES = ("spec.json", "calibration.json", "deployment.json", "plan.json", "records.json",
         "closure.json", "selection.json", "reference.json", "verdict.json")


def seal(destination, data, key, key_id):
    path = Path(destination)
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name in FILES:
        write(path / name, data[name])
    manifest = dict(version=VERSION, files={name: digest(data[name]) for name in FILES})
    write(path / "manifest.json", manifest)
    write(path / "transcript.sig", sign(manifest, key, key_id, "auditor", "artifact"))


def check_deployment(deployment, s, trust, endpoint=None, now=None):
    p = authenticate(deployment, trust, "provider", "deployment", s["identities"]["provider"])
    fields(p, "version origin spec_sha256 model_sha256 epoch expires_unix receipts exact_token_ids")
    require(p["version"] == VERSION and p["spec_sha256"] == digest(s) and
            p["model_sha256"] == digest(s["model"]), "deployment contract mismatch")
    require(p["receipts"] is True and p["exact_token_ids"] is True, "missing deployment capabilities")
    require(isinstance(p["epoch"], str) and p["epoch"] and integer(p["expires_unix"]), "invalid epoch/expiry")
    require(endpoint is None or p["origin"] == endpoint, "endpoint origin mismatch")
    require(now is None or p["expires_unix"] > now, "expired deployment")
    return p


class AuditRunner:
    def __init__(self, provider, reference, auditor_key, reference_key, trust):
        self.provider, self.reference = provider, reference
        self.auditor_key, self.reference_key, self.trust = auditor_key, reference_key, trust

    def run(self, spec, calibration, prompts, destination):
        s = validate_spec(spec)
        cal = authenticate(calibration, self.trust, "calibration", "calibration", s["calibration"]["publisher_id"])
        require(digest(calibration) == s["calibration"]["sha256"], "wrong calibration bundle")
        n = s["audit"]["scheduled_requests"]
        require(len(prompts) == n, "prompt count must equal frozen request schedule")
        for ids in prompts:
            token_ids(ids)
            require(len(ids) + s["generation"]["max_output_tokens"] <= s["generation"]["context_limit"],
                    "prompt exceeds reject-overflow context contract")
        session, seed = secrets.token_hex(32), secrets.token_bytes(32)
        reasons, unsupported = [], False
        started = time.time()
        deployment = None
        try:
            deployment = self.provider.deployment()
            if s["mode"] == "receipted":
                if deployment is None:
                    unsupported = True
                    reasons.append("required_receipts_unavailable")
                else:
                    check_deployment(deployment, s, self.trust, getattr(self.provider, "endpoint", None), started)
        except Exception as exc:
            reasons.append("deployment_error:" + type(exc).__name__)
        if s["mode"] == "observed":
            deployment = None
        requests = [dict(session=session, nonce=secrets.token_hex(32), sequence=i,
                         spec_sha256=digest(s), prompt_token_ids=p, generation=s["generation"])
                    for i, p in enumerate(prompts)]
        plan = sign(dict(version=VERSION, session=session, created_unix=int(started),
                         endpoint=getattr(self.provider, "endpoint", "local"), spec_sha256=digest(s),
                         deployment_sha256=digest(deployment), calibration_sha256=digest(calibration),
                         selection_commitment=commitment(session, seed), requests=requests),
                    self.auditor_key, s["identities"]["auditor"], "auditor", "plan")
        budget = s["budget"]
        if n > budget["max_requests"] or n * s["generation"]["max_output_tokens"] > budget["max_generated_tokens"]:
            reasons.append("generation_budget_exhausted")
        if budget["max_cost_usd"] is not None:
            reasons.append("monetary_budget_requires_priced_adapter")
        if not reasons:
            try:
                self.provider.start(plan)
            except Exception as exc:
                reasons.append("plan_delivery_error:" + type(exc).__name__)
        records = []
        halted = bool(reasons)
        for req in requests:
            row = dict(request=req, capture=None, error=None, observed_unix=None)
            if halted:
                row["error"] = "not_sent_due_to_preflight_failure"
            else:
                try:
                    row["capture"] = self.provider.complete(req)
                    row["observed_unix"] = time.time()
                except Exception as exc:
                    row["error"] = type(exc).__name__ + ":" + str(exc)[:300]
            records.append(row)
        close_payload = dict(session=session, plan_sha256=digest(plan), count=n, records_sha256=digest(records))
        provider_close = None
        if s["mode"] == "receipted" and not halted:
            try:
                provider_close = self.provider.close(close_payload, records)
            except Exception as exc:
                reasons.append("closure_error:" + type(exc).__name__)
        closure = dict(payload=close_payload, provider_signature=provider_close,
                       preflight_reasons=reasons, unsupported=unsupported)
        opening = dict(seed=b64(seed), indices=select(seed, digest(closure), session, n,
                                                    s["audit"]["audited_responses"]))
        data = {"spec.json": s, "calibration.json": calibration, "deployment.json": deployment,
                "plan.json": plan, "records.json": records, "closure.json": closure,
                "selection.json": opening, "reference.json": None, "verdict.json": None}
        reasons = inspect_records(data, self.trust)
        chosen = [records[i] for i in opening["indices"]]
        if not reasons:
            cost = sum(len(r["request"]["prompt_token_ids"]) +
                       len(r["capture"]["payload"]["output_token_ids"]) - 1 for r in chosen)
            if cost > budget["max_reference_prefill_tokens"]:
                reasons.append("reference_prefill_budget_exhausted")
            else:
                try:
                    result = self.reference.score(chosen, s)
                    result.update(session=session, selection_sha256=digest(opening),
                                  records_sha256=digest(records), spec_sha256=digest(s))
                    data["reference.json"] = sign(result, self.reference_key, s["identities"]["reference"],
                                                  "reference", "scores")
                except Exception as exc:
                    reasons.append("reference_error:" + type(exc).__name__)
        # Signed execution errors are retained separately from the statistical calculation.
        data["closure.json"]["execution_reasons"] = reasons
        # Selection root excludes post-selection execution results.
        data["verdict.json"] = compute_verdict(data, self.trust)
        seal(destination, data, self.auditor_key, s["identities"]["auditor"])
        return data["verdict.json"]


def selection_root(closure):
    return digest({k: v for k, v in closure.items() if k != "execution_reasons"})


def inspect_records(data, trust):
    s, plan_env = data["spec.json"], data["plan.json"]
    plan = authenticate(plan_env, trust, "auditor", "plan", s["identities"]["auditor"])
    fields(plan, "version session created_unix endpoint spec_sha256 deployment_sha256 calibration_sha256 selection_commitment requests")
    require(plan["version"] == VERSION and plan["spec_sha256"] == digest(s) and
            plan["calibration_sha256"] == digest(data["calibration.json"]) and
            plan["deployment_sha256"] == digest(data["deployment.json"]), "plan binding mismatch")
    require(sha(plan["session"]) and sha(plan["selection_commitment"]) and integer(plan["created_unix"]),
            "malformed session, commitment or timestamp")
    records, close = data["records.json"], data["closure.json"]
    require(set(close) in ({"payload", "provider_signature", "preflight_reasons", "unsupported"},
                          {"payload", "provider_signature", "preflight_reasons", "unsupported", "execution_reasons"}),
            "invalid closure fields")
    require(type(close["unsupported"]) is bool and isinstance(close["preflight_reasons"], list),
            "invalid closure status")
    n = s["audit"]["scheduled_requests"]
    require(len(records) == len(plan["requests"]) == n, "omitted scheduled records")
    require(close["payload"] == dict(session=plan["session"], plan_sha256=digest(plan_env),
                                     count=n, records_sha256=digest(records)), "closure mismatch")
    reasons = list(close["preflight_reasons"])
    if s["mode"] == "receipted" and data["deployment.json"] is not None:
        check_deployment(data["deployment.json"], s, trust, plan["endpoint"], plan["created_unix"])
    previous, nonces, chain_known = None, set(), True
    for i, (row, request) in enumerate(zip(records, plan["requests"])):
        fields(row, "request capture error observed_unix")
        fields(request, "session nonce sequence spec_sha256 prompt_token_ids generation")
        require(row["request"] == request and request["session"] == plan["session"] and
                request["sequence"] == i and request["nonce"] not in nonces and
                request["spec_sha256"] == digest(s) and request["generation"] == s["generation"],
                "request identity/order/contract mismatch")
        require(sha(request["nonce"]), "malformed request nonce")
        nonces.add(request["nonce"])
        token_ids(request["prompt_token_ids"])
        require(len(request["prompt_token_ids"]) + s["generation"]["max_output_tokens"] <=
                s["generation"]["context_limit"], "request context overflow")
        if row["error"] or row["capture"] is None:
            reasons.append("incomplete_scheduled_data")
            chain_known = False
            continue
        cap = row["capture"]
        fields(cap, "request_wire response_wire payload receipt")
        from .crypto import loads
        require(loads(unb64(cap["request_wire"])) == request, "request wire mismatch")
        wire = loads(unb64(cap["response_wire"]))
        require(wire == dict(payload=cap["payload"], receipt=cap["receipt"]), "response wire mismatch")
        validate_output(cap["payload"], request, s)
        if s["mode"] == "receipted":
            if cap["receipt"] is None:
                reasons.append("missing_final_receipt")
                chain_known = False
                continue
            p = authenticate(cap["receipt"], trust, "provider", "receipt", s["identities"]["provider"])
            require(p == cap["payload"] and p["deployment_sha256"] == digest(data["deployment.json"]),
                    "receipt payload/deployment mismatch")
            require(not chain_known or p["previous_receipt_sha256"] == previous, "receipt chain mismatch")
            previous = digest(cap["receipt"])
            chain_known = True
    if s["mode"] == "receipted":
        if close["provider_signature"] is None:
            reasons.append("missing_provider_closure")
        else:
            require(authenticate(close["provider_signature"], trust, "provider", "closure",
                                 s["identities"]["provider"]) == close["payload"], "provider closure mismatch")
    return reasons


def compute_verdict(data, trust):
    s = validate_spec(data["spec.json"])
    cal = authenticate(data["calibration.json"], trust, "calibration", "calibration", s["calibration"]["publisher_id"])
    require(digest(data["calibration.json"]) == s["calibration"]["sha256"], "calibration hash mismatch")
    reasons = inspect_records(data, trust) + data["closure.json"].get("execution_reasons", [])
    budget, audit = s["budget"], s["audit"]
    if audit["scheduled_requests"] > budget["max_requests"] or audit["scheduled_requests"] * s["generation"]["max_output_tokens"] > budget["max_generated_tokens"]:
        reasons.append("generation_budget_exhausted")
    if budget["max_cost_usd"] is not None:
        reasons.append("monetary_budget_requires_priced_adapter")
    plan, opening = data["plan.json"]["payload"], data["selection.json"]
    fields(opening, "seed indices")
    seed = unb64(opening["seed"])
    require(commitment(plan["session"], seed) == plan["selection_commitment"], "seed commitment mismatch")
    require(opening["indices"] == select(seed, selection_root(data["closure.json"]), plan["session"],
                                         s["audit"]["scheduled_requests"], s["audit"]["audited_responses"]),
            "selection mismatch")
    result, scores = None, None
    if data["reference.json"] is not None:
        result = authenticate(data["reference.json"], trust, "reference", "scores", s["identities"]["reference"])
        fields(result, "model profile per_token_scores scores token_count prefill_tokens gpu_seconds session selection_sha256 records_sha256 spec_sha256")
        require(integer(result["token_count"], 1) and integer(result["prefill_tokens"], 1) and
                type(result["gpu_seconds"]) in (int, float) and math.isfinite(result["gpu_seconds"]) and
                result["gpu_seconds"] >= 0, "invalid reference measurements")
        require(result["session"] == plan["session"] and result["selection_sha256"] == digest(opening) and
                result["records_sha256"] == digest(data["records.json"]) and result["spec_sha256"] == digest(s),
                "reference binding mismatch")
        require(result["profile"] == s["reference"] and result["model"] == s["model"], "reference profile mismatch")
        selected = [data["records.json"][i] for i in opening["indices"]]
        count = sum(len(r["capture"]["payload"]["output_token_ids"]) for r in selected)
        prefill = sum(len(r["request"]["prompt_token_ids"]) + len(r["capture"]["payload"]["output_token_ids"]) - 1
                      for r in selected)
        require(result["token_count"] == count and result["prefill_tokens"] == prefill, "reference counts mismatch")
        per_token = result["per_token_scores"]
        require(len(per_token) == count, "missing per-token scores")
        for row in per_token:
            valid_scores(row)
        scores = {d: sum(row[d] for row in per_token) / count for d in DETECTORS}
        require(scores == result["scores"], "reference aggregation mismatch")
        if prefill > s["budget"]["max_reference_prefill_tokens"]:
            reasons.append("reference_prefill_budget_exhausted")
    verdict = decision(s, cal, scores, reasons, data["closure.json"]["unsupported"])
    verdict.update(version=VERSION, spec_sha256=digest(s), calibration_sha256=digest(data["calibration.json"]),
                   model_sha256=digest(s["model"]), transcript_root=digest(data["records.json"]),
                   deployment_epoch=(data["deployment.json"] or {}).get("payload", {}).get("epoch"),
                   scheduled_requests=len(data["records.json"]),
                   audited_responses=len(opening["indices"]) if result else 0,
                   audited_tokens=result["token_count"] if result else 0, scores=scores,
                   costs=dict(reference_prefill_tokens=result["prefill_tokens"] if result else 0,
                              reference_gpu_seconds=result["gpu_seconds"] if result else None,
                              cost_usd=None,
                              wire_bytes=sum(len(unb64(r["capture"][w])) for r in data["records.json"]
                                             if r["capture"] for w in ("request_wire", "response_wire"))))
    if any(reason in reasons for reason in ("missing_final_receipt", "missing_provider_closure", "incomplete_scheduled_data")):
        verdict["provider_endorsement"] = False
    return verdict


def verify(destination, trust):
    try:
        path = Path(destination)
        manifest = read(path / "manifest.json")
        signed = authenticate(read(path / "transcript.sig"), trust, "auditor", "artifact")
        require(signed == manifest and manifest["version"] == VERSION and
                set(manifest["files"]) == set(FILES), "artifact manifest mismatch")
        require({p.name for p in path.iterdir()} == set(FILES) | {"manifest.json", "transcript.sig"},
                "missing or unmanifested artifact files")
        require(all(not (path / name).is_symlink() for name in (*FILES, "manifest.json", "transcript.sig")),
                "artifact symlinks forbidden")
        data = {name: read(path / name) for name in FILES}
        require(all(digest(data[name]) == manifest["files"][name] for name in FILES), "artifact file digest mismatch")
        require(read(path / "transcript.sig")["key_id"] == data["spec.json"]["identities"]["auditor"],
                "wrong artifact signer")
        recalculated = compute_verdict(data, trust)
        require(recalculated == data["verdict.json"], "recorded verdict disagrees with decision calculation")
        return dict(integrity="valid", verdict=recalculated)
    except Untrusted as exc:
        return dict(integrity="unverifiable", error=str(exc))
    except Exception as exc:
        return dict(integrity="invalid", error=type(exc).__name__ + ": " + str(exc))


def replay(destination, trust, reference):
    verified = verify(destination, trust)
    if verified["integrity"] != "valid":
        return dict(replay="unsupported", verification=verified)
    path = Path(destination)
    recorded = read(path / "reference.json")
    if recorded is None:
        return dict(replay="unsupported", reason="no recorded reference result")
    try:
        spec, rows, opening = read(path / "spec.json"), read(path / "records.json"), read(path / "selection.json")
        result = reference.score([rows[i] for i in opening["indices"]], spec)
        a, b = result["per_token_scores"], recorded["payload"]["per_token_scores"]
        require(len(a) == len(b), "replay score count mismatch")
        delta = max(abs(x[d] - y[d]) for x, y in zip(a, b) for d in DETECTORS)
        data = {name: read(path / name) for name in FILES}
        cal = data["calibration.json"]["payload"]
        rerun = decision(spec, cal, result["scores"], data["closure.json"]["execution_reasons"],
                         data["closure.json"]["unsupported"])
        boundary = rerun["outcome"] != verified["verdict"]["outcome"]
        return dict(replay="disagreement" if boundary or delta > spec["reference"]["replay_atol"] else "agrees",
                    max_absolute_score_delta=delta, boundary_disagreement=boundary,
                    gpu_seconds=result["gpu_seconds"], recorded_verdict_unchanged=True)
    except Exception as exc:
        return dict(replay="unsupported", reason=type(exc).__name__ + ": " + str(exc))
