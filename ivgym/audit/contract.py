"""Deliberately narrow executable v1 contract, distinct from the draft template."""
from __future__ import annotations

import math
import re

from .crypto import VERSION, digest, require

DETECTORS = ["unfiltered_nll_v1", "capped_log_rank_v1"]
SELECTION = "hmac-sha256-fisher-yates-rejection-v1"
FEATURES = {"names": DETECTORS, "temperature": "claimed", "rank": "strict-greater",
            "rank_cap": 50, "transform": "log1p(rank)", "aggregate": "mean_over_tokens"}


def fields(obj, names):
    require(isinstance(obj, dict) and set(obj) == set(names.split()), "unexpected schema fields")


def integer(value, low=0):
    return type(value) is int and value >= low


def sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def token_ids(ids, nonempty=True):
    require(isinstance(ids, list) and (ids or not nonempty) and
            all(integer(x) for x in ids), "exact nonnegative token IDs required")


def validate_spec(s):
    fields(s, "version mode model generation reference calibration audit budget identities")
    require(s["version"] == VERSION, "unsupported spec version (draft templates need provisioning)")
    require(s["mode"] in ("observed", "receipted"), "unsupported protocol mode")
    m = s["model"]
    fields(m, "repository revision files_sha256 tokenizer_sha256 chat_template_sha256 adapters claimed_dtype")
    require(isinstance(m["repository"], str) and m["repository"], "model repository missing")
    require(isinstance(m["revision"], str) and re.fullmatch(r"[0-9a-f]{40}", m["revision"]),
            "immutable HF revision required")
    require(all(sha(m[k]) for k in ("files_sha256", "tokenizer_sha256", "chat_template_sha256")),
            "unresolved model digests")
    require(m["adapters"] == [] and m["claimed_dtype"] == "bfloat16", "unsupported model profile")
    g = s["generation"]
    fields(g, "temperature top_p top_k max_output_tokens eos_token_ids context_limit context_policy input_mode")
    require(type(g["temperature"]) in (int, float) and math.isfinite(g["temperature"]) and
            g["temperature"] > 0, "positive finite temperature required")
    require(g["top_p"] == 1.0 and g["top_k"] is None,
            "v1 supports full softmax only; filtered sampler profiles need separate calibration")
    require(integer(g["max_output_tokens"], 1) and integer(g["context_limit"], 2), "invalid lengths")
    token_ids(g["eos_token_ids"], nonempty=False)
    require(g["context_policy"] == "reject_overflow" and g["input_mode"] == "raw_token_ids",
            "hidden templates, reasoning and truncation are unsupported")
    fields(s["reference"], "executor_id software_sha256 hardware attention dtype replay_atol")
    r = s["reference"]
    require(sha(r["software_sha256"]) and r["dtype"] == "bfloat16" and
            r["attention"] in ("eager", "sdpa") and isinstance(r["hardware"], str), "unresolved reference")
    require(type(r["replay_atol"]) in (int, float) and math.isfinite(r["replay_atol"]) and
            0 <= r["replay_atol"] <= 0.01, "invalid replay tolerance")
    a = s["audit"]
    fields(a, "scheduled_requests audited_responses selection detectors feature_sha256 alpha alpha_scope traffic_profile")
    require(integer(a["scheduled_requests"], 1) and integer(a["audited_responses"], 1) and
            a["audited_responses"] <= a["scheduled_requests"], "invalid fixed schedule")
    require(a["selection"] == SELECTION and a["detectors"] == DETECTORS and
            a["feature_sha256"] == digest(FEATURES), "unsupported detector/selection policy")
    require(type(a["alpha"]) in (int, float) and 0 < a["alpha"] < 1 and
            a["alpha_scope"] == "one_audit_only", "invalid alpha or unsupported campaign")
    require(isinstance(a["traffic_profile"], str) and a["traffic_profile"], "traffic profile required")
    fields(s["calibration"], "sha256 publisher_id require_validated_operating_point")
    require(sha(s["calibration"]["sha256"]) and
            type(s["calibration"]["require_validated_operating_point"]) is bool, "unresolved calibration")
    fields(s["budget"], "max_requests max_generated_tokens max_reference_prefill_tokens max_cost_usd")
    require(all(integer(s["budget"][k]) for k in ("max_requests", "max_generated_tokens",
                                                 "max_reference_prefill_tokens")), "invalid budget")
    cost = s["budget"]["max_cost_usd"]
    require(cost is None or (type(cost) in (int, float) and math.isfinite(cost) and cost >= 0),
            "invalid monetary budget")
    fields(s["identities"], "auditor provider reference")
    require(all(isinstance(v, str) and v for v in s["identities"].values()), "signer IDs required")
    require(r["executor_id"] == s["identities"]["reference"], "reference identity mismatch")
    return s


def profile(s):
    """Applicability binds the entire fixed block design, not just GPU name."""
    return digest({k: s[k] for k in ("model", "generation", "reference", "audit")})


def decision(s, calibration, scores, reasons=(), unsupported=False):
    reasons = list(reasons)
    a = s["audit"]
    threshold = a["alpha"] / len(DETECTORS)
    ps, resolution = {}, {}
    require(isinstance(calibration, dict) and {"version", "profile_sha256", "conditions"} <= set(calibration) and
            set(calibration) <= {"version", "profile_sha256", "conditions", "status", "power_reference",
                                 "source_artifact_roots", "partitions"}, "invalid calibration fields")
    require(calibration.get("status", "exploratory") in ("provisioning", "exploratory"), "unsupported calibration status")
    require(calibration["version"] == VERSION, "unsupported calibration version")
    if calibration["profile_sha256"] != profile(s):
        reasons.append("calibration_profile_not_applicable")
    if s["calibration"]["require_validated_operating_point"]:
        # This MVP cannot certify deployment coverage from a publisher boolean.
        reasons.append("validated_operating_point_not_available_in_v1")
    cells = calibration["conditions"]
    require(isinstance(cells, dict) and (cells or calibration.get("status") == "provisioning"), "empty honest calibration")
    if not cells:
        reasons.append("calibration_provisioning")
    seen = set()
    for name, blocks in cells.items():
        require(isinstance(name, str) and isinstance(blocks, list) and blocks, "empty condition")
        for block in blocks:
            fields(block, "block_id scores")
            require(isinstance(block["block_id"], str) and block["block_id"] not in seen,
                    "duplicate calibration block")
            seen.add(block["block_id"])
            valid_scores(block["scores"])
    for d in DETECTORS:
        resolution[d] = max((1 / (len(rows) + 1) for rows in cells.values()), default=1.0)
        if scores is not None and cells:
            valid_scores(scores)
            ps[d] = {h: (1 + sum(row["scores"][d] >= scores[d] for row in rows)) / (len(rows) + 1)
                     for h, rows in cells.items()}
    if max(resolution.values()) > threshold:
        reasons.append("insufficient_calibration_resolution")
    if scores is None:
        reasons.append("reference_scores_unavailable")
    combined = {d: max(p.values()) for d, p in ps.items()}
    outcome = ("unsupported" if unsupported else "inconclusive" if reasons else
               "inconsistent" if any(p <= threshold for p in combined.values()) else
               "no_deviation_detected")
    return dict(outcome=outcome, reasons=sorted(set(reasons)), p_values=combined,
                condition_p_values=ps, threshold=threshold, minimum_p_values=resolution,
                alpha=a["alpha"], alpha_scope=a["alpha_scope"],
                scope="captured_requests_in_one_deployment_epoch",
                provider_endorsement=s["mode"] == "receipted" and not unsupported,
                reference_trust="adopter_controlled_or_explicitly_trusted_signer",
                operating_point="exploratory_not_deployment_validated",
                assumptions=["independent exchangeable complete blocks within a represented condition",
                             "traffic and execution applicability require independent trust",
                             "local plan signature does not prevent discarding entire sessions"],
                power_reference=calibration.get("power_reference"),
                tested_claim="behavioral consistency; no proof of precision, GPU identity or compute")


def valid_scores(scores):
    require(isinstance(scores, dict) and set(scores) == set(DETECTORS) and
            all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in scores.values()),
            "invalid detector scores")
