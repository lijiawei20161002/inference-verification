"""Independent service orchestration around the original experiment detectors."""

from __future__ import annotations
import time
import numpy as np
from ..clock import ClockSlope
from .errors import ServiceError
from .provider import ProviderClient, validate_target


class References:
    """Operator-controlled models, with one resident GPU reference at a time."""

    def __init__(self, settings):
        self.settings, self.models, self.tokenizers = settings, {}, {}

    def config(self, name):
        if name not in self.settings.models:
            raise ServiceError(
                "unknown_reference_model",
                "Reference model is not configured by this service.",
                404,
            )
        return self.settings.models[name]

    def encode(self, name, text):
        if name in self.models:
            return self.models[name].encode(text)
        config = self.config(name)
        if name not in self.tokenizers:
            from transformers import AutoTokenizer

            self.tokenizers[name] = AutoTokenizer.from_pretrained(
                config.snapshot, local_files_only=True, trust_remote_code=False
            )
        return self.tokenizers[name].encode(text, add_special_tokens=False)

    def get(self, name):
        config = self.config(name)
        if name not in self.models:
            from ..backends.reference import HFReferenceModel

            self.models.clear()
            import gc

            gc.collect()
            self.models[name] = HFReferenceModel(config)
        return self.models[name]


class VerificationEngine:
    def __init__(self, settings, references=None, provider=None):
        self.settings = settings
        self.references = references or References(settings)
        self.provider = provider or ProviderClient(settings)

    def decide(self, name, score, design, threshold, count):
        result = dict(
            status="inconclusive",
            score=score,
            sample_count=count,
            p_value=None,
            threshold=threshold,
            calibration_id=None,
            reason="no_applicable_calibration",
            design=design,
        )
        matches = [c for c in self.settings.calibrations if c.design == design]
        if not matches:
            return result
        if len(matches) != 1:
            raise ServiceError(
                "ambiguous_calibration",
                "Operator configuration has duplicate matching calibration designs.",
                500,
            )
        cal = matches[0]
        ps = {
            h: (1 + sum(v >= score for v in values)) / (len(values) + 1)
            for h, values in cal.conditions.items()
        }
        p = max(ps.values())
        minimum = max(1 / (len(values) + 1) for values in cal.conditions.values())
        result.update(
            p_value=p,
            condition_p_values=ps,
            minimum_p_value=minimum,
            calibration_id=cal.id,
            calibration_provenance=cal.provenance,
        )
        if minimum > threshold:
            result["reason"] = "insufficient_calibration_resolution"
        else:
            result.update(
                status="inconsistent" if p <= threshold else "no_deviation_detected",
                reason=None,
            )
        return result

    def verify(self, request):
        started = time.perf_counter()
        provenance = "service_observed" if request.target else "client_reported"
        captures, pairs, observations = request.captures, request.timing_pairs, []
        if request.target:
            validate_target(
                request.target.base_url, self.settings.allowed_provider_urls
            )
            encode = lambda text: self.references.encode(request.model, text)
            # Reject an unaffordable reference request before querying the provider.
            upper = sum(
                len(encode(p)) + request.max_tokens - 1 for p in request.prompts
            )
            if upper > self.settings.max_reference_tokens:
                raise ServiceError(
                    "reference_budget_exceeded",
                    "Requested token verification exceeds the service prefill limit.",
                    413,
                )
            captures, pairs, observations = self.provider.collect(request, encode)
        names = [v for v in request.verifiers if v != "clock_slope"]
        result, reference_info = {}, None
        prefill, reference_seconds = 0, 0.0
        threshold = request.alpha / len(request.verifiers)
        if names and captures:
            try:
                model = self.references.get(request.model)
                prepared = [model.prepare(c) for c in captures]
            except ServiceError:
                raise
            except (ValueError, IndexError) as exc:
                raise ServiceError("invalid_token_capture", str(exc)) from exc
            prefill = sum(len(p) + len(o) - 1 for p, o, _ in prepared)
            if prefill > self.settings.max_reference_tokens:
                raise ServiceError(
                    "reference_budget_exceeded",
                    "Captured token verification exceeds the service prefill limit.",
                    413,
                )
            supported = list(names)
            if "token_difr" in supported and any(
                mode != "exact_ids" or c.sampler_contract != "ivgym_numpy_gumbel_v1"
                for c, (_, _, mode) in zip(captures, prepared)
            ):
                supported.remove("token_difr")
                result["token_difr"] = dict(
                    status="unsupported",
                    reason="exact_ids_and_shared_sampler_contract_required",
                )
            values = {name: [] for name in supported}
            for c, prepared_row in zip(captures, prepared):
                if not supported:
                    break
                scores, elapsed = model.score(
                    prepared_row, c, request.sampling, supported
                )
                reference_seconds += elapsed
                for name in supported:
                    values[name].extend(scores[name])
            if not supported:
                prefill = 0
            reference_info = model.identity
            for name, rows in values.items():
                if not rows or not np.isfinite(rows).all():
                    raise ServiceError(
                        "invalid_reference_scores",
                        "Reference produced invalid scores.",
                        500,
                    )
                design = dict(
                    model=request.model,
                    reference=model.identity,
                    detector=name,
                    sampling=request.sampling.model_dump(),
                    provenance=provenance,
                    input_modes=[row[2] for row in prepared],
                    prompt_lengths=[len(row[0]) for row in prepared],
                    output_lengths=[len(row[1]) for row in prepared],
                    aggregate="mean_over_tokens",
                    sampler_contract="ivgym_numpy_gumbel_v1"
                    if name == "token_difr"
                    else None,
                )
                result[name] = self.decide(
                    name, float(np.mean(rows)), design, threshold, len(rows)
                )
                if request.include_token_scores:
                    result[name]["token_scores"] = rows
        else:
            for name in names:
                result[name] = dict(
                    status="unsupported", reason="token_evidence_missing"
                )
        if "clock_slope" in request.verifiers:
            if any(o.get("role") == "clock_error" for o in observations):
                result["clock_slope"] = dict(
                    status="unsupported", reason="incomplete_or_unpaired_clock_probes"
                )
            elif not pairs:
                result["clock_slope"] = dict(
                    status="unsupported", reason="paired_clock_evidence_missing"
                )
            elif len({p.unit for p in pairs}) != 1:
                result["clock_slope"] = dict(
                    status="unsupported", reason="mixed_timing_units"
                )
            else:
                scores = np.concatenate(
                    [
                        ClockSlope().score_pairs(p.short_itl_ms, p.long_itl_ms)
                        for p in pairs
                    ]
                )
                design = dict(
                    model=request.model,
                    detector="clock_slope",
                    provenance=provenance,
                    unit=pairs[0].unit,
                    contexts=[
                        [p.short_context_tokens, p.long_context_tokens] for p in pairs
                    ],
                    intervals_per_pair=[len(p.short_itl_ms) for p in pairs],
                    aggregate="mean_negative_context_difference",
                )
                result["clock_slope"] = self.decide(
                    "clock_slope", float(scores.mean()), design, threshold, len(scores)
                )
                result["clock_slope"]["details"] = dict(
                    delta_itl_ms=float(-scores.mean()),
                    short_mean_ms=float(
                        np.mean([x for p in pairs for x in p.short_itl_ms])
                    ),
                    long_mean_ms=float(
                        np.mean([x for p in pairs for x in p.long_itl_ms])
                    ),
                    unit=pairs[0].unit,
                    interpretation="A smaller long-minus-short gap increases the anomaly score.",
                )
        statuses = [r["status"] for r in result.values()]
        outcome = (
            "inconsistent"
            if "inconsistent" in statuses
            else "no_deviation_detected"
            if all(s == "no_deviation_detected" for s in statuses)
            else "unsupported"
            if all(s == "unsupported" for s in statuses)
            else "inconclusive"
        )
        return dict(
            outcome=outcome,
            verifiers=result,
            alpha=request.alpha,
            combination="bonferroni_over_requested_verifiers",
            parties=dict(
                requester="API caller",
                provider="untrusted inference endpoint",
                verifier="independent third-party service",
            ),
            evidence_provenance=provenance,
            reference=reference_info,
            observations=observations,
            costs=dict(
                reference_prefill_tokens=prefill,
                reference_seconds=reference_seconds,
                wall_seconds=time.perf_counter() - started,
            ),
            scope="behavioral token and timing evidence; not hardware attestation",
            calibration_assumption="Honest calibration blocks must be exchangeable with this request; matching a design does not prove traffic coverage.",
        )
