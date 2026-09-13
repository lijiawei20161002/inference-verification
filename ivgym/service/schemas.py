from __future__ import annotations
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

TokenId = Annotated[int, Field(strict=True, ge=0, le=2**31 - 1)]
Detector = Literal["token_difr", "cross_entropy", "token_toploc", "clock_slope"]


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Sampling(Schema):
    temperature: float = Field(default=1.0, gt=0, le=10)
    top_k: int | None = Field(default=None, ge=1, le=1000000)
    top_p: float = Field(default=1.0, gt=0, le=1)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)


class Capture(Schema):
    prompt_token_ids: list[TokenId] | None = Field(
        default=None, min_length=1, max_length=32768
    )
    output_token_ids: list[TokenId] | None = Field(
        default=None, min_length=1, max_length=2048
    )
    prompt: str | None = Field(default=None, min_length=1, max_length=131072)
    output: str | None = Field(default=None, min_length=1, max_length=32768)
    prompt_id: int = Field(default=0, ge=0, le=2**31 - 1)
    sampler_contract: Literal["ivgym_numpy_gumbel_v1"] | None = None

    @model_validator(mode="after")
    def complete_input(self):
        ids = self.prompt_token_ids is not None or self.output_token_ids is not None
        if ids and (self.prompt_token_ids is None or self.output_token_ids is None):
            raise ValueError("supply both prompt_token_ids and output_token_ids")
        if not ids and (self.prompt is None or self.output is None):
            raise ValueError("supply exact IDs or both prompt and output text")
        if ids and (self.prompt is not None or self.output is not None):
            raise ValueError("choose exact IDs or text, not both")
        return self


class TimingPair(Schema):
    short_context_tokens: int = Field(ge=1, le=1000000)
    long_context_tokens: int = Field(ge=2, le=1000000)
    short_itl_ms: list[Annotated[float, Field(ge=0, le=600000)]] = Field(
        min_length=1, max_length=4096
    )
    long_itl_ms: list[Annotated[float, Field(ge=0, le=600000)]] = Field(
        min_length=1, max_length=4096
    )
    unit: Literal["token", "sse_chunk", "device_token"] = "token"

    @model_validator(mode="after")
    def paired(self):
        if self.long_context_tokens <= self.short_context_tokens:
            raise ValueError("long context must exceed short context")
        if len(self.short_itl_ms) != len(self.long_itl_ms):
            raise ValueError(
                "pair equally many intervals; do not silently truncate captures"
            )
        return self


class Target(Schema):
    base_url: str = Field(min_length=8, max_length=2048)
    model: str = Field(min_length=1, max_length=200)
    api_key: SecretStr | None = None
    sampler_contract: Literal["ivgym_numpy_gumbel_v1"] | None = None


class ClockProbes(Schema):
    short_prompt: str = Field(min_length=1, max_length=131072)
    long_prompt: str = Field(min_length=2, max_length=131072)
    pairs: int = Field(default=2, ge=1, le=8)


class VerificationRequest(Schema):
    model: str = Field(
        min_length=1,
        max_length=200,
        description="Reference model ID configured by the independent service",
    )
    verifiers: list[Detector] = Field(
        default_factory=lambda: ["token_toploc", "cross_entropy", "clock_slope"],
        min_length=1,
        max_length=4,
    )
    sampling: Sampling = Field(default_factory=Sampling)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    captures: list[Capture] = Field(default_factory=list, max_length=16)
    timing_pairs: list[TimingPair] = Field(default_factory=list, max_length=16)
    target: Target | None = None
    prompts: list[Annotated[str, Field(min_length=1, max_length=131072)]] = Field(
        default_factory=list, max_length=16
    )
    clock_probes: ClockProbes | None = None
    max_tokens: int = Field(default=32, ge=2, le=256)
    include_token_scores: bool = False

    @model_validator(mode="after")
    def source(self):
        if len(set(self.verifiers)) != len(self.verifiers):
            raise ValueError("duplicate verifier")
        if self.target:
            if self.captures or self.timing_pairs:
                raise ValueError(
                    "live targets cannot be mixed with client-supplied captures"
                )
            if not self.prompts and self.clock_probes is None:
                raise ValueError("live verification needs prompts and/or clock_probes")
            if self.prompts and not any(v != "clock_slope" for v in self.verifiers):
                raise ValueError("prompts require a requested token verifier")
            if self.clock_probes and "clock_slope" not in self.verifiers:
                raise ValueError("clock_probes require clock_slope")
        elif self.prompts or self.clock_probes:
            raise ValueError("prompts and clock_probes require a target")
        elif not self.captures and not self.timing_pairs:
            raise ValueError("provide a live target or captured token/timing evidence")
        return self


class ReferenceConfig(Schema):
    repository: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    snapshot: str
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    attention: Literal["eager", "sdpa"] = "eager"
    device: Literal["cuda", "cpu"] = "cuda"


class Calibration(Schema):
    id: str
    design: dict
    conditions: dict[str, list[float]]
    provenance: str = Field(min_length=1)

    @model_validator(mode="after")
    def nonempty(self):
        if not self.conditions or any(
            not values for values in self.conditions.values()
        ):
            raise ValueError("honest calibration conditions must be nonempty")
        return self


class Settings(Schema):
    models: dict[str, ReferenceConfig] = Field(default_factory=dict)
    calibrations: list[Calibration] = Field(default_factory=list)
    calibration_files: list[str] = Field(default_factory=list)
    allowed_provider_urls: list[str] = Field(default_factory=list)
    max_pending_jobs: int = Field(default=8, ge=1, le=128)
    max_retained_jobs: int = Field(default=128, ge=8, le=10000)
    job_ttl_seconds: int = Field(default=3600, ge=1, le=86400)
    max_reference_tokens: int = Field(default=16384, ge=2, le=1000000)
    provider_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    max_request_bytes: int = Field(
        default=2 * 1024 * 1024, ge=1024, le=16 * 1024 * 1024
    )


Outcome = Literal[
    "inconsistent", "no_deviation_detected", "inconclusive", "unsupported"
]


class DetectorResult(Schema):
    status: Outcome
    reason: str | None = None
    score: float | None = None
    sample_count: int | None = None
    p_value: float | None = None
    threshold: float | None = None
    calibration_id: str | None = None
    calibration_provenance: str | None = None
    condition_p_values: dict[str, float] | None = None
    minimum_p_value: float | None = None
    design: dict | None = None
    details: dict | None = None
    token_scores: list[float] | None = None


class VerificationResponse(Schema):
    id: str
    outcome: Outcome
    verifiers: dict[str, DetectorResult]
    alpha: float
    combination: str
    parties: dict[str, str]
    evidence_provenance: Literal["service_observed", "client_reported"]
    reference: dict | None
    observations: list[dict]
    costs: dict[str, int | float]
    scope: str
    calibration_assumption: str
