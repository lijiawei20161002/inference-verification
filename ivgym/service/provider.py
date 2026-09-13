"""Untrusted provider transport. Timings come from this service's monotonic clock.

No receipt middleware or provider-side verifier is required. The public adapter
uses the OpenAI-compatible streaming /completions endpoint. Exact-ID extensions
are optional; ordinary text output is labeled as reconstructed text evidence.
"""

from __future__ import annotations
import json
import time
from urllib.parse import urlsplit
import httpx
from .errors import ServiceError
from .schemas import Capture, TimingPair


def validate_target(url, allowed):
    p = urlsplit(url)
    if (
        p.scheme not in ("http", "https")
        or not p.hostname
        or p.username
        or p.password
        or p.query
        or p.fragment
    ):
        raise ServiceError(
            "invalid_provider_url",
            "Use an HTTP(S) base URL without credentials, query or fragment.",
        )
    url = url.rstrip("/")
    if url not in {u.rstrip("/") for u in allowed}:
        raise ServiceError(
            "provider_not_allowed",
            "This provider URL is not enabled by the service operator.",
            403,
        )
    return url


class ProviderClient:
    def __init__(self, settings, client_factory=httpx.Client):
        self.settings, self.client_factory = settings, client_factory

    def completion(self, target, prompt, sampling, max_tokens, prompt_id, encode):
        url = validate_target(target.base_url, self.settings.allowed_provider_urls)
        body = dict(
            model=target.model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            seed=sampling.seed,
            stream=True,
        )
        if sampling.top_k is not None:
            body["top_k"] = sampling.top_k
        if target.sampler_contract:
            body["ivgym_prompt_id"] = prompt_id
        headers = {"Accept": "text/event-stream"}
        if target.api_key:
            headers["Authorization"] = "Bearer " + target.api_key.get_secret_value()
        fragments, ids, arrivals, counts = [], [], [], []
        prompt_ids = None
        exact, done, finish = True, False, None
        total = 0
        started = time.perf_counter()
        try:
            with self.client_factory(
                timeout=self.settings.provider_timeout_seconds,
                follow_redirects=False,
                trust_env=False,
            ) as client:
                with client.stream(
                    "POST", url + "/completions", json=body, headers=headers
                ) as response:
                    if response.status_code != 200:
                        raise ServiceError(
                            "provider_http_error",
                            f"Provider returned HTTP {response.status_code}.",
                            502,
                        )
                    if "text/event-stream" not in response.headers.get(
                        "content-type", ""
                    ):
                        raise ServiceError(
                            "provider_not_streaming",
                            "Provider must return an SSE completion stream.",
                            502,
                        )
                    event = []
                    for line in response.iter_lines():
                        total += len(line.encode("utf-8")) + 1
                        if (
                            total > 2 * 1024 * 1024
                            or time.perf_counter() - started
                            > self.settings.provider_timeout_seconds
                        ):
                            raise ServiceError(
                                "provider_limit",
                                "Provider response exceeded the byte or wall-time limit.",
                                502,
                            )
                        if line.startswith("data:"):
                            event.append(line[5:].lstrip())
                            continue
                        if line or not event:
                            continue
                        raw = "\n".join(event)
                        event = []
                        if raw == "[DONE]":
                            done = True
                            break
                        chunk = json.loads(raw)
                        if "prompt_token_ids" in chunk:
                            candidate = chunk["prompt_token_ids"]
                            if prompt_ids is not None and candidate != prompt_ids:
                                raise ServiceError(
                                    "provider_protocol_error",
                                    "Provider changed prompt token IDs.",
                                    502,
                                )
                            prompt_ids = candidate
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue  # Usage-only frame.
                        if len(choices) != 1 or choices[0].get("index", 0) != 0:
                            raise ServiceError(
                                "provider_protocol_error",
                                "Expected one ordered completion choice.",
                                502,
                            )
                        choice = choices[0]
                        text = choice.get("text", "")
                        tokens = choice.get("token_ids")
                        if not isinstance(text, str):
                            raise ValueError("invalid completion text")
                        if text or tokens:
                            if finish is not None:
                                raise ValueError("content after finish")
                            arrivals.append((time.perf_counter() - started) * 1000)
                            fragments.append(text)
                            if tokens is None:
                                exact = False
                                counts.append(None)
                            else:
                                if (
                                    not isinstance(tokens, list)
                                    or not tokens
                                    or any(type(t) is not int or t < 0 for t in tokens)
                                ):
                                    raise ValueError("invalid token IDs")
                                ids.extend(tokens)
                                counts.append(len(tokens))
                            if len(arrivals) > max_tokens * 8 or len(ids) > max_tokens:
                                raise ServiceError(
                                    "provider_limit",
                                    "Provider exceeded the output limit.",
                                    502,
                                )
                        if choice.get("finish_reason") is not None:
                            finish = choice["finish_reason"]
            if not done or finish not in ("stop", "length") or not arrivals:
                raise ServiceError(
                    "incomplete_provider_response",
                    "Provider stream has no complete final output.",
                    502,
                )
            if exact and prompt_ids is not None and ids:
                if prompt_ids != encode(prompt):
                    raise ServiceError(
                        "provider_prompt_mismatch",
                        "Returned prompt IDs differ from the requested raw prompt.",
                        502,
                    )
                capture = Capture(
                    prompt_token_ids=prompt_ids,
                    output_token_ids=ids,
                    prompt_id=prompt_id,
                    sampler_contract=target.sampler_contract,
                )
            else:
                capture = Capture(
                    prompt=prompt, output="".join(fragments), prompt_id=prompt_id
                )
            gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
            unit = "token" if exact and all(n == 1 for n in counts) else "sse_chunk"
            return capture, dict(
                intervals_ms=gaps,
                unit=unit,
                ttft_ms=arrivals[0],
                events=len(arrivals),
                response_body_bytes=total,
                wall_seconds=time.perf_counter() - started,
            )
        except ServiceError:
            raise
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise ServiceError(
                "provider_protocol_error",
                "Provider request failed or returned malformed completion evidence.",
                502,
            ) from exc

    def collect(self, request, encode):
        captures, pairs, observations = [], [], []
        for i, prompt in enumerate(request.prompts):
            cap, obs = self.completion(
                request.target, prompt, request.sampling, request.max_tokens, i, encode
            )
            captures.append(cap)
            observations.append(dict(role="token", **obs))
        probes = request.clock_probes
        if probes:
            low_n, high_n = (
                len(encode(probes.short_prompt)),
                len(encode(probes.long_prompt)),
            )
            if high_n <= low_n:
                raise ServiceError(
                    "invalid_clock_probes",
                    "The long probe must contain more reference tokens than the short probe.",
                )
            for i in range(probes.pairs):
                measured = {}
                # Alternate order to reduce a deterministic short-first load confound.
                for role, prompt in (
                    [("short", probes.short_prompt), ("long", probes.long_prompt)]
                    if i % 2 == 0
                    else [("long", probes.long_prompt), ("short", probes.short_prompt)]
                ):
                    _, obs = self.completion(
                        request.target,
                        prompt,
                        request.sampling,
                        request.max_tokens,
                        1000 + 2 * i + (role == "long"),
                        encode,
                    )
                    measured[role] = obs
                    observations.append(dict(role=role, **obs))
                lo, hi = measured["short"], measured["long"]
                if (
                    not lo["intervals_ms"]
                    or len(lo["intervals_ms"]) != len(hi["intervals_ms"])
                    or lo["unit"] != hi["unit"]
                ):
                    observations.append(
                        dict(role="clock_error", reason="unpaired_clock_streams")
                    )
                    continue
                pairs.append(
                    TimingPair(
                        short_context_tokens=low_n,
                        long_context_tokens=high_n,
                        short_itl_ms=lo["intervals_ms"],
                        long_itl_ms=hi["intervals_ms"],
                        unit=lo["unit"],
                    )
                )
        return captures, pairs, observations
