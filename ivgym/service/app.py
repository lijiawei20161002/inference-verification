"""Public third-party verification API. No provider-side deployment is needed."""

from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
import os
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from .engine import VerificationEngine
from .errors import ServiceError
from .jobs import Jobs
from .schemas import VerificationRequest, VerificationResponse


class BodyLimit:
    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        chunks = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunks.extend(message.get("body", b""))
            if len(chunks) > self.limit:
                await JSONResponse(
                    status_code=413,
                    content={
                        "error": {
                            "code": "request_too_large",
                            "message": "Request body exceeds the service limit.",
                        }
                    },
                )(scope, receive, send)
                return
            if not message.get("more_body"):
                break
        delivered = False

        async def replay_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {
                    "type": "http.request",
                    "body": bytes(chunks),
                    "more_body": False,
                }
            return await receive()

        await self.app(scope, replay_receive, send)


def create_app(settings=None, engine=None):
    if settings is None:
        filename = os.environ.get("IVGYM_SERVICE_CONFIG")
        from .calibrate import load_settings

        settings = load_settings(filename)
    engine = engine or VerificationEngine(settings)
    jobs = Jobs(engine, settings)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await asyncio.to_thread(jobs.close)

    app = FastAPI(
        title="IVGym Verification Service",
        version="0.2.0",
        lifespan=lifespan,
        description="An independent third party verifies inference providers using the token and clock detectors from IVGym experiments. Anyone can submit requests; references and calibration are owned by the service.",
    )
    app.add_middleware(BodyLimit, limit=settings.max_request_bytes)
    app.state.engine, app.state.jobs, app.state.settings = engine, jobs, settings

    @app.exception_handler(ServiceError)
    async def service_error(request, exc):
        return JSONResponse(
            status_code=exc.status,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Do not echo entire request objects or caller-supplied provider credentials.
        errors = [
            {"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_request", "details": errors}},
        )

    @app.get("/health", tags=["Service"])
    def health():
        return dict(status="ok", service="independent_verifier", api_version="v1")

    @app.get("/v1/models", tags=["Service"])
    def models():
        return dict(
            models=[
                dict(id=k, repository=v.repository, revision=v.revision)
                for k, v in settings.models.items()
            ],
            verifiers=["token_difr", "cross_entropy", "token_toploc", "clock_slope"],
            enabled_provider_urls=settings.allowed_provider_urls,
        )

    @app.get("/v1/calibrations", tags=["Service"])
    def calibrations():
        return dict(
            calibrations=[
                dict(
                    id=c.id,
                    design=c.design,
                    provenance=c.provenance,
                    blocks_per_condition={h: len(v) for h, v in c.conditions.items()},
                )
                for c in settings.calibrations
            ]
        )

    @app.post(
        "/v1/verify",
        tags=["Verification"],
        response_model=VerificationResponse,
        response_model_exclude_unset=True,
        summary="Submit evidence or a provider target and wait for verification",
    )
    async def verify(request: VerificationRequest):
        key, future = jobs.submit(request)
        result = await asyncio.shield(asyncio.wrap_future(future))
        return dict(id=key, **result)

    @app.post(
        "/v1/verifications",
        status_code=202,
        tags=["Verification"],
        summary="Queue a verification and poll its status",
    )
    def submit(request: VerificationRequest):
        key, _ = jobs.submit(request)
        return dict(
            id=key,
            status=jobs.get(key)["status"],
            status_url=f"/v1/verifications/{key}",
        )

    @app.get("/v1/verifications/{job_id}", tags=["Verification"])
    def job(job_id: str):
        return jobs.get(job_id)

    return app
