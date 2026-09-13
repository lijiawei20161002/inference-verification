"""Bounded single-worker GPU queue; public jobs have unguessable IDs and a TTL."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, BoundedSemaphore
import time
import uuid
from .errors import ServiceError


class Jobs:
    def __init__(self, engine, settings):
        self.engine, self.settings = engine, settings
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ivgym-verifier"
        )
        self.capacity = BoundedSemaphore(settings.max_pending_jobs)
        self.lock = Lock()
        self.jobs = {}

    def _purge(self):
        now = time.time()
        complete = [
            (j["completed_at"], key)
            for key, j in self.jobs.items()
            if j["status"] in ("succeeded", "failed")
        ]
        for completed, key in sorted(complete):
            if (
                now - completed > self.settings.job_ttl_seconds
                or len(self.jobs) >= self.settings.max_retained_jobs
            ):
                self.jobs.pop(key, None)

    def submit(self, request):
        if not self.capacity.acquire(blocking=False):
            raise ServiceError(
                "queue_full", "Verification queue is full; retry later.", 429
            )
        key = uuid.uuid4().hex
        try:
            with self.lock:
                self._purge()
                self.jobs[key] = dict(
                    id=key,
                    status="queued",
                    created_at=time.time(),
                    completed_at=None,
                    result=None,
                    error=None,
                )
            future = self.executor.submit(self._run, key, request)
            return key, future
        except Exception:
            with self.lock:
                self.jobs.pop(key, None)
            self.capacity.release()
            raise

    def _run(self, key, request):
        try:
            with self.lock:
                self.jobs[key]["status"] = "running"
            result = self.engine.verify(request)
            with self.lock:
                self.jobs[key].update(
                    status="succeeded", result=result, completed_at=time.time()
                )
            return result
        except Exception as exc:
            error = (
                dict(code=exc.code, message=exc.message, http_status=exc.status)
                if isinstance(exc, ServiceError)
                else dict(
                    code="verification_failed",
                    message="Verification could not complete; check the service model configuration.",
                    http_status=500,
                )
            )
            with self.lock:
                self.jobs[key].update(
                    status="failed", error=error, completed_at=time.time()
                )
            raise ServiceError(
                error["code"], error["message"], error["http_status"]
            ) from None
        finally:
            self.capacity.release()

    def get(self, key):
        with self.lock:
            j = self.jobs.get(key)
            if (
                j
                and j["completed_at"]
                and time.time() - j["completed_at"] > self.settings.job_ttl_seconds
            ):
                self.jobs.pop(key)
                j = None
            if j is None:
                raise ServiceError(
                    "job_not_found", "Verification job not found or expired.", 404
                )
            return dict(j)

    def close(self):
        self.executor.shutdown(wait=True)
