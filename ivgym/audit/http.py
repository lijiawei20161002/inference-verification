"""Exact-token HTTP capture and receipt middleware around a generation callable.

The /ivgym extension is explicitly negotiated, not a standard OpenAI endpoint.
The bundled server is for local experiments, not a hardened public service.
"""
from __future__ import annotations

import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .crypto import VERSION, authenticate, b64, canonical, digest, loads, require, sign
from .protocol import output_payload, validate_output

MAX_BODY = 32 * 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("audit endpoint redirects are not permitted")


class HTTPProviderClient:
    def __init__(self, endpoint, timeout=120):
        self.endpoint = endpoint.rstrip("/")
        require(self.endpoint.startswith(("http://", "https://")), "HTTP(S) endpoint required")
        self.timeout = timeout
        self.opener = urllib.request.build_opener(NoRedirect)

    def exchange(self, route, payload=None):
        raw = canonical(payload) if payload is not None else None
        req = urllib.request.Request(self.endpoint + "/ivgym/" + route, data=raw,
                                     headers={"Content-Type": "application/json"})
        with self.opener.open(req, timeout=self.timeout) as response:
            wire = response.read(MAX_BODY + 1)
            require(len(wire) <= MAX_BODY, "response body too large")
            return raw, wire, loads(wire)

    def deployment(self):
        try:
            return self.exchange("deployment")[2]
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 405, 501):
                return None
            raise

    def start(self, plan):
        self.exchange("start", plan)

    def complete(self, request):
        raw, wire, response = self.exchange("completions", request)
        return dict(request_wire=b64(raw), response_wire=b64(wire),
                    payload=response["payload"], receipt=response.get("receipt"))

    def close(self, closure, records):
        return self.exchange("close", dict(closure=closure, records=records))[2]


class ReceiptMiddleware:
    def __init__(self, spec, key, trust, generate, origin, epoch, receipts=True):
        self.spec, self.key, self.trust, self.generate = spec, key, trust, generate
        self.receipts, self.sessions = receipts, {}
        self.deployment = sign(dict(version=VERSION, origin=origin, spec_sha256=digest(spec),
                                    model_sha256=digest(spec["model"]), epoch=epoch,
                                    expires_unix=int(time.time()) + 86400,
                                    receipts=True, exact_token_ids=True),
                               key, spec["identities"]["provider"], "provider", "deployment") if receipts else None

    def handle(self, route, body):
        s = self.spec
        if route == "deployment":
            return self.deployment
        if route == "start":
            p = authenticate(body, self.trust, "auditor", "plan", s["identities"]["auditor"])
            require(p["spec_sha256"] == digest(s) and p["session"] not in self.sessions,
                    "wrong contract or replayed session")
            require(p["deployment_sha256"] == digest(self.deployment if s["mode"] == "receipted" else None),
                    "wrong deployment")
            require(len(p["requests"]) == s["audit"]["scheduled_requests"], "wrong schedule length")
            self.sessions[p["session"]] = dict(plan=body, responses=[], previous=None, closed=False)
            return {"accepted": True}
        if route == "completions":
            state = self.sessions[body["session"]]
            require(not state["closed"] and body == state["plan"]["payload"]["requests"][len(state["responses"])] and
                    body["spec_sha256"] == digest(s), "request replay, reorder or contract mismatch")
            tokens, text, finish = self.generate(body)
            payload = output_payload(body, self.deployment, tokens, text, finish, state["previous"])
            validate_output(payload, body, s)
            receipt = sign(payload, self.key, s["identities"]["provider"], "provider", "receipt") if self.receipts else None
            response = dict(payload=payload, receipt=receipt)
            state["responses"].append(response)
            state["previous"] = digest(receipt) if receipt else None
            return response
        if route == "close":
            c, records = body["closure"], body["records"]
            state = self.sessions[c["session"]]
            require(not state["closed"] and c["plan_sha256"] == digest(state["plan"]) and
                    c["records_sha256"] == digest(records) and
                    c["count"] == len(records) == len(state["responses"]), "incomplete or replayed closure")
            for i, (row, response) in enumerate(zip(records, state["responses"])):
                require(row["request"] == state["plan"]["payload"]["requests"][i] and
                        row["capture"]["payload"] == response["payload"] and
                        row["capture"]["receipt"] == response["receipt"] and row["error"] is None,
                        "closure changed provider response")
            state["closed"] = True
            return sign(c, self.key, s["identities"]["provider"], "provider", "closure") if self.receipts else None
        raise ValueError("unknown route")


def make_server(address, middleware):
    import threading
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.respond(None)

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", "0"))
                require(0 < n <= MAX_BODY, "invalid request body length")
                self.respond(loads(self.rfile.read(n)))
            except Exception as exc:
                self.send_json(400, {"error": type(exc).__name__ + ":" + str(exc)})

        def respond(self, body):
            if not self.path.startswith("/ivgym/"):
                self.send_json(404, {"error": "exact-token IVGym extension required"})
                return
            try:
                with lock:
                    result = middleware.handle(self.path[len("/ivgym/"):], body)
                self.send_json(200, result)
            except Exception as exc:
                self.send_json(400, {"error": type(exc).__name__ + ":" + str(exc)})

        def send_json(self, status, obj):
            raw = canonical(obj)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    return ThreadingHTTPServer(address, Handler)
