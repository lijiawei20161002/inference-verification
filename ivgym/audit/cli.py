"""Command line interface; offline verification does not import torch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .crypto import load_key, new_key, public, read, require
from .protocol import AuditRunner, replay, verify


def reference(config):
    from .hf import HFModel, HFReferenceExecutor
    return HFReferenceExecutor(HFModel(**read(config)))


class LazyReference:
    def __init__(self, config):
        self.config, self.executor = config, None

    def score(self, records, spec):
        if self.executor is None:
            self.executor = reference(self.config)
        return self.executor.score(records, spec)


def main(argv=None):
    p = argparse.ArgumentParser(prog="ivgym", description="Experimental authenticated statistical audit v1")
    sub = p.add_subparsers(dest="command", required=True)
    key = sub.add_parser("keygen", help="create a mode-0600 Ed25519 private key; print its public key")
    key.add_argument("path")
    for command in ("verify", "replay"):
        parser = sub.add_parser(command)
        parser.add_argument("transcript")
        parser.add_argument("--trust", required=True)
        if command == "replay":
            parser.add_argument("--reference", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("endpoint")
    audit.add_argument("--model", help="optional assertion of model repository name")
    for name in ("spec", "calibration", "prompts", "reference", "trust", "auditor-key", "reference-key", "out"):
        audit.add_argument("--" + name, required=True)
    serve = sub.add_parser("serve", help="local HF test server with exact-token receipt extension")
    for name in ("spec", "reference", "provider-key", "trust"):
        serve.add_argument("--" + name, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--epoch", default="local-experiment-1")
    serve.add_argument("--temperature-override", type=float)
    args = p.parse_args(argv)
    try:
        if args.command == "keygen":
            print(json.dumps({"public_key": public(new_key(args.path))}))
            return 0
        trust = read(args.trust)
        if args.command == "verify":
            result = verify(args.transcript, trust)
        elif args.command == "replay":
            # Authenticate before loading any heavyweight reference.
            checked = verify(args.transcript, trust)
            result = (dict(replay="unsupported", verification=checked) if checked["integrity"] != "valid" else
                      replay(args.transcript, trust, reference(args.reference)))
        elif args.command == "audit":
            from .http import HTTPProviderClient
            spec = read(args.spec)
            require(args.model is None or args.model == spec["model"]["repository"], "model assertion mismatch")
            runner = AuditRunner(HTTPProviderClient(args.endpoint), LazyReference(args.reference),
                                 load_key(args.auditor_key), load_key(args.reference_key), trust)
            result = runner.run(spec, read(args.calibration), read(args.prompts), args.out)
        else:
            from .http import ReceiptMiddleware, make_server
            spec, model = read(args.spec), reference(args.reference).model
            middleware = ReceiptMiddleware(spec, load_key(args.provider_key), trust,
                lambda req: model.generate(req, args.temperature_override),
                f"http://{args.host}:{args.port}", args.epoch, receipts=spec["mode"] == "receipted")
            server = make_server((args.host, args.port), middleware)
            print(f"Listening on http://{args.host}:{args.port}", flush=True)
            try:
                server.serve_forever()
            finally:
                server.server_close()
            return 0
        print(json.dumps(result, indent=2, allow_nan=False))
        if result.get("integrity") in ("invalid", "unverifiable") or result.get("replay") in ("unsupported", "disagreement"):
            return 1
        return 2 if result.get("outcome") in ("unsupported", "inconclusive") else 0
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__ + ": " + str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
