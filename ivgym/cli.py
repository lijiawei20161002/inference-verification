"""Primary service CLI; the prior artifact experiment stays an optional tool."""

from __future__ import annotations
import argparse
import os
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Preserve artifact commands, while 'serve' now means the independent verifier.
    if argv and argv[0] in ("audit", "verify", "replay", "keygen"):
        from .audit.cli import main as artifact_main

        return artifact_main(argv)
    if argv and argv[0] == "artifact-provider":
        from .audit.cli import main as artifact_main

        return artifact_main(["serve", *argv[1:]])
    parser = argparse.ArgumentParser(
        prog="ivgym", description="Independent token and clock verification service"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Run the third-party verification API")
    serve.add_argument(
        "--config",
        help="Operator-owned reference models, provider allowlist and calibration",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8000, type=int)
    calibration = sub.add_parser(
        "calibrate",
        help="Operator-only calibration from a trusted live control provider",
    )
    for name in ("config", "request", "out", "provenance"):
        calibration.add_argument("--" + name, required=True)
    calibration.add_argument("--blocks", type=int, default=128)
    args = parser.parse_args(argv)
    if args.command == "calibrate":
        from .service.calibrate import main as calibrate_main

        return calibrate_main(args)
    if args.config:
        os.environ["IVGYM_SERVICE_CONFIG"] = args.config
    import uvicorn

    uvicorn.run(
        "ivgym.service.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
