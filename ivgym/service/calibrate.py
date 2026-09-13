"""Operator-only provisioning of honest score profiles from fresh live requests.

Never exposed as a public endpoint. The operator must independently trust the
chosen control provider; the target of an audit cannot define its own null.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from .engine import VerificationEngine
from .errors import ServiceError
from .schemas import Settings, VerificationRequest


def load_settings(path=None):
    settings = (
        Settings.model_validate_json(Path(path).read_text()) if path else Settings()
    )
    from .schemas import Calibration

    for filename in settings.calibration_files:
        source = Path(filename)
        if not source.is_absolute() and path:
            source = Path(path).resolve().parent / source
        bundle = json.loads(source.read_text())
        if bundle.get("version") != "ivgym.service.calibration.v1":
            raise ValueError("unsupported service calibration file")
        settings.calibrations.extend(
            Calibration.model_validate(p) for p in bundle["profiles"]
        )
    designs = [json.dumps(c.design, sort_keys=True) for c in settings.calibrations]
    if len(designs) != len(set(designs)):
        raise ValueError("duplicate calibration designs in service configuration")
    return settings


def collect(settings, request, blocks, provenance, engine=None):
    if request.target is None:
        raise ServiceError(
            "live_control_required",
            "Calibration must collect fresh requests from an operator-trusted live control.",
        )
    if blocks < 2 or blocks > 10000 or not provenance.strip():
        raise ValueError("provide 2..10000 blocks and an explicit control provenance")
    engine = engine or VerificationEngine(settings)
    profiles = {}
    for i in range(blocks):
        result = engine.verify(request)
        for name, row in result["verifiers"].items():
            if row.get("score") is None:
                raise ValueError(
                    f"control cannot supply required detector {name}: {row.get('reason')}"
                )
            key = hashlib.sha256(
                json.dumps(row["design"], sort_keys=True).encode()
            ).hexdigest()
            profile = profiles.setdefault(
                key,
                dict(
                    id=f"{name}-{key[:16]}",
                    design=row["design"],
                    conditions={"trusted_control": []},
                    provenance=provenance,
                ),
            )
            profile["conditions"]["trusted_control"].append(row["score"])
        print(f"Collected honest control block {i + 1}/{blocks}", flush=True)
    return dict(
        version="ivgym.service.calibration.v1",
        blocks_collected=blocks,
        profiles=list(profiles.values()),
        caveat="Operator asserts control provenance and exchangeability. This does not certify a deployment error rate.",
    )


def main(args):
    settings = load_settings(args.config)
    request = VerificationRequest.model_validate_json(Path(args.request).read_text())
    out = Path(args.out)
    if out.exists():
        raise ValueError("calibration output already exists")
    bundle = collect(settings, request, args.blocks, args.provenance)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(bundle, f, indent=2, allow_nan=False)
    print(f"Wrote {len(bundle['profiles'])} service-owned profiles to {out}")
    return 0
