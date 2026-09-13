"""Reproducible complete-block H100 protocol experiment, with real NF4 weights.

Run: python -m experiments.exp_audit_protocol_gpu --snapshot /path/to/snapshot
No token bootstrap. Each development/calibration/test block is freshly generated
over HTTP, committed before generation, selected after provider-signed closure,
independently scored, sealed and verified. This is a single-host pilot.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import threading
import time
from pathlib import Path

from ivgym.audit.contract import DETECTORS, FEATURES, SELECTION, profile
from ivgym.audit.crypto import VERSION, digest, new_key, public, read, sign, write
from ivgym.audit.hf import HFModel, HFReferenceExecutor, software
from ivgym.audit.http import HTTPProviderClient, ReceiptMiddleware, make_server
from ivgym.audit.protocol import AuditRunner, replay, verify


def prompt_block(rng, tokenizer, count):
    prompts = []
    for _ in range(count):
        a, b, c = [rng.randrange(100, 99999) for _ in range(3)]
        templates = [f"Explain how to compute {a} plus {b}. Start with the calculation:",
                     f"Write a Python function that sorts the list [{a}, {b}, {c}].\n",
                     f"A fictional island has {a} residents and {b} trees. Describe a day there:\n",
                     f"Explain a database with {a} records split across {b % 31 + 2} servers:\n"]
        prompts.append(tokenizer.encode(rng.choice(templates), add_special_tokens=False))
    return prompts


def interval(k, n):
    from scipy.stats import beta
    return [float(beta.ppf(.025, k, n - k + 1)) if k else 0.,
            float(beta.ppf(.975, k + 1, n - k)) if k < n else 1.]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--repository", default="Qwen/Qwen3-0.6B")
    p.add_argument("--out", default="runs/audit-h100")
    p.add_argument("--calibration-blocks", type=int, default=63)
    p.add_argument("--test-blocks", type=int, default=20)
    p.add_argument("--attack-blocks", type=int, default=10)
    p.add_argument("--tokens", type=int, default=24)
    p.add_argument("--skip-nf4", action="store_true")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    keys_dir = out / "keys"
    keys_dir.mkdir(mode=0o700)
    keys = {role: new_key(keys_dir / (role + ".pem")) for role in ("auditor", "provider", "reference", "calibration")}
    trust = {role: {role: public(key)} for role, key in keys.items()}
    write(out / "trust.json", trust)
    cfg = dict(snapshot=str(Path(args.snapshot).resolve()), repository=args.repository,
               revision=Path(args.snapshot).name, attention="eager")
    write(out / "reference-config.json", cfg)
    started = time.perf_counter()
    print("Loading separate provider and reference model instances", flush=True)
    provider_model = HFModel(**cfg)
    reference_model = HFModel(**cfg)
    reference = HFReferenceExecutor(reference_model)
    import torch
    torch.manual_seed(20260913)
    write(out / "software.json", software())
    write(out / "model-files.json", reference_model.files)
    spec = dict(version=VERSION, mode="receipted", model=reference_model.manifest,
        generation=dict(temperature=1., top_p=1., top_k=None, max_output_tokens=args.tokens,
                        eos_token_ids=[provider_model.tokenizer.eos_token_id], context_limit=512,
                        context_policy="reject_overflow", input_mode="raw_token_ids"),
        reference=reference_model.profile(),
        calibration=dict(sha256="0" * 64, publisher_id="calibration", require_validated_operating_point=False),
        audit=dict(scheduled_requests=4, audited_responses=2, selection=SELECTION,
                   detectors=DETECTORS, feature_sha256=digest(FEATURES), alpha=.05,
                   alpha_scope="one_audit_only", traffic_profile="synthetic-independent-numeric-sources-v1"),
        budget=dict(max_requests=4, max_generated_tokens=4 * args.tokens,
                    max_reference_prefill_tokens=2048, max_cost_usd=None),
        identities=dict(auditor="auditor", provider="provider", reference="reference"))
    pending = sign(dict(version=VERSION, profile_sha256=profile(spec), conditions={},
                        status="provisioning", power_reference=None), keys["calibration"],
                   "calibration", "calibration", "calibration")
    spec["calibration"]["sha256"] = digest(pending)
    seen_prompts, blocks = set(), []

    def run_block(label, index, s, bundle, model, rng, temperature=None):
        prompts = prompt_block(rng, provider_model.tokenizer, 4)
        for prompt in prompts:
            assert digest(prompt) not in seen_prompts, "source crossed block partitions"
            seen_prompts.add(digest(prompt))
        # Reserve a port, then bind it into the signed manifest.
        server = make_server(("127.0.0.1", 0), None)
        port = server.server_port
        server.server_close()
        endpoint = f"http://127.0.0.1:{port}"
        middleware = ReceiptMiddleware(s, keys["provider"], trust,
                         lambda req: model.generate(req, temperature), endpoint, f"{label}-{index}")
        server = make_server(("127.0.0.1", port), middleware)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .05}, daemon=True)
        thread.start()
        destination = out / label / f"block-{index:04d}"
        try:
            t = time.perf_counter()
            runner = AuditRunner(HTTPProviderClient(endpoint), reference, keys["auditor"], keys["reference"], trust)
            verdict = runner.run(s, bundle, prompts, destination)
            wall = time.perf_counter() - t
            checked = verify(destination, trust)
            assert checked["integrity"] == "valid", checked
            assert verdict["scores"] is not None, verdict
            rows = read(destination / "records.json")
            generated = sum(len(r["capture"]["payload"]["output_token_ids"]) for r in rows)
            result = dict(partition=label, block_id=f"{label}-{index}", artifact=str(destination.relative_to(out)),
                          outcome=verdict["outcome"], scores=verdict["scores"], p_values=verdict["p_values"],
                          costs=verdict["costs"], wall_seconds=wall, generated_tokens=generated,
                          source_sha256=digest(prompts), transcript_sha256=digest(read(destination / "manifest.json")))
            blocks.append(result)
            write(out / "progress.json", blocks)
            print(f"{label} {index + 1}: {verdict['outcome']} scores={verdict['scores']} wall={wall:.2f}s", flush=True)
            return result
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    # Detector transforms are frozen in source before all three partitions.
    dev_rng, cal_rng, test_rng = random.Random(1001), random.Random(2002), random.Random(3003)
    for i in range(3):
        run_block("development", i, spec, pending, provider_model, dev_rng)
    honest_cal = [run_block("calibration", i, spec, pending, provider_model, cal_rng)
                  for i in range(args.calibration_blocks)]
    bundle = sign(dict(version=VERSION, profile_sha256=profile(spec), status="exploratory",
                        conditions={"h100_hf_eager_bf16": [dict(block_id=b["block_id"], scores=b["scores"]) for b in honest_cal]},
                        source_artifact_roots={b["block_id"]: b["transcript_sha256"] for b in honest_cal},
                        partitions=dict(development=3, calibration=args.calibration_blocks, evaluation="fresh held-out blocks"),
                        power_reference=None), keys["calibration"], "calibration", "calibration", "calibration")
    spec["calibration"]["sha256"] = digest(bundle)
    write(out / "spec.json", spec)
    write(out / "calibration.json", bundle)
    for i in range(args.test_blocks):
        run_block("honest", i, spec, bundle, provider_model, test_rng)
    for i in range(args.attack_blocks):
        run_block("temperature_2", i, spec, bundle, provider_model, test_rng, temperature=2.)
    if not args.skip_nf4:
        print("Loading actual bitsandbytes NF4 weights for deviating HTTP provider", flush=True)
        quantized = HFModel(**cfg, quantization="nf4")
        from bitsandbytes.nn import Linear4bit
        nf4_layers = sum(isinstance(m, Linear4bit) for m in quantized.model.modules())
        assert nf4_layers > 0
        write(out / "nf4-manifest.json", dict(base_model=quantized.manifest, linear4bit_layers=nf4_layers,
                    quantization="NF4", compute_dtype="bfloat16", double_quant=True,
                    provider_parameter_bytes=quantized.model.get_memory_footprint(),
                    bf16_parameter_bytes=provider_model.model.get_memory_footprint()))
        for i in range(args.attack_blocks):
            run_block("real_nf4", i, spec, bundle, quantized, test_rng)
    replay_results = {}
    for label in ("honest", "temperature_2", "real_nf4"):
        path = out / label / "block-0000"
        if path.exists():
            replay_results[label] = replay(path, trust, reference)
            assert replay_results[label]["replay"] == "agrees", replay_results[label]
    summary = {}
    for label in ("honest", "temperature_2", "real_nf4"):
        rows = [b for b in blocks if b["partition"] == label]
        if not rows:
            continue
        n, k = len(rows), sum(b["outcome"] == "inconsistent" for b in rows)
        summary[label] = dict(blocks=n, flags=k, flag_rate=k/n, binomial_95_interval=interval(k, n),
                             abstentions=sum(b["outcome"] in ("inconclusive", "unsupported") for b in rows),
                             mean_scores={d: sum(b["scores"][d] for b in rows)/n for d in DETECTORS},
                             mean_wall_seconds=sum(b["wall_seconds"] for b in rows)/n,
                             mean_reference_gpu_seconds=sum(b["costs"]["reference_gpu_seconds"] for b in rows)/n,
                             mean_reference_prefill_tokens=sum(b["costs"]["reference_prefill_tokens"] for b in rows)/n,
                             mean_wire_bytes=sum(b["costs"]["wire_bytes"] for b in rows)/n)
    report = dict(version=VERSION, hardware=reference_model.profile(), software=software(),
                  model=spec["model"], spec_sha256=digest(spec), calibration_sha256=digest(bundle),
                  selection=dict(scheduled=4, audited=2, max_output_tokens=args.tokens),
                  alpha=.05, calibration_blocks=args.calibration_blocks,
                  operating_point="single-H100 exploratory pilot; not deployment validated",
                  confidence_interval_scope="descriptive conditional on one frozen calibration; independence and representativeness assumed",
                  summary=summary, replay=replay_results, total_wall_seconds=time.perf_counter()-started,
                  limitations=["no cross-hardware, vLLM, batching or routing coverage",
                               "no rare-FPR certification; 1e-4 operating point requires far more fresh blocks",
                               "receipt signer and reference are locally trusted, not hardware-attested",
                               "synthetic prompt domain and one serving session/day",
                               "no privacy backend, exact-relation proof or external plan witness"], blocks=blocks)
    write(out / "report.json", report)
    write(out / "report.sig", sign(report, keys["auditor"], "auditor", "auditor", "experiment-report"))
    print(json.dumps({k: v for k, v in report.items() if k != "blocks"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
