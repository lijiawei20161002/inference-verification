"""Freeze honest evidence once, replay it on multiple GPUs, compare paired scores.

Each output directory is new and contains a status manifest. Capture saves exact
prompt/token IDs and projected activation evidence; replay never regenerates text.
Only the clean arm is measured. Scores use the existing harness and shape backend.
Activation-DiFR is an additional-visibility control, not public token evidence.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import time

import numpy as np

from experiments.exp_benign_shape_dprime_gpu import ShapeBackend, paired_d_prime
from ivgym import attacks, harness, verifiers
from ivgym.backends.hf_gpu import DEFAULT_PROMPTS
from ivgym.core import SamplingSpec, Sequence, TokenStep
from scripts.run_research_job import gpu_uuid, host_lock

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DETECTORS = ("token_difr", "cross_entropy", "token_toploc", "activation_difr")
SHAPES = ("chunked_b1", "chunked_b4", "chunked_b1_sdpa", "sequential")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_hashes():
    paths = list((ROOT / "ivgym").rglob("*.py")) + [
        Path(__file__), ROOT / "experiments/exp_benign_shape_dprime_gpu.py",
        ROOT / "scripts/run_research_job.py",
    ]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def provenance():
    import torch
    return {
        "hostname": socket.gethostname(), "device": torch.cuda.get_device_name(0),
        "gpu_uuid": gpu_uuid(os.environ.get("CUDA_VISIBLE_DEVICES", "0")),
        "driver": subprocess.check_output([
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
        ], text=True).strip(),
        "software": {p: importlib.metadata.version(p) for p in
                     ("torch", "transformers", "numpy", "accelerate")},
        "cuda_runtime": torch.version.cuda,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "torch_threads": torch.get_num_threads(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": source_hashes(),
    }


def make_backend(model, revision, **kwargs):
    from huggingface_hub import snapshot_download
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Model revision must be an immutable 40-character SHA")
    snapshot = snapshot_download(model, revision=revision, local_files_only=True)
    return ShapeBackend(model_name=snapshot, ref_noise=False, verifier_sigma=0.0,
                        act_benign_sigma=0.0, lazy_reference=True, **kwargs)


def validate_capture(data):
    if data.get("schema") != "ivgym-frozen-capture-v1":
        raise ValueError("Unsupported capture schema")
    if (data.get("dtype") != "bfloat16" or data.get("injected_noise") is not False
            or data.get("projection") != {"seed": 123, "dimension": 32}
            or data.get("generation_shape") != "sequential_eager"):
        raise ValueError("Unsupported capture/scoring contract")
    if not re.fullmatch(r"[0-9a-f]{40}", data["revision"]):
        raise ValueError("Capture revision must be immutable")
    if type(data["n_tokens"]) is not int or data["n_tokens"] < 1:
        raise ValueError("Invalid token count")
    rows = data["sequences"]
    if not rows or len(rows) != data["n_prompts"]:
        raise ValueError("Capture sequence count mismatch")
    if len({r["prompt_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate prompt ID")
    for row in rows:
        if type(row["prompt_id"]) is not int or row["prompt_id"] < 0:
            raise ValueError("Invalid prompt ID")
        if not row["prompt_tokens"] or len(row["claimed_tokens"]) != data["n_tokens"]:
            raise ValueError("Missing prompt or incomplete generated tokens")
        ids = row["prompt_tokens"] + row["claimed_tokens"]
        if any(type(t) is not int or not 0 <= t < data["vocab"] for t in ids):
            raise ValueError("Token outside model vocabulary")
        fps = np.asarray(row["fingerprints"], dtype=float)
        if fps.shape != (data["n_tokens"], 32) or not np.isfinite(fps).all():
            raise ValueError("Invalid activation fingerprints")
    SamplingSpec(**data["sampling"])
    return data


def load_capture(path):
    with gzip.open(path, "rt") as f:
        return validate_capture(json.load(f))


def restore_sequence(row, spec):
    return Sequence(row["prompt_id"], "honest", [
        TokenStep(j, token, spec, np.asarray(fp, dtype=float))
        for j, (token, fp) in enumerate(zip(row["claimed_tokens"], row["fingerprints"]))
    ])


def capture(args, dest, meta):
    if not 1 <= args.prompts <= len(DEFAULT_PROMPTS) or args.tokens < 1:
        raise ValueError("Use positive lengths and distinct prompts from the built-in bank")
    backend = make_backend(args.model, args.revision)
    spec = SamplingSpec(seed=args.seed)
    data = dict(schema="ivgym-frozen-capture-v1", model=args.model,
                revision=args.revision, dtype="bfloat16", sampling=asdict(spec),
                n_prompts=args.prompts, n_tokens=args.tokens, vocab=backend.vocab,
                hidden_dim=backend.hidden_dim, max_prompt_tokens=32,
                projection=dict(seed=123, dimension=32),
                generation_shape="sequential_eager", injected_noise=False,
                provenance=meta["provenance"], sequences=[])
    for i in range(args.prompts):
        seq = backend.generate(i, args.tokens, spec, attacks.get("honest"), True)
        data["sequences"].append(dict(
            prompt_id=i, prompt_text=backend.prompts[i],
            prompt_tokens=backend._prompt_ids(i)[0].cpu().tolist(),
            claimed_tokens=[s.claimed_token for s in seq.steps],
            fingerprints=[s.fingerprint.tolist() for s in seq.steps]))
        backend._served_cache.clear()
        backend.drop_reference_cache()
        print(f"Captured {i+1}/{args.prompts} sequences", flush=True)
    validate_capture(data)
    artifact = dest / "capture.json.gz"
    with gzip.open(artifact, "wt") as f:
        json.dump(data, f, allow_nan=False)
    meta.update(capture_sha256=sha256(artifact), n_prompts=args.prompts,
                n_tokens=args.tokens, artifact=artifact.name)


def replay(args, dest, meta):
    data = load_capture(args.capture)
    capture_digest = sha256(args.capture)
    if capture_digest != args.capture_sha256:
        raise ValueError("Capture hash differs from the frozen capture manifest")
    if data["provenance"]["source_sha256"] != meta["provenance"]["source_sha256"]:
        raise ValueError("Capture and replay use different source code")
    backend = make_backend(data["model"], data["revision"])
    if (backend.vocab, backend.hidden_dim) != (data["vocab"], data["hidden_dim"]):
        raise ValueError("Model dimensions differ from capture")
    spec = SamplingSpec(**data["sampling"])
    dets = [verifiers.get(d) for d in DETECTORS]
    n, t = data["n_prompts"], data["n_tokens"]
    scores = {f"{shape}/{d}": np.empty((n, t), dtype=float)
              for shape in (*SHAPES, "rerun") for d in DETECTORS}
    for i, row in enumerate(data["sequences"]):
        seq = restore_sequence(row, spec)
        pid = seq.prompt_id
        # Replay the saved IDs directly, never retokenize or regenerate the text.
        prompt = backend._torch.tensor([row["prompt_tokens"]], device=backend.device)
        backend._claimed[pid] = row["claimed_tokens"]
        backend._prompt_len[pid] = len(row["prompt_tokens"])
        for shape in (*SHAPES, "rerun"):
            backend.ref_shape = "chunked_b1" if shape == "rerun" else shape
            backend.drop_reference_cache()
            backend._populate_ref_cache(pid, prompt, row["claimed_tokens"], t)
            # ShapeBackend has already populated exactly t rows, so the harness
            # prefill call is a cache hit even though generation was lazy.
            result = harness.verify(backend, [seq], spec, dets)
            for d in DETECTORS:
                scores[f"{shape}/{d}"][i] = result.scores[d]
        backend.drop_reference_cache()
        print(f"Replayed {i+1}/{n} sequences, {len(SHAPES)} shapes + control", flush=True)
    controls = {}
    comparisons = {}
    for d in DETECTORS:
        base = scores[f"chunked_b1/{d}"].ravel()
        controls[d] = bool(np.array_equal(base, scores[f"rerun/{d}"].ravel()))
        for shape in SHAPES[1:]:
            comparisons.setdefault(shape, {})[d] = paired_d_prime(
                base, scores[f"{shape}/{d}"].ravel(), n, t, args.bootstrap)
    np.savez_compressed(dest / "scores.npz", **scores)
    summary = dict(capture_sha256=capture_digest, n_prompts=n, n_tokens=t,
                   same_shape_exact=controls, comparisons=comparisons,
                   bootstrap=dict(unit="sequence", draws=args.bootstrap,
                                  seed=0, interval=0.90),
                   scope="Paired clean honest replay; no attack power or deployment FPR")
    write_json(dest / "summary.json", summary)
    meta.update(capture_sha256=capture_digest, n_prompts=n, n_tokens=t,
                scores_sha256=sha256(dest / "scores.npz"),
                summary_sha256=sha256(dest / "summary.json"),
                same_shape_exact=controls)
    if not all(controls.values()):
        raise RuntimeError("Same-shape replay was not exactly reproducible")


def compare_runs(base_dir, alt_dir, bootstrap=400):
    directories = [Path(base_dir), Path(alt_dir)]
    metas = [json.loads((d / "manifest.json").read_text()) for d in directories]
    for directory, meta in zip(directories, metas):
        if meta["status"] != "complete" or meta["command"] != "replay":
            raise ValueError("Both runs must be completed replays")
        if sha256(directory / "scores.npz") != meta["scores_sha256"]:
            raise ValueError("Score artifact hash mismatch")
        if not all(meta["same_shape_exact"].values()):
            raise ValueError("Failed same-shape control")
    for key in ("capture_sha256", "n_prompts", "n_tokens"):
        if metas[0][key] != metas[1][key]:
            raise ValueError(f"Cannot pair runs with different {key}")
    for key in ("source_sha256", "software", "cuda_runtime", "tf32_matmul",
                "tf32_cudnn", "torch_threads"):
        if metas[0]["provenance"][key] != metas[1]["provenance"][key]:
            raise ValueError(f"Replay conditions differ: {key}")
    result = dict(capture_sha256=metas[0]["capture_sha256"],
                  provenance=[m["provenance"] for m in metas], comparisons={},
                  scope="Paired honest scores; hardware and driver may both differ")
    with np.load(directories[0] / "scores.npz", allow_pickle=False) as base, \
            np.load(directories[1] / "scores.npz", allow_pickle=False) as alt:
        for shape in SHAPES:
            result["comparisons"][shape] = {}
            for detector in DETECTORS:
                key = f"{shape}/{detector}"
                expected = (metas[0]["n_prompts"], metas[0]["n_tokens"])
                if base[key].shape != expected or alt[key].shape != expected:
                    raise ValueError("Score dimensions differ from manifest")
                result["comparisons"][shape][detector] = paired_d_prime(
                    base[key].ravel(), alt[key].ravel(), *expected, bootstrap)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--model", default=MODEL)
    cap.add_argument("--revision", default=REVISION)
    cap.add_argument("--prompts", type=int, default=80)
    cap.add_argument("--tokens", type=int, default=256)
    cap.add_argument("--seed", type=int, default=42)
    rep = sub.add_parser("replay")
    rep.add_argument("--capture", type=Path, required=True)
    rep.add_argument("--capture-sha256", required=True)
    rep.add_argument("--bootstrap", type=int, default=400)
    for p in (cap, rep):
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--node", required=True)
    cmp = sub.add_parser("compare")
    cmp.add_argument("--baseline", type=Path, required=True)
    cmp.add_argument("--alternate", type=Path, required=True)
    cmp.add_argument("--output", type=Path, required=True)
    cmp.add_argument("--bootstrap", type=int, default=400)
    args = parser.parse_args()
    if hasattr(args, "bootstrap") and args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    if args.command == "compare":
        result = compare_runs(args.baseline, args.alternate, args.bootstrap)
        with args.output.open("x") as f:
            json.dump(result, f, indent=2, allow_nan=False)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    meta = dict(schema="ivgym-frozen-run-v1", command=args.command, node=args.node,
                status="running", started_utc=datetime.now(timezone.utc).isoformat())
    started = time.monotonic()
    try:
        with ExitStack() as locks:
            lock_root = Path(tempfile.gettempdir())
            locks.enter_context(host_lock(lock_root / f"ivgym-research-{os.getuid()}.lock",
                                          shared=True))
            uuid = gpu_uuid(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
            locks.enter_context(host_lock(lock_root / f"ivgym-research-{os.getuid()}-{uuid}.lock"))
            meta["provenance"] = provenance()
            write_json(args.output / "manifest.json", meta)
            {"capture": capture, "replay": replay}[args.command](args, args.output, meta)
            meta["status"] = "complete"
    except BaseException as exc:
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        meta.update(elapsed_seconds=time.monotonic() - started,
                    finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output / "manifest.json", meta)


if __name__ == "__main__":
    main()
