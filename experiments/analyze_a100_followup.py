"""Validate archived 18 October measurements and rebuild their numerical summary.

    python -m experiments.analyze_a100_followup --check

Requires NumPy, but no CUDA, model weights, or network access.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from experiments.exp_frozen_replay_gpu import DETECTORS, SHAPES, paired_d_prime

DATA = Path(__file__).resolve().parents[1] / "docs/results/a100_followup_20261018"


def build_summary():
    provenance = json.loads((DATA / "provenance.json").read_text())
    for name, digest in provenance["files"].items():
        if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Artifact hash mismatch: {name}")
    timing = {}
    for name, contexts in (
        ("clock-window", {256, 512, 2048, 8192, 16384, 32768}),
        ("clock-stability", {256, 1024, 4096, 8192, 16384, 32768}),
    ):
        manifest = json.loads((DATA / f"{name}-manifest.json").read_text())
        if manifest["status"] != "complete" or manifest["returncode"] != 0:
            raise ValueError(f"Incomplete timing run: {name}")
        if provenance["files"][f"{name}.json"] != manifest["artifact_sha256"]:
            raise ValueError(f"Manifest/artifact mismatch: {name}")
        if any(manifest["model_refs_before"].get(model) != revision
               or manifest["model_refs_after"].get(model) != revision
               for model, revision in manifest["model_lock"].items()):
            raise ValueError(f"Model revision changed: {name}")
        data = json.loads((DATA / f"{name}.json").read_text())
        if (data["steps"], data["reps"]) != (192, 4):
            raise ValueError(f"Unexpected timing configuration: {name}")
        if len(data["cells"]) != 6 or {c["ctx"] for c in data["cells"]} != contexts:
            raise ValueError(f"Incomplete timing grid: {name}")
        for cell in data["cells"]:
            if (cell["B"] != 1 or cell["mode"] != "graph"
                    or len(cell["itl_ms"]) != 768 or len(cell["rep_means"]) != 4
                    or not all(math.isfinite(x) and x >= 0 for x in cell["itl_ms"])
                    or not math.isclose(mean(cell["itl_ms"]), cell["mean"], abs_tol=0.000051, rel_tol=0)
                    or not math.isclose(mean(cell["rep_means"]), cell["mean"], abs_tol=1e-9, rel_tol=0)):
                raise ValueError(f"Invalid samples/means: {name}/{cell['ctx']}")
        timing[name] = {c["ctx"]: c["mean"] for c in data["cells"]}
    capture = json.loads((DATA / "capture-manifest.json").read_text())
    replay = json.loads((DATA / "replay-manifest.json").read_text())
    saved = json.loads((DATA / "token-summary.json").read_text())
    if (capture["status"] != "complete" or replay["status"] != "complete"
            or capture["capture_sha256"] != replay["capture_sha256"]
            or saved["capture_sha256"] != capture["capture_sha256"]
            or replay["scores_sha256"] != provenance["files"]["scores.npz"]
            or replay["summary_sha256"] != provenance["files"]["token-summary.json"]):
        raise ValueError("Incomplete or mismatched frozen evidence")
    n, t = capture["n_prompts"], capture["n_tokens"]
    if (n, t) != (80, 256) or (replay["n_prompts"], replay["n_tokens"]) != (n, t):
        raise ValueError("Unexpected frozen capture size")
    comparisons = {}
    with np.load(DATA / "scores.npz", allow_pickle=False) as scores:
        for d in DETECTORS:
            base = scores[f"chunked_b1/{d}"]
            if base.shape != (n, t) or not np.array_equal(base, scores[f"rerun/{d}"]):
                raise ValueError(f"Failed same-shape control: {d}")
            for shape in SHAPES[1:]:
                alternate = scores[f"{shape}/{d}"]
                if alternate.shape != (n, t):
                    raise ValueError(f"Incomplete scores: {shape}/{d}")
                row = paired_d_prime(base.ravel(), alternate.ravel(), n, t, 400)
                if row != saved["comparisons"][shape][d]:
                    raise ValueError(f"Bootstrap summary does not reproduce: {shape}/{d}")
                comparisons.setdefault(shape, {})[d] = row
    smoke = json.loads((DATA / "smoke-process-comparison.json").read_text())
    if any(r["frac_tokens_changed"] != 0 for row in smoke["comparisons"].values() for r in row.values()):
        raise ValueError("Failed fresh-process smoke control")
    previous_path = DATA.parents[2] / provenance["previous_token_artifact"]
    previous_bytes = previous_path.read_bytes()
    if hashlib.sha256(previous_bytes).hexdigest() != provenance["previous_token_sha256"]:
        raise ValueError("Previous A100 token artifact hash mismatch")
    previous = json.loads(previous_bytes)["arms"]["clean"]["shapes"]
    window, stability = timing["clock-window"], timing["clock-stability"]
    shared = window.keys() & stability.keys()
    return dict(
        scope="A100 only; paired H100 replay and deployment FPR remain open",
        verified_artifacts=len(provenance["files"]), capture_sha256=capture["capture_sha256"],
        n_prompts=n, tokens_per_prompt=t, same_shape_exact=True,
        smoke_cross_process_exact=True, token_comparisons=comparisons,
        previous_a100_aggregate_statistics_exact=(comparisons == previous),
        window=[dict(context=c, mean_ms=v, gap_from_256_ms=v-window[256])
                for c, v in sorted(window.items())],
        clock_window_gap_ms=window[32768]-window[256],
        clock_stability_gap_ms=stability[32768]-stability[256],
        max_shared_context_difference_ms=max(abs(window[c]-stability[c]) for c in shared),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary = build_summary()
    path = DATA / "summary.json"
    if args.check:
        if json.loads(path.read_text()) != summary:
            raise ValueError("Saved campaign summary differs")
        print("Verified hashes, timing coverage/means, frozen pairing, rerun controls, and all bootstrap intervals.")
    else:
        path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(path)


if __name__ == "__main__":
    main()
