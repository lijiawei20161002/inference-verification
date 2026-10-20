"""Summarize evaluation-only score tails in a validated quantization archive.

The capped-event ledger identifies observations to investigate. It cannot
distinguish filtered-out tokens from finite clipped margins without new captures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.analyze_real_quantization import ARMS, digest, load_run

ARCHIVE_SCHEMA = "ivgym-real-quantization-plan-v1"
DIFR_CAP = 30.0
RANK_CAP = 50.0


def summarize_tail(difr_scores, rank_scores):
    """Describe concentration, not false-positive rates or causal mechanisms."""
    d = np.asarray(difr_scores, dtype=float)
    r = np.asarray(rank_scores, dtype=float)
    if d.ndim != 2 or d.shape != r.shape or min(d.shape) < 1:
        raise ValueError("Expected matching nonempty sequence-by-position arrays")
    if (not np.isfinite(d).all() or not np.isfinite(r).all()
            or (d < 0).any() or (d > DIFR_CAP).any()
            or (r < 0).any() or (r > RANK_CAP).any()):
        raise ValueError("Scores violate the archived finite, nonnegative cap contract")
    capped = d == DIFR_CAP
    mass = float(d.sum())
    return {
        "tokens": int(d.size),
        "nonzero_scores": int(np.count_nonzero(d)),
        "capped_scores": int(capped.sum()),
        "capped_token_fraction": float(capped.mean()),
        "capped_score_mass_fraction": float(d[capped].sum() / mass) if mass else None,
        "capped_also_rank_cap_50": int(np.count_nonzero(capped & (r == RANK_CAP))),
        "sequences_with_capped_score": int(np.any(capped, axis=1).sum()),
        "mean_score": float(d.mean()),
    }


def analyze_archive(directory):
    """Validate source hashes/captures and return a summary plus capped events."""
    directory = Path(directory)
    plan, arrays = load_run(directory)
    if plan.get("schema") != ARCHIVE_SCHEMA:
        raise ValueError("Unsupported archive schema; score caps must be checked explicitly")
    corpus = json.loads((directory / "corpus.json").read_text(encoding="utf-8"))
    nc = plan["calibration"]
    report = {
        "schema": "ivgym-score-tail-diagnostic-v1",
        "archive": str(directory),
        "source": str(directory / "scores.npz"),
        "sha256": digest(directory / "scores.npz"),
        "verified_archive": True,
        "source_hashes": {
            name: digest(directory / name)
            for name in ("plan.json", "corpus.json", "scores.npz", "manifest.json")
        },
        "split": {"calibration": nc, "evaluation": plan["evaluation"]},
        "tokens_per_sequence": plan["tokens"],
        "caps": {"token_difr": DIFR_CAP, "token_rank": RANK_CAP},
        "position_convention": "Zero-based generated-token position, excluding the prompt",
        "interpretation": (
            "Score concentration only, not a false-positive rate or proof of support mismatch. "
            "Cap=30 may represent a filtered-out token or a finite clipped margin. "
            "Raw support masks and unfiltered logits are needed to attribute individual events. "
            "token_toploc is the local rank baseline, not an evaluation of original TOPLOC."
        ),
        "arms": {},
    }
    events = []
    for arm in ARMS:
        d = arrays[f"{arm}/token_difr"][nc:]
        r = arrays[f"{arm}/token_toploc"][nc:]
        report["arms"][arm] = summarize_tail(d, r)
        for sequence, position in np.argwhere(d == DIFR_CAP):
            sequence, position = int(sequence), int(position)
            row = corpus[nc + sequence]
            events.append({
                "event_id": f"{arm}:{row['prompt_id']}:{position}",
                "arm": arm,
                "partition": "evaluation",
                "corpus_index": nc + sequence,
                "evaluation_index": sequence,
                "prompt_id": row["prompt_id"],
                "sampling_seed": row["sampling_seed"],
                "position": position,
                "token_difr": float(d[sequence, position]),
                "token_rank": float(r[sequence, position]),
                "rank_also_capped": bool(r[sequence, position] == RANK_CAP),
                "cause_status": "unresolved_from_scores",
            })
    report["capped_event_count"] = len(events)
    return report, events


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New summary JSON file")
    parser.add_argument("--events", type=Path, help="New capped-event JSONL file")
    args = parser.parse_args(argv)
    outputs = [p for p in (args.output, args.events) if p is not None]
    if len({p.resolve() for p in outputs}) != len(outputs):
        parser.error("Summary and event outputs must be different files")
    for path in outputs:
        if path.exists():
            parser.error(f"Refusing to overwrite an existing output: {path}")
        if not path.parent.is_dir():
            parser.error(f"Create the output directory first: {path.parent}")
    report, events = analyze_archive(args.archive)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.events is not None:
        with args.events.open("x", encoding="utf-8") as stream:
            for event in events:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
    print(f"Validated archive; summarized {len(events)} capped evaluation events: {args.output}")


if __name__ == "__main__":
    main()
