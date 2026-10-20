"""Compatibility entry point for the deck's derived numerical diagnostic.

Reusable analysis lives in experiments.numerical. This adapter may regenerate
the deck's derived JSON; it never writes to the source experiment archive.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.numerical.sampler_contract import check_examples, contract_examples
from experiments.numerical.score_tails import analyze_archive

ARCHIVE = ROOT / "docs/results/real_quantization_l40s_20261018/pilot"
OUT = Path(__file__).resolve().parent / "qa/technical/score-tail-diagnostic.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    result, _ = analyze_archive(ARCHIVE)
    result["source"] = str(ARCHIVE.relative_to(ROOT) / "scores.npz")
    examples = contract_examples()
    check_examples(examples)
    for key in ("synthetic_top_k_example", "top_p_conformance_example"):
        result[key] = examples[key]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(args.output)
    print(json.dumps(result["arms"], indent=2))


if __name__ == "__main__":
    main()
