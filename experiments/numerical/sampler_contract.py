"""Small CPU examples that pin the legacy sampler's boundary conventions.

These are synthetic illustrations, not model evidence or a GPU conformance test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ivgym.sampling import NEG_INF, apply_top_p, filtered_logits, gumbel_max_sample


def contract_examples():
    provider = np.log(np.array([0.4, 0.3001, 0.2999]))
    reference = np.log(np.array([0.4, 0.2999, 0.3001]))
    noise = np.array([0.0, 1.0, 0.0])
    claimed = gumbel_max_sample(provider, 1.0, noise, top_k=2)
    nucleus = np.log(np.array([0.6, 0.3, 0.1]))
    return {
        "schema": "ivgym-sampler-contract-examples-v1",
        "status": "synthetic_examples_only",
        "legacy_contract": {
            "filter_order": "top-k, then top-p, then temperature-scaled Gumbel noise",
            "top_k_ties": "Retain every token at or above the kth logit",
            "top_p_boundary": "Keep cumulative probability <= p; always retain the first token",
            "top_p_dtype": "Preserve the input NumPy floating-point dtype",
            "tie_order": "Use the existing NumPy argsort/argmax behavior; no new tie policy",
            "rng": "NumPy default_rng, not a vLLM RNG compatibility claim",
        },
        "synthetic_top_k_example": {
            "label": "Illustration, not an observed model event",
            "provider_probabilities": np.exp(provider).tolist(),
            "reference_probabilities": np.exp(reference).tolist(),
            "shared_gumbel": noise.tolist(),
            "top_k": 2,
            "claimed_token": claimed,
            "reference_token": gumbel_max_sample(reference, 1.0, noise, top_k=2),
            "claimed_filtered_by_reference": bool(
                filtered_logits(reference, 2, None)[claimed] <= NEG_INF / 2),
            "without_top_k_provider_token": gumbel_max_sample(provider, 1.0, noise),
            "without_top_k_reference_token": gumbel_max_sample(reference, 1.0, noise),
        },
        "top_p_conformance_example": {
            "probabilities": [0.6, 0.3, 0.1],
            "top_p": 0.7,
            "local_kept_indices": np.flatnonzero(
                apply_top_p(nucleus, 0.7) > NEG_INF / 2).tolist(),
            "including_crossing_token_would_keep": [0, 1],
            "interpretation": (
                "Legacy behavior excludes the crossing token except when it is first. "
                "The historical comment has been corrected without changing behavior. "
                "Any new convention requires a separately versioned protocol and calibration."
            ),
        },
    }


def check_examples(result):
    """Fail explicitly if the illustrated legacy behavior has changed."""
    top_k = result["synthetic_top_k_example"]
    expected = {
        "claimed_token": 1, "reference_token": 0,
        "claimed_filtered_by_reference": True,
        "without_top_k_provider_token": 1, "without_top_k_reference_token": 1,
    }
    if any(top_k[key] != value for key, value in expected.items()):
        raise ValueError("Legacy top-k example changed; investigate before comparing archives")
    if result["top_p_conformance_example"]["local_kept_indices"] != [0]:
        raise ValueError("Legacy top-p boundary changed; a new protocol version is required")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New JSON file; defaults to stdout")
    args = parser.parse_args(argv)
    result = contract_examples()
    check_examples(result)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        print(args.output)


if __name__ == "__main__":
    main()
