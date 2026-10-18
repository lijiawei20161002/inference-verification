"""CPU-only summaries for matched real-weight quantization experiments.

Pairing is by fresh prompt and sampling seed, not by generated token. Effect
intervals resample whole evaluation sequences jointly across provider arms.
Thresholds use raw sequence means and only BF16 calibration observations.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ARMS = ("bf16", "nf4", "int8")
DETECTORS = ("token_difr", "cross_entropy", "token_toploc", "activation_difr")
DIAGNOSTICS = ("logit_mae", "logit_rms", "logit_max_abs")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def gini(values):
    x = np.sort(np.asarray(values, dtype=float).ravel())
    if not len(x) or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Gini needs finite nonnegative observations")
    return float(2 * np.dot(np.arange(1, len(x) + 1), x) / (len(x) * x.sum())
                 - (len(x) + 1) / len(x)) if x.sum() else 0.0


def effect(honest, alternate, bootstrap=1000, seed=0, winsor=True):
    """Descriptive eval-pool d', with the legacy 99.9% cap recomputed per draw.

    A zero baseline variance makes d' undefined, even if clipping erased the
    difference. Never turn that case into an apparent zero effect.
    """
    h, a = np.asarray(honest, float), np.asarray(alternate, float)
    if h.ndim != 2 or h.shape != a.shape or min(h.shape) < 1:
        raise ValueError("Expected paired nonempty sequence-by-position arrays")
    if not np.isfinite(h).all() or not np.isfinite(a).all() or bootstrap < 1:
        raise ValueError("Nonfinite evidence or invalid bootstrap count")

    def statistic(b, c):
        cap = float(np.percentile(b, 99.9)) if winsor else None
        x, y = (np.minimum(b, cap), np.minimum(c, cap)) if winsor else (b, c)
        sd = float(x.std())
        return dict(d_prime=float((y.mean() - x.mean()) / sd) if sd > 0 else None,
                    cap=cap, honest_mean=float(x.mean()), alternate_mean=float(y.mean()),
                    honest_sd=sd)

    result = statistic(h, a)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap):
        idx = rng.integers(0, len(h), len(h))
        value = statistic(h[idx], a[idx])["d_prime"]
        if value is not None:
            draws.append(value)
    result.update(bootstrap=bootstrap, undefined_bootstrap_draws=bootstrap - len(draws),
                  # Do not silently discard degenerate draws to produce a bound.
                  ci90=np.quantile(draws, [0.05, 0.95]).tolist()
                  if len(draws) == bootstrap and result["d_prime"] is not None else None,
                  status="defined" if result["d_prime"] is not None
                  else "undefined_zero_baseline_variance")
    return result


def threshold(calibration_means, alpha=0.05):
    x = np.asarray(calibration_means, float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or not 0 < alpha < 1:
        raise ValueError("Invalid calibration observations or alpha")
    k = math.ceil((len(x) + 1) * (1 - alpha))
    return dict(alpha=alpha, calibration_sequences=len(x), order_statistic=k,
                threshold=float(np.sort(x)[k - 1]) if k <= len(x) else None,
                rule="reject iff raw sequence mean > threshold",
                status="finite" if k <= len(x) else "insufficient_calibration")


def rejection_rate(means, cutoff):
    """Wilson 95% interval for fresh sequence rejection counts, not token counts."""
    if cutoff is None:
        return None
    n = len(means)
    k = int(np.count_nonzero(np.asarray(means) > cutoff))
    p, z = k / n, 1.959963984540054
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return dict(rejected=k, sequences=n, rate=p,
                wilson95=[max(0.0, center - half), min(1.0, center + half)])


def validate_arrays(plan, arrays):
    if (any(type(plan[k]) is not int or plan[k] < 1
            for k in ("calibration", "evaluation", "tokens", "bootstrap"))
            or plan["arms"] != list(ARMS)):
        raise ValueError("Invalid run dimensions or arms")
    shape = (plan["calibration"] + plan["evaluation"], plan["tokens"])
    expected = {f"{a}/{d}" for a in ARMS for d in (*DETECTORS, *DIAGNOSTICS)}
    expected |= {f"rerun/{a}/{d}" for a in ARMS for d in DETECTORS}
    if set(arrays) != expected:
        raise ValueError("Incomplete or unexpected evidence arrays")
    for name, values in arrays.items():
        if values.shape != shape or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid score/diagnostic array: {name}")
    for arm in ARMS:
        for detector in DETECTORS:
            if not np.array_equal(arrays[f"{arm}/{detector}"],
                                  arrays[f"rerun/{arm}/{detector}"]):
                raise ValueError(f"Reference rerun differs: {arm}/{detector}")


def summarize(plan, arrays):
    validate_arrays(plan, arrays)
    nc = plan["calibration"]
    result = dict(schema="ivgym-real-quantization-summary-v1", plan=plan,
                  same_shape_rerun_exact=True, detectors={}, logit_drift={})
    for detector in DETECTORS:
        h = arrays[f"bf16/{detector}"]
        cut = threshold(h[:nc].mean(axis=1), plan["alpha"])
        row = dict(calibration=cut, evaluation={}, effect_vs_bf16={})
        for arm in ARMS:
            a = arrays[f"{arm}/{detector}"][nc:]
            row["evaluation"][arm] = rejection_rate(a.mean(axis=1), cut["threshold"])
            if arm != "bf16":
                row["effect_vs_bf16"][arm] = {
                    label: effect(h[nc:], a, plan["bootstrap"], winsor=winsor)
                    for label, winsor in (("winsorized", True), ("raw", False))}
        result["detectors"][detector] = row
    for arm in ARMS:
        mae = arrays[f"{arm}/logit_mae"][nc:]
        result["logit_drift"][arm] = dict(
            mean_absolute_logit_difference=float(mae.mean()),
            rms_logit_difference=float(np.sqrt(np.mean(arrays[f"{arm}/logit_rms"][nc:] ** 2))),
            max_absolute_logit_difference=float(arrays[f"{arm}/logit_max_abs"][nc:].max()),
            # Concentration across token positions of vocabulary-mean |delta|.
            position_mae_gini=gini(mae),
            mean_within_sequence_position_mae_gini=float(np.mean([gini(x) for x in mae])))
    return result


def load_run(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Run is not complete")
    expected_files = {"plan.json", "corpus.json", "scores.npz", "summary.json"}
    expected_files |= {f"capture_{arm}.jsonl.gz" for arm in ARMS}
    if set(manifest["artifacts_sha256"]) != expected_files:
        raise ValueError("Incomplete artifact hash inventory")
    for name, expected in manifest["artifacts_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    plan = json.loads((directory / "plan.json").read_text())
    corpus = json.loads((directory / "corpus.json").read_text())
    n, t = plan["calibration"] + plan["evaluation"], plan["tokens"]
    if len(corpus) != n or [r["prompt_id"] for r in corpus] != list(range(n)):
        raise ValueError("Incomplete or reordered corpus")
    if len({tuple(r["prompt_tokens"]) for r in corpus}) != n:
        raise ValueError("Prompt token collision across independent blocks")
    seeds = [r["sampling_seed"] for r in corpus]
    if (len(set(seeds)) != n
            or any(type(s) is not int or not 0 < s < 2**31 for s in seeds)):
        raise ValueError("Invalid or reused sequence sampling seeds")
    for i, row in enumerate(corpus):
        if row["partition"] != ("calibration" if i < plan["calibration"] else "evaluation"):
            raise ValueError("Partition mismatch")
    for arm in ARMS:
        info = manifest["arms"][arm]
        if (not info["reference_logits_rerun_exact"]
                or not info["reference_hidden_rerun_exact"]):
            raise ValueError("Reference recomputation was not stable")
        if arm != "bf16" and not info["quantized_linear_modules"]:
            raise ValueError("No actual quantized linear modules")
        with gzip.open(directory / f"capture_{arm}.jsonl.gz", "rt") as f:
            capture = [json.loads(line) for line in f]
        if len(capture) != n:
            raise ValueError("Incomplete capture")
        for row, evidence in zip(corpus, capture):
            if evidence["prompt_id"] != row["prompt_id"]:
                raise ValueError("Capture/corpus alignment mismatch")
            if len(evidence["claimed_tokens"]) != t:
                raise ValueError("Wrong captured sequence length")
            ids = row["prompt_tokens"] + evidence["claimed_tokens"]
            if any(type(v) is not int or not 0 <= v < manifest["vocab"] for v in ids):
                raise ValueError("Token outside the model vocabulary")
            fp = np.asarray(evidence["fingerprints"])
            if fp.shape != (t, 32) or not np.isfinite(fp).all():
                raise ValueError("Invalid captured activation evidence")
    with np.load(directory / "scores.npz", allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}
    validate_arrays(plan, arrays)
    return plan, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(*load_run(args.directory))
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
