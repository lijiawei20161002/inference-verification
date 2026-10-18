"""Real-weight experiment: seed contract, independent holdout and degenerate nulls."""
import gzip
import json

import numpy as np
import pytest

from experiments.analyze_real_quantization import (
    ARMS, DETECTORS, DIAGNOSTICS, digest, effect, gini, load_run, rejection_rate,
    summarize, threshold, validate_arrays, write_json,
)
from experiments.exp_real_quantization_gpu import make_corpus, score
from ivgym import verifiers
from ivgym.core import SamplingSpec, VContext
from ivgym.sampling import gumbel_max_sample, gumbel_noise, position_seed


def fixture_arrays():
    plan = dict(calibration=32, evaluation=8, tokens=4, bootstrap=20,
                alpha=0.05, arms=list(ARMS))
    arrays = {f"{arm}/{d}": np.arange(160, dtype=float).reshape(40, 4) / 100
              for arm in ARMS for d in (*DETECTORS, *DIAGNOSTICS)}
    for arm in ARMS:
        for d in DETECTORS:
            arrays[f"rerun/{arm}/{d}"] = arrays[f"{arm}/{d}"].copy()
    return plan, arrays


def test_score_matches_existing_evidence_contract():
    row = dict(prompt_id=17, sampling_seed=1234)
    spec = SamplingSpec(seed=row["sampling_seed"])
    logits = np.array([[0, 1, 2, 3], [3, 1, 2, 0]], dtype=np.float32)
    hidden = np.array([[1, 2], [3, 4]], dtype=np.float32)
    proj = np.eye(2)
    gum = np.array([gumbel_noise(4, position_seed(spec.seed, 17, p)) for p in range(2)])
    claimed = [gumbel_max_sample(lg, 1, g, spec.top_k, spec.top_p)
               for lg, g in zip(logits, gum)]
    got = score(row, claimed, hidden, logits, hidden, proj)
    ctx = VContext(prompt_id=17, claimed_tokens=claimed, sampling=spec,
                   ref_logits=logits.astype(float), gumbel=gum,
                   fingerprints=list(hidden), ref_fingerprints=list(hidden))
    for d in DETECTORS:
        np.testing.assert_array_equal(got[d], verifiers.get(d).evidence(ctx))
    np.testing.assert_array_equal(got["token_difr"], [0, 0])
    np.testing.assert_array_equal(got["activation_difr"], [0, 0])
    perturbed = score(row, claimed, hidden + 1, logits, hidden, proj)
    np.testing.assert_allclose(perturbed["activation_difr"], np.sqrt(2))


def test_fresh_corpus_preserves_all_tokens_and_partition_independence():
    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return list(text.encode())

    corpus = make_corpus(Tokenizer(), 32, 64, 123)
    assert len({tuple(r["prompt_tokens"]) for r in corpus}) == 96
    assert len({r["sampling_seed"] for r in corpus}) == 96
    assert [r["partition"] for r in corpus] == ["calibration"] * 32 + ["evaluation"] * 64
    again = make_corpus(Tokenizer(), 32, 64, 123)
    assert corpus == again
    changed_cal_size = make_corpus(Tokenizer(), 16, 64, 123)[16:]
    assert [r["text"] for r in changed_cal_size] == [r["text"] for r in corpus[32:]]
    assert [r["sampling_seed"] for r in changed_cal_size] == [r["sampling_seed"] for r in corpus[32:]]


def test_zero_null_variance_and_degenerate_bootstraps_are_explicit():
    row = effect(np.zeros((4, 8)), np.ones((4, 8)), bootstrap=20)
    assert row["d_prime"] is None and row["ci90"] is None
    assert row["undefined_bootstrap_draws"] == 20
    assert row["status"] == "undefined_zero_baseline_variance"
    row = effect(np.array([[0], [1]]), np.array([[1], [2]]), bootstrap=100, winsor=False)
    assert row["d_prime"] == 2
    assert row["undefined_bootstrap_draws"] > 0 and row["ci90"] is None


def test_effect_pairing_and_sequence_cluster_bootstrap():
    h = np.tile([0., 1., 2., 3.], (10, 1))
    np.testing.assert_array_equal(effect(h, h, bootstrap=40)["ci90"], [0, 0])
    # Entire sequences share their shift. Treating 400 tokens as independent
    # observations would produce a much narrower interval than ten clusters.
    h = np.tile(h, (1, 10))
    a = h + np.arange(10)[:, None]
    result = effect(h, a, bootstrap=400, winsor=False)
    assert result["ci90"][1] - result["ci90"][0] > 2


def test_calibration_threshold_does_not_see_heldout_or_attack_values():
    plan, arrays = fixture_arrays()
    first = summarize(plan, arrays)
    for arm in ARMS:
        for d in DETECTORS:
            arrays[f"{arm}/{d}"][32:] += 100
            arrays[f"rerun/{arm}/{d}"][32:] += 100
    second = summarize(plan, arrays)
    for d in DETECTORS:
        assert first["detectors"][d]["calibration"] == second["detectors"][d]["calibration"]
    assert threshold(np.arange(32.))["threshold"] == 31
    assert threshold(np.arange(2.))["status"] == "insufficient_calibration"
    assert rejection_rate([31, 32], 31)["rejected"] == 1  # strict inequality
    zero = rejection_rate([0] * 64, 1)
    assert zero["rate"] == 0 and zero["wilson95"][1] > 0.05


@pytest.mark.parametrize("error", ["rerun", "nonfinite", "missing", "shape"])
def test_invalid_evidence_rejected(error):
    plan, arrays = fixture_arrays()
    if error == "rerun":
        arrays["rerun/nf4/token_difr"][0, 0] += 1
    elif error == "nonfinite":
        arrays["int8/logit_mae"][0, 0] = np.nan
    elif error == "missing":
        del arrays["bf16/token_toploc"]
    else:
        arrays["nf4/token_difr"] = np.zeros((20, 8))
    with pytest.raises(ValueError):
        validate_arrays(plan, arrays)


def test_gini_measures_concentration_across_positions():
    assert gini([0, 0, 0]) == 0
    assert gini([1, 1, 1]) == 0
    assert gini([0, 0, 3]) == pytest.approx(2 / 3)
    assert gini([1, 2, 9]) == gini([10, 20, 90])


def write_fixture(directory, error=None):
    plan, arrays = fixture_arrays()
    corpus = [dict(prompt_id=i, partition="calibration" if i < 32 else "evaluation",
                   prompt_tokens=[i + 1], sampling_seed=i + 101) for i in range(40)]
    if error == "partition":
        corpus[32]["partition"] = "calibration"
    elif error == "duplicate_prompt":
        corpus[32]["prompt_tokens"] = corpus[0]["prompt_tokens"]
    elif error == "reused_seed":
        corpus[32]["sampling_seed"] = corpus[0]["sampling_seed"]
    write_json(directory / "plan.json", plan)
    write_json(directory / "corpus.json", corpus)
    write_json(directory / "summary.json", summarize(plan, arrays))
    np.savez_compressed(directory / "scores.npz", **arrays)
    info = dict(reference_logits_rerun_exact=True, reference_hidden_rerun_exact=True,
                quantized_linear_modules=["test_layer"])
    manifest = dict(status="complete", vocab=100, arms={arm: dict(info) for arm in ARMS})
    if error == "unstable_reference":
        manifest["arms"]["nf4"]["reference_logits_rerun_exact"] = False
    elif error == "unquantized":
        manifest["arms"]["nf4"]["quantized_linear_modules"] = []
    for arm in ARMS:
        with gzip.open(directory / f"capture_{arm}.jsonl.gz", "wt") as f:
            for i in range(40):
                ids = [0] * 4 if error != "out_of_vocab" else [101] * 4
                row = dict(prompt_id=i, claimed_tokens=ids,
                           fingerprints=np.zeros((4, 32)).tolist())
                f.write(json.dumps(row) + "\n")
    manifest["artifacts_sha256"] = {p.name: digest(p) for p in directory.iterdir()}
    if error == "missing_hash":
        del manifest["artifacts_sha256"]["scores.npz"]
    write_json(directory / "manifest.json", manifest)


@pytest.mark.parametrize("error", ["partition", "duplicate_prompt", "reused_seed",
                                    "unstable_reference", "unquantized", "out_of_vocab",
                                    "missing_hash"])
def test_hash_consistent_but_invalid_experiment_rejected(tmp_path, error):
    write_fixture(tmp_path, error)
    with pytest.raises(ValueError):
        load_run(tmp_path)


def test_complete_evidence_loads_and_detects_later_corruption(tmp_path):
    write_fixture(tmp_path)
    plan, arrays = load_run(tmp_path)
    assert plan["calibration"] == 32
    assert arrays["nf4/token_difr"].shape == (40, 4)
    with (tmp_path / "corpus.json").open("a") as f:
        f.write(" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_run(tmp_path)
