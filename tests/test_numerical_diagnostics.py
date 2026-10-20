"""Regression checks for archive provenance, evaluation splits, and boundaries."""
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.numerical import score_tails
from experiments.numerical.sampler_contract import check_examples, contract_examples
from ivgym.sampling import NEG_INF, apply_top_k, apply_top_p, gumbel_max_sample

ARCHIVE = (Path(__file__).resolve().parents[1]
           / "docs/results/real_quantization_l40s_20261018/pilot")


def test_zero_mass_is_undefined_not_a_claim_of_concentration():
    report = score_tails.summarize_tail(np.zeros((2, 3)), np.zeros((2, 3)))
    assert report["capped_score_mass_fraction"] is None
    assert report["capped_scores"] == report["nonzero_scores"] == 0
    assert report["tokens"] == 6
    json.dumps(report, allow_nan=False)


def test_score_mass_and_token_frequency_are_distinct():
    report = score_tails.summarize_tail([[30, 1], [0, 0]], [[50, 1], [0, 0]])
    assert report["capped_token_fraction"] == 0.25
    assert report["capped_score_mass_fraction"] == pytest.approx(30 / 31)
    assert report["sequences_with_capped_score"] == 1
    assert report["capped_also_rank_cap_50"] == 1


@pytest.mark.parametrize("difr,rank", [
    ([], []), ([1], [1]), ([[]], [[]]),
    ([[1, 2]], [[1]]), ([[float("nan")]], [[0]]),
    ([[0]], [[float("inf")]]), ([[-1]], [[0]]),
    ([[31]], [[0]]), ([[0]], [[-1]]), ([[0]], [[51]]),
])
def test_malformed_scores_are_rejected(difr, rank):
    with pytest.raises(ValueError):
        score_tails.summarize_tail(difr, rank)


def test_archived_evaluation_counts_and_event_identity():
    report, events = score_tails.analyze_archive(ARCHIVE)
    assert report["verified_archive"] is True
    assert report["split"] == {"calibration": 32, "evaluation": 64}
    assert report["sha256"] == (
        "423776dac131938176feb9762a1cae8029c76ae2efab023be857b40bc1fe62cd")
    assert report["capped_event_count"] == len(events) == 395
    expected = {"bf16": (144, 49), "nf4": (903, 252), "int8": (345, 94)}
    for arm, (nonzero, capped) in expected.items():
        row = report["arms"][arm]
        assert row["tokens"] == 64 * 128
        assert row["nonzero_scores"] == nonzero
        assert row["capped_scores"] == capped
        assert len([e for e in events if e["arm"] == arm]) == capped
    corpus = json.loads((ARCHIVE / "corpus.json").read_text())
    assert len({e["event_id"] for e in events}) == len(events)
    with np.load(ARCHIVE / "scores.npz", allow_pickle=False) as arrays:
        for event in events:
            i, position = event["corpus_index"], event["position"]
            assert i == event["evaluation_index"] + 32
            assert 32 <= i < 96 and 0 <= position < 128
            assert event["partition"] == corpus[i]["partition"] == "evaluation"
            assert event["prompt_id"] == corpus[i]["prompt_id"]
            assert event["sampling_seed"] == corpus[i]["sampling_seed"]
            assert event["token_difr"] == arrays[f"{event['arm']}/token_difr"][i, position]
            assert event["token_rank"] == arrays[f"{event['arm']}/token_toploc"][i, position]
            assert event["cause_status"] == "unresolved_from_scores"


def test_cli_writes_separate_outputs_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(score_tails, "analyze_archive",
                        lambda _: ({"example": True}, [{"event_id": "bf16:32:0"}]))
    summary, events = tmp_path / "summary.json", tmp_path / "events.jsonl"
    args = ["unused", "--output", str(summary), "--events", str(events)]
    score_tails.main(args)
    assert json.loads(summary.read_text()) == {"example": True}
    assert json.loads(events.read_text()) == {"event_id": "bf16:32:0"}
    before = summary.read_bytes(), events.read_bytes()
    with pytest.raises(SystemExit):
        score_tails.main(args)
    assert (summary.read_bytes(), events.read_bytes()) == before


def test_cli_rejects_aliasing_outputs_before_reading_archive(tmp_path):
    out = tmp_path / "same.json"
    with pytest.raises(SystemExit):
        score_tails.main(["nonexistent", "--output", str(out), "--events", str(out)])
    assert not out.exists()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_legacy_nucleus_excludes_crossing_token_and_preserves_dtype(dtype):
    logits = np.log(np.array([0.6, 0.3, 0.1], dtype=dtype))
    before = logits.copy()
    filtered = apply_top_p(logits, 0.7)
    assert filtered.dtype == dtype
    assert np.flatnonzero(filtered > NEG_INF / 2).tolist() == [0]
    np.testing.assert_array_equal(logits, before)


def test_legacy_top_k_retains_boundary_ties_and_zero_temperature_is_greedy():
    logits = np.array([3.0, 2.0, 2.0, 1.0])
    assert np.flatnonzero(apply_top_k(logits, 2) > NEG_INF / 2).tolist() == [0, 1, 2]
    assert gumbel_max_sample(logits, 0.0, np.array([0.0, 1e3, 2e3, 3e3])) == 0


def test_synthetic_examples_are_labeled_and_protocol_drift_is_detected():
    result = contract_examples()
    check_examples(result)
    assert result["status"] == "synthetic_examples_only"
    result["top_p_conformance_example"]["local_kept_indices"] = [0, 1]
    with pytest.raises(ValueError, match="protocol version"):
        check_examples(result)
