"""Hand-calculated metric references and corrupted saved-evidence rejection."""
import csv
import json
import math

import pytest

from tools.validate_evidence import audit_saved_evaluation, reference_query


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def saved_evidence(tmp_path):
    # a's true candidate ranks first; b's true candidate ranks second with error -2.
    rows = [
        dict(split="test", query_id="a", candidate_id="a", score=1, rank_min=1, rank_max=1, display_rank=1),
        dict(split="test", query_id="a", candidate_id="b", score=0, rank_min=2, rank_max=2, display_rank=2),
        dict(split="test", query_id="b", candidate_id="a", score=0, rank_min=1, rank_max=1, display_rank=1),
        dict(split="test", query_id="b", candidate_id="b", score=-1, rank_min=2, rank_max=2, display_rank=2),
    ]
    write_csv(tmp_path / "demo_test_rankings.csv", rows)
    write_csv(tmp_path / "independent_review_normalized.csv",
              [dict(tree_id=q, independent_status="confirmed") for q in ("a", "b")])
    metrics = {"queries_sampled": 2, "queries_verified": 2, "gallery_size": 2,
        "audit_counts": {"confirmed": 2},
        "verified_retrieval": {"top1": .5, "top5": 1, "mrr": .75},
        "verified_query_regression": {
            "positive_pairs": {"count": 2, "mae": 1, "rmse": math.sqrt(2)},
            "negative_pairs": {"count": 2, "mae": 0, "rmse": 0},
            "all_pairs": {"count": 4, "mae": .5, "rmse": 1}, "balanced_class_mae": .5},
        "all_sample_filename_diagnostic": {"retrieval": {"top1": .5, "top5": 1, "mrr": .75,
            "uniform_random_reference": {"top1": .5, "top5": 1, "mrr": .75}}},
        "intervals_95": {
            "top1": {"method": "Wilson_95", "lower": .09453120573423074, "upper": .9054687942657693},
            "top5": {"method": "Wilson_95", "lower": .34238022750665303, "upper": 1}}}
    (tmp_path / "test_metrics.json").write_text(json.dumps({"demo": metrics}), encoding="utf-8")
    return tmp_path, rows, metrics


def test_hand_calculated_metrics_and_constant_reference(saved_evidence):
    folder, _, _ = saved_evidence
    report = audit_saved_evaluation(folder, ["a", "b"], ["a", "b"])
    assert report["passed"] and report["models"]["demo"]["pairs_checked"] == 4
    assert report["models"]["demo"]["verified_retrieval"] == {"top1": .5, "top5": 1, "mrr": .75}
    assert report["constant_zero_reference"]["all_pairs_rmse"] == pytest.approx(1 / math.sqrt(2))
    assert len(report["source_sha256"]) == 3


def test_all_equal_scores_do_not_get_credit_from_id_sorting():
    rows = [dict(query_id="a", candidate_id=str(i), score=0, display_rank=i+1, rank_min=1, rank_max=6)
            for i in range(5)]
    rows.append(dict(query_id="a", candidate_id="a", score=0, display_rank=6, rank_min=1, rank_max=6))
    values = reference_query(rows)
    assert values == pytest.approx({"top1": 1 / 6, "top5": 5 / 6,
                                    "mrr": sum(1 / r for r in range(1, 7)) / 6})


@pytest.mark.parametrize("corruption", ["score", "gallery", "rank", "metric", "audit", "infinite"])
def test_corrupted_evidence_is_rejected(saved_evidence, corruption):
    folder, rows, metrics = saved_evidence
    if corruption == "score":
        rows[0]["score"] = 2  # Preserves ranks but changes regression error.
    elif corruption == "gallery":
        rows.pop()
    elif corruption == "rank":
        rows[0]["rank_min"] = 2
    elif corruption == "metric":
        metrics["verified_retrieval"]["top1"] = 1
        (folder / "test_metrics.json").write_text(json.dumps({"demo": metrics}), encoding="utf-8")
    elif corruption == "audit":
        write_csv(folder / "independent_review_normalized.csv",
                  [dict(tree_id="a", independent_status="confirmed"),
                   dict(tree_id="b", independent_status="pending")])
    else:
        rows[0]["score"] = "inf"
    write_csv(folder / "demo_test_rankings.csv", rows)
    with pytest.raises(ValueError):
        audit_saved_evaluation(folder, ["a", "b"], ["a", "b"])


def test_unconfirmed_queries_cannot_inherit_verified_accuracy(saved_evidence):
    folder, _, metrics = saved_evidence
    write_csv(folder / "independent_review_normalized.csv",
              [dict(tree_id=q, independent_status="uncertain") for q in ("a", "b")])
    metrics.update(queries_verified=0, audit_counts={"uncertain": 2},
                   verified_retrieval={"top1": None, "top5": None, "mrr": None},
                   verified_query_regression=None)
    path = folder / "test_metrics.json"
    path.write_text(json.dumps({"demo": metrics}), encoding="utf-8")
    assert audit_saved_evaluation(folder, ["a", "b"], ["a", "b"])["models"]["demo"]["verified_queries"] == 0
    metrics["verified_retrieval"]["top1"] = 1
    path.write_text(json.dumps({"demo": metrics}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unverified"):
        audit_saved_evaluation(folder, ["a", "b"], ["a", "b"])
