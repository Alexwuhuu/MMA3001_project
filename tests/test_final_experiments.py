import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.baseline import make_pipeline, retrieval_results
from src.final_experiments import (bootstrap_interval, build_features, cross_pairs,
                                   development_experiments, paired_comparison,
                                   proportion_interval, validate_locked_models, verified_metrics)
from src.reproduce import verify_export
from src.utils import file_digest
from tests.test_svr import baseline_data  # Reuse the numeric development fixture.


def config():
    return {"bootstrap_repeats": 500, "seed": 42, "timing_repeats": 5,
            "reference_size": 256, "reference_ransac_threshold_px": 3., "cache_dir": "cache"}


def test_verified_subset_keeps_every_query_and_never_labels_rejections():
    ids = ["R01N01", "R01N02", "R01N03", "R01N04"]
    frame = cross_pairs(ids, ids, "test")
    frame["score"] = frame.query_id.eq(frame.candidate_id).astype(float)
    audit = pd.DataFrame({"tree_id": ids, "independent_status": ["confirmed", "rejected", "uncertain", "pending"]})
    rank, queries, metrics = verified_metrics(frame, audit, {"score_tie_atol": 1e-12, "score_tie_rtol": 1e-10}, config())
    assert len(rank) == 16 and len(queries) == 4
    assert metrics["queries_sampled"] == 4 and metrics["queries_verified"] == 1
    assert metrics["verified_retrieval"]["top1"] == 1
    assert queries.loc[~queries.verified_eligible, "verified_top1"].isna().all()
    assert metrics["verified_query_regression"]["positive_pairs"]["count"] == 1
    assert metrics["verified_query_regression"]["negative_pairs"]["count"] == 3
    assert metrics["intervals_95"]["top1"]["lower"] < 1
    empty_audit = audit.assign(independent_status="uncertain")
    _, _, empty = verified_metrics(frame, empty_audit, {"score_tie_atol": 0, "score_tie_rtol": 0}, config())
    assert empty["verified_retrieval"]["top1"] is None
    assert empty["verified_query_regression"] is None
    with pytest.raises(ValueError, match="every frozen query"):
        verified_metrics(frame, audit.iloc[:-1], {}, config())


def test_uncertainty_respects_fractional_ties_and_paired_query_alignment():
    assert proportion_interval([0]*10)["upper"] > 0
    assert proportion_interval([1]*10)["lower"] < 1
    assert proportion_interval([.25, .5])["method"] == "query_bootstrap_fractional_ties"
    assert bootstrap_interval([.1, .2, .3]) == bootstrap_interval([.1, .2, .3])
    left = pd.DataFrame({"query_id": ["a", "b", "c"], "verified_eligible": [True, True, False],
                         "top1": [1., 0., 1.], "top5": [1., 1., 1.], "reciprocal_rank": [1., .2, 1.]})
    right = left.iloc[::-1].copy()
    right.loc[right.query_id.eq("b"), "top1"] = 1.
    result = paired_comparison(left, right, config())
    assert result.iloc[0].svr_minus_linear == .5 and result.iloc[0].verified_queries == 2


def test_feature_construction_rejects_split_leakage_before_image_io(tmp_path):
    cases = pd.DataFrame({"tree_id": ["a", "b"], "split": ["train", "test"]})
    pairs = pd.DataFrame({"split": ["train"], "query_id": ["a"], "candidate_id": ["b"]})
    with pytest.raises(ValueError, match="split boundaries"):
        build_features(tmp_path, cases, pairs, {}, tmp_path/"cache")


def test_development_variants_never_extract_test_or_change_shared_samples(tmp_path, baseline_data, monkeypatch):
    import src.final_experiments as experiment
    baseline = baseline_data
    columns = baseline["bundle"]["feature_columns"]
    cases = baseline["query_metrics"][["query_id", "split", "row"]].rename(columns={"query_id": "tree_id"})
    cases = pd.concat([cases, pd.DataFrame([{"tree_id": "R99N01", "split": "test", "row": "R99"}])], ignore_index=True)
    full = pd.concat(baseline["rankings"], ignore_index=True)
    calls = []
    def mock_features(root, supplied, pairs, cfg, cache):
        assert "test" not in set(supplied.split) and "test" not in set(pairs.split)
        calls.append((cfg["image_size"], cfg["crop"], cfg["ransac_threshold_px"]))
        return pairs.merge(full[[*experiment.KEYS, *columns]], on=experiment.KEYS, validate="one_to_one"), {}
    monkeypatch.setattr(experiment, "build_features", mock_features)
    baseline["bundle"]["settings"].update(image_size=256, crop="center", workers=1)
    svr = {"selection": {"parameters": {"C": 1, "gamma": "scale", "epsilon": .1}},
           "summary": {"settings": {"tol": .001, "cache_size_mb": 32, "max_iter": -1}}}
    # Match the reference to the deliberately different fixture SVR.
    fit, _, _ = experiment._fit(svr["selection"]["parameters"], svr["summary"]["settings"],
                                baseline["training_pairs"][columns], baseline["training_pairs"].target,
                                baseline["training_pairs"].sample_weight)
    svr["bundle"] = {"pipeline": fit}
    saved = baseline["training_pairs"].copy(deep=True)
    table, fingerprints = development_experiments(tmp_path, cases, baseline, svr, config(), tmp_path)
    assert len(table) == 12 and len(fingerprints) == 6
    assert calls == [(256, "full", 3.), (128, "center", 1.5), (384, "center", 4.5)]
    assert len(json.loads(table.iloc[0].fit_seconds_repetitions)) == 5
    pd.testing.assert_frame_equal(saved, baseline["training_pairs"])


def test_export_replay_and_tamper_detection(tmp_path):
    columns = ["hsv_hellinger"]
    X = pd.DataFrame({"hsv_hellinger": [0., .8, .9, .1]})
    pipe = make_pipeline().fit(X, [1, 0, 0, 1])
    bundle = {"pipeline": pipe, "feature_columns": columns}
    joblib.dump(bundle, tmp_path/"model.joblib")
    features = cross_pairs(["a", "b"], ["a", "b"], "test")
    features[columns] = X
    features.to_csv(tmp_path/"features.csv", index=False)
    scored = features.assign(score=pipe.predict(X))
    metric_settings = {"score_tie_atol": 1e-12, "score_tie_rtol": 1e-10}
    ranks, _, _ = retrieval_results(scored, metric_settings)
    ranks.to_csv(tmp_path/"ranks.csv", index=False)
    replay = {"metric_settings": metric_settings, "models": {"linear": {"model_file": "model.joblib",
        "feature_file": "features.csv", "ranking_file": "ranks.csv", "feature_columns": columns}}}
    (tmp_path/"replay.json").write_text(json.dumps(replay))
    paths = list(tmp_path.iterdir())
    manifest = {"files": {p.name: {"bytes": p.stat().st_size, "sha256": file_digest(p)} for p in paths}}
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    result = verify_export(tmp_path)
    assert result["model_replays"][0]["pairs_replayed"] == 4
    (tmp_path/"features.csv").write_text("changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_export(tmp_path)


def test_final_preflight_rejects_changed_parameters_preprocessing_and_protocol(tmp_path, baseline_data):
    from src.final_experiments import _fit
    baseline = baseline_data
    columns = baseline["bundle"]["feature_columns"]
    sample = baseline["training_pairs"]
    params = {"C": 1., "gamma": "scale", "epsilon": .1}
    fitted, _, _ = _fit(params, {"tol": .001, "cache_size_mb": 32, "max_iter": -1},
                         sample[columns], sample.target, sample.sample_weight)
    for name in ("queries.csv", "gallery.csv", "protocol.json"):
        (tmp_path/name).write_text(name)
    hashes = {name: file_digest(tmp_path/name) for name in ("queries.csv", "gallery.csv", "protocol.json")}
    baseline["summary"]["frozen_protocol_hashes"] = hashes
    selection = {"parameters": params, "feature_columns": columns, "frozen_protocol_hashes": hashes,
                 "training_pair_identity_hash": baseline["summary"]["training_pair_identity_hash"]}
    svr = {"selection": selection, "bundle": {"pipeline": fitted, "feature_columns": columns, "selection": selection}}
    validate_locked_models(baseline, svr, {"folder": tmp_path})
    fitted.named_steps["regressor"].C = 10
    with pytest.raises(ValueError, match="hyperparameters changed"):
        validate_locked_models(baseline, svr, {"folder": tmp_path})
    fitted.named_steps["regressor"].C = 1
    fitted.named_steps["scaler"].mean_[0] += 1
    with pytest.raises(AssertionError):
        validate_locked_models(baseline, svr, {"folder": tmp_path})
    fitted.named_steps["scaler"].mean_[0] -= 1
    (tmp_path/"queries.csv").write_text("resampled")
    with pytest.raises(ValueError, match="protocol differs"):
        validate_locked_models(baseline, svr, {"folder": tmp_path})
