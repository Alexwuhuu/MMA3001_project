from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.baseline import make_pipeline, regression_errors, retrieval_results, training_pairs
from src.features import stable_digest
from src.svr import _inputs, load_svr_config, parameter_grid, run_svr, select_trial
from src.utils import file_digest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def baseline_data():
    """Tiny numeric fixture for algorithm checks, not agricultural validation."""
    rng = np.random.default_rng(4)
    columns = ["good_matches", "hsv_hellinger", "reprojection_rmse_px"]
    settings = {"seed": 42, "negatives_per_query": 10, "same_row_negatives": 5,
                "score_tie_atol": 1e-12, "score_tie_rtol": 1e-10}
    cases = pd.DataFrame([{"tree_id": f"R{row:02d}N{i:02d}", "split": split, "row": f"R{row:02d}",
                           "label_basis": "filename_unverified", "human_verified": False}
                          for row, split in [(1, "train"), (2, "train"), (3, "validation")] for i in range(1, 7)])
    frames = []
    for split in ("train", "validation"):
        group = cases[cases.split.eq(split)]
        records = []
        for q in group.tree_id:
            for c in group.tree_id:
                records.append({"split": split, "query_id": q, "candidate_id": c,
                                "good_matches": int(rng.integers(0, 15)),
                                "hsv_hellinger": rng.uniform(.1, .4) if q == c else rng.uniform(.3, .9),
                                "reprojection_rmse_px": np.nan})
        frames.append(pd.DataFrame(records))
    sample = training_pairs(cases, settings).merge(frames[0], on=["split", "query_id", "candidate_id"], validate="one_to_one")
    pipe = make_pipeline().fit(sample[columns], sample.target, regressor__sample_weight=sample.sample_weight)
    rankings, query_frames, metrics = [], [], {}
    for frame in frames:
        frame["score"] = pipe.predict(frame[columns])
        ranked, queries, retrieval = retrieval_results(frame, settings)
        split = frame.split.iloc[0]
        queries["split"] = split
        queries = queries.merge(cases, left_on=["split", "query_id"], right_on=["split", "tree_id"])
        rankings.append(ranked)
        query_frames.append(queries)
        metrics[split] = {"retrieval": retrieval, "regression": regression_errors(frame)}
    return {"training_pairs": sample, "rankings": rankings, "query_metrics": pd.concat(query_frames),
            "bundle": {"feature_columns": columns, "pipeline": pipe, "settings": settings, "screen_settings": {}},
            "metrics": metrics,
            "summary": {"versions": {"fixture": "synthetic"}, "frozen_protocol_hashes": {},
                        "training_pair_identity_hash": stable_digest(sample[["query_id", "candidate_id", "target", "sample_weight"]].to_dict("records"))}}


def test_grid_matches_the_agreed_eighteen_settings():
    cfg = load_svr_config(ROOT)
    assert len(parameter_grid(cfg)) == 18
    for altered in [dict(cfg, C=[0]), dict(cfg, epsilon=[-.1]), dict(cfg, gamma=["auto"]),
                    dict(cfg, kernel="linear"), dict(cfg, timing_repeats=0), dict(cfg, max_iter=0)]:
        with pytest.raises(ValueError):
            parameter_grid(altered)


def test_selection_priorities_and_failed_trials():
    rows = pd.DataFrame([
        {"trial_id": 0, "status": "ok", "validation_top1": .5, "validation_mrr": .6, "median_query_ms": 1.},
        {"trial_id": 1, "status": "ok", "validation_top1": .6, "validation_mrr": .65, "median_query_ms": .01},
        {"trial_id": 2, "status": "ok", "validation_top1": .6, "validation_mrr": .7, "median_query_ms": 2.},
        {"trial_id": 3, "status": "ok", "validation_top1": .6, "validation_mrr": .7, "median_query_ms": 1.},
        {"trial_id": 4, "status": "not_converged", "validation_top1": 1., "validation_mrr": 1., "median_query_ms": .001}])
    assert select_trial(rows).trial_id == 3
    with pytest.raises(ValueError, match="No converged"):
        select_trial(rows[rows.status.ne("ok")])


@pytest.mark.parametrize("change", ["test_pairs", "metadata_feature", "changed_feature", "changed_weight", "overlapping_rows", "partial_gallery"])
def test_rejects_leakage_or_changed_shared_inputs(baseline_data, change):
    bad = deepcopy(baseline_data)
    if change == "test_pairs":
        bad["rankings"][1]["split"] = "test"
    elif change == "metadata_feature":
        bad["bundle"]["feature_columns"] += ["row"]
    elif change == "changed_feature":
        bad["training_pairs"].loc[0, "good_matches"] += 1
    elif change == "changed_weight":
        bad["training_pairs"].loc[0, "sample_weight"] *= 2
    elif change == "overlapping_rows":
        bad["query_metrics"].loc[bad["query_metrics"].split.eq("validation"), "row"] = "R01"
    else:
        bad["rankings"][1] = bad["rankings"][1].iloc[:-1]
    with pytest.raises(ValueError):
        _inputs(bad)


def test_svr_search_persistence_and_frozen_selection(tmp_path, baseline_data):
    cfg = load_svr_config(ROOT)
    cfg.update(output_dir="outputs/svr", selection_file="config/svr_selection.json", timing_repeats=2)
    original = deepcopy(baseline_data)
    result = run_svr(tmp_path, baseline_data, cfg, render=False)
    assert len(result["trials"]) == 18 and result["trials"].selected.sum() == 1
    assert result["summary"]["test_queries_scored"] == 0
    assert not result["summary"]["selection_reused"]
    assert result["summary"]["same_train_fitted_imputer_scaler_as_baseline"]
    assert not result["summary"]["refitted_on_train_plus_validation"]
    saved = joblib.load(result["output_dir"] / "models/rbf_svr.joblib")
    features = baseline_data["rankings"][1][saved["feature_columns"]]
    np.testing.assert_allclose(saved["pipeline"].predict(features), result["bundle"]["pipeline"].predict(features))
    for name in ("statistics_",):
        np.testing.assert_equal(saved["pipeline"].named_steps["imputer"].__dict__[name],
                                baseline_data["bundle"]["pipeline"].named_steps["imputer"].__dict__[name])
    before = file_digest(tmp_path / cfg["selection_file"])
    repeated = run_svr(tmp_path, baseline_data, cfg, render=False)
    assert repeated["summary"]["selection_reused"]
    assert repeated["selection"] == result["selection"]
    assert file_digest(tmp_path / cfg["selection_file"]) == before
    with pytest.raises(ValueError, match="Locked SVR inputs/grid changed"):
        run_svr(tmp_path, baseline_data, dict(cfg, C=[.2, 1., 10.]), render=False)
    pd.testing.assert_frame_equal(original["training_pairs"], baseline_data["training_pairs"])
    for left, right in zip(original["rankings"], baseline_data["rankings"]):
        pd.testing.assert_frame_equal(left, right)
