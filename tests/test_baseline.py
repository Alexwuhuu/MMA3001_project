import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.appearance import extract_appearance, pair_appearance
from src.baseline import make_pipeline, match_tree, retrieval_results, run_baseline, training_pairs
from src.main import run_data_preparation
from src.numerical import fit_affine_lstsq, fit_affine_ransac, transform_points, run_numerical_verification
from src.screening import load_screening_config, normalize_independent_review, run_targeted_review
from src.utils import file_digest
from tools.synthetic_data import create_synthetic_dataset

ROOT = Path(__file__).resolve().parents[1]


def settings():
    return {**load_screening_config(ROOT), **json.loads((ROOT / "config/baseline.json").read_text())}


def test_independent_entry_normalization_preserves_source():
    original = pd.DataFrame({"independent_status": ["confrimed", "pending", "pending", "rejected"],
                             "human_notes": ["Matching shapes.", "Foliage; cannot identify the tree.", "", "Different knots."],
                             "reviewer": ["user", "user", "", "user"]})
    saved = original.copy(deep=True)
    result = normalize_independent_review(original)
    assert result.independent_status.tolist() == ["confirmed", "uncertain", "pending", "rejected"]
    pd.testing.assert_frame_equal(original, saved)


def test_affine_recovery_outliers_and_degenerate_inputs(tmp_path):
    rng = np.random.default_rng(42)
    points = rng.uniform(0, 256, (100, 2))
    truth = np.array([[1.05, .1, 8], [-.05, .98, 3]])
    target = transform_points(points, truth)
    fitted, _ = fit_affine_lstsq(points, target)
    np.testing.assert_allclose(fitted, truth, atol=1e-10)
    noisy = target + rng.normal(0, .3, target.shape)
    noisy[:30] = rng.uniform(0, 256, (30, 2))
    ordinary, _ = fit_affine_lstsq(points, noisy)
    robust, _, _ = fit_affine_ransac(points, noisy)
    probe = rng.uniform(0, 256, (200, 2))
    assert np.linalg.norm(transform_points(probe, robust)-transform_points(probe, truth)) < 0.1 * np.linalg.norm(transform_points(probe, ordinary)-transform_points(probe, truth))
    for a, b in [(points[:2], target[:2]), (np.ones((4, 2)), np.ones((4, 2))),
                 (points*np.nan, target)]:
        with pytest.raises(ValueError):
            fit_affine_lstsq(a, b)
    result = run_numerical_verification(tmp_path, render=False)
    assert result["summary"]["noiseless_passed"] and result["summary"]["robust_improvement_observed"]


def test_appearance_missing_zero_unsupported_and_content_cache(tmp_path):
    rgb = tmp_path / "rgb.png"
    depth = tmp_path / "depth.png"
    Image.fromarray(np.full((48, 64, 3), [100, 120, 80], dtype=np.uint8)).save(rgb)
    cfg = settings()
    missing = extract_appearance(rgb, "", cfg, tmp_path / "cache")
    assert missing["missing_depth"] == 1 and np.isnan(missing["depth_mean"])
    Image.fromarray(np.zeros((48, 64), dtype=np.uint8)).save(depth)
    zero = extract_appearance(rgb, depth, cfg, tmp_path / "cache")
    assert zero["missing_depth"] == 0 and zero["depth_no_nonzero"] == 1
    assert pair_appearance(zero, zero)["hsv_hellinger"] == 0
    Image.fromarray(np.full((48, 64), 100, dtype=np.uint8)).save(depth)
    valid = extract_appearance(rgb, depth, cfg, tmp_path / "cache")
    assert valid["depth_mean"] == 100 and valid["depth_zero_fraction"] == 0
    assert pair_appearance(valid, valid)["depth_hellinger"] == 0
    extract_appearance(rgb, depth, dict(cfg, image_size=128), tmp_path / "cache")
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 4
    Image.fromarray(np.ones((2, 2), dtype=np.uint8)).save(depth)
    assert extract_appearance(rgb, depth, cfg, tmp_path / "cache")["missing_depth"] == 1


def test_train_only_imputation_scaling_and_all_missing_features():
    X = pd.DataFrame({"a": [1., 2., 3., np.nan], "b": [np.nan]*4})
    pipe = make_pipeline().fit(X, [0, 0, 1, 1])
    mean = pipe.named_steps["scaler"].mean_.copy()
    predictions = pipe.predict(pd.DataFrame({"a": [1e20], "b": [5.]}))
    np.testing.assert_equal(mean, pipe.named_steps["scaler"].mean_)
    assert predictions.shape == (1,) and len(mean) == 2


def test_all_ties_receive_random_credit_independent_of_ids():
    cfg = settings()
    pairs = pd.DataFrame({"query_id": ["z"]*8, "candidate_id": [*"abcdefg", "z"], "score": [0.]*8})
    for frame in [pairs, pairs.iloc[::-1]]:
        _, queries, metrics = retrieval_results(frame, cfg)
        assert metrics["top1"] == 1/8 and metrics["top5"] == 5/8
        assert metrics["mrr"] == pytest.approx(np.mean(1/np.arange(1, 9)))
        assert queries.iloc[0].rank_max == 8


def test_complete_baseline_preserves_protocol_and_roundtrips_inference(tmp_path):
    root = tmp_path / "baseline_project"
    (root / "config").mkdir(parents=True)
    create_synthetic_dataset(root / "data")
    prep_cfg = {"data_root": "data", "output_dir": "outputs/prep", "seed": 42,
                "review_file": "config/pair_review.csv", "review_pairs_per_row": 2,
                "review_all_rows": ["R04"], "contact_sheet_pairs_per_page": 8, "require_human_review": False}
    (root / "config/experiment.json").write_text(json.dumps(prep_cfg))
    prep = run_data_preparation(root, render=False)
    cfg = settings()
    cfg.update(evaluation_queries=2, timing_repeats=2, workers=2)
    screen = run_targeted_review(root, prep, cfg, render=False)
    protected = [root / "config/pair_review.csv", *screen["evaluation"]["folder"].glob("*")]
    hashes = {p: file_digest(p) for p in protected}
    pairs = training_pairs(prep["cases"], cfg)
    assert len(pairs) == 18*11
    np.testing.assert_allclose(pairs.groupby("target").sample_weight.sum(), len(pairs)/2)
    assert not pairs.duplicated(["query_id", "candidate_id"]).any()
    for query, group in pairs.groupby("query_id"):
        negatives = group[group.target.eq(0)].candidate_id
        assert query not in set(negatives)
        assert negatives.str[:3].eq(query[:3]).sum() == 5
    result = run_baseline(root, prep, screen, cfg, render=False)
    assert result["summary"]["test_queries_scored"] == 0
    assert result["metrics"]["validation"]["retrieval"]["gallery_size"] == 6
    assert all(file_digest(p) == digest for p, digest in hashes.items())
    model = joblib.load(result["output_dir"] / "models/linear_regression.joblib")
    group = prep["cases"][prep["cases"].split.eq("validation")]
    query = group.iloc[0]
    gallery = group[["tree_id", "winter_rgb", "winter_depth"]].rename(columns={"tree_id": "candidate_id", "winter_rgb": "rgb", "winter_depth": "depth"})
    inference = match_tree({"rgb": query.summer_rgb, "depth": query.summer_depth}, gallery, model, root)
    cached = result["rankings"][1]
    cached = cached[cached.query_id.eq(query.tree_id)].set_index("candidate_id")
    np.testing.assert_allclose(inference.score, cached.loc[inference.candidate_id].score, atol=1e-10)
