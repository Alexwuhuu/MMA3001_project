from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.features import ImageFeatures, extract_pair_features
from src.svr import parameter_grid
from tools.feature_optimization import load_config, select_mrr
from tools.matching_diagnostics import hubness
from tools.relative_features import (BASE_COLUMNS, SEASON_COLUMNS, build_development, context_features,
    depth_rank_mask, fit_season_calibration, pair_orb, training_fills)

ROOT = Path(__file__).resolve().parents[1]


def test_mrr_first_selection_and_expanded_epsilon():
    config = load_config(ROOT)
    assert len(parameter_grid(config)) == 36
    assert set(config["epsilon"]) == {.01, .1, .2, .3}
    rows = pd.DataFrame([
        {"trial_id": 0, "status": "ok", "validation_top1": .5, "validation_top5": .8, "validation_mrr": .6, "median_query_ms": 1},
        {"trial_id": 1, "status": "ok", "validation_top1": .4, "validation_top5": .6, "validation_mrr": .7, "median_query_ms": 2},
        {"trial_id": 2, "status": "not_converged", "validation_top1": 1., "validation_top5": 1., "validation_mrr": 1., "median_query_ms": 0}])
    assert select_mrr(rows).trial_id == 1
    rows["representation"] = ["pair_only", "raw_all", "pair_only"]
    assert select_mrr(rows, config["selection_eligible"]).trial_id == 0


def test_depth_rank_heuristic_handles_missing_and_reversed_encoding():
    config = load_config(ROOT)
    h, w = 80, 96
    depth = np.tile(np.arange(1, w+1, dtype=np.uint8), (h, 1))
    mask, fallback, ranks = depth_rank_mask(depth, depth.shape, config)
    inverse, inverse_fallback, _ = depth_rank_mask(256-depth.astype(int), depth.shape, config)
    assert not fallback and not inverse_fallback
    np.testing.assert_array_equal(mask, inverse)
    assert 0 < mask.mean() < 1
    for missing in (None, np.zeros((h, w), dtype=np.uint8)):
        result, fallback, _ = depth_rank_mask(missing, (h, w), config)
        assert fallback and result.all()


def test_shared_knn_ratios_reproduce_original_feature_definition():
    rng = np.random.default_rng(42)
    xy = rng.uniform(10, 120, (40, 2)).astype(np.float32)
    descriptors = rng.integers(0, 256, (40, 32), dtype=np.uint8)
    query = ImageFeatures("q", xy, descriptors)
    candidate = ImageFeatures("c", xy*1.03+[3, -2], descriptors.copy())
    config = load_config(ROOT)
    actual = pair_orb(query, candidate, config["ratios"], config)
    for ratio, values in zip(config["ratios"], actual):
        expected = extract_pair_features(query, candidate, dict(config, ratio_threshold=ratio))
        np.testing.assert_allclose(list(values.values()), list(expected.values()), rtol=1e-10, atol=1e-10, equal_nan=True)
    empty = ImageFeatures("empty", np.empty((0, 2), np.float32), np.empty((0, 32), np.uint8))
    assert pair_orb(empty, candidate, [.9], config)[0]["missing_query_descriptors"] == 1


def full_frame():
    frame = pd.DataFrame({"split": ["train"]*9, "query_id": np.repeat(["a", "b", "c"], 3),
                          "candidate_id": np.tile(["a", "b", "c"], 3)})
    for column in BASE_COLUMNS+SEASON_COLUMNS:
        frame[column] = np.tile([0., 1., 2.], 3)
    return frame


def test_context_uses_full_gallery_and_is_label_and_order_invariant():
    frame = full_frame()
    fills = dict.fromkeys(BASE_COLUMNS+SEASON_COLUMNS, 0.)
    context, columns = context_features(frame, "relative_season", fills)
    assert not set(["query_keypoints", "candidate_keypoints", "missing_query_depth", "missing_candidate_depth"]) & set(columns)
    assert len(columns) == 65
    assert context.loc[0, "ransac_inliers__percentile"] == 0
    assert context.loc[2, "ransac_inliers__percentile"] == 1
    assert context.loc[0, "hamming_mean__percentile"] == 1
    assert np.isfinite(context[columns]).all().all()
    shuffled = frame.sample(frac=1, random_state=1).assign(target=np.arange(9)%2)
    second, _ = context_features(shuffled, "relative_season", fills)
    np.testing.assert_allclose(context.sort_values(["query_id", "candidate_id"])[columns],
                               second.sort_values(["query_id", "candidate_id"])[columns])
    with pytest.raises(ValueError, match="full gallery"):
        context_features(frame.iloc[:-1], "query_relative", fills)


def test_context_fills_fit_sampled_train_not_validation_or_full_negative_pool():
    frame = full_frame()
    frame.loc[frame.candidate_id.eq("c"), "hamming_mean"] = 10000.
    validation = frame.assign(split="validation", query_id="v"+frame.query_id, candidate_id="v"+frame.candidate_id)
    validation["hamming_mean"] = 1e20
    sample = frame[frame.candidate_id.ne("c")][["split", "query_id", "candidate_id"]]
    fills = training_fills(pd.concat([frame, validation], ignore_index=True), sample)
    assert fills["hamming_mean"] == .5


def test_season_calibration_is_train_only_and_test_rejected_before_io(tmp_path):
    appearance = {"hsv": np.array([.3, .7]), "depth_mean": 30., "depth_std": 4., "depth_zero_fraction": .1}
    records = pd.DataFrame([{"split": split, "season": season, "appearance": deepcopy(appearance)}
                            for split in ("train", "validation") for season in ("summer", "winter")])
    first = fit_season_calibration(records)
    records.loc[records.split.eq("validation"), "appearance"] = pd.Series([
        {**appearance, "depth_mean": 1e10}, {**appearance, "depth_mean": -1e10}], index=records.index[records.split.eq("validation")])
    second = fit_season_calibration(records)
    for season in first:
        np.testing.assert_array_equal(first[season]["mean"], second[season]["mean"])
        assert second[season]["images"] == 1
    cases = pd.DataFrame({"split": ["test"], "tree_id": ["R01N01"]})
    with pytest.raises(ValueError, match="development identities only"):
        build_development(tmp_path, cases, {}, "raw")


def test_hubness_distributes_cutoff_ties_instead_of_favoring_ids():
    frame = pd.DataFrame({"query_id": ["q"]*10, "candidate_id": list("abcdefghij"),
                          "rank_min": 1, "rank_max": 10, "display_rank": np.arange(1, 11)})
    counts, summary = hubness(frame)
    np.testing.assert_allclose(counts.expected_top5, .5)
    assert summary["expected_top5_total"] == 5
    assert summary["effective_candidates"] == pytest.approx(10)


def test_complete_optimized_experiment_reuses_pairs_locks_and_never_reads_test(tmp_path, monkeypatch):
    import json
    import tools.feature_optimization as optimization
    import tools.relative_features as relative
    from src.baseline import run_baseline
    from src.main import portable_frame, run_data_preparation
    from src.screening import load_screening_config, run_targeted_review
    from src.svr import load_svr_config, run_svr
    from src.utils import file_digest, read_rgb
    from tools.synthetic_data import create_synthetic_dataset
    root = tmp_path/"project"
    (root/"config").mkdir(parents=True)
    create_synthetic_dataset(root/"data")
    prep_cfg = json.loads((ROOT/"config/experiment.json").read_text())
    prep_cfg.update(data_root="data", output_dir="outputs/data_preparation")
    (root/"config/experiment.json").write_text(json.dumps(prep_cfg))
    prep = run_data_preparation(root, render=False)
    screen_cfg = load_screening_config(ROOT)
    screen_cfg.update(evaluation_queries=2, workers=2)
    screen = run_targeted_review(root, prep, screen_cfg, render=False)
    baseline_cfg = {**screen_cfg, **json.loads((ROOT/"config/baseline.json").read_text())}
    baseline_cfg["timing_repeats"] = 1
    baseline = run_baseline(root, prep, screen, baseline_cfg, render=False)
    svr_cfg = load_svr_config(ROOT)
    svr_cfg.update(C=[1.], gamma=["scale"], epsilon=[.1], timing_repeats=1)
    run_svr(root, baseline, svr_cfg, render=False)
    cases = portable_frame(prep["cases"], root)
    forbidden = {str(Path(path).resolve()) for r in prep["cases"][prep["cases"].split.eq("test")].itertuples()
                 for path in (r.summer_rgb, r.winter_rgb, r.summer_depth, r.winter_depth) if path}
    def guarded_hash(path):
        assert str(Path(path).resolve()) not in forbidden
        return file_digest(path)
    def guarded_rgb(path):
        assert str(Path(path).resolve()) not in forbidden
        return read_rgb(path)
    monkeypatch.setattr(optimization, "file_digest", guarded_hash)
    monkeypatch.setattr(relative, "file_digest", guarded_hash)
    monkeypatch.setattr(relative, "read_rgb", guarded_rgb)
    config = load_config(ROOT)
    config.update(image_size=128, orb_nfeatures=80, preprocessing=["raw", "clahe_depth"],
        representations=["without_query", "query_relative", "relative_season"],
        C=[1.], gamma=[.01], epsilon=[.2, .3], workers=2)
    development = (cases[cases.split.ne("test")], baseline["training_pairs"], baseline["bundle"], pd.concat(baseline["rankings"], ignore_index=True))
    optimized = optimization.run_optimization(root, config, development=development, render=True)
    assert optimized["summary"]["test_images_read"] == 0
    assert optimized["summary"]["test_queries_scored"] == 0
    assert optimized["summary"]["historical_artifacts_unchanged"]
    assert optimized["summary"]["training_pairs"] == len(baseline["training_pairs"])
    assert optimized["summary"]["training_context_gallery"] == 18
    lock_hash = file_digest(root/config["selection_file"])
    repeated = optimization.run_optimization(root, config, development=development, render=False)
    assert repeated["summary"]["selection_reused"]
    assert file_digest(root/config["selection_file"]) == lock_hash
    for name in optimized["bundles"]:
        columns = optimized["bundles"][name]["feature_columns"]
        np.testing.assert_allclose(optimized["bundles"][name]["pipeline"].predict(optimized["validation"][columns]),
                                   repeated["bundles"][name]["pipeline"].predict(repeated["validation"][columns]))
    with pytest.raises(ValueError, match="inputs changed"):
        optimization.run_optimization(root, dict(config, epsilon=[.1, .2]), development=development, render=False)
