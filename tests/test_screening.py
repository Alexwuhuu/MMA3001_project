import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import ImageFeatures, extract_image_features, extract_pair_features, preprocess_rgb
from src.main import run_data_preparation
from src.screening import freeze_evaluation, load_screening_config, rank_review_cases, run_targeted_review
from src.utils import file_digest, read_review_csv
from tools.synthetic_data import create_synthetic_dataset


@pytest.fixture
def screen_project(tmp_path):
    root = tmp_path / "screen_project"
    (root / "config").mkdir(parents=True)
    create_synthetic_dataset(root / "data")
    config = {"data_root": "data", "output_dir": "outputs/data_preparation", "seed": 42,
              "review_file": "config/pair_review.csv", "review_pairs_per_row": 2,
              "review_all_rows": ["R04"], "contact_sheet_pairs_per_page": 8, "require_human_review": False}
    (root / "config/experiment.json").write_text(json.dumps(config), encoding="utf-8")
    settings = load_screening_config(Path(__file__).resolve().parents[1])
    settings.update(evaluation_queries=2, workers=2, max_review_cases=10)
    first = run_data_preparation(root, render=False)
    review = read_review_csv(root / config["review_file"])
    train_id = first["cases"].loc[first["cases"].split.eq("train"), "tree_id"].iloc[0]
    review.loc[review.tree_id.eq(train_id), ["human_status", "human_notes"]] = ["rejected", "人工发现错配"]
    review.to_csv(root / config["review_file"], index=False, encoding="utf-8-sig")
    return root, settings, run_data_preparation(root, render=False)


def test_legacy_review_encoding_and_bytes_are_preserved(screen_project):
    root, _, prep = screen_project
    path = root / "config/pair_review.csv"
    review = read_review_csv(path)
    review.to_csv(path, index=False, encoding="gb18030")
    before = file_digest(path)
    rerun = run_data_preparation(root, render=False)
    assert file_digest(path) == before
    assert len(rerun["cases"]) == len(prep["cases"]) == 29
    assert rerun["cases"].label_basis.eq("filename_unverified").all()
    assert not rerun["cases"].human_verified.any()


def test_frozen_test_sample_is_stable_and_preserves_independent_notes(screen_project):
    root, settings, prep = screen_project
    first = freeze_evaluation(prep["cases"], root, settings)
    path = first["folder"] / "independent_review.csv"
    review = first["review"]
    review.loc[0, ["independent_status", "human_notes"]] = ["uncertain", "主干遮挡"]
    review.to_csv(path, index=False, encoding="utf-8-sig")
    second = freeze_evaluation(prep["cases"].sample(frac=1, random_state=8), root, settings)
    assert first["queries"].tree_id.tolist() == second["queries"].tree_id.tolist()
    assert second["review"].loc[0, "independent_status"] == "uncertain"
    assert len(second["gallery"]) == prep["cases"].split.eq("test").sum()
    assert set(second["queries"].row).issubset(set(prep["cases"].loc[prep["cases"].split.eq("test"), "row"]))
    with pytest.raises(ValueError, match="do not automatically resample"):
        freeze_evaluation(prep["cases"], root, dict(settings, evaluation_seed=99))


def test_synthetic_affine_outliers_are_rejected_by_ransac():
    settings = load_screening_config(Path(__file__).resolve().parents[1])
    rng = np.random.default_rng(6)
    xy = rng.uniform(10, 220, (40, 2)).astype(np.float32)
    target = xy @ np.array([[1.03, -0.08], [0.08, 1.03]], dtype=np.float32).T + [5, 7]
    target[-10:] = rng.uniform(10, 220, (10, 2))
    descriptors = rng.integers(0, 256, (40, 32), dtype=np.uint8)
    q, c = ImageFeatures("q", xy, descriptors), ImageFeatures("c", target.astype(np.float32), descriptors.copy())
    result = extract_pair_features(q, c, settings)
    assert result["good_matches"] == 40
    assert 28 <= result["ransac_inliers"] <= 31
    assert result["reprojection_rmse_px"] < 1e-3
    assert result["geometry_failed"] == 0


def test_missing_and_collinear_correspondences_are_retained():
    settings = load_screening_config(Path(__file__).resolve().parents[1])
    empty = ImageFeatures("empty", np.empty((0, 2)), np.empty((0, 32), dtype=np.uint8))
    values = extract_pair_features(empty, empty, settings)
    assert values["missing_query_descriptors"] == 1 and values["geometry_failed"] == 1
    assert np.isnan(values["hamming_mean"])
    rng = np.random.default_rng(2)
    line = np.c_[np.arange(20), np.arange(20)].astype(np.float32)
    descriptors = rng.integers(0, 256, (20, 32), dtype=np.uint8)
    values = extract_pair_features(ImageFeatures("a", line, descriptors),
                                   ImageFeatures("b", line + 4, descriptors.copy()), settings)
    assert values["good_matches"] == 20 and values["geometry_failed"] == 1


def test_image_cache_changes_with_preprocessing_settings(screen_project):
    root, settings, prep = screen_project
    path = prep["cases"].iloc[0].summer_rgb
    cache = root / ".cache/cache_check"
    a = extract_image_features(path, settings, cache)
    b = extract_image_features(path, dict(settings, image_size=128), cache)
    assert len(list(cache.glob("*.npz"))) == 2
    assert a.image_hash == b.image_hash
    assert preprocess_rgb(np.zeros((480, 640, 3), dtype=np.uint8)).shape == (256, 256, 3)


def test_screening_has_full_split_galleries_and_no_label_mutation(screen_project):
    root, settings, prep = screen_project
    path = root / "config/pair_review.csv"
    before = file_digest(path)
    result = run_targeted_review(root, prep, settings, render=False)
    assert file_digest(path) == before
    pairs = result["pair_features"]
    assert not pairs.split.eq("test").any()
    for split, group in prep["cases"].groupby("split"):
        if split != "test":
            chosen = pairs[pairs.split.eq(split)]
            assert len(chosen) == len(group) ** 2
            assert set(chosen.candidate_id) == set(group.tree_id)
    assert not result["summary"]["automatic_labels_written"]
    assert (root / settings["evaluation_dir"] / "protocol.json").exists()
    assert result["evaluation"]["review"].independent_status.eq("pending").all()
    # A cached rerun cannot silently promote pending labels or reselect test queries.
    second = run_targeted_review(root, prep, settings, render=False)
    assert second["summary"]["timings"]["cached_queries"] == result["summary"]["screened_queries"]
    pd.testing.assert_frame_equal(result["pair_features"], second["pair_features"], check_dtype=False)


def test_zero_geometry_stays_tied_and_does_not_invent_review_priorities(screen_project):
    root, settings, prep = screen_project
    cases = prep["cases"][prep["cases"].split.eq("train")].head(3)
    query_id = cases.iloc[0].tree_id
    pairs = pd.DataFrame({
        "query_id": [query_id] * 3, "candidate_id": cases.tree_id.tolist(),
        "evidence_score": [0.0] * 3, "good_matches": [0, 4, 2],
        "hamming_mean": [np.nan, 30, 50], "ransac_inliers": [0, 3, 0],
        "ransac_inlier_ratio": [0, 0.75, 0],
    })
    result = rank_review_cases(pairs, cases, settings).iloc[0]
    assert result.original_rank_min == 1 and result.original_rank_max == 3
    assert result.evidence_status == "insufficient_geometry"
    assert result.alternative_display_basis == "descriptors_only_weak"
    assert not result.priority_supported
