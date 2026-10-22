from pathlib import Path
import importlib.metadata
import json
import shutil

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.main import get_runtime_versions, load_config, portable_frame, prepare_review, run_data_preparation, validate_split
from src.utils import assign_splits, index_dataset, matched_cases, parse_filename, read_depth
from tools.synthetic_data import create_synthetic_dataset


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    config = {
        "data_root": "data/TreeID", "output_dir": "outputs/data_preparation", "seed": 42,
        "review_file": "config/pair_review.csv", "review_pairs_per_row": 2,
        "review_all_rows": ["R04"], "contact_sheet_pairs_per_page": 8,
        "require_human_review": False,
    }
    (root / "data").mkdir()
    create_synthetic_dataset(root / "data/TreeID")
    (root / "config/experiment.json").write_text(json.dumps(config), encoding="utf-8")
    return root, config


def test_corrected_r04_prefix_is_identity():
    parsed = parse_filename("R04N01_Winter_D-R04N05_RGB_12-17-20.png")
    assert (parsed["row"], parsed["number"], parsed["modality"]) == ("R04", 1, "D")
    assert parse_filename("FANK_Sat.png") is None


def test_u_never_becomes_a_labelled_pair(project):
    root, _ = project
    index = index_dataset(root / "data/TreeID")
    cases, excluded = matched_cases(index)
    assert len(index) == 62 and len(cases) == 30 and excluded.empty
    assert index[index["kind"].eq("U")].shape[0] == 2
    assert cases["tree_id"].str.contains("N").all()


def test_row_split_is_deterministic_and_disjoint(project):
    root, _ = project
    index = index_dataset(root / "data/TreeID")
    cases, _ = matched_cases(index)
    a, b = assign_splits(cases), assign_splits(cases.sample(frac=1, random_state=10))
    assert a.set_index("tree_id")["split"].to_dict() == b.set_index("tree_id")["split"].to_dict()
    assert a.groupby("split")["row"].nunique().to_dict() == {"test": 1, "train": 3, "validation": 1}
    assert validate_split(a, index)["passed"].all()


def test_exact_duplicate_content_across_splits_is_blocked(project):
    root, _ = project
    index = index_dataset(root / "data/TreeID")
    cases, _ = matched_cases(index)
    split = assign_splits(cases)
    train = split[split["split"].eq("train")].iloc[0]
    test = split[split["split"].eq("test")].iloc[0]
    shutil.copyfile(train["summer_rgb"], test["summer_rgb"])
    changed = index_dataset(root / "data/TreeID")
    with pytest.raises(ValueError, match="leakage"):
        validate_split(split, changed)


def test_review_preserves_decisions_and_rejects_invalid_status(project):
    root, config = project
    cases, _ = matched_cases(index_dataset(root / "data/TreeID"))
    review = prepare_review(cases, root, config)
    assert review["required_review"].sum() == 14  # 2 per row, plus four additional R04 pairs
    assert review["human_status"].eq("pending").all()
    review.loc[0, ["human_status", "human_notes"]] = ["confirmed", "Checked the trunk visually"]
    review.to_csv(root / config["review_file"], index=False)
    assert prepare_review(cases, root, config).loc[0, "human_status"] == "confirmed"
    review.loc[0, "human_status"] = "automatically_approved"
    review.to_csv(root / config["review_file"], index=False)
    with pytest.raises(ValueError, match="human_status"):
        prepare_review(cases, root, config)


def test_missing_and_invalid_depth_is_explicit(project):
    root, _ = project
    folder = root / "data/TreeID/Dataset (With Summer GT)"
    depth_path = folder / "R01N01_Summer_D-12-00-00.png"
    depth_path.unlink()
    rgb = np.zeros((64, 96, 3), dtype=np.uint8)
    rgb[..., 1] = 10
    invalid = folder / "R01N02_Summer_D-12-00-00.png"
    Image.fromarray(rgb).save(invalid)
    index = index_dataset(root / "data/TreeID")
    assert set(index["depth_status"]) >= {"missing_depth", "unsupported_coloured_depth"}
    cases, excluded = matched_cases(index)
    assert len(cases) == 30 and excluded.empty  # valid RGB pairs are retained
    assert read_depth(None, (64, 96))[1] == "missing_depth"


def test_corrupt_rgb_is_retained_in_audit_and_excluded_with_reason(project):
    root, _ = project
    path = root / "data/TreeID/Dataset (With Summer GT)/R01N01_Summer_RGB-12-00-00.png"
    path.write_bytes(b"not a PNG")
    index = index_dataset(root / "data/TreeID")
    cases, excluded = matched_cases(index)
    assert len(cases) == 29
    assert excluded.iloc[0]["tree_id"] == "R01N01"
    assert index["rgb_status"].eq("corrupt_rgb").sum() == 1


def test_duplicate_modalities_are_not_silently_overwritten(project):
    root, _ = project
    folder = root / "data/TreeID/Dataset (With Summer GT)"
    shutil.copyfile(folder / "R01N01_Summer_RGB-12-00-00.png", folder / "R01N01_Summer_RGB-12-01-00.png")
    with pytest.raises(ValueError, match="Duplicate"):
        index_dataset(root / "data/TreeID")


def test_rejected_pairs_and_pending_review_are_honest(project):
    root, config = project
    first = run_data_preparation(root, config, render=False)
    review = pd.read_csv(root / config["review_file"], keep_default_na=False)
    review.loc[0, "human_status"] = "rejected"
    review.to_csv(root / config["review_file"], index=False)
    second = run_data_preparation(root, config, render=False)
    assert first["summary"]["review_pending"] == 14
    assert len(second["cases"]) == 29
    assert second["exclusions"]["reason"].eq("human_rejected").sum() == 1
    assert not second["summary"]["models_trained"]
    config["require_human_review"] = True
    with pytest.raises(ValueError, match="manual review"):
        run_data_preparation(root, config, render=False)


def test_exported_paths_are_portable_and_rerun_is_stable(project):
    root, config = project
    a = run_data_preparation(root, config, render=False)
    digest = a["summary"]["split_manifest_sha256"]
    b = run_data_preparation(root, config, render=False)
    assert digest == b["summary"]["split_manifest_sha256"]
    exported = pd.read_csv(root / config["output_dir"] / "split_manifest.csv")
    assert exported["summer_rgb"].str.startswith("data/").all()
    assert (root / exported.iloc[0]["summer_rgb"]).is_file()
    assert portable_frame(a["cases"], root)["winter_rgb"].str.startswith("data/").all()


def test_local_config_and_environment_override(project, monkeypatch):
    root, _ = project
    (root / "config/local.json").write_text('{"seed": 12}', encoding="utf-8")
    monkeypatch.setenv("MMA3001_DATA_ROOT", "/alternate/data")
    config = load_config(root)
    assert config["seed"] == 12 and config["data_root"] == "/alternate/data"


def test_data_preparation_does_not_require_notebook_automation_packages(project, monkeypatch):
    root, config = project
    original = importlib.metadata.version

    def without_automation(name):
        if name in {"nbformat", "nbclient"}:
            raise importlib.metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(importlib.metadata, "version", without_automation)
    result = run_data_preparation(root, config, render=False)
    assert result["summary"]["eligible_n_pairs"] == 30
    assert result["summary"]["leakage_checks_passed"]


def test_missing_metadata_is_reported_without_stopping_computation(monkeypatch):
    original = importlib.metadata.version

    def missing_pandas_record(name):
        if name == "pandas":
            raise importlib.metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(importlib.metadata, "version", missing_pandas_record)
    assert get_runtime_versions()["pandas"] is None
