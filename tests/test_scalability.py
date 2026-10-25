"""Guard development-only workloads and timing semantics on tiny image fixtures."""
from copy import deepcopy
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.appearance import APPEARANCE_COLUMNS
from src.baseline import make_pipeline
from src.features import FEATURE_COLUMNS
from tools.benchmark_scalability import run_benchmark, save_results, select_workload, summarize, STAGES


def fixture_cases():
    return pd.DataFrame([{"tree_id": f"{row}N{i}", "row": row, "split": split}
                         for split, row in [("train", "R01"), ("validation", "R02")]
                         for i in range(3)])


def fixture_settings():
    return {"seed": 42, "queries_per_split": 2, "timing_repeats": 2,
            "gallery_sizes": {"train": [1, 3], "validation": [1, 3]},
            "output_dir": "result", "small_results_dir": "aggregate"}


def test_workload_is_fixed_nested_and_rejects_test_before_io(tmp_path):
    cases, settings = fixture_cases(), fixture_settings()
    work = select_workload(cases, settings)
    assert work == select_workload(cases.sample(frac=1, random_state=10), settings)
    for split, spec in work.items():
        assert len(set(spec["query_ids"])) == 2
        assert set(spec["gallery_ids"]) == set(cases.loc[cases.split.eq(split), "tree_id"])
        assert spec["gallery_ids"][:1] == spec["gallery_ids"][:3][:1]
    bad = cases.copy()
    bad.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="never test"):
        run_benchmark(tmp_path, bad, {}, settings)
    bad = cases.copy()
    bad.loc[bad.split.eq("validation"), "row"] = "R01"
    with pytest.raises(ValueError, match="rows overlap"):
        select_workload(bad, settings)
    settings["gallery_sizes"]["validation"] = [1, 4]
    with pytest.raises(ValueError, match="gallery sizes"):
        select_workload(cases, settings)


def test_staged_image_benchmark_accounts_once_and_preserves_models(tmp_path):
    cases, settings = fixture_cases(), fixture_settings()
    cfg = {"image_size": 64, "crop": "center", "orb_nfeatures": 50, "ratio_threshold": .75,
           "ransac_threshold_px": 3., "ransac_max_iters": 2000, "ransac_confidence": .99,
           "seed": 42, "hue_bins": 16, "saturation_bins": 8, "depth_bins": 16,
           "score_tie_atol": 1e-12, "score_tie_rtol": 1e-10}
    rng = np.random.default_rng(8)
    for season in ["summer", "winter"]:
        for i, row in cases.iterrows():
            path = f"{row.tree_id}_{season}.png"
            Image.fromarray(rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)).save(tmp_path/path)
            cases.loc[i, f"{season}_rgb"] = path
            cases.loc[i, f"{season}_depth"] = ""
    columns = FEATURE_COLUMNS+APPEARANCE_COLUMNS
    X = pd.DataFrame(rng.normal(size=(10, len(columns))), columns=columns)
    pipe = make_pipeline().fit(X, np.arange(10) % 2)
    coefficient = pipe.named_steps["regressor"].coef_.copy()
    bundle = {"pipeline": pipe, "feature_columns": columns, "settings": cfg}
    result = run_benchmark(tmp_path, cases, {"linear": bundle, "copy": deepcopy(bundle)}, settings)
    raw, caches = result["timings"], result["cache_timings"]
    assert len(raw) == 2*2*2*2*2  # splits, sizes, queries, repeats, models
    assert len(caches) == 2*2*2  # shared cache is built once, not per model/query
    assert (raw[STAGES] >= 0).all().all()
    np.testing.assert_allclose(raw.online_seconds, raw[STAGES[:4]].sum(axis=1))
    joined = raw.merge(caches, on=["split", "gallery_size", "repeat"], validate="many_to_one")
    np.testing.assert_allclose(joined.first_query_seconds, joined.online_seconds+joined.cache_build_seconds)
    np.testing.assert_array_equal(coefficient, pipe.named_steps["regressor"].coef_)
    assert result["protocol"]["checks"]["repeated_scores_and_ranks_agree"]
    assert len(result["protocol"]["image_sha256"]) == 10  # 4 query + 6 gallery RGBs
    assert result["per_query_summary"]["count"].eq(2).all()
    save_results(result, tmp_path, settings)
    for folder in (tmp_path/"result", tmp_path/"aggregate"):
        assert len(pd.read_csv(folder/"timings.csv")) == len(raw)
        assert (folder/"scaling.png").stat().st_size > 1000


def test_dispersion_uses_observations_not_sum_of_stage_medians():
    raw = pd.DataFrame({"group": ["a"]*3, "seconds": [1., 2., 9.]})
    row = summarize(raw, ["group"], ["seconds"]).iloc[0]
    assert row["median"] == 2 and row.q25 == 1.5 and row.q75 == 5.5
    assert row["min"] == 1 and row["max"] == 9 and row["count"] == 3


def test_notebook_does_not_display_real_timings_as_synthetic_results(capsys):
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root/"apple_tree_matching.ipynb", as_version=4)
    cell = next(c for c in notebook.cells if "scalability_results" in c.metadata.get("tags", []))
    exec(compile(cell.source, "scalability-results", "exec"), {"SYNTHETIC": True})
    assert "not part of synthetic CI" in capsys.readouterr().out
