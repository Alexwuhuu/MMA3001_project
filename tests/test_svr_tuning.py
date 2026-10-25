from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from tests.test_svr import baseline_data
from tools import tune_svr_parameters as tuning


def test_mrr_first_selection_and_nonconvergence():
    trials = pd.DataFrame([
        dict(trial_id=0, status="ok", validation_mrr=.3, validation_top1=.2, validation_top5=.4, median_query_ms=1),
        dict(trial_id=1, status="ok", validation_mrr=.4, validation_top1=.1, validation_top5=.5, median_query_ms=2),
        dict(trial_id=2, status="not_converged", validation_mrr=1, validation_top1=1, validation_top5=1, median_query_ms=0)])
    assert tuning.select_trial(trials).trial_id == 1
    with pytest.raises(ValueError, match="No converged"):
        tuning.select_trial(trials[trials.status.ne("ok")])


def test_original_features_tuning_reload_and_class_errors(tmp_path, baseline_data, monkeypatch):
    cfg = tuning.load_config(tuning.ROOT)
    cfg.update(C=[.1, 1.], gamma=["scale"], epsilon=[.1, .3], timing_repeats=1)
    before = deepcopy(baseline_data)
    result = tuning.run_tuning(tmp_path, baseline_data, cfg, render=True)
    assert len(result["trials"]) == 4
    assert result["summary"]["test_queries_scored"] == 0
    assert len(result["replay"]["model_replays"]) == 2
    # A clean checkout can contain a tracked lock but no ignored model/results folder.
    cold = tmp_path / "clean_checkout"
    from src.utils import write_json
    write_json(cold / cfg["selection_file"], result["selection"])
    rebuilt = tuning.run_tuning(cold, baseline_data, cfg, render=False)
    assert rebuilt["summary"]["selection_reused"]
    assert rebuilt["selection"] == result["selection"]
    assert all(r["scores_and_ranks_reproduced"] for r in rebuilt["replay"]["model_replays"])
    sample, val = baseline_data["training_pairs"], baseline_data["rankings"][1]
    columns = result["bundle"]["feature_columns"]
    pipe = result["bundle"]["pipeline"]
    from src.baseline import regression_errors
    trial = result["trials"].loc[result["trials"].selected].iloc[0]
    for split, frame in [("train", sample), ("validation", val)]:
        errors = regression_errors(frame[tuning.KEYS].assign(score=pipe.predict(frame[columns])))
        for group in ("positive_pairs", "negative_pairs"):
            assert np.isclose(trial[f"{split}_{group}_mae"], errors[group]["mae"])
    monkeypatch.setattr(tuning, "_fit", lambda *args: pytest.fail("Cached rerun must not fit"))
    cached = tuning.run_tuning(tmp_path, baseline_data, cfg)
    assert cached["summary"]["selection_reused"]
    assert cached["selection"] == result["selection"]
    with pytest.raises(ValueError, match="Locked tuning inputs/grid changed"):
        tuning.run_tuning(tmp_path, baseline_data, dict(cfg, epsilon=[.2]))
    pd.testing.assert_frame_equal(before["training_pairs"], baseline_data["training_pairs"])
    bad = deepcopy(baseline_data)
    bad["rankings"][1]["split"] = "test"
    with pytest.raises(ValueError):
        tuning.run_tuning(tmp_path, bad, cfg)
