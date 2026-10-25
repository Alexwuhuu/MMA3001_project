"""MRR-first SVR parameter refinement using the original stage-5 feature matrix."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.baseline import regression_errors, retrieval_results
from src.features import stable_digest
from src.reproduce import verify_export
from src.svr import _fit, _frame_hash, _inputs, _time_prediction, parameter_grid
from src.utils import file_digest, write_json

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["split", "query_id", "candidate_id"]
ORDER = ["highest_validation_mrr", "highest_validation_top1", "highest_validation_top5",
         "lowest_median_query_ms", "lowest_trial_id"]


def load_config(root):
    """Read the separate original-feature SVR refinement settings.

    Args:
        root (str | Path): Project containing config/svr_tuning.json.

    Returns:
        dict: Grid, MRR-first order, solver and destination settings.

    Raises:
        OSError: Configuration cannot be read.
        ValueError: Configuration JSON is malformed.
    """
    return json.loads((Path(root) / "config/svr_tuning.json").read_text(encoding="utf-8"))


def load_baseline(root):
    """Reload original development features and sample weights without refitting.

    Args:
        root (str | Path): Project containing the historical baseline outputs.

    Returns:
        dict: Baseline-compatible bundle, training_pairs, full rankings, query_metrics, metrics
        and summary for run_tuning.

    Raises:
        ValueError: Historical development inputs fail validation.
        OSError: Saved artifacts or reused appearance caches are inaccessible.
    """
    from tools.matching_diagnostics import load_development
    _, sample, bundle, frame = load_development(root)
    folder = Path(root) / "outputs/baseline_linear"
    return {"bundle": bundle, "training_pairs": sample,
            "rankings": [frame[frame.split.eq(s)].copy() for s in ("train", "validation")],
            "query_metrics": pd.read_csv(folder / "query_metrics.csv"),
            "metrics": json.loads((folder / "metrics.json").read_text()),
            "summary": json.loads((folder / "run_summary.json").read_text())}


def select_trial(trials):
    """Prefer MRR, Top-1, Top-5, median scoring time and finally trial ID.

    Args:
        trials (pandas.DataFrame): Refinement trial rows with convergence/metrics/latency
            fields.

    Returns:
        pandas.Series: Winning converged trial.

    Raises:
        ValueError: No converged trial is available.
    """
    valid = trials[trials.status.eq("ok")]
    if valid.empty:
        raise ValueError("No converged SVR configuration")
    return valid.sort_values(
        ["validation_mrr", "validation_top1", "validation_top5", "median_query_ms", "trial_id"],
        ascending=[False, False, False, True, True], kind="stable").iloc[0]


def _fingerprint(baseline, columns, sample, frames, settings):
    """Bind refinement choices to original features/samples, settings, versions and source bytes.

    Args:
        baseline (dict): Original fitted settings and summary versions.
        columns (list[str]): Original numeric feature order.
        sample (pandas.DataFrame): Exact sampled training features/targets/weights.
        frames (dict): Complete train/validation feature tables.
        settings (dict): Refinement configuration; output-only fields are excluded.

    Returns:
        str: Stable SHA-256 input fingerprint.

    Raises:
        OSError: A source file cannot be hashed.
    """
    sources = ["tools/tune_svr_parameters.py", "tools/matching_diagnostics.py", "src/svr.py",
               "src/baseline.py", "src/features.py", "src/appearance.py", "src/utils.py"]
    return stable_digest({
        "features": columns,
        "sample": _frame_hash(sample[[*KEYS, "target", "sample_weight", *columns]]),
        "galleries": {s: _frame_hash(f[[*KEYS, *columns]]) for s, f in frames.items()},
        "settings": {k: v for k, v in settings.items()
                     if k not in {"output_dir", "selection_file", "small_results_dir"}},
        "metric_settings": {k: baseline["bundle"]["settings"][k]
                            for k in ("score_tie_atol", "score_tie_rtol")},
        "versions": baseline["summary"]["versions"],
        "sources": {p: file_digest(ROOT / p) for p in sources}})


def _render(trials, comparison, output):
    """Plot original-feature parameter sensitivity and validation comparison.

    Args:
        trials (pandas.DataFrame): Refinement grid results.
        comparison (pandas.DataFrame): Per-model/split retrieval summaries.
        output (Path): Existing output directory.

    Returns:
        None: Two figures are written under figures/.

    Raises:
        OSError: Figures cannot be written.
    """
    folder = output / "figures"
    folder.mkdir(exist_ok=True)
    valid = trials[trials.status.eq("ok")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for (c, gamma), group in valid.groupby(["C", "gamma"], sort=False):
        group = group.sort_values("epsilon")
        label = f"C={c:g}, gamma={gamma}"
        for ax, column in zip(axes, ("validation_mrr", "validation_balanced_class_mae", "support_vector_fraction")):
            ax.plot(group.epsilon, group[column], "o-", label=label, markersize=4)
    for ax, title in zip(axes, ("Validation MRR (selection)", "Validation balanced MAE", "Support-vector fraction")):
        ax.set(xlabel="epsilon", title=title, xticks=sorted(valid.epsilon.unique()))
    axes[-1].legend(fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(folder / "parameter_sensitivity.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    val = comparison[comparison.split.eq("validation")]
    width = .8 / len(val)
    for i, row in enumerate(val.itertuples()):
        ax.bar(np.arange(3) + i * width, [row.top1, row.top5, row.mrr], width, label=row.model)
    ax.set(xticks=np.arange(3) + (len(val)-1)*width/2, xticklabels=["Top-1", "Top-5", "MRR"],
           title="Original features: validation parameter refinement", ylabel="Metric")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "validation_comparison.png", dpi=140)
    plt.close(fig)


def _result(output, selection, reused):
    """Verify numeric replay and saved parameters before exposing a refinement result.

    Args:
        output (Path): Saved refinement output directory.
        selection (dict): Expected locked trial parameters.
        reused (bool): Whether the caller is reusing an existing selection.

    Returns:
        dict: output_dir, trusted bundle, selection, summary, trials, comparison and replay.

    Raises:
        ValueError: The saved model/parameters or export integrity differ from the lock.
        AssertionError: Saved predictions/ranks fail replay.
        OSError: Saved artifacts cannot be read.
    """
    checked = verify_export(output)
    bundle = joblib.load(output / "models/rbf_svr.joblib")
    if bundle["selection"] != selection:
        raise ValueError("Saved model and parameter lock differ")
    for key, value in selection["parameters"].items():
        if getattr(bundle["pipeline"].named_steps["regressor"], key) != value:
            raise ValueError("Saved model parameters differ from lock")
    summary = json.loads((output / "run_summary.json").read_text())
    summary["selection_reused"] = reused
    return {"output_dir": output, "bundle": bundle, "selection": selection, "summary": summary,
            "trials": pd.read_csv(output / "search_results.csv"),
            "comparison": pd.read_csv(output / "development_comparison.csv"), "replay": checked}


def run_tuning(root, baseline, settings=None, original_svr=None, render=True):
    """Refine SVR on the original feature matrix using only training/validation.

    The exact baseline samples and preprocessing are reused. Existing matching
    locks fix choices; a verified manifest can return saved models without fitting.

    Args:
        root (str | Path): Project base for sources and destinations.
        baseline (dict): Unmodified baseline-compatible development result.
        settings (dict | None): Refinement settings; None loads config/svr_tuning.json.
        original_svr (dict | None): Optional original metrics included in comparisons only.
        render (bool): Whether fresh search results receive figures.

    Returns:
        dict: Verified output_dir, bundle, selection, summary, trials, comparison and replay. No
        test queries are scored.

    Raises:
        ValueError: Grid/order/shared inputs/lock/model validation fails or fitting does not
            converge.
        AssertionError: Repeated predictions, shared preprocessing or numeric replay differ.
        OSError: Configuration, caches or artifacts are inaccessible.
    """
    root = Path(root)
    cfg = dict(settings or load_config(root))
    grid = parameter_grid(cfg)
    if cfg["selection_order"] != ORDER:
        raise ValueError("This parameter revision requires MRR-first selection")
    columns, sample, frames = _inputs(baseline)
    fingerprint = _fingerprint(baseline, columns, sample, frames, cfg)
    output, lock_path = root / cfg["output_dir"], root / cfg["selection_file"]
    lock = None
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock["input_fingerprint"] != fingerprint:
            raise ValueError("Locked tuning inputs/grid changed; use a new selection/output path")
        # Reuse verified artifacts rather than reselecting on noisy timing measurements.
        if (output / "manifest.json").is_file():
            return _result(output, lock, reused=True)
    output.mkdir(parents=True, exist_ok=True)
    validation = frames["validation"]
    metric_cfg = baseline["bundle"]["settings"]
    X, y, weights = sample[columns], sample.target, sample.sample_weight
    rows, pipelines = [], {}
    for trial_id, params in enumerate(grid):
        pipe, seconds, converged = _fit(params, cfg, X, y, weights)
        reg = pipe.named_steps["regressor"]
        row = {"trial_id": trial_id, **params, "gamma": str(params["gamma"]),
               "status": "ok" if converged else "not_converged", "fit_seconds": seconds,
               "support_vectors": len(reg.support_), "support_vector_fraction": len(reg.support_)/len(sample),
               "resolved_gamma": float(reg._gamma), "iterations": int(reg.n_iter_)}
        if converged:
            scores, durations = _time_prediction(pipe, validation[columns], cfg["timing_repeats"])
            scored = validation[KEYS].assign(score=scores)
            _, _, retrieval = retrieval_results(scored, metric_cfg)
            row.update({f"validation_{k}": retrieval[k] for k in ("top1", "top5", "mrr")})
            for split, frame in (("train", sample), ("validation", validation)):
                error_frame = frame[KEYS].assign(score=pipe.predict(frame[columns])) if split == "train" else scored
                errors = regression_errors(error_frame)
                for group in ("all_pairs", "positive_pairs", "negative_pairs"):
                    for metric in ("mae", "rmse"):
                        row[f"{split}_{group}_{metric}"] = errors[group][metric]
                row[f"{split}_balanced_class_mae"] = errors["balanced_class_mae"]
            row["median_query_ms"] = float(1000*np.median(durations)/validation.query_id.nunique())
            row["batch_predict_seconds_repetitions"] = json.dumps(durations)
            pipelines[trial_id] = pipe
        rows.append(row)
        pd.DataFrame(rows).to_csv(output / "search_results.csv", index=False)
        print(f"Tuning {trial_id+1}/{len(grid)}: {params} | {row['status']} | "
              f"validation MRR={row.get('validation_mrr', float('nan')):.5f}", flush=True)
    table = pd.DataFrame(rows)
    best = select_trial(table)
    if lock is not None:
        locked_trial = table[table.trial_id.eq(lock["selected_trial_id"])].iloc[0]
        if locked_trial.status != "ok" or any(
            not np.isclose(locked_trial[f"validation_{key}"], value, rtol=0, atol=1e-12)
            for key in ("mrr", "top1", "top5")
            for value in (best[f"validation_{key}"], lock[f"validation_{key}"])
        ):
            raise ValueError("Locked tuning choice no longer reproduces the best validation metrics")
        best = locked_trial
    selected_id = int(best.trial_id)
    params, pipeline = grid[selected_id], pipelines[selected_id]
    reference = pipeline.predict(validation[columns])
    fit_times = []
    for _ in range(cfg["timing_repeats"]):
        pipeline, seconds, converged = _fit(params, cfg, X, y, weights)
        if not converged:
            raise ValueError("Selected configuration failed repeated fitting")
        fit_times.append(seconds)
    np.testing.assert_allclose(pipeline.predict(validation[columns]), reference, rtol=1e-10, atol=1e-10)
    for step, attributes in (("imputer", ["statistics_"]), ("scaler", ["mean_", "scale_"])):
        for attribute in attributes:
            np.testing.assert_allclose(getattr(pipeline.named_steps[step], attribute),
                getattr(baseline["bundle"]["pipeline"].named_steps[step], attribute), rtol=0, atol=0, equal_nan=True)
    selection = {"model": "RBF-SVR (original-feature parameter refinement)", "parameters": params,
                 "selected_trial_id": selected_id, "selection_order": ORDER, "selection_split": "validation",
                 "input_fingerprint": fingerprint, "feature_columns": columns, "grid_trials": len(grid),
                 "training_pair_identity_hash": baseline["summary"]["training_pair_identity_hash"],
                 "validation_mrr": float(best.validation_mrr), "validation_top1": float(best.validation_top1),
                 "validation_top5": float(best.validation_top5), "selection_query_ms": float(best.median_query_ms),
                 "fit_split": "train_only", "test_queries_scored": 0}
    if lock is not None:
        if params != lock["parameters"]:
            raise ValueError("Locked tuning parameters differ from the selected trial")
        selection = lock
    table["selected"] = table.trial_id.eq(selected_id)
    table.to_csv(output / "search_results.csv", index=False)
    metrics = {}
    for split, frame in frames.items():
        scored = frame[KEYS].assign(score=pipeline.predict(frame[columns]))
        ranked, queries, retrieval = retrieval_results(scored, metric_cfg)
        ranked.to_csv(output / f"{split}_rankings.csv", index=False)
        queries.to_csv(output / f"{split}_query_metrics.csv", index=False)
        metrics[split] = {"retrieval": retrieval, "regression": regression_errors(scored)}
    comparison = []
    methods = {"LinearRegression": baseline["metrics"], "RBF-SVR tuned": metrics}
    if original_svr is not None:
        methods["RBF-SVR original"] = original_svr["metrics"]
    for name, values in methods.items():
        for split, values_split in values.items():
            comparison.append({"model": name, "split": split, **values_split["retrieval"],
                "positive_mae": values_split["regression"]["positive_pairs"]["mae"],
                "negative_mae": values_split["regression"]["negative_pairs"]["mae"]})
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(output / "development_comparison.csv", index=False)
    summary = {"parameters": params, "grid_trials": len(grid), "converged_trials": int(table.status.eq("ok").sum()),
               "feature_count": len(columns), "training_pair_count": len(sample),
               "support_vector_count": len(pipeline.named_steps["regressor"].support_),
               "support_vector_fraction": len(pipeline.named_steps["regressor"].support_)/len(sample),
               "median_fit_seconds": float(np.median(fit_times)), "fit_seconds_repetitions": fit_times,
               "median_query_ms": float(best.median_query_ms), "input_fingerprint": fingerprint,
               "same_training_pairs_features_weights_as_baseline": True, "train_only_preprocessing": True,
               "test_queries_scored": 0, "refitted_on_train_plus_validation": False,
               "settings": cfg, "validation_role": "repeated development selection; not independent test accuracy"}
    bundle = {**baseline["bundle"], "pipeline": pipeline, "summary": summary, "selection": selection}
    models = output / "models"
    models.mkdir(exist_ok=True)
    for name, obj in (("rbf_svr", bundle), ("linear_regression", baseline["bundle"]),
                      ("imputer", pipeline.named_steps["imputer"]), ("scaler", pipeline.named_steps["scaler"])):
        joblib.dump(obj, models / f"{name}.joblib")
    validation[[*KEYS, *columns]].to_csv(output / "validation_features.csv", index=False)
    linear_ranked, _, _ = retrieval_results(validation[KEYS].assign(
        score=baseline["bundle"]["pipeline"].predict(validation[columns])), metric_cfg)
    linear_ranked.to_csv(output / "linear_validation_rankings.csv", index=False)
    write_json(output / "selected_config.json", selection)
    write_json(output / "metrics.json", metrics)
    write_json(output / "run_summary.json", summary)
    write_json(output / "replay.json", {"metric_settings": {k: metric_cfg[k] for k in ("score_tie_atol", "score_tie_rtol")},
        "models": {name: {"model_file": f"models/{model}.joblib", "feature_file": "validation_features.csv",
                          "ranking_file": ranking, "feature_columns": columns}
                   for name, model, ranking in [("LinearRegression", "linear_regression", "linear_validation_rankings.csv"),
                                                ("RBF-SVR tuned", "rbf_svr", "validation_rankings.csv")]}})
    if render:
        _render(table, comparison, output)
    write_json(output / "manifest.json", {"files": {
        p.relative_to(output).as_posix(): {"sha256": file_digest(p), "bytes": p.stat().st_size}
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "manifest.json"}})
    checked = _result(output, selection, reused=lock is not None)
    if lock is None:
        write_json(lock_path, selection)
    small = root / cfg["small_results_dir"]
    small.mkdir(parents=True, exist_ok=True)
    for name in ("search_results.csv", "development_comparison.csv", "metrics.json", "run_summary.json", "selected_config.json"):
        shutil.copyfile(output / name, small / name)
    if render:
        for path in (output / "figures").glob("*.png"):
            shutil.copyfile(path, small / path.name)
    return checked


def main():
    """Run the original-feature refinement using trusted saved baseline and original-SVR metrics.

    Returns:
        None: The validation comparison is printed and refinement artifacts are written.

    Raises:
        ValueError: Saved development inputs or refinement lock fails validation.
        OSError: Required artifacts are inaccessible.
    """
    plt.switch_backend("Agg")
    original = {"metrics": json.loads((ROOT / "outputs/svr_main/metrics.json").read_text())}
    result = run_tuning(ROOT, load_baseline(ROOT), original_svr=original)
    print(result["comparison"].to_string(index=False))


if __name__ == "__main__":
    main()
