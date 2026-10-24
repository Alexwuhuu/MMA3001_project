"""Stage 6: select RBF-SVR on validation rows using the exact linear-baseline data.

The locked model is fitted on training pairs only. No independent test features,
predictions, or human audit outcomes influence the parameter search.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from time import perf_counter
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVR

from .appearance import APPEARANCE_COLUMNS
from .baseline import make_pipeline, regression_errors, retrieval_results
from .features import FEATURE_COLUMNS, stable_digest
from .utils import file_digest, write_json


def load_svr_config(root: Path) -> dict:
    settings = json.loads((Path(root) / "config/svr.json").read_text(encoding="utf-8"))
    parameter_grid(settings)
    return settings


def parameter_grid(settings: dict) -> list[dict]:
    """Validate the small, explicit grid before any training or output writes."""
    if settings["kernel"] != "rbf":
        raise ValueError("Stage 6 uses the agreed RBF kernel")
    for name in ("C", "gamma", "epsilon"):
        values = settings[name]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must be a non-empty list of distinct values")
        for value in values:
            if name == "gamma" and value == "scale":
                continue
            if (not isinstance(value, (int, float)) or not np.isfinite(value) or
                    value < 0 or (name != "epsilon" and value == 0)):
                raise ValueError(f"Invalid {name}: {value}")
    if (not isinstance(settings["timing_repeats"], int) or settings["timing_repeats"] < 1 or
            settings["cache_size_mb"] <= 0 or settings["tol"] <= 0 or
            not isinstance(settings["max_iter"], int) or settings["max_iter"] == 0 or settings["max_iter"] < -1):
        raise ValueError("Invalid timing, solver, or cache settings")
    return [dict(zip(("C", "gamma", "epsilon"), values))
            for values in itertools.product(settings["C"], settings["gamma"], settings["epsilon"])]


def select_trial(trials: pd.DataFrame) -> pd.Series:
    """Prefer Top-1, then MRR, then median scoring time; trial ID resolves exact ties."""
    valid = trials[trials.status.eq("ok")]
    if valid.empty:
        raise ValueError("No converged SVR configuration is available for selection")
    return valid.sort_values(["validation_top1", "validation_mrr", "median_query_ms", "trial_id"],
                             ascending=[False, False, True, True], kind="stable").iloc[0]


def _frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def _inputs(baseline: dict) -> tuple[list[str], pd.DataFrame, dict]:
    """Reject split leakage, incomplete galleries, changed samples, and metadata features."""
    columns = list(baseline["bundle"]["feature_columns"])
    if not columns or len(columns) != len(set(columns)) or not set(columns).issubset(FEATURE_COLUMNS + APPEARANCE_COLUMNS):
        raise ValueError("Use the baseline's numerical feature allowlist; metadata is forbidden")
    sample = baseline["training_pairs"].copy(deep=True)
    frames = {str(frame.split.iloc[0]): frame.copy(deep=True) for frame in baseline["rankings"] if not frame.empty}
    if set(frames) != {"train", "validation"} or len(baseline["rankings"]) != 2:
        raise ValueError("SVR requires only complete train/validation feature tables")
    for split, frame in frames.items():
        ids = set(frame.query_id)
        if (not frame.split.eq(split).all() or frame.duplicated(["query_id", "candidate_id"]).any() or
                set(frame.candidate_id) != ids or len(frame) != len(ids)**2 or
                not frame.groupby("query_id").size().eq(len(ids)).all()):
            raise ValueError("Each query must use its entire split gallery, without test pairs")
        if np.isinf(frame[columns].to_numpy(dtype=float)).any():
            raise ValueError("Feature values must be finite or missing (NaN)")
    if set(frames["train"].query_id) & set(frames["validation"].query_id):
        raise ValueError("Train/validation identities overlap")
    provenance = baseline["query_metrics"]
    if (not set(provenance.split).issubset({"train", "validation"}) or
            set(provenance.loc[provenance.split.eq("train"), "row"]) &
            set(provenance.loc[provenance.split.eq("validation"), "row"])):
        raise ValueError("Train/validation orchard rows must be disjoint")
    if (not sample.split.eq("train").all() or sample.duplicated(["query_id", "candidate_id"]).any() or
            not np.array_equal(sample.target.to_numpy(), sample.query_id.eq(sample.candidate_id).astype(int).to_numpy()) or
            not np.isfinite(sample.sample_weight).all() or not sample.sample_weight.gt(0).all()):
        raise ValueError("Invalid shared baseline targets, weights, or training identities")
    identity_hash = stable_digest(sample[["query_id", "candidate_id", "target", "sample_weight"]].to_dict("records"))
    if identity_hash != baseline["summary"]["training_pair_identity_hash"]:
        raise ValueError("Baseline training pairs/weights changed")
    shared = sample.merge(frames["train"][["query_id", "candidate_id", *columns]],
                          on=["query_id", "candidate_id"], how="left", suffixes=("", "_gallery"),
                          validate="one_to_one", indicator=True)
    if not shared._merge.eq("both").all() or not np.allclose(
            shared[columns].to_numpy(dtype=float), shared[[c+"_gallery" for c in columns]].to_numpy(dtype=float),
            rtol=0, atol=0, equal_nan=True):
        raise ValueError("SVR must reuse the baseline's exact training feature values")
    return columns, sample, frames


def _pipeline(params, settings):
    return make_pipeline().set_params(regressor=SVR(kernel="rbf", **params, tol=settings["tol"],
                                                    cache_size=settings["cache_size_mb"], max_iter=settings["max_iter"]))


def _fit(params, settings, X, y, weights):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        started = perf_counter()
        pipeline = _pipeline(params, settings).fit(X, y, regressor__sample_weight=weights)
        seconds = perf_counter() - started
    converged = pipeline.named_steps["regressor"].fit_status_ == 0
    converged &= not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return pipeline, seconds, bool(converged)


def _time_prediction(pipeline, X, repeats):
    durations = []
    prediction = None
    for _ in range(repeats):
        started = perf_counter()
        prediction = pipeline.predict(X)
        durations.append(perf_counter()-started)
    return prediction, durations


def run_svr(root: Path, baseline: dict, settings: dict | None = None, render=True) -> dict:
    """Search validation-only parameters, freeze a configuration, and save its train-only model.

    baseline is the unmodified run_baseline result, with full feature tables and
    exact sampled training pairs. Test data is neither required nor accepted.
    A matching existing lock is reused so timing jitter cannot change its choice;
    a changed input/grid fingerprint raises rather than silently overwriting it.
    """
    root = Path(root).resolve()
    settings = dict(settings or load_svr_config(root))
    grid = parameter_grid(settings)
    columns, sample, frames = _inputs(baseline)
    extraction_settings = baseline["bundle"]["settings"]
    metric_settings = {key: extraction_settings[key] for key in ("score_tie_atol", "score_tie_rtol")}
    identifiers = ["split", "query_id", "candidate_id"]
    fingerprint = stable_digest({"features": columns,
        "sample": _frame_hash(sample[[*identifiers, "target", "sample_weight", *columns]]),
        "train_gallery": _frame_hash(frames["train"][[*identifiers, *columns]]),
        "validation_gallery": _frame_hash(frames["validation"][[*identifiers, *columns]]),
        "settings": {k: v for k, v in settings.items() if k not in {"output_dir", "selection_file"}},
        "metrics": metric_settings, "versions": baseline["summary"]["versions"],
        "implementation_sha256": file_digest(Path(__file__)),
        "ranking_implementation_sha256": file_digest(Path(__file__).with_name("baseline.py"))})
    lock_path = root / settings["selection_file"]
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else None
    selection_reused = lock is not None
    if lock is not None and lock["input_fingerprint"] != fingerprint:
        raise ValueError("Locked SVR inputs/grid changed; use a separate selection_file for a new experiment")
    output = root / settings["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    X, y, weights = sample[columns], sample.target, sample.sample_weight
    validation = frames["validation"]
    query_count = validation.query_id.nunique()
    trials, fitted_models = [], {}
    for trial_id, params in enumerate(grid):
        pipeline, fit_seconds, converged = _fit(params, settings, X, y, weights)
        regressor = pipeline.named_steps["regressor"]
        row = {"trial_id": trial_id, "C": params["C"], "gamma": str(params["gamma"]), "epsilon": params["epsilon"],
               "status": "ok" if converged else "not_converged", "fit_seconds": fit_seconds,
               "support_vectors": len(regressor.support_), "iterations": int(regressor.n_iter_),
               "resolved_gamma": float(regressor._gamma)}
        if converged:
            scores, durations = _time_prediction(pipeline, validation[columns], settings["timing_repeats"])
            val_scored = validation[[*identifiers]].copy().assign(score=scores)
            _, _, retrieval = retrieval_results(val_scored, metric_settings)
            val_error = regression_errors(val_scored)
            train_scored = sample[[*identifiers]].copy().assign(score=pipeline.predict(X))
            train_error = regression_errors(train_scored)
            row.update({f"validation_{key}": retrieval[key] for key in ("top1", "top5", "mrr")})
            for split, errors in [("train", train_error), ("validation", val_error)]:
                for group in ("all_pairs", "positive_pairs", "negative_pairs"):
                    for metric in ("mae", "rmse"):
                        row[f"{split}_{group}_{metric}"] = errors[group][metric]
                row[f"{split}_balanced_class_mae"] = errors["balanced_class_mae"]
            row["median_query_ms"] = float(1000*np.median(durations)/query_count)
            row["batch_predict_seconds_repetitions"] = json.dumps(durations)
            fitted_models[trial_id] = pipeline
        trials.append(row)
        print(f"SVR {trial_id+1}/{len(grid)}: C={params['C']}, gamma={params['gamma']}, epsilon={params['epsilon']} | "
              f"{row['status']} | validation Top-1={row.get('validation_top1', float('nan')):.4f}, "
              f"MRR={row.get('validation_mrr', float('nan')):.4f}", flush=True)
        pd.DataFrame(trials).to_csv(output / "search_results.csv", index=False)
    table = pd.DataFrame(trials)
    best = select_trial(table)
    selected_id = int(best.trial_id)
    if lock is not None:
        selected_id = lock["selected_trial_id"]
        selected = table[table.trial_id.eq(selected_id)].iloc[0]
        if (selected.status != "ok" or
                not np.isclose(selected.validation_top1, best.validation_top1, rtol=0, atol=1e-12) or
                not np.isclose(selected.validation_mrr, best.validation_mrr, rtol=0, atol=1e-12) or
                not np.isclose(selected.validation_top1, lock["validation_top1"], rtol=0, atol=1e-12) or
                not np.isclose(selected.validation_mrr, lock["validation_mrr"], rtol=0, atol=1e-12)):
            raise ValueError("Locked configuration no longer reproduces the best validation metrics")
        best = selected
    params = grid[selected_id]
    selected_pipeline = fitted_models[selected_id]
    reference_scores = selected_pipeline.predict(validation[columns])
    fit_durations = []
    for _ in range(settings["timing_repeats"]):
        candidate, duration, converged = _fit(params, settings, X, y, weights)
        if not converged:
            raise ValueError("Selected SVR did not converge on repeated fitting")
        fit_durations.append(duration)
        selected_pipeline = candidate
    np.testing.assert_allclose(selected_pipeline.predict(validation[columns]), reference_scores, rtol=1e-10, atol=1e-10)
    for step in ("imputer", "scaler"):
        attributes = ("statistics_",) if step == "imputer" else ("mean_", "scale_")
        for attribute in attributes:
            np.testing.assert_allclose(getattr(selected_pipeline.named_steps[step], attribute),
                                       getattr(baseline["bundle"]["pipeline"].named_steps[step], attribute),
                                       rtol=0, atol=0, equal_nan=True)
    table["selected"] = table.trial_id.eq(selected_id)
    table.to_csv(output / "search_results.csv", index=False)
    if lock is None:
        lock = {"model": "RBF-SVR", "selected_trial_id": selected_id, "parameters": params,
                "input_fingerprint": fingerprint, "selection_split": "validation",
                "selection_order": ["highest_top1", "highest_mrr", "lowest_median_query_ms", "lowest_trial_id"],
                "validation_top1": float(best.validation_top1), "validation_mrr": float(best.validation_mrr),
                "selection_query_ms": float(best.median_query_ms), "grid_trials": len(grid),
                "training_pair_identity_hash": baseline["summary"]["training_pair_identity_hash"],
                "feature_columns": columns, "versions": baseline["summary"]["versions"],
                "frozen_protocol_hashes": baseline["summary"]["frozen_protocol_hashes"],
                "fit_split": "train_only", "test_queries_scored": 0}
        write_json(lock_path, lock)
    write_json(output / "selected_config.json", lock)
    ranked_frames, query_frames, metrics = [], [], {}
    for split, frame in frames.items():
        scored = frame[[*identifiers]].copy().assign(score=selected_pipeline.predict(frame[columns]))
        ranked, queries, retrieval = retrieval_results(scored, metric_settings)
        ranked.to_csv(output / f"{split}_rankings.csv", index=False)
        queries.insert(0, "split", split)
        metadata = baseline["query_metrics"][["split", "query_id", "row", "human_verified", "label_basis"]]
        queries = queries.merge(metadata, on=["split", "query_id"], validate="one_to_one")
        query_frames.append(queries)
        ranked_frames.append(ranked)
        metrics[split] = {"retrieval": retrieval, "regression": regression_errors(scored),
                          "label_basis": "filename identities with recorded human provenance",
                          "evaluation_role": "resubstitution" if split == "train" else "model selection; development only"}
    query_metrics = pd.concat(query_frames, ignore_index=True)
    query_metrics.to_csv(output / "query_metrics.csv", index=False)
    comparison = []
    for model, model_metrics in [("LinearRegression", baseline["metrics"]), ("RBF-SVR", metrics)]:
        for split in ("train", "validation"):
            values = model_metrics[split]
            comparison.append({"model": model, "split": split,
                **{key: values["retrieval"][key] for key in ("queries", "gallery_size", "top1", "top5", "mrr")},
                "positive_mae": values["regression"]["positive_pairs"]["mae"],
                "negative_mae": values["regression"]["negative_pairs"]["mae"]})
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(output / "development_comparison.csv", index=False)
    regressor = selected_pipeline.named_steps["regressor"]
    support = sample.iloc[regressor.support_][["query_id", "candidate_id", "target", "sample_weight"]].copy()
    support.insert(0, "training_sample_index", regressor.support_)
    support["dual_coefficient"] = regressor.dual_coef_.ravel()
    support.to_csv(output / "support_vectors.csv", index=False)
    _, predict_times = _time_prediction(selected_pipeline, validation[columns], settings["timing_repeats"])
    summary = {"model": "RBF-SVR", "parameters": params, "grid_trials": len(grid),
               "converged_trials": int(table.status.eq("ok").sum()), "selected_trial_id": selected_id,
               "selection_reused": selection_reused,
               "training_pair_count": len(sample), "feature_count": len(columns), "feature_columns": columns,
               "training_pair_identity_hash": baseline["summary"]["training_pair_identity_hash"],
               "same_training_pairs_features_weights_as_baseline": True,
               "same_train_fitted_imputer_scaler_as_baseline": True, "refitted_on_train_plus_validation": False,
               "support_vector_count": len(support), "support_vector_fraction": len(support)/len(sample),
               "support_vector_array_bytes": int(regressor.support_vectors_.nbytes),
               "resolved_gamma": float(regressor._gamma), "iterations": int(regressor.n_iter_),
               "fit_seconds_repetitions": fit_durations, "median_fit_seconds": float(np.median(fit_durations)),
               "batch_predict_seconds_repetitions": predict_times,
               "median_batch_predict_seconds": float(np.median(predict_times)),
               "median_amortized_score_ms_per_validation_query": float(1000*np.median(predict_times)/query_count),
               "timing_scope": "Fit on shared training samples; batched scoring of cached full validation features, excluding extraction/ranking",
               "scores_are_probabilities": False, "test_queries_scored": 0,
               "input_fingerprint": fingerprint, "settings": settings, "versions": baseline["summary"]["versions"],
               "frozen_protocol_hashes": baseline["summary"]["frozen_protocol_hashes"],
               "limitation": "Validation was used for model selection and mainly uses filename labels; it is not independent test accuracy."}
    bundle = {"pipeline": selected_pipeline, "feature_columns": columns, "settings": extraction_settings,
              "screen_settings": baseline["bundle"]["screen_settings"], "summary": summary, "selection": lock}
    (output / "models").mkdir(exist_ok=True)
    model_path = output / "models/rbf_svr.joblib"
    joblib.dump(bundle, model_path)
    summary["saved_model_sha256"] = file_digest(model_path)
    write_json(output / "metrics.json", metrics)
    write_json(output / "run_summary.json", summary)
    if render:
        render_search(table, comparison, output)
    return {"bundle": bundle, "summary": summary, "metrics": metrics, "trials": table,
            "comparison": comparison, "rankings": ranked_frames, "query_metrics": query_metrics,
            "selection": lock, "output_dir": output}


def render_search(trials, comparison, output):
    folder = output / "figures"
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    valid = trials[trials.status.eq("ok")]
    for (gamma, epsilon), group in valid.groupby(["gamma", "epsilon"], sort=False):
        label = f"gamma={gamma}, epsilon={epsilon}"
        group = group.sort_values("C")
        axes[0].plot(group.C, group.validation_top1, "o-", label=label)
        axes[1].plot(group.C, group.validation_mrr, "o-", label=label)
    for ax, ylabel in zip(axes, ("Validation Top-1", "Validation MRR")):
        ax.set(xscale="log", xlabel="C", ylabel=ylabel, title="Parameter selection on held-out rows")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(folder / "parameter_search.png", dpi=135)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    val = comparison[comparison.split.eq("validation")]
    for i, row in enumerate(val.itertuples()):
        ax.bar(np.arange(3)+i*.35, [row.top1, row.top5, row.mrr], .35, label=row.model)
    ax.set(xticks=np.arange(3)+.175, xticklabels=["Top-1", "Top-5", "MRR"],
           ylabel="Metric value", title="Development comparison (not independent test results)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "development_comparison.png", dpi=135)
    plt.close(fig)
