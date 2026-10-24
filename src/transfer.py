"""Frozen ResNet18 + PCA pair features, with linear and SVR regression comparisons.

This extension keeps the original ORB experiments and all human/test records.
The only learned image projection is PCA fitted on training images. Both
regressors reuse the original sampled pair IDs, labels, and weights exactly.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .baseline import make_pipeline, rank_candidates, regression_errors, retrieval_results
from .deep_features import embed_paths, fit_projection, pair_columns, pair_matrix, build_pair_tables
from .features import stable_digest
from .main import portable_frame
from .svr import _fit, _time_prediction, _frame_hash, load_svr_config, parameter_grid, select_trial
from .utils import file_digest, write_json

PAIR_KEYS = ["split", "query_id", "candidate_id"]
SAMPLE_KEYS = [*PAIR_KEYS, "target", "label_basis", "sample_weight"]


def load_reference(root: Path) -> tuple[pd.DataFrame, dict]:
    """Read existing stage-5 artifacts without retraining or overwriting the original run."""
    folder = root / "outputs/baseline_linear"
    cases = pd.read_csv(root / "outputs/data_preparation/split_manifest.csv", keep_default_na=False)
    reference = {"training_pairs": pd.read_csv(folder / "training_pairs.csv", float_precision="round_trip"),
                 "summary": json.loads((folder / "run_summary.json").read_text()),
                 "metrics": json.loads((folder / "metrics.json").read_text())}
    return cases, reference


def validate_development(cases, reference):
    cases = cases[cases.split.isin(["train", "validation"])].copy()
    if (cases.empty or cases.tree_id.duplicated().any() or not cases.tree_id.str.fullmatch(r"R\d+N\d+").all() or
            set(cases.split) != {"train", "validation"}):
        raise ValueError("Need unique eligible N identities for train and validation")
    if set(cases.loc[cases.split.eq("train"), "row"]) & set(cases.loc[cases.split.eq("validation"), "row"]):
        raise ValueError("Train and validation orchard rows must be disjoint")
    sample = reference["training_pairs"][SAMPLE_KEYS].copy()
    train_ids = set(cases.loc[cases.split.eq("train"), "tree_id"])
    if (not sample.split.eq("train").all() or set(sample.query_id) != train_ids or
            not set(sample.candidate_id).issubset(train_ids) or sample.duplicated(PAIR_KEYS).any() or
            not np.array_equal(sample.target, sample.query_id.eq(sample.candidate_id).astype(int)) or
            not np.isfinite(sample.sample_weight).all() or not sample.sample_weight.gt(0).all()):
        raise ValueError("Reuse the original valid training pairs and positive weights")
    digest = stable_digest(sample[["query_id", "candidate_id", "target", "sample_weight"]].to_dict("records"))
    if digest != reference["summary"]["training_pair_identity_hash"]:
        raise ValueError("Original training pair identities/labels/weights changed")
    return cases.sort_values("tree_id"), sample


def evaluate_pipeline(pipeline, frames, columns, cases, metric_settings, output, model_name):
    metrics, queries, rankings = {}, [], []
    folder = output / model_name
    folder.mkdir(parents=True, exist_ok=True)
    for split, frame in frames.items():
        scored = frame[PAIR_KEYS].copy().assign(score=pipeline.predict(frame[columns]))
        ranked, per_query, retrieval = retrieval_results(scored, metric_settings)
        ranked.to_csv(folder / f"{split}_rankings.csv", index=False)
        per_query.insert(0, "split", split)
        per_query = per_query.merge(cases[["tree_id", "row", "human_verified", "label_basis"]],
                                    left_on="query_id", right_on="tree_id", validate="one_to_one").drop(columns="tree_id")
        queries.append(per_query)
        rankings.append(ranked)
        metrics[split] = {"retrieval": retrieval, "regression": regression_errors(scored)}
    per_query = pd.concat(queries, ignore_index=True)
    per_query.to_csv(folder / "query_metrics.csv", index=False)
    write_json(folder / "metrics.json", metrics)
    return {"metrics": metrics, "query_metrics": per_query, "rankings": rankings}


def run_transfer(root: Path, cases: pd.DataFrame, reference: dict, settings: dict,
                 svr_settings: dict | None = None, render=True) -> dict:
    """Train and select only on development data; no test-image inference is performed."""
    root = Path(root).resolve()
    settings = dict(settings)
    cases, sample = validate_development(portable_frame(cases, root), reference)
    svr_settings = dict(svr_settings or load_svr_config(root))
    svr_settings["timing_repeats"] = settings["timing_repeats"]
    grid = parameter_grid(svr_settings)
    if settings["timing_repeats"] < 1:
        raise ValueError("timing_repeats must be positive")
    records = []
    for row in cases.itertuples():
        for season in ("Summer", "Winter"):
            records.append({"tree_id": row.tree_id, "split": row.split, "season": season,
                            "rgb": getattr(row, f"{season.lower()}_rgb")})
    images = pd.DataFrame(records)
    vectors, encoder_info = embed_paths(images.rgb, root, settings)
    images["embedding"] = images.rgb.map(vectors)
    projection = fit_projection(images, settings["pca_components"])
    frames = build_pair_tables(images, projection)
    columns = pair_columns(projection.n_components_)
    training = sample.merge(frames["train"], on=PAIR_KEYS, how="left", validate="one_to_one", indicator=True)
    if not training._merge.eq("both").all():
        raise ValueError("A reference training pair has no corresponding feature vector")
    training = training.drop(columns="_merge")
    X, y, weights = training[columns], training.target, training.sample_weight
    validation = frames["validation"]
    metric_settings = {k: reference["summary"]["settings"][k] for k in ("score_tie_atol", "score_tie_rtol")}
    versions = {**reference["summary"]["versions"], **encoder_info["versions"]}
    fingerprint = stable_digest({"settings": settings, "svr_grid": grid,
        "svr_solver": {k: svr_settings[k] for k in ("tol", "max_iter", "cache_size_mb")},
        "features": columns, "training": _frame_hash(training),
        "validation": _frame_hash(validation), "encoder": encoder_info["signature"],
        "versions": versions, "code": file_digest(Path(__file__))})
    lock_path = root / settings["selection_file"]
    lock = json.loads(lock_path.read_text()) if lock_path.is_file() else None
    reused = lock is not None
    if lock is not None and lock["input_fingerprint"] != fingerprint:
        raise ValueError("ResNet experiment differs from its selection lock; use a separate selection_file")
    output = root / settings["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    images.drop(columns="embedding").assign(rgb_sha256=images.rgb.map(lambda p: file_digest(root/p))).to_csv(output / "embedding_manifest.csv", index=False)
    training.to_csv(output / "training_pairs.csv", index=False)
    fit_times = []
    for _ in range(settings["timing_repeats"]):
        started = perf_counter()
        linear = make_pipeline().fit(X, y, regressor__sample_weight=weights)
        fit_times.append(perf_counter()-started)
    _, linear_prediction_times = _time_prediction(linear, validation[columns], settings["timing_repeats"])
    linear_result = evaluate_pipeline(linear, frames, columns, cases, metric_settings, output, "linear")
    trials, pipelines = [], {}
    for i, params in enumerate(grid):
        pipeline, fit_seconds, converged = _fit(params, svr_settings, X, y, weights)
        row = {"trial_id": i, **params, "gamma": str(params["gamma"]), "status": "ok" if converged else "not_converged",
               "fit_seconds": fit_seconds, "support_vectors": len(pipeline.named_steps["regressor"].support_)}
        if converged:
            scores, durations = _time_prediction(pipeline, validation[columns], settings["timing_repeats"])
            val_scored = validation[PAIR_KEYS].copy().assign(score=scores)
            _, _, retrieval = retrieval_results(val_scored, metric_settings)
            row.update({f"validation_{k}": retrieval[k] for k in ("top1", "top5", "mrr")})
            for split, scored in [("train", training[PAIR_KEYS].copy().assign(score=pipeline.predict(X))), ("validation", val_scored)]:
                errors = regression_errors(scored)
                row[f"{split}_balanced_class_mae"] = errors["balanced_class_mae"]
                for category in ("positive_pairs", "negative_pairs"):
                    for metric in ("mae", "rmse"):
                        row[f"{split}_{category}_{metric}"] = errors[category][metric]
            row["median_query_ms"] = float(1000*np.median(durations)/validation.query_id.nunique())
            row["prediction_seconds_repetitions"] = json.dumps(durations)
            pipelines[i] = pipeline
        trials.append(row)
        print(f"ResNet SVR {i+1}/{len(grid)}: {params} | Top-1={row.get('validation_top1',float('nan')):.4f}, "
              f"MRR={row.get('validation_mrr',float('nan')):.4f}", flush=True)
        pd.DataFrame(trials).to_csv(output / "search_results.csv", index=False)
    table = pd.DataFrame(trials)
    best = select_trial(table)
    selected_id = int(best.trial_id) if lock is None else lock["selected_trial_id"]
    selected_row = table[table.trial_id.eq(selected_id)].iloc[0]
    if selected_row.status != "ok" or not np.allclose([selected_row.validation_top1, selected_row.validation_mrr],
                                                     [best.validation_top1, best.validation_mrr], rtol=0, atol=1e-12):
        raise ValueError("Locked ResNet SVR no longer reproduces the best validation metrics")
    params = grid[selected_id]
    table["selected"] = table.trial_id.eq(selected_id)
    table.to_csv(output / "search_results.csv", index=False)
    selected = pipelines[selected_id]
    reference_predictions = selected.predict(validation[columns])
    svr_fit_times = []
    for _ in range(settings["timing_repeats"]):
        selected, duration, converged = _fit(params, svr_settings, X, y, weights)
        if not converged:
            raise ValueError("Selected ResNet SVR did not converge on repeated fitting")
        svr_fit_times.append(duration)
    np.testing.assert_allclose(selected.predict(validation[columns]), reference_predictions, atol=1e-10, rtol=1e-10)
    np.testing.assert_equal(linear.named_steps["scaler"].mean_, selected.named_steps["scaler"].mean_)
    svr_result = evaluate_pipeline(selected, frames, columns, cases, metric_settings, output, "svr")
    _, svr_prediction_times = _time_prediction(selected, validation[columns], settings["timing_repeats"])
    if lock is None:
        lock = {"model": "ResNet18_PCA_RBF-SVR", "parameters": params, "selected_trial_id": selected_id,
                "input_fingerprint": fingerprint, "weights_sha256": encoder_info["weights_sha256"],
                "training_pair_identity_hash": reference["summary"]["training_pair_identity_hash"],
                "feature_columns": columns, "pca_fit_split": "train_images_only",
                "selection_split": "validation", "validation_top1": float(best.validation_top1),
                "validation_mrr": float(best.validation_mrr), "versions": versions, "test_queries_scored": 0}
        write_json(lock_path, lock)
    write_json(output / "selected_config.json", lock)
    summary = {"encoder": encoder_info, "pca_components": projection.n_components_,
        "pca_training_images": int(images.split.eq("train").sum()),
        "pca_explained_variance_ratio_sum": float(projection.explained_variance_ratio_.sum()),
        "feature_columns": columns, "training_pairs": len(training), "settings": settings,
        "training_pair_identity_hash": reference["summary"]["training_pair_identity_hash"],
        "same_pairs_targets_weights_for_both_regressors_and_original_baseline": True,
        "image_feature_input": "RGB only; grayscale D is retained in the original ORB comparison",
        "selection_reused": reused, "selected_svr_parameters": params, "grid_trials": len(grid),
        "converged_trials": int(table.status.eq("ok").sum()), "test_images_embedded": 0, "test_queries_scored": 0,
        "support_vector_count": len(selected.named_steps["regressor"].support_),
        "versions": versions, "input_fingerprint": fingerprint,
        "linear_median_fit_seconds": float(np.median(fit_times)), "svr_median_fit_seconds": float(np.median(svr_fit_times)),
        "linear_fit_seconds_repetitions": fit_times, "svr_fit_seconds_repetitions": svr_fit_times,
        "linear_predict_seconds_repetitions": linear_prediction_times, "svr_predict_seconds_repetitions": svr_prediction_times,
        "timing_scope": "Fitting and batch scoring cached pair features; excludes CNN/PCA extraction and ranking",
        "frozen_protocol_hashes": reference["summary"]["frozen_protocol_hashes"],
        "limitation": "Validation used for selection; mostly filename labels. This representation also omits D, so changes are not attributable to CNN alone."}
    bundles = {}
    (output / "models").mkdir(exist_ok=True)
    for name, pipeline in [("linear", linear), ("svr", selected)]:
        bundle = {"pipeline": pipeline, "projection": projection, "feature_columns": columns,
                  "settings": settings, "encoder_info": encoder_info, "metric_settings": metric_settings,
                  "selection": lock if name == "svr" else None}
        bundles[name] = bundle
        joblib.dump(bundle, output / f"models/{name}.joblib")
    joblib.dump(projection, output / "models/pca.joblib")
    comparison = []
    for name, values in [("ResNet18+LinearRegression", linear_result), ("ResNet18+RBF-SVR", svr_result)]:
        for split, metric in values["metrics"].items():
            comparison.append({"representation": "frozen_ResNet18_PCA_RGB", "model": name, "split": split,
                               **{k: metric["retrieval"][k] for k in ("queries", "gallery_size", "top1", "top5", "mrr")}})
    original = root / settings["original_comparison_file"]
    comparison = pd.DataFrame(comparison)
    if original.is_file():
        old = pd.read_csv(original).assign(representation="ORB_HSV_D")
        comparison = pd.concat([old[comparison.columns], comparison], ignore_index=True)
    comparison.to_csv(output / "development_comparison.csv", index=False)
    write_json(output / "run_summary.json", summary)
    if render:
        render_comparison(comparison, output)
    return {"summary": summary, "bundles": bundles, "linear": linear_result, "svr": svr_result,
            "trials": table, "comparison": comparison, "output_dir": output, "selection": lock}


def match_resnet(query_rgb, gallery: pd.DataFrame, bundle: dict, root: Path) -> pd.DataFrame:
    """Rank an RGB query against candidate_id/rgb gallery columns with the saved model."""
    if gallery.empty or gallery.candidate_id.duplicated().any():
        raise ValueError("Gallery must contain non-empty, unique candidate IDs")
    vectors, info = embed_paths([str(query_rgb), *gallery.rgb.tolist()], root, bundle["settings"])
    if info["weights_sha256"] != bundle["encoder_info"]["weights_sha256"]:
        raise ValueError("Inference checkpoint differs from the fitted experiment")
    query = np.repeat(vectors[str(query_rgb)][None, :], len(gallery), axis=0)
    candidates = np.stack([vectors[p] for p in gallery.rgb])
    features = pd.DataFrame(pair_matrix(query, candidates, bundle["projection"]), columns=bundle["feature_columns"])
    ranked = gallery[["candidate_id"]].copy()
    ranked["score"] = bundle["pipeline"].predict(features)
    return rank_candidates(ranked, bundle["metric_settings"]["score_tie_atol"], bundle["metric_settings"]["score_tie_rtol"])


def render_comparison(comparison, output):
    val = comparison[comparison.split.eq("validation")]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = .8/len(val)
    for i, row in enumerate(val.itertuples()):
        bars = ax.bar(np.arange(3)+i*width, [row.top1, row.top5, row.mrr], width, label=f"{row.representation}: {row.model}")
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set(xticks=np.arange(3)+(len(val)-1)*width/2, xticklabels=["Top-1", "Top-5", "MRR"],
           ylabel="Metric value", title="Full-gallery validation: development results")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, 1.22), ncol=2)
    fig.tight_layout()
    (output / "figures").mkdir(exist_ok=True)
    fig.savefig(output / "figures/development_comparison.png", dpi=135)
    plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    cases, reference = load_reference(root)
    settings = json.loads((root / "config/resnet.json").read_text())
    run_transfer(root, cases, reference, settings)
