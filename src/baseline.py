"""Stage 5: weighted Linear Regression with train-only preprocessing.

Validation uses the full validation winter gallery. This module deliberately
does not score the frozen independent test set or select SVR parameters.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .appearance import APPEARANCE_COLUMNS, appearance_signature, extract_appearance, pair_appearance
from .features import FEATURE_COLUMNS, extract_image_features, extract_pair_features, stable_digest
from .main import portable_frame
from .utils import file_digest, read_rgb, write_json


def training_pairs(cases: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Sample one filename-positive and distinct train-only negatives per query.

    Prefer five same-row negatives; fill from other rows, then remaining trees.
    Labels describe filename identities, not independent pairwise human truth.
    """
    train = cases[cases.split.eq("train")].sort_values("tree_id")
    if train.empty or train.tree_id.duplicated().any():
        raise ValueError("Training identities must be non-empty and unique")
    count = settings["negatives_per_query"]
    if count < 1 or len(train) <= count:
        raise ValueError("Training gallery must contain at least negatives_per_query + 1 trees")
    rng, records = np.random.default_rng(settings["seed"]), []
    for query in train.itertuples():
        same = train[train.row.eq(query.row) & train.tree_id.ne(query.tree_id)].tree_id.to_numpy()
        preferred = rng.choice(same, min(len(same), settings["same_row_negatives"], count), replace=False).tolist()
        others = train[train.row.ne(query.row)].tree_id.to_numpy()
        take = min(count - len(preferred), len(others))
        negatives = preferred + rng.choice(others, take, replace=False).tolist()
        remaining = train[~train.tree_id.isin([query.tree_id, *negatives])].tree_id.to_numpy()
        negatives += rng.choice(remaining, count - len(negatives), replace=False).tolist()
        records.append({"split": "train", "query_id": query.tree_id, "candidate_id": query.tree_id,
                        "target": 1, "label_basis": query.label_basis})
        records.extend({"split": "train", "query_id": query.tree_id, "candidate_id": candidate,
                        "target": 0, "label_basis": "different_filename_identity"} for candidate in negatives)
    result = pd.DataFrame(records)
    frequency = result.target.value_counts()
    result["sample_weight"] = result.target.map({label: len(result) / (2 * n) for label, n in frequency.items()})
    return result


def make_pipeline() -> Pipeline:
    """Preserve all-missing features; median imputation and scaling fit on train only."""
    return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("scaler", StandardScaler()), ("regressor", LinearRegression())])


def add_appearance(cases: pd.DataFrame, pairs: pd.DataFrame, root: Path, settings: dict) -> pd.DataFrame:
    """Combine cached ORB/geometric features with per-image color/D summaries."""
    if not set(pairs.split).issubset({"train", "validation"}):
        raise ValueError("Baseline construction must not include test pairs")
    identities = cases[cases.split.isin(pairs.split.unique())].set_index("tree_id")
    images = {}
    for number, (identity, row) in enumerate(identities.iterrows(), 1):
        for season in ("summer", "winter"):
            depth = row[f"{season}_depth"]
            images[(identity, season)] = extract_appearance(
                root / row[f"{season}_rgb"], root / depth if depth else "", settings, root / settings["cache_dir"])
        if number % 100 == 0 or number == len(identities):
            print(f"Appearance summaries: {number}/{len(identities)} trees", flush=True)
    records = [pair_appearance(images[(r.query_id, "summer")], images[(r.candidate_id, "winter")])
               for r in pairs.itertuples()]
    return pd.concat([pairs.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def rank_candidates(frame: pd.DataFrame, atol=1e-12, rtol=1e-10) -> pd.DataFrame:
    """Sort by continuous score and preserve tolerance-based rank ranges.

    IDs break display ties only. Metric functions average over tied positions.
    """
    ordered = frame.sort_values(["score", "candidate_id"], ascending=[False, True], kind="stable").copy()
    scores = ordered.score.to_numpy()
    if not np.isfinite(scores).all():
        raise ValueError("Model scores must be finite")
    lower, upper = np.empty(len(scores), dtype=int), np.empty(len(scores), dtype=int)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and np.isclose(scores[stop], scores[start], atol=atol, rtol=rtol):
            stop += 1
        lower[start:stop], upper[start:stop] = start + 1, stop
        start = stop
    ordered["rank_min"], ordered["rank_max"] = lower, upper
    ordered["display_rank"] = np.arange(1, len(ordered) + 1)
    return ordered


def retrieval_results(frame: pd.DataFrame, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Closed-set retrieval against the complete split gallery; ties get expected credit."""
    rankings, queries = [], []
    for query_id, group in frame.groupby("query_id", sort=True):
        ordered = rank_candidates(group, settings["score_tie_atol"], settings["score_tie_rtol"])
        correct = ordered[ordered.candidate_id.eq(query_id)]
        if len(correct) != 1 or ordered.candidate_id.duplicated().any():
            raise ValueError("Each closed-set query needs one true filename identity and unique candidates")
        row = correct.iloc[0]
        positions = np.arange(row.rank_min, row.rank_max + 1)
        queries.append({"query_id": query_id, "gallery_size": len(group),
                        "rank_min": int(row.rank_min), "rank_max": int(row.rank_max),
                        "top1": float(np.mean(positions <= 1)), "top5": float(np.mean(positions <= 5)),
                        "reciprocal_rank": float(np.mean(1 / positions)),
                        "original_score": float(row.score), "top_candidate": ordered.iloc[0].candidate_id})
        rankings.append(ordered)
    query_metrics = pd.DataFrame(queries)
    metrics = {"queries": len(queries), "gallery_size": int(query_metrics.gallery_size.iloc[0]),
               "top1": float(query_metrics.top1.mean()), "top5": float(query_metrics.top5.mean()),
               "mrr": float(query_metrics.reciprocal_rank.mean()),
               "true_identity_tied_queries": int(query_metrics.rank_max.gt(query_metrics.rank_min).sum())}
    gallery = metrics["gallery_size"]
    metrics["uniform_random_reference"] = {"top1": 1/gallery, "top5": min(5, gallery)/gallery,
                                           "mrr": float(np.mean(1/np.arange(1, gallery+1)))}
    return pd.concat(rankings, ignore_index=True), query_metrics, metrics


def regression_errors(frame: pd.DataFrame) -> dict:
    labels = frame.query_id.eq(frame.candidate_id).astype(int).to_numpy()
    errors = frame.score.to_numpy() - labels
    result = {}
    for name, mask in [("all_pairs", np.ones(len(frame), dtype=bool)), ("positive_pairs", labels == 1),
                        ("negative_pairs", labels == 0)]:
        selected = errors[mask]
        result[name] = {"count": int(mask.sum()), "mae": float(np.mean(np.abs(selected))),
                        "rmse": float(np.sqrt(np.mean(selected**2)))}
    result["balanced_class_mae"] = (result["positive_pairs"]["mae"] + result["negative_pairs"]["mae"]) / 2
    return result


def run_baseline(root: Path, preparation: dict, screening: dict, settings: dict, render=True) -> dict:
    """Fit the linear baseline and export training/validation results only."""
    root = Path(root).resolve()
    settings = dict(settings)
    output = root / settings["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    cases = portable_frame(preparation["cases"], root)
    pairs = screening["pair_features"].copy()
    if not {"train", "validation"}.issubset(set(pairs.split)) or pairs.split.eq("test").any():
        raise ValueError("Baseline requires train and validation, with no test pairs")
    start = perf_counter()
    combined = add_appearance(cases, pairs, root, settings)
    appearance_seconds = perf_counter() - start
    columns = FEATURE_COLUMNS + (APPEARANCE_COLUMNS if settings["include_depth"] else ["hsv_hellinger"])
    sampled = training_pairs(cases, settings)
    sampled = sampled.merge(combined[["split", "query_id", "candidate_id", *columns]],
                            on=["split", "query_id", "candidate_id"], how="left", validate="one_to_one", indicator=True)
    if not sampled._merge.eq("both").all():
        raise ValueError("Missing sampled pair feature rows")
    sampled = sampled.drop(columns="_merge")
    X, y, weights = sampled[columns], sampled.target, sampled.sample_weight
    pipeline = make_pipeline()
    fit_times = []
    for _ in range(settings["timing_repeats"]):
        start = perf_counter()
        fitted = clone(pipeline).fit(X, y, regressor__sample_weight=weights)
        fit_times.append(perf_counter()-start)
    validation = combined[combined.split.eq("validation")]
    predict_times = []
    for _ in range(settings["timing_repeats"]):
        start = perf_counter()
        fitted.predict(validation[columns])
        predict_times.append(perf_counter()-start)
    all_rankings, all_queries, metrics = [], [], {}
    for split in ("train", "validation"):
        frame = combined[combined.split.eq(split)].copy()
        frame["score"] = fitted.predict(frame[columns])
        ranked, query_metrics, retrieval = retrieval_results(frame, settings)
        query_metrics.insert(0, "split", split)
        query_metrics = query_metrics.merge(cases[["tree_id", "row", "human_verified", "label_basis"]],
                                            left_on="query_id", right_on="tree_id", validate="one_to_one").drop(columns="tree_id")
        metrics[split] = {"retrieval": retrieval, "regression": regression_errors(frame),
                          "label_basis": "filename identities with recorded human provenance",
                          "evaluation_role": "resubstitution" if split == "train" else "held-out rows; development only"}
        ranked[["split", "query_id", "candidate_id", "score", "display_rank", "rank_min", "rank_max"]].to_csv(
            output / f"{split}_rankings.csv", index=False)
        all_rankings.append(ranked)
        all_queries.append(query_metrics)
    query_metrics = pd.concat(all_queries, ignore_index=True)
    query_metrics.to_csv(output / "query_metrics.csv", index=False)
    sampled.to_csv(output / "training_pairs.csv", index=False)
    scaled = fitted[:-1].transform(X)
    weighted_design = np.c_[scaled, np.ones(len(X))] * np.sqrt(weights.to_numpy())[:, None]
    rank = int(np.linalg.matrix_rank(weighted_design))
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "opencv-python-headless")}
    frozen_folder = screening["evaluation"]["folder"]
    protocol_hashes = {name: file_digest(frozen_folder / name) for name in ("queries.csv", "gallery.csv", "protocol.json")}
    audit = screening["evaluation"]["review"]
    audit.to_csv(output / "independent_review_normalization.csv", index=False, encoding="utf-8-sig")
    summary = {"model": "LinearRegression", "main_method_reserved": "RBF-SVR (stage 6)",
               "test_queries_scored": 0, "independent_review_effective_counts": audit.independent_status.value_counts().to_dict(),
               "training_pair_count": len(sampled), "training_positive_count": int(y.sum()),
               "training_negative_count": int((y == 0).sum()), "sample_weight_sum_per_class": sampled.groupby("target").sample_weight.sum().to_dict(),
               "feature_columns": columns, "feature_count": len(columns),
               "all_missing_training_features": [c for c in columns if X[c].isna().all()],
               "weighted_affine_design_rank": rank, "weighted_affine_design_columns": weighted_design.shape[1],
               "scores_are_probabilities": False, "score_min": float(min(f.score.min() for f in all_rankings)),
               "score_max": float(max(f.score.max() for f in all_rankings)),
               "imputer_scaler_fit_on_train_only": True, "candidate_gallery_restricted_by_row": False,
               "appearance_feature_seconds": appearance_seconds, "fit_seconds_repetitions": fit_times,
               "median_fit_seconds": float(np.median(fit_times)), "batch_predict_seconds_repetitions": predict_times,
               "median_batch_predict_seconds": float(np.median(predict_times)),
               "median_amortized_score_ms_per_validation_query": float(1000*np.median(predict_times)/metrics["validation"]["retrieval"]["queries"]),
               "timing_scope": "CPU fitting and batched scoring of cached features; excludes image extraction and ranking",
               "feature_matrix_bytes": int(combined[columns].to_numpy().nbytes),
               "settings": settings, "versions": versions, "frozen_protocol_hashes": protocol_hashes,
               "training_pair_identity_hash": stable_digest(sampled[["query_id", "candidate_id", "target", "sample_weight"]].to_dict("records")),
               "limitation": "Development scores mostly use filename labels; no independent test accuracy is claimed."}
    bundle = {"pipeline": fitted, "feature_columns": columns, "settings": settings,
              "summary": summary, "screen_settings": screening["summary"]["settings"]}
    (output / "models").mkdir(exist_ok=True)
    joblib.dump(bundle, output / "models/linear_regression.joblib")
    pd.DataFrame({"feature": columns, "standardized_coefficient": fitted.named_steps["regressor"].coef_,
                  "imputed_median": fitted.named_steps["imputer"].statistics_,
                  "training_scale": fitted.named_steps["scaler"].scale_}).to_csv(output / "coefficients.csv", index=False)
    write_json(output / "metrics.json", metrics)
    write_json(output / "run_summary.json", summary)
    if render:
        render_results(cases, all_rankings[1], query_metrics, metrics, root, output)
    return {"bundle": bundle, "metrics": metrics, "summary": summary, "query_metrics": query_metrics,
            "rankings": all_rankings, "output_dir": output, "training_pairs": sampled}


def render_results(cases, validation, query_metrics, metrics, root, output):
    folder = output / "figures"
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = ["top1", "top5", "mrr"]
    for i, split in enumerate(("train", "validation")):
        axes[0].bar(np.arange(3)+i*.35, [metrics[split]["retrieval"][n] for n in names], .35, label=split)
    axes[0].set(xticks=np.arange(3)+.175, xticklabels=names, ylim=(0, 1), title="Linear Regression: filename-label retrieval")
    axes[0].legend()
    val = query_metrics[query_metrics.split.eq("validation")].copy()
    joined = val.merge(cases[["tree_id", "number"]], left_on="query_id", right_on="tree_id")
    image = axes[1].scatter(joined.row.str[1:].astype(int), joined.number, c=joined.top1, cmap="RdYlGn", vmin=0, vmax=1)
    axes[1].set(xlabel="R: row index", ylabel="N: tree number", title="Validation index diagram (not metric coordinates)")
    axes[1].invert_yaxis()
    fig.colorbar(image, ax=axes[1], label="Top-1 credit")
    fig.tight_layout()
    fig.savefig(folder / "retrieval_overview.png", dpi=130)
    plt.close(fig)
    identities = cases.set_index("tree_id")
    # Fixed IDs avoid selecting only attractive successes.
    selected = val.sort_values("query_id").head(3)
    fig, axes = plt.subplots(len(selected), 3, figsize=(11, 3*len(selected)), squeeze=False)
    for axis_row, record in zip(axes, selected.itertuples()):
        paths = [(identities.loc[record.query_id].summer_rgb, f"{record.query_id}: Summer query"),
                 (identities.loc[record.query_id].winter_rgb, f"{record.query_id}: filename pair"),
                 (identities.loc[record.top_candidate].winter_rgb, f"{record.top_candidate}: displayed rank 1")]
        for axis, (path, title) in zip(axis_row, paths):
            axis.imshow(read_rgb(root / path)); axis.set_title(title, fontsize=9); axis.axis("off")
    fig.suptitle("Validation examples: filename labels, not independent identity truth", fontsize=11)
    fig.tight_layout()
    fig.savefig(folder / "validation_examples.png", dpi=120)
    plt.close(fig)


def match_tree(query: dict, gallery: pd.DataFrame, model: dict, root: Path) -> pd.DataFrame:
    """Rank one RGB/D query against every supplied winter candidate, without labels.

    query: rgb/depth paths; gallery: candidate_id, rgb, optional depth columns.
    model: trusted local bundle returned by run_baseline/joblib; root: path base.
    Returns every continuous score and rank range, with display ranks for top-five
    inspection. Corrupt RGB raises; missing/unsupported D is retained with flags.
    """
    root = Path(root)
    settings, screen = model["settings"], model["screen_settings"]
    def image(record):
        rgb, depth = root / record["rgb"], record.get("depth", "")
        orb = extract_image_features(rgb, screen, root / screen["cache_dir"] / "inference_images")
        appearance = extract_appearance(rgb, root / depth if depth else "", settings, root / settings["cache_dir"])
        return orb, appearance
    if gallery.empty or gallery.candidate_id.duplicated().any():
        raise ValueError("Gallery must contain non-empty, unique candidate IDs")
    qorb, qapp = image(query)
    rows = []
    for candidate in gallery.to_dict("records"):
        corb, capp = image(candidate)
        rows.append({"candidate_id": candidate["candidate_id"], **extract_pair_features(qorb, corb, screen),
                     **pair_appearance(qapp, capp)})
    frame = pd.DataFrame(rows)
    frame["score"] = model["pipeline"].predict(frame[model["feature_columns"]])
    return rank_candidates(frame, settings["score_tie_atol"], settings["score_tie_rtol"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    from .main import run_data_preparation
    from .screening import run_targeted_review
    root = Path(__file__).resolve().parents[1]
    preparation = run_data_preparation(root, render=False)
    screening = run_targeted_review(root, preparation=preparation, render=False)
    settings = {**screening["summary"]["settings"], **json.loads((root / "config/baseline.json").read_text())}
    run_baseline(root, preparation, screening, settings, render=not args.no_render)
