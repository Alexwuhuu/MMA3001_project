"""Low-cost, validation-only feature direction, hubness and rank diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.appearance import APPEARANCE_COLUMNS
from src.baseline import add_appearance, retrieval_results
from src.features import FEATURE_COLUMNS
from src.utils import write_json

QUERY_ONLY = ["query_keypoints", "missing_query_descriptors", "missing_query_depth", "query_depth_no_nonzero"]
CANDIDATE_ONLY = ["candidate_keypoints", "missing_candidate_descriptors", "missing_candidate_depth", "candidate_depth_no_nonzero"]


def load_development(root):
    """Reload trusted historical development features without reading test images or fitting.

    Args:
        root (str | Path): Project containing preparation, baseline and targeted-review outputs.

    Returns:
        tuple: Development cases, original training sample, trusted baseline bundle and complete
        development feature frame.

    Raises:
        ValueError: Development rows overlap or pair tables include test data.
        OSError: Required saved artifacts/images/caches are inaccessible.
    """
    root = Path(root)
    cases = pd.read_csv(root/"outputs/data_preparation/split_manifest.csv", keep_default_na=False)
    cases = cases[cases.split.isin(["train", "validation"])].copy()
    if set(cases.loc[cases.split.eq("train"), "row"]) & set(cases.loc[cases.split.eq("validation"), "row"]):
        raise ValueError("Development orchard rows overlap")
    sample = pd.read_csv(root/"outputs/baseline_linear/training_pairs.csv", float_precision="round_trip")
    bundle = joblib.load(root/"outputs/baseline_linear/models/linear_regression.joblib")
    pairs = pd.read_csv(root/"outputs/targeted_review/pair_features.csv", float_precision="round_trip")
    if not set(pairs.split).issubset({"train", "validation"}):
        raise ValueError("Original feature input must exclude test")
    frame = add_appearance(cases, pairs, root, bundle["settings"])
    return cases, sample, bundle, frame


def _auc(y, values):
    """Compute descriptive AUC only when both target classes are represented.

    Args:
        y (array-like): Binary filename-equality targets.
        values (array-like): Aligned finite feature scores.

    Returns:
        float | None: ROC AUC, or None for a single/empty target class.

    Raises:
        ValueError: Targets or scores are incompatible.
    """
    return float(roc_auc_score(y, values)) if len(np.unique(y)) == 2 else None


def feature_diagnostics(validation, sample, columns, settings):
    """Measure signal, direction and missingness while keeping full galleries for MRR.

    Args:
        validation (pandas.DataFrame): Validation-only complete pair feature frame.
        sample (pandas.DataFrame): Original training-only sample used for median fills.
        columns (list[str]): Numeric feature names to diagnose.
        settings (dict): Retrieval score tie tolerances.

    Returns:
        pandas.DataFrame: Per-feature finite coverage, class missingness, AUCs and full-gallery
        retrieval metrics in both directions. These are development diagnostics.

    Raises:
        ValueError: Inputs have the wrong split or invalid closed-set gallery scores.
    """
    if not validation.split.eq("validation").all() or not sample.split.eq("train").all():
        raise ValueError("Diagnostics require validation and train-only imputation")
    y = validation.query_id.eq(validation.candidate_id).astype(int)
    rows = []
    for column in columns:
        raw = validation[column].astype(float)
        finite = np.isfinite(raw)
        median = sample[column].median()
        median = float(median) if pd.notna(median) else 0.
        values = raw.where(finite, median)
        item = {"feature": column, "finite_pairs": int(finite.sum()),
                "finite_positives": int((finite & y.eq(1)).sum()), "finite_negatives": int((finite & y.eq(0)).sum()),
                "missing_positive_fraction": float((~finite)[y.eq(1)].mean()),
                "missing_negative_fraction": float((~finite)[y.eq(0)].mean()),
                "auc_higher_finite": _auc(y[finite], raw[finite]),
                "auc_lower_finite": _auc(y[finite], -raw[finite]),
                "auc_missingness": _auc(y, (~finite).astype(int)),
                "auc_higher_train_imputed": _auc(y, values),
                "train_median_fill": median,
                "query_only": column in QUERY_ONLY, "candidate_only": column in CANDIDATE_ONLY,
                "query_constant_fraction": float(validation.assign(v=values).groupby("query_id").v.nunique().le(1).mean())}
        for direction, sign in [("higher", 1), ("lower", -1)]:
            scored = validation[["split", "query_id", "candidate_id"]].assign(score=sign*values)
            _, _, metrics = retrieval_results(scored, settings)
            item.update({f"{direction}_{m}": metrics[m] for m in ("top1", "top5", "mrr")})
        rows.append(item)
    return pd.DataFrame(rows)


def hubness(rankings):
    """Count expected candidate appearances while distributing boundary ties fairly.

    Args:
        rankings (pandas.DataFrame): query_id/candidate_id, rank_min/rank_max and display_rank
            rows.

    Returns:
        tuple[pandas.DataFrame, dict]: Candidate expected/displayed Top-1/Top-5 counts and
        concentration summary, including HHI and effective candidate count.

    Raises:
        ValueError: A query-candidate ranking is duplicated.
    """
    if rankings.duplicated(["query_id", "candidate_id"]).any():
        raise ValueError("Duplicate query-candidate rankings")
    data = rankings.copy()
    for k in (1, 5):
        width = data.rank_max-data.rank_min+1
        data[f"expected_top{k}"] = np.clip(np.minimum(k, data.rank_max)-data.rank_min+1, 0, None)/width
        data[f"display_top{k}"] = data.display_rank.le(k).astype(int)
    counts = data.groupby("candidate_id")[["expected_top1", "expected_top5", "display_top1", "display_top5"]].sum()
    counts["query_count"] = data.query_id.nunique()
    counts["top5_query_fraction"] = counts.expected_top5/counts.query_count
    total = counts.expected_top5.sum()
    share = counts.expected_top5/total
    summary = {"queries": int(data.query_id.nunique()), "candidates": len(counts),
               "expected_top5_total": float(total), "max_top5_query_fraction": float(counts.top5_query_fraction.max()),
               "top10_candidate_share": float(share.nlargest(10).sum()),
               "top5_hhi": float((share**2).sum()), "effective_candidates": float(1/(share**2).sum()),
               "zero_top5_candidates": int(counts.expected_top5.eq(0).sum())}
    return counts.reset_index().sort_values(["expected_top5", "candidate_id"], ascending=[False, True]), summary


def rank_distribution(queries):
    """Describe filename-positive ranks with expected positions and a tie-aware CDF.

    Args:
        queries (pandas.DataFrame): Nonempty per-query rank_min/rank_max and gallery_size.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]: Copy with expected_rank and a
        rank/true_pair_cdf table up to the largest gallery size.
    """
    data = queries.copy()
    data["expected_rank"] = (data.rank_min+data.rank_max)/2
    rows = []
    for k in range(1, int(data.gallery_size.max())+1):
        credit = np.clip(np.minimum(k, data.rank_max)-data.rank_min+1, 0, None)/(data.rank_max-data.rank_min+1)
        rows.append({"rank": k, "true_pair_cdf": float(credit.mean())})
    return data, pd.DataFrame(rows)


def run_diagnostics(root, output=None, development=None, render=True, reference_dirs=None):
    """Export validation feature directions, candidate hubness and rank distributions.

    Args:
        root (str | Path): Project base for historical outputs.
        output (str | Path | None): Destination; None uses outputs/validation_diagnostics.
        development (tuple | None): Existing load_development result; None reloads it.
        render (bool): Whether to write diagnostic figures.
        reference_dirs (dict | None): Model-to-directory mapping; None uses historical
            linear/SVR outputs.

    Returns:
        dict: features, hubs, ranks, cdf, summary, output_dir and the development tuple for
        reuse.

    Raises:
        ValueError: Development inputs or reference ranking tables fail validation.
        OSError: Historical inputs or diagnostic outputs are inaccessible.
    """
    root = Path(root)
    output = Path(output or root/"outputs/validation_diagnostics")
    output.mkdir(parents=True, exist_ok=True)
    cases, sample, bundle, frame = development or load_development(root)
    validation = frame[frame.split.eq("validation")].copy()
    columns = FEATURE_COLUMNS+APPEARANCE_COLUMNS
    features = feature_diagnostics(validation, sample, columns, bundle["settings"])
    features.to_csv(output/"single_feature_signal.csv", index=False)
    hubs, rank_rows, cdfs, summaries = [], [], [], {}
    folders = reference_dirs or {"LinearRegression": "outputs/baseline_linear", "RBF-SVR": "outputs/svr_main"}
    for model, folder in folders.items():
        rankings = pd.read_csv(root/folder/"validation_rankings.csv")
        queries = pd.read_csv(root/folder/"query_metrics.csv")
        queries = queries[queries.split.eq("validation")]
        counts, stats = hubness(rankings)
        attributes = validation.groupby("candidate_id")[CANDIDATE_ONLY].first().reset_index()
        counts = counts.merge(attributes, on="candidate_id", validate="one_to_one")
        counts["row"] = counts.candidate_id.map(cases.set_index("tree_id").row)
        counts.insert(0, "model", model)
        hubs.append(counts)
        query_ranks, cdf = rank_distribution(queries)
        query_ranks.insert(0, "model", model)
        cdf.insert(0, "model", model)
        rank_rows.append(query_ranks)
        cdfs.append(cdf)
        stats["candidate_attribute_spearman"] = {c: (float(counts[c].corr(counts.expected_top5, method="spearman"))
            if counts[c].nunique() > 1 and counts.expected_top5.nunique() > 1 else None) for c in CANDIDATE_ONLY}
        stats["rank_quantiles"] = {str(q): float(query_ranks.expected_rank.quantile(q)) for q in (.25, .5, .75, .9)}
        stats["true_pairs_below_gallery_midpoint"] = int(query_ranks.expected_rank.gt(stats["candidates"]/2).sum())
        summaries[model] = stats
    hubs = pd.concat(hubs, ignore_index=True)
    rank_rows = pd.concat(rank_rows, ignore_index=True)
    cdfs = pd.concat(cdfs, ignore_index=True)
    hubs.to_csv(output/"candidate_hubness.csv", index=False)
    rank_rows.to_csv(output/"true_pair_ranks.csv", index=False)
    cdfs.to_csv(output/"rank_cdf.csv", index=False)
    summary = {"split": "validation", "queries": validation.query_id.nunique(),
               "candidate_count": validation.candidate_id.nunique(), "pair_count": len(validation),
               "models": summaries, "test_images_read": 0, "test_predictions": 0,
               "label_basis": "filename identities; not new human confirmations",
               "missing_policy": "Finite-only AUC with coverage; all-gallery MRR uses train-median fills and expected tie credit",
               "direction_policy": "Both directions are descriptive validation diagnostics, not an independent evaluation",
               "query_constants": "No within-query signal alone; RBF may condition on them, so removal needs an ablation"}
    write_json(output/"diagnostic_summary.json", summary)
    if render:
        render_diagnostics(features, hubs, rank_rows, cdfs, output)
    return {"features": features, "hubs": hubs, "ranks": rank_rows, "cdf": cdfs,
            "summary": summary, "output_dir": output, "development": (cases, sample, bundle, frame)}


def render_diagnostics(features, hubs, ranks, cdf, output):
    """Plot feature-direction signal and validation candidate/rank concentration.

    Args:
        features (pandas.DataFrame): Per-feature AUC/retrieval diagnostic rows.
        hubs (pandas.DataFrame): Per-model candidate hubness rows.
        ranks (pandas.DataFrame): Per-model true-pair expected ranks.
        cdf (pandas.DataFrame): Per-model rank CDF rows.
        output (Path): Existing directory for diagnostic PNGs.

    Returns:
        None: Diagnostic figures are written and closed.

    Raises:
        OSError: Figures cannot be written.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    names = features.feature.tolist()
    positions = np.arange(len(names))
    axes[0].scatter(features.auc_higher_finite, positions, label="higher is better")
    axes[0].scatter(features.auc_lower_finite, positions, label="lower is better", marker="x")
    axes[0].axvline(.5, c="gray", ls="--")
    axes[0].set(yticks=positions, yticklabels=names, xlabel="Finite-value validation AUC", xlim=(0, 1))
    axes[0].legend(fontsize=8)
    axes[1].scatter(features.higher_mrr, positions, label="higher")
    axes[1].scatter(features.lower_mrr, positions, label="lower", marker="x")
    axes[1].set(yticks=positions, yticklabels=[], xlabel="Complete-gallery single-feature MRR")
    axes[1].legend()
    fig.suptitle("Validation feature diagnostics; missing coverage is in the table")
    fig.tight_layout(); fig.savefig(output/"single_feature_signal.png", dpi=140); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for j, model in enumerate(("LinearRegression", "RBF-SVR")):
        group = hubs[hubs.model.eq(model)].head(20)
        axes[0, j].bar(group.candidate_id, group.expected_top5)
        axes[0, j].set(ylabel="Expected Top-5 appearances", title=model)
        axes[0, j].tick_params(axis="x", labelrotation=70, labelsize=7)
        selected = ranks[ranks.model.eq(model)]
        axes[1, 0].hist(selected.expected_rank, bins=np.linspace(1, selected.gallery_size.max()+1, 15), histtype="step", label=model)
        curve = cdf[cdf.model.eq(model)]
        axes[1, 1].plot(curve["rank"], curve.true_pair_cdf, label=model)
    axes[1, 0].set(xlabel="True filename-pair expected rank", ylabel="Queries")
    axes[1, 1].set(xlabel="Rank cutoff", ylabel="True-pair CDF")
    axes[1, 0].legend(); axes[1, 1].legend()
    fig.suptitle("Complete validation gallery: candidate hubs and true-pair ranks")
    fig.tight_layout(); fig.savefig(output/"hubness_and_ranks.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    plt.switch_backend("Agg")
    result = run_diagnostics(ROOT)
    print(result["features"].sort_values("lower_mrr", ascending=False).to_string(index=False))
    print(json.dumps(result["summary"], indent=2))
