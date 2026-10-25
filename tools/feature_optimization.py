"""Separate MRR-first development experiment; never score the historical test."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.baseline import make_pipeline, regression_errors, retrieval_results
from src.features import stable_digest
from src.features import preprocess_rgb
from src.reproduce import verify_export
from src.svr import _fit, _frame_hash, parameter_grid
from src.utils import file_digest, read_depth, read_rgb, write_json
from tools.matching_diagnostics import hubness, load_development, rank_distribution, run_diagnostics
from tools.relative_features import (BASE_COLUMNS, SEASON_COLUMNS, KEYS, build_development,
    context_features, depth_rank_mask, match_tree_optimized, training_fills)


def load_config(root):
    """Validate the predeclared MRR-first image/representation experiment.

    Args:
        root (str | Path): Project containing config/optimization.json.

    Returns:
        dict: Validated image, ratio, solver and selection settings.

    Raises:
        OSError: Configuration cannot be read.
        ValueError: JSON, MRR order, ORB settings or grid parameters are invalid.
    """
    settings = json.loads((Path(root)/"config/optimization.json").read_text())
    if settings["selection_order"][0] != "highest_validation_mrr":
        raise ValueError("The revised plan selects MRR first")
    if settings["ratios"] != [.85, .90] or settings["image_size"] != 384 or settings["orb_nfeatures"] != 1000:
        raise ValueError("Use the predeclared ORB experiment")
    parameter_grid(settings)
    return settings


def select_mrr(table, eligible=None):
    """Select a converged trial by MRR, Top-1, Top-5, scoring time and trial ID.

    Args:
        table (pandas.DataFrame): Trial rows containing status and validation/latency fields.
        eligible (sequence[str] | None): Optional representation allowlist; None accepts all
            representations.

    Returns:
        pandas.Series: Best eligible converged trial.

    Raises:
        ValueError: No eligible trial has status "ok".
    """
    valid = table[table.status.eq("ok")]
    if eligible is not None:
        valid = valid[valid.representation.isin(eligible)]
    if valid.empty:
        raise ValueError("No converged eligible development configuration")
    return valid.sort_values(["validation_mrr", "validation_top1", "validation_top5", "median_query_ms", "trial_id"],
                              ascending=[False, False, False, True, True], kind="stable").iloc[0]


def source_fingerprint(root, cases, sample, settings):
    """Hash development image contents and input/source evidence without reading test images.

    Args:
        root (Path): Project base for image and artifact paths.
        cases (pandas.DataFrame): Explicit train/validation identities and seasonal paths.
        sample (pandas.DataFrame): Original sample keys, targets and weights.
        settings (dict): Full experiment settings; destination-only paths are excluded from the
            digest.

    Returns:
        tuple[str, dict]: Stable input fingerprint and detailed
        image/sample/settings/version/source evidence.

    Raises:
        ValueError: A case belongs to a non-development split.
        OSError: A required source/image cannot be hashed.
    """
    if not set(cases.split).issubset({"train", "validation"}):
        raise ValueError("Fingerprint image inputs must exclude test")
    images = []
    for row in cases.sort_values("tree_id").itertuples():
        for season in ("summer", "winter"):
            rgb, depth = getattr(row, season+"_rgb"), getattr(row, season+"_depth")
            images.append({"identity": row.tree_id, "split": row.split, "season": season,
                           "rgb_sha256": file_digest(root/rgb),
                           "D_sha256": file_digest(root/depth) if depth and (root/depth).is_file() else "missing"})
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "opencv-python-headless", "scipy", "joblib")}
    code_root = Path(__file__).resolve().parents[1]
    sources = [code_root/p for p in ("tools/feature_optimization.py", "tools/relative_features.py", "tools/matching_diagnostics.py",
               "src/baseline.py", "src/svr.py", "src/features.py", "src/appearance.py")]
    evidence = {"images": images, "sample_hash": _frame_hash(sample[[*KEYS, "target", "sample_weight"]]),
                "settings": {k: v for k, v in settings.items() if k not in {"output_dir", "cache_dir", "selection_file", "small_results_dir", "export_dir", "diagnostics_dir"}},
                "versions": versions, "sources": {p.relative_to(code_root).as_posix(): file_digest(p) for p in sources}}
    return stable_digest(evidence), evidence


def input_records(root, settings, development=None):
    """Validate original training pair labels/weights before optimized feature extraction.

    Args:
        root (Path): Project base for saved development inputs.
        settings (dict): Experiment settings retained by this input-loading interface.
        development (tuple | None): Existing load_development tuple; None loads historical
            inputs.

    Returns:
        tuple: cases, sample, baseline bundle, original feature table and training-pair identity
        hash.

    Raises:
        ValueError: Samples contain test/out-of-training IDs, duplicates or changed
            labels/weights.
        OSError: Historical artifacts are inaccessible.
    """
    cases, sample, baseline, old_features = development or load_development(root)
    train = set(cases.loc[cases.split.eq("train"), "tree_id"])
    if (not sample.split.eq("train").all() or not set(sample.query_id).issubset(train) or
            not set(sample.candidate_id).issubset(train) or sample.duplicated(KEYS).any() or
            not np.array_equal(sample.target, sample.query_id.eq(sample.candidate_id).astype(int))):
        raise ValueError("Reuse the original train-only pair labels and weights")
    pair_hash = stable_digest(sample[["query_id", "candidate_id", "target", "sample_weight"]].to_dict("records"))
    if pair_hash != baseline["summary"]["training_pair_identity_hash"]:
        raise ValueError("Original sample identities/weights changed")
    return cases, sample, baseline, old_features, pair_hash


def score_pipeline(pipeline, train, validation, columns, settings):
    """Score one fitted representation on complete validation galleries.

    Args:
        pipeline (sklearn.pipeline.Pipeline): Training-fitted regression pipeline.
        train (pandas.DataFrame): Original sampled training pairs with features.
        validation (pandas.DataFrame): Complete validation pair features.
        columns (list[str]): Exact numerical feature order.
        settings (dict): Score tie tolerances.

    Returns:
        tuple[dict, pandas.DataFrame, pandas.DataFrame, dict]: Flat trial fields, full rankings,
        per-query metrics and nested retrieval/regression results. The latency field here is
        one measured batch; search_svr replaces it with repeated-run medians.

    Raises:
        ValueError: Features, scores or gallery identities are invalid.
    """
    started = perf_counter()
    scores = pipeline.predict(validation[columns])
    seconds = perf_counter()-started
    scored = validation[KEYS].assign(score=scores)
    ranked, queries, metrics = retrieval_results(scored, settings)
    errors = regression_errors(scored)
    train_errors = regression_errors(train[KEYS].assign(score=pipeline.predict(train[columns])))
    fields = {f"validation_{key}": metrics[key] for key in ("top1", "top5", "mrr")}
    fields["median_query_ms"] = 1000*seconds/queries.query_id.nunique()
    for split, values in [("train", train_errors), ("validation", errors)]:
        fields[f"{split}_balanced_mae"] = values["balanced_class_mae"]
        for label in ("positive_pairs", "negative_pairs"):
            for metric in ("mae", "rmse"):
                fields[f"{split}_{label}_{metric}"] = values[label][metric]
    return fields, ranked, queries, {"retrieval": metrics, "regression": errors, "train_sample_regression": train_errors}


def screen_features(root, cases, sample, settings, output):
    """Screen predeclared image/representation variants with a fixed pilot SVR.

    Only eligible SVR rows can select the representation. Raw all-column controls
    and linear comparisons remain descriptive development results.

    Args:
        root (Path): Base for development images/caches.
        cases (pandas.DataFrame): Development-only case metadata.
        sample (pandas.DataFrame): Original training pairs/labels/weights.
        settings (dict): Preprocessing, ratios, representations, eligibility and pilot_svr
            settings.
        output (Path): Existing destination for progressive screening tables and image audits.

    Returns:
        tuple[pandas.DataFrame, dict]: All screening rows and winning train/validation inputs
        with columns, fills, calibration and representation/image choices.

    Raises:
        ValueError: Feature construction loses samples or no eligible model converges.
        AssertionError: Winner bookkeeping disagrees with the ordered trial table.
        OSError: Feature caches or result files are inaccessible.
    """
    records, best_inputs = [], None
    current_best = -np.inf
    variant_hashes, audits = {}, []
    for preprocessing in settings["preprocessing"]:
        data = build_development(root, cases, settings, preprocessing)
        audit = data["audit"].assign(preprocessing=preprocessing)
        audits.append(audit)
        audit.to_csv(output/f"{preprocessing}_image_audit.csv", index=False)
        for ratio_text, frame in data["frames"].items():
            fills = training_fills(frame, sample)
            for representation in settings["representations"]:
                transformed, columns = context_features(frame, representation, fills)
                train = sample[[*KEYS, "target", "label_basis", "sample_weight"]].merge(
                    transformed[transformed.split.eq("train")], on=KEYS, how="left", validate="one_to_one", indicator=True)
                if not train._merge.eq("both").all():
                    raise ValueError("Context transform lost an original training pair")
                train = train.drop(columns="_merge")
                validation = transformed[transformed.split.eq("validation")].copy()
                slug = f"{preprocessing}_{ratio_text}_{representation}"
                variant_hashes[slug] = {"train": _frame_hash(train), "validation": _frame_hash(validation)}
                models = [("LinearRegression", make_pipeline().fit(train[columns], train.target,
                           regressor__sample_weight=train.sample_weight), True)]
                svr, fit_seconds, converged = _fit(settings["pilot_svr"], settings, train[columns], train.target, train.sample_weight)
                models.append(("RBF-SVR", svr, converged))
                for name, pipeline, ok in models:
                    trial_id = len(records)
                    row = {"trial_id": trial_id, "preprocessing": preprocessing, "ratio": float(ratio_text),
                           "representation": representation, "model": name, "features": len(columns),
                           "status": "ok" if ok else "not_converged", "eligible": representation in settings["selection_eligible"]}
                    if ok:
                        fields, ranked, queries, metrics = score_pipeline(pipeline, train, validation, columns, settings)
                        row.update(fields)
                        reg = pipeline.named_steps["regressor"]
                        row["support_vectors"] = len(reg.support_) if name == "RBF-SVR" else 0
                        row["support_vector_fraction"] = len(reg.support_)/len(train) if name == "RBF-SVR" else 0.
                        _, hub = hubness(ranked)
                        row["max_top5_query_fraction"] = hub["max_top5_query_fraction"]
                        row["top10_candidate_share"] = hub["top10_candidate_share"]
                        folder = output/"screening_query_metrics"
                        folder.mkdir(exist_ok=True)
                        queries.to_csv(folder/f"{slug}_{name}.csv", index=False)
                        if name == "RBF-SVR" and row["eligible"] and row["validation_mrr"] >= current_best:
                            candidate_table = pd.DataFrame([*records, row])
                            winning = select_mrr(candidate_table[candidate_table.model.eq("RBF-SVR")], settings["selection_eligible"])
                            if int(winning.trial_id) == trial_id:
                                current_best = row["validation_mrr"]
                                best_inputs = {"train": train.copy(), "validation": validation.copy(), "columns": columns,
                                    "fills": fills, "calibration": data["calibration"], "preprocessing": preprocessing,
                                    "ratio": float(ratio_text), "representation": representation, "screen_trial_id": trial_id}
                    records.append(row)
                print(f"Screen {slug}: linear MRR={records[-2].get('validation_mrr',float('nan')):.4f}; "
                      f"SVR MRR={records[-1].get('validation_mrr',float('nan')):.4f}", flush=True)
                pd.DataFrame(records).to_csv(output/"feature_screening.csv", index=False)
        del data
    if best_inputs is None:
        raise ValueError("No eligible feature representation converged")
    table = pd.DataFrame(records)
    selected = select_mrr(table[table.model.eq("RBF-SVR")], settings["selection_eligible"])
    if int(selected.trial_id) != best_inputs["screen_trial_id"]:
        raise AssertionError("Feature-screen winner bookkeeping mismatch")
    pd.concat(audits, ignore_index=True).to_csv(output/"image_audit.csv", index=False)
    write_json(output/"feature_input_hashes.json", variant_hashes)
    return table, best_inputs


def search_svr(inputs, settings, output):
    """Search the declared RBF grid on the selected representation using validation MRR.

    Args:
        inputs (dict): Winning train, validation and columns from screen_features.
        settings (dict): RBF grid/solver/timing and retrieval settings.
        output (Path): Existing destination for progressive svr_search.csv.

    Returns:
        tuple[pandas.DataFrame, dict, sklearn.pipeline.Pipeline, int]: Trial table, selected
        parameters, fitted winning pipeline and trial ID.

    Raises:
        ValueError: Grid/input validation fails or no trial converges.
        OSError: Search results cannot be written.
    """
    train, validation, columns = inputs["train"], inputs["validation"], inputs["columns"]
    trials, pipelines = [], {}
    grid = parameter_grid(settings)
    for trial_id, params in enumerate(grid):
        pipeline, seconds, converged = _fit(params, settings, train[columns], train.target, train.sample_weight)
        row = {"trial_id": trial_id, "C": params["C"], "gamma": str(params["gamma"]), "epsilon": params["epsilon"],
               "status": "ok" if converged else "not_converged", "fit_seconds": seconds,
               "support_vectors": len(pipeline.named_steps["regressor"].support_),
               "support_vector_fraction": len(pipeline.named_steps["regressor"].support_)/len(train)}
        if converged:
            fields, _, _, _ = score_pipeline(pipeline, train, validation, columns, settings)
            durations = []
            for _ in range(settings["timing_repeats"]):
                started = perf_counter(); pipeline.predict(validation[columns]); durations.append(perf_counter()-started)
            fields["median_query_ms"] = 1000*float(np.median(durations))/validation.query_id.nunique()
            row.update(fields)
            row["predict_seconds_repetitions"] = json.dumps(durations)
            pipelines[trial_id] = pipeline
        trials.append(row)
        pd.DataFrame(trials).to_csv(output/"svr_search.csv", index=False)
        print(f"MRR-first SVR {trial_id+1}/{len(grid)} {params}: MRR={row.get('validation_mrr',float('nan')):.4f}, "
              f"Top-1={row.get('validation_top1',float('nan')):.4f}, SV={row['support_vector_fraction']:.1%}", flush=True)
    trials = pd.DataFrame(trials)
    best = select_mrr(trials)
    return trials, grid[int(best.trial_id)], pipelines[int(best.trial_id)], int(best.trial_id)


def render_results(result):
    """Plot optimized feature screening, parameter sensitivity and validation comparison.

    Args:
        result (dict): Completed optimization result with
            screening/trials/comparison/output_dir.

    Returns:
        None: Aggregate PNG figures are written and closed.

    Raises:
        OSError: Figures cannot be written.
    """
    output = result["output_dir"]
    folder = output/"figures"; folder.mkdir(exist_ok=True)
    screening = result["screening"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for model, marker in [("LinearRegression", "o"), ("RBF-SVR", "x")]:
        data = screening[screening.model.eq(model)].copy()
        axes[0].scatter(np.arange(len(data)), data.validation_mrr, marker=marker, label=model)
        axes[1].scatter(data.validation_mrr, data.max_top5_query_fraction, marker=marker, label=model)
    labels = screening[screening.model.eq("RBF-SVR")].apply(lambda r: f"{r.preprocessing}/{r.ratio}/{r.representation}", axis=1)
    axes[0].set(xticks=np.arange(len(labels)), xticklabels=labels, ylabel="Validation MRR")
    axes[0].tick_params(axis="x", rotation=90, labelsize=6)
    axes[1].set(xlabel="Validation MRR", ylabel="Largest candidate Top-5 query fraction")
    axes[0].legend(); axes[1].legend()
    fig.suptitle("Feature screen: six image settings, five representations; no test scores")
    fig.tight_layout(); fig.savefig(folder/"feature_screening.png", dpi=150); plt.close(fig)
    comparison = result["comparison"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ("top1", "top5", "mrr")):
        ax.bar(comparison.model, comparison[metric])
        ax.set(ylabel=metric, title="Validation only")
        ax.tick_params(axis="x", labelrotation=35, labelsize=7)
    fig.tight_layout(); fig.savefig(folder/"validation_comparison.png", dpi=140); plt.close(fig)
    trials = result["trials"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for (gamma, epsilon), group in trials[trials.status.eq("ok")].groupby(["gamma", "epsilon"]):
        group = group.sort_values("C")
        label = f"g={gamma}, eps={epsilon}"
        axes[0].plot(group.C, group.validation_mrr, "o-", label=label)
        axes[1].plot(group.C, group.support_vector_fraction, "o-")
        axes[2].plot(group.C, group.validation_balanced_mae, "o-")
    for ax, title in zip(axes, ("Validation MRR", "Support-vector fraction", "Validation balanced MAE")):
        ax.set(xscale="log", xlabel="C", title=title)
    axes[0].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(folder/"expanded_epsilon.png", dpi=140); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, frame in result["queries"].items():
        ranks, curve = rank_distribution(frame)
        axes[0].hist(ranks.expected_rank, bins=np.linspace(1, ranks.gallery_size.max()+1, 15), histtype="step", label=name)
        axes[1].plot(curve["rank"], curve.true_pair_cdf, label=name)
    axes[0].set(xlabel="True filename-pair rank", ylabel="Validation queries")
    axes[1].set(xlabel="Rank cutoff", ylabel="True-pair CDF")
    axes[0].legend(); axes[1].legend()
    fig.tight_layout(); fig.savefig(folder/"optimized_ranks.png", dpi=140); plt.close(fig)


def render_mask_examples(root, cases, settings, output):
    """Illustrate the foreground heuristic on fixed training IDs without validating masks.

    Args:
        root (Path): Base for seasonal RGB/D paths.
        cases (pandas.DataFrame): Cases containing training identities.
        settings (dict): Crop/size and heuristic foreground settings.
        output (Path): Existing destination for figures/foreground_examples.png.

    Returns:
        None: Training example figure is written and closed.

    Raises:
        ValueError: An RGB image is unreadable.
        OSError: The figure cannot be written.
    """
    chosen = cases[cases.split.eq("train")].sort_values("tree_id").head(2)
    fig, axes = plt.subplots(2*len(chosen), 3, figsize=(10, 3*2*len(chosen)), squeeze=False)
    k = 0
    for row in chosen.itertuples():
        for season in ("summer", "winter"):
            rgb = read_rgb(root/getattr(row, season+"_rgb"))
            path = getattr(row, season+"_depth")
            depth, status = read_depth(root/path if path else "", rgb.shape[:2])
            image = preprocess_rgb(rgb, settings["image_size"], settings["crop"])
            if depth is not None:
                h, w = depth.shape; side = min(h, w)
                depth = depth[(h-side)//2:(h-side)//2+side, (w-side)//2:(w-side)//2+side]
                depth = cv2.resize(depth, (settings["image_size"], settings["image_size"]), interpolation=cv2.INTER_NEAREST)
            mask, fallback, ranks = depth_rank_mask(depth, image.shape[:2], settings)
            axes[k, 0].imshow(image)
            axes[k, 0].set_title(f"{row.tree_id} {season}")
            axes[k, 1].imshow(ranks, cmap="viridis", vmin=0, vmax=1)
            axes[k, 1].set_title("Nonzero D intensity percentile")
            axes[k, 2].imshow(image)
            axes[k, 2].imshow(np.where(mask, np.nan, 1), cmap="gray", alpha=.65, vmin=0, vmax=1)
            axes[k, 2].set_title(f"Selected {mask.mean():.1%}; fallback={fallback}")
            for ax in axes[k]: ax.axis("off")
            k += 1
    fig.suptitle("Training-only mask examples: intensity heuristic, not verified trunk segmentation")
    fig.tight_layout(); fig.savefig(output/"figures/foreground_examples.png", dpi=120); plt.close(fig)


def _json_calibration(calibration):
    """Convert seasonal NumPy calibration arrays to JSON-compatible lists.

    Args:
        calibration (dict): Seasonal mean/scale arrays and image counts.

    Returns:
        dict: Same seasonal/field structure with arrays converted to lists.
    """
    return {s: {k: a.tolist() if isinstance(a, np.ndarray) else a for k, a in record.items()} for s, record in calibration.items()}


def protect_existing(root):
    """Snapshot historical source, label, protocol and result hashes without extracting features.

    Args:
        root (Path): Project directory containing historical source/configuration/outputs.

    Returns:
        dict[str, str]: Existing protected project-relative paths mapped to SHA-256 digests.

    Raises:
        OSError: A protected file cannot be hashed.
    """
    paths = [*root.joinpath("src").glob("*.py"), root/"config/pair_review.csv",
             *root.joinpath("config/evaluation").glob("*"), root/"config/svr_selection.json",
             root/"config/resnet_svr_selection.json", root/"config/final_experiments_lock.json"]
    for folder in ("baseline_linear", "svr_main", "resnet_matching", "final_experiments"):
        paths += [p for p in (root/"outputs"/folder).rglob("*") if p.is_file() and p.suffix in {".csv", ".json", ".joblib", ".png"}]
    return {p.relative_to(root).as_posix(): file_digest(p) for p in paths if p.is_file()}


def run_optimization(root, settings=None, development=None, render=True):
    """Run or reuse the separate validation-only MRR-first representation experiment.

    Historical source/labels/results are checked for changes around execution. A
    matching lock reuses verified saved artifacts; changed source/input hashes
    require new selection/output paths. No historical test inference is performed.

    Args:
        root (str | Path): Project base.
        settings (dict | None): Experiment configuration; None loads config/optimization.json.
        development (tuple | None): Optional existing load_development tuple.
        render (bool): Whether a fresh run writes aggregate and mask-example figures.

    Returns:
        dict: summary, selection, screening, trials, comparison, bundles, queries, rankings,
        metrics, validation, cases and output_dir; fresh runs also include diagnostics.

    Raises:
        ValueError: Inputs/lock/artifacts change, no eligible fit converges, or protected
            historical files change.
        AssertionError: Shared preprocessing, selected metrics or saved-image inference fails to
            reproduce.
        OSError: Sources/caches/models/results are inaccessible.
    """
    root = Path(root).resolve()
    cfg = dict(settings or load_config(root))
    protected = protect_existing(root)
    cases, sample, baseline, old_features, pair_hash = input_records(root, cfg, development)
    fingerprint, evidence = source_fingerprint(root, cases, sample, cfg)
    output = root/cfg["output_dir"]; output.mkdir(parents=True, exist_ok=True)
    lock_path = root/cfg["selection_file"]
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text())
        # A saved choice belongs to its source bytes and inputs, including docstrings.
        if lock["input_fingerprint"] != fingerprint:
            raise ValueError("Optimized experiment inputs changed: preserve this lock and use a new selection/output path")
        result = load_run(root, cfg, cases=cases)
        if protect_existing(root) != protected:
            raise ValueError("A historical result/label file changed")
        result["summary"]["selection_reused"] = True
        print("Reusing validated optimized artifacts and MRR-first selection; no test scoring.", flush=True)
        return result
    reference_dirs = {"LinearRegression": cfg.get("baseline_dir", "outputs/baseline_linear"),
                      "RBF-SVR": cfg.get("svr_dir", "outputs/svr_main")}
    diagnostics = run_diagnostics(root, root/cfg["diagnostics_dir"], (cases, sample, baseline, old_features),
                                  render=render, reference_dirs=reference_dirs)
    write_json(output/"input_evidence.json", evidence)
    screening, selected = screen_features(root, cases, sample, cfg, output)
    trials, params, svr, selected_id = search_svr(selected, cfg, output)
    columns, train, validation = selected["columns"], selected["train"], selected["validation"]
    fit_times = {"LinearRegression": [], "RBF-SVR": []}
    for _ in range(cfg["timing_repeats"]):
        started = perf_counter()
        linear = make_pipeline().fit(train[columns], train.target, regressor__sample_weight=train.sample_weight)
        fit_times["LinearRegression"].append(perf_counter()-started)
        svr, seconds, converged = _fit(params, cfg, train[columns], train.target, train.sample_weight)
        if not converged:
            raise ValueError("Selected SVR did not converge on refit")
        fit_times["RBF-SVR"].append(seconds)
    for step, attrs in [("imputer", ["statistics_"]), ("scaler", ["mean_", "scale_"])]:
        for attribute in attrs:
            np.testing.assert_allclose(getattr(linear.named_steps[step], attribute), getattr(svr.named_steps[step], attribute), rtol=0, atol=0)
    best_screen = screening.loc[screening.trial_id.eq(selected["screen_trial_id"])].iloc[0]
    best_grid = trials.loc[trials.trial_id.eq(selected_id)].iloc[0]
    lock = {"model": "optimized_ORB_RBF-SVR", "selection_split": "validation", "selection_order": cfg["selection_order"],
            "input_fingerprint": fingerprint, "parameters": params, "selected_trial_id": selected_id,
            "feature_screen_trial_id": selected["screen_trial_id"], "preprocessing": selected["preprocessing"],
            "ratio": selected["ratio"], "representation": selected["representation"], "feature_columns": columns,
            "feature_screen_validation_mrr": float(best_screen.validation_mrr), "validation_mrr": float(best_grid.validation_mrr),
            "validation_top1": float(best_grid.validation_top1), "validation_top5": float(best_grid.validation_top5),
            "training_pair_identity_hash": pair_hash, "train_context_candidates": int(cases.split.eq("train").sum()),
            "validation_context_candidates": int(cases.split.eq("validation").sum()),
            "season_calibration_fit": "training_images_only", "train_only_fit": True,
            "test_images_read": 0, "test_queries_scored": 0,
            "train_features_hash": _frame_hash(train), "validation_features_hash": _frame_hash(validation),
            "staged_selection": "image/representation by pilot SVR MRR, then 36-configuration SVR grid"}
    write_json(lock_path, lock)
    write_json(output/"selected_config.json", lock)
    train.to_csv(output/"training_pairs.csv", index=False)
    validation.to_csv(output/"validation_features.csv", index=False)
    write_json(output/"feature_schema.json", {"columns": columns, "representation": selected["representation"],
        "fills_fit_on_sampled_training_pairs": selected["fills"], "context_requires_complete_gallery": True,
        "season_statistics": _json_calibration(selected["calibration"]), "mask": "central D-intensity percentile band; not validated trunk segmentation or metric depth"})
    bundles, queries, rankings, metrics, comparison = {}, {}, {}, {}, []
    model_folder = output/"models"; model_folder.mkdir(exist_ok=True)
    for name, pipeline in [("LinearRegression", linear), ("RBF-SVR", svr)]:
        fields, ranked, per_query, values = score_pipeline(pipeline, train, validation, columns, cfg)
        np.testing.assert_allclose([fields["validation_mrr"], fields["validation_top1"]],
            [best_grid.validation_mrr, best_grid.validation_top1] if name == "RBF-SVR" else [fields["validation_mrr"], fields["validation_top1"]], rtol=0, atol=1e-12)
        bundle = {"pipeline": pipeline, "settings": cfg, "feature_columns": columns, "fills": selected["fills"],
                  "calibration": selected["calibration"], "preprocessing": selected["preprocessing"],
                  "ratio": selected["ratio"], "representation": selected["representation"], "selection": lock,
                  "inference_api": "tools.relative_features.match_tree_optimized; complete gallery required"}
        bundles[name], queries[name], rankings[name], metrics[name] = bundle, per_query, ranked, values
        model_file = model_folder/f"{name}.joblib"
        joblib.dump(bundle, model_file)
        reloaded = joblib.load(model_file)
        np.testing.assert_allclose(reloaded["pipeline"].predict(validation[columns]), pipeline.predict(validation[columns]), rtol=0, atol=0)
        ranked.to_csv(output/f"{name}_validation_rankings.csv", index=False)
        per_query.to_csv(output/f"{name}_query_metrics.csv", index=False)
        counts, hub = hubness(ranked)
        counts.to_csv(output/f"{name}_candidate_hubness.csv", index=False)
        metrics[name]["hubness"] = hub
        comparison.append({"model": "Optimized "+name, **{k: values["retrieval"][k] for k in ("queries", "gallery_size", "top1", "top5", "mrr")},
                           "max_top5_query_fraction": hub["max_top5_query_fraction"],
                           "median_fit_seconds": float(np.median(fit_times[name]))})
    for name, folder in [("Original LinearRegression", reference_dirs["LinearRegression"]),
                         ("Original RBF-SVR", reference_dirs["RBF-SVR"])]:
        values = json.loads((root/folder/"metrics.json").read_text())["validation"]["retrieval"]
        comparison.append({"model": name, **{k: values[k] for k in ("queries", "gallery_size", "top1", "top5", "mrr")}})
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(output/"validation_comparison.csv", index=False)
    write_json(output/"metrics.json", metrics)
    summary = {"settings": cfg, "input_fingerprint": fingerprint, "selection_reused": False,
        "selected_preprocessing": selected["preprocessing"], "ratio": selected["ratio"], "representation": selected["representation"],
        "parameters": params, "feature_count": len(columns), "feature_screen_configurations": len(screening)//2,
        "grid_trials": len(trials), "training_pairs": len(train), "training_pair_identity_hash": pair_hash,
        "training_context_gallery": int(cases.split.eq("train").sum()), "validation_context_gallery": int(cases.split.eq("validation").sum()),
        "support_vectors": len(svr.named_steps["regressor"].support_), "support_vector_fraction": len(svr.named_steps["regressor"].support_)/len(train),
        "fit_seconds_repetitions": fit_times, "historical_artifacts_unchanged": protected == protect_existing(root),
        "test_images_read": 0, "test_queries_scored": 0, "scope": "Repeated validation selection; development results only, no new independent accuracy claim"}
    if not summary["historical_artifacts_unchanged"]:
        raise ValueError("A historical method, label or result changed")
    write_json(output/"run_summary.json", summary)
    write_json(output/"historical_hashes.json", protected)
    result = {"summary": summary, "selection": lock, "screening": screening, "trials": trials, "comparison": comparison,
              "bundles": bundles, "queries": queries, "rankings": rankings, "metrics": metrics,
              "validation": validation, "cases": cases, "diagnostics": diagnostics, "output_dir": output}
    if render:
        render_results(result)
        render_mask_examples(root, cases, cfg, output)
    inference_checks(root, result)
    write_artifact_manifest(output)
    return result


def write_artifact_manifest(output):
    """Record hashes of reusable optimization artifacts after inference checks.

    Args:
        output (Path): Existing optimization result directory.

    Returns:
        None: artifact_manifest.json is written, excluding itself, export_summary.json and
        bytecode caches.

    Raises:
        OSError: An artifact cannot be hashed or the manifest cannot be written.
    """
    paths = [p for p in output.rglob("*") if p.is_file() and p.name not in {"artifact_manifest.json", "export_summary.json"}
             and "__pycache__" not in p.parts]
    write_json(output/"artifact_manifest.json", {p.relative_to(output).as_posix(): file_digest(p) for p in paths})


def load_run(root, settings, cases=None):
    """Verify and reload trusted fitted optimization artifacts without repeating selection.

    Args:
        root (Path): Project base for outputs and optional case manifest.
        settings (dict): output_dir for the saved experiment.
        cases (pandas.DataFrame | None): Optional development cases; None reloads and filters
            the preparation manifest.

    Returns:
        dict: Saved summary/selection/tables/metrics, trusted model bundles, development cases
        and output_dir.

    Raises:
        ValueError: An artifact hash differs from artifact_manifest.json.
        OSError: A saved artifact cannot be read.
    """
    output = root/settings["output_dir"]
    inventory = json.loads((output/"artifact_manifest.json").read_text())
    for name, digest in inventory.items():
        if file_digest(output/name) != digest:
            raise ValueError(f"Optimized artifact changed: {name}")
    names = ("LinearRegression", "RBF-SVR")
    if cases is None:
        cases = pd.read_csv(root/"outputs/data_preparation/split_manifest.csv", keep_default_na=False)
        cases = cases[cases.split.isin(["train", "validation"])].copy()
    return {"summary": json.loads((output/"run_summary.json").read_text()),
            "selection": json.loads((output/"selected_config.json").read_text()),
            "screening": pd.read_csv(output/"feature_screening.csv"), "trials": pd.read_csv(output/"svr_search.csv"),
            "comparison": pd.read_csv(output/"validation_comparison.csv"),
            "bundles": {n: joblib.load(output/f"models/{n}.joblib") for n in names},
            "queries": {n: pd.read_csv(output/f"{n}_query_metrics.csv") for n in names},
            "rankings": {n: pd.read_csv(output/f"{n}_validation_rankings.csv", float_precision="round_trip") for n in names},
            "metrics": json.loads((output/"metrics.json").read_text()),
            "validation": pd.read_csv(output/"validation_features.csv", float_precision="round_trip"),
            "cases": cases, "output_dir": output}


def inference_checks(root, result):
    """Verify complete-gallery image inference against saved validation scores.

    Args:
        root (Path): Base for validation images/caches.
        result (dict): Optimization bundles, cases, reference rankings and output_dir.

    Returns:
        list[dict]: Model/query/gallery-size/pass records, also written to
        inference_checks.json.

    Raises:
        AssertionError: Image-based scores differ from saved full-gallery scores.
        OSError: Images/caches/check records are inaccessible.
    """
    val = result["cases"][result["cases"].split.eq("validation")].sort_values("tree_id")
    query = val.iloc[0]
    gallery = val[["tree_id", "winter_rgb", "winter_depth"]].rename(columns={"tree_id": "candidate_id", "winter_rgb": "rgb", "winter_depth": "depth"})
    checks = []
    for name, bundle in result["bundles"].items():
        ranked = match_tree_optimized({"rgb": query.summer_rgb, "depth": query.summer_depth}, gallery, bundle, root)
        reference = result["rankings"][name]
        reference = reference[reference.query_id.eq(query.tree_id)].set_index("candidate_id")
        np.testing.assert_allclose(ranked.score, reference.loc[ranked.candidate_id].score, rtol=1e-9, atol=1e-9)
        checks.append({"model": name, "query": query.tree_id, "candidates": len(gallery), "passed": True})
    write_json(result["output_dir"]/"inference_checks.json", checks)
    return checks


def export_optimization(root, result, settings=None):
    """Export the separate validation models and numeric replay evidence.

    Original images, foreground photographs and large caches are excluded. This
    export does not rewrite the historical final-test export.

    Args:
        root (str | Path): Project source/configuration directory.
        result (dict): Completed optimization result with fitted bundles and output_dir.
        settings (dict | None): Export settings; None uses result summary settings.

    Returns:
        dict: export_dir, manifest_sha256 and checksum/model-replay verification results.

    Raises:
        ValueError: Export checksum/feature/gallery verification fails.
        AssertionError: Exported scores or ranks fail numeric replay.
        OSError: Source/result/model copies or summaries cannot be written.
    """
    import nbformat
    root = Path(root)
    cfg = settings or result["summary"]["settings"]
    output = root/cfg["export_dir"]; output.mkdir(parents=True, exist_ok=True)
    known = set()
    def copy(source, name):
        path = output/name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path); known.add(name)
    for folder in ("src", "tools"):
        for path in (root/folder).glob("*.py"):
            copy(path, f"{folder}/{path.name}")
    for path in (root/"config").rglob("*"):
        if path.is_file() and path.name != "local.json":
            copy(path, path.relative_to(root).as_posix())
    for path in (root/"doc").glob("*.md"):
        copy(path, f"doc/{path.name}")
    for name in ("README.md", "LICENSE", ".gitattributes", ".gitignore"):
        copy(root/name, name)
    for path in result["output_dir"].glob("*"):
        if path.is_file() and path.suffix in {".json", ".csv"} and path.name not in {"artifact_manifest.json", "export_summary.json"}:
            copy(path, f"results/{path.name}")
    for path in (result["output_dir"]/"figures").glob("*.png"):
        if path.name != "foreground_examples.png":
            copy(path, f"results/figures/{path.name}")
    replay = {"models": {}, "metric_settings": {k: cfg[k] for k in ("score_tie_atol", "score_tie_rtol")}}
    for name, bundle in result["bundles"].items():
        file = f"models/{name}.joblib"
        copy(result["output_dir"]/file, file)
        replay["models"][name] = {"model_file": file, "feature_file": "results/validation_features.csv",
                                   "ranking_file": f"results/{name}_validation_rankings.csv", "feature_columns": bundle["feature_columns"]}
    write_json(output/"replay.json", replay); known.add("replay.json")
    notebook = nbformat.read(root/"apple_tree_matching.ipynb", as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []; cell.execution_count = None
    nbformat.write(notebook, output/"apple_tree_matching.ipynb"); known.add("apple_tree_matching.ipynb")
    text = """# Optimized validation experiment

Install the pinned core config/requirements.txt with Python 3.12.
From this copied bundle run `python -m src.reproduce --verify .` to checksum and
replay both complete validation rankings without source images or CNN weights.
The fitted pipelines are in models/. For image inference use
`tools.relative_features.match_tree_optimized(query, gallery, bundle, root)`;
gallery must contain all intended candidates, with candidate_id/rgb/depth paths.
Context features depend on this complete gallery, never a sampled 11-pair table.

The original data and large feature caches are not packaged. D masking is a
central intensity-rank heuristic, not validated trunk segmentation or metric
depth. Calibration and imputation fit training images/pairs only. This model was
selected repeatedly on validation MRR; these are development scores. Historical
test results are not recomputed, and no new independent test accuracy is claimed.

For a complete image-backed rerun, supply original images and the original
stages 1-6 development inputs, then use the main notebook's optimization-only
runner: `python tools/run_notebook.py --optimization-only --in-place`.
The primary single notebook also retains the previous experiments. This bundle
replays the new validation models; the separate original export replays the
historical test. Changed inputs require separate output/selection paths.
"""
    (output/"RUN.md").write_text(text, encoding="utf-8"); known.add("RUN.md")
    write_json(output/"manifest.json", {"files": {n: {"sha256": file_digest(output/n), "bytes": (output/n).stat().st_size} for n in sorted(known)}})
    checked = verify_export(output)
    small = root/cfg["small_results_dir"]; small.mkdir(parents=True, exist_ok=True)
    for name in ("validation_comparison.csv", "feature_screening.csv", "svr_search.csv", "run_summary.json", "metrics.json", "selected_config.json", "inference_checks.json"):
        shutil.copyfile(result["output_dir"]/name, small/name)
    for path in (result["output_dir"]/"figures").glob("*.png"):
        if path.name != "foreground_examples.png":
            shutil.copyfile(path, small/path.name)
    for path in (root/cfg["diagnostics_dir"]).glob("*"):
        if path.is_file():
            target = small/"diagnostics"; target.mkdir(exist_ok=True)
            shutil.copyfile(path, target/path.name)
    summary = {"export_dir": str(output), "manifest_sha256": file_digest(output/"manifest.json"), **checked}
    write_json(result["output_dir"]/"export_summary.json", summary)
    return summary


def main():
    # CLI runs must not leave Tk objects for garbage collection in feature workers.
    """Run --diagnostics-only or the separate optimization/export CLI with headless plotting.

    Returns:
        None: Results are written and diagnostic/comparison summaries printed.

    Raises:
        SystemExit: Arguments are invalid or --help is requested.
        ValueError: Development inputs, locks or artifacts fail validation.
        OSError: Required inputs or outputs are inaccessible.
    """
    plt.switch_backend("Agg")
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-only", action="store_true")
    args = parser.parse_args()
    if args.diagnostics_only:
        result = run_diagnostics(ROOT)
        print(json.dumps(result["summary"], indent=2))
    else:
        result = run_optimization(ROOT)
        print(result["comparison"].to_string(index=False))
        print(export_optimization(ROOT, result))


if __name__ == "__main__":
    main()
