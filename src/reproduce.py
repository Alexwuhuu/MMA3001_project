"""Stage 8: portable source/model/results export and checksum/prediction replay.

No original images, pretrained checkpoint, large development matrices or caches
are packaged. Image inference uses the original seasonal filenames supplied by
the user; the numeric test snapshot also permits an image-free replay.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys

import joblib
import numpy as np
import pandas as pd

from .baseline import match_tree, retrieval_results
from .utils import file_digest, write_json


def verify_export(folder, replay=True):
    """Verify packaged file hashes and optionally replay trusted model scores/ranks.

    Checksums establish file integrity rather than model trust. Use replay=True only
    for bundles from a trusted source because replay loads joblib artifacts.

    Args:
        folder (str | Path): Export containing manifest.json and, for replay, replay.json.
        replay (bool): Whether to load fitted models and reproduce every stored score and rank.

    Returns:
        dict: checksummed_files, per-model model_replays, and original-image/large-cache
        exclusion declarations.

    Raises:
        ValueError: Manifest paths/hashes, feature order or query/gallery identities differ.
        AssertionError: Scores or display/tie rank columns do not reproduce.
        OSError: A manifest, feature table or model cannot be read.
    """
    folder = Path(folder).resolve()
    manifest = json.loads((folder/"manifest.json").read_text())
    for name, record in manifest["files"].items():
        path = (folder/name).resolve()
        if not path.is_relative_to(folder) or not path.is_file():
            raise ValueError(f"Missing or unsafe export path: {name}")
        if path.stat().st_size != record["bytes"] or file_digest(path) != record["sha256"]:
            raise ValueError(f"Export checksum mismatch: {name}")
    checked = []
    if replay:
        config = json.loads((folder/"replay.json").read_text())
        for name, info in config["models"].items():
            # Integrity checks precede loading; the caller must still trust the model source.
            model = joblib.load(folder/info["model_file"])
            features = pd.read_csv(folder/info["feature_file"], float_precision="round_trip")
            reference = pd.read_csv(folder/info["ranking_file"], float_precision="round_trip")
            if model["feature_columns"] != info["feature_columns"]:
                raise ValueError("Exported feature order changed")
            scored = features[["split", "query_id", "candidate_id"]].copy()
            scored["score"] = model["pipeline"].predict(features[info["feature_columns"]])
            ranked, _, _ = retrieval_results(scored, config["metric_settings"])
            keys = ["query_id", "candidate_id"]
            ranked, reference = ranked.sort_values(keys), reference.sort_values(keys)
            if not np.array_equal(ranked[keys].to_numpy(), reference[keys].to_numpy()):
                raise ValueError("Exported query/gallery changed")
            np.testing.assert_allclose(ranked.score, reference.score, rtol=1e-10, atol=1e-10)
            np.testing.assert_array_equal(ranked[["rank_min", "rank_max", "display_rank"]],
                                          reference[["rank_min", "rank_max", "display_rank"]])
            checked.append({"model": name, "pairs_replayed": len(ranked), "scores_and_ranks_reproduced": True})
    return {"checksummed_files": len(manifest["files"]), "model_replays": checked,
            "contains_original_images": False, "contains_large_caches": False}


def export_run(root, final, preparation, baseline, svr, settings, notebook_path=None):
    """Package source, clean notebook, fitted models and numeric replay evidence.

    Original images, checkpoint weights and large caches are excluded. Saved image
    inference and numeric replay are verified before the export summary is written.
    A fresh export_dir is required when the previous file inventory differs.

    Args:
        root (str | Path): Project base for source/configuration and saved outputs.
        final (dict): Completed run_final result with fitted bundles and test features.
        preparation (dict): Preparation result containing cases for the portable manifest.
        baseline (dict): Original baseline bundle with extraction/tie settings.
        svr (dict): Original selected SVR result retained by the stage-8 interface.
        settings (dict): export_dir and small_results_dir destinations.
        notebook_path (str | Path | None): Source notebook to clean; None uses the main
            notebook.

    Returns:
        dict: Export paths, manifest hash, file/byte counts, score/rank replay and saved-image
        inference checks. Small aggregate results are copied for version control.

    Raises:
        ValueError: Stale export inventory or replay integrity checks fail.
        AssertionError: Saved preprocessing, image inference or replay differs from its
            reference.
        OSError: Sources/models/results cannot be copied or written.
    """
    root = Path(root).resolve()
    output = root/settings["export_dir"]
    output.mkdir(parents=True, exist_ok=True)
    # Never leave obsolete files in an existing export. Reuse requires the same
    # inventory; accidental leftovers are caught by the manifest inventory check.
    known = set()

    def copy(source, destination):
        target = output/destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        known.add(target.relative_to(output).as_posix())
        return target

    def write(name, value):
        write_json(output/name, value)
        known.add(name)

    for path in sorted((root/"src").glob("*.py")):
        copy(path, "src/"+path.name)
    for path in sorted((root/"tools").glob("*.py")):
        copy(path, "tools/"+path.name)
    for path in sorted((root/"config").rglob("*")):
        if path.is_file() and path.suffix in {".json", ".csv", ".txt", ".in"} and path.name != "local.json":
            copy(path, path.relative_to(root).as_posix())
    for name in ("README.md", "LICENSE", ".gitignore", ".gitattributes", "pytest.ini"):
        if (root/name).is_file():
            copy(root/name, name)
    for path in sorted((root/"tests").glob("*.py")):
        copy(path, "tests/"+path.name)
    for path in sorted((root/"doc").glob("*.md")):
        copy(path, "doc/"+path.name)
    # Copy a clean source notebook so exports never contain stale pre-export
    # status/output cells. The workspace notebook retains the executed results.
    import nbformat
    notebook = nbformat.read(notebook_path or root/"apple_tree_matching.ipynb", as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    nbformat.write(notebook, output/"apple_tree_matching.ipynb")
    known.add("apple_tree_matching.ipynb")
    copy(root/"outputs/data_preparation/split_manifest.csv" if not isinstance(preparation.get("cases"), pd.DataFrame)
         else _save_cases(preparation["cases"], root, final["output_dir"]), "results/split_manifest.csv")
    model_map = {}
    folder = final["output_dir"]
    # Small tables and aggregate figures are shared; example photographs and
    # large development pair tables remain in local outputs only.
    for path in sorted(folder.glob("*")):
        if path.is_file() and path.name != "export_summary.json" and path.suffix in {".csv", ".json", ".txt", ".prof"}:
            copy(path, "results/"+path.name)
    aggregate_figures = ["test_comparison.png", "development_sensitivity.png", "parameter_error_sensitivity.png", "performance.png", "test_index.png"]
    for name in aggregate_figures:
        if (folder/"figures"/name).is_file():
            copy(folder/"figures"/name, "results/figures/"+name)
    for name, bundle in final["bundles"].items():
        slug = {"LinearRegression": "linear", "RBF-SVR": "svr",
                "ResNet18+LinearRegression": "resnet_linear", "ResNet18+RBF-SVR": "resnet_svr"}[name]
        model_file = f"models/{slug}.joblib"
        (output/"models").mkdir(exist_ok=True)
        joblib.dump(bundle, output/model_file)
        known.add(model_file)
        # Separate preprocessing allows replay to compare it with the full pipeline.
        joblib.dump(bundle["pipeline"][:-1], output/f"models/{slug}_preprocessor.joblib")
        known.add(f"models/{slug}_preprocessor.joblib")
        if "projection" in bundle:
            joblib.dump(bundle["projection"], output/f"models/{slug}_pca.joblib")
            known.add(f"models/{slug}_pca.joblib")
        model_map[name] = {"model_file": model_file, "feature_columns": bundle["feature_columns"],
                           "feature_file": "results/"+("supplementary_test_pair_features.csv" if name.startswith("ResNet") else "test_pair_features.csv"),
                           "ranking_file": f"results/{name}_test_rankings.csv"}
    metric_settings = {k: baseline["bundle"]["settings"][k] for k in ("score_tie_atol", "score_tie_rtol")}
    write("replay.json", {"models": model_map, "metric_settings": metric_settings,
                           "scores": "continuous regression values, not calibrated probabilities"})
    distributions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata.get("Name")}
    write("environment.json", {"python": sys.version, "platform": platform.platform(),
                                "packages": dict(sorted(distributions.items(), key=lambda x: x[0].lower())),
                                "pinned_core_file": "config/requirements.txt",
                                "optional_torch_file": "config/requirements-resnet.txt"})
    write("feature_schema.json", {name: {"ordered_columns": b["feature_columns"],
        "preprocessing": b["settings"], "imputer_fit": "training_pairs_only", "scaler_fit": "training_pairs_only",
        "metadata_features": [], "D_units": "uint8 intensity; not metric depth"} for name, b in final["bundles"].items()})
    instructions = """# Reproduce this exported experiment

Use Python 3.12 and install `config/requirements.txt`. For the supplementary
ResNet cells also install `config/requirements-resnet.txt` using the CPU or CUDA
index described in README.md, and run `python tools/download_resnet_weights.py`.
The official checkpoint is intentionally not included.

1. For an image-free replay of the final scores and complete-gallery rankings,
   run `python -m src.reproduce --verify .` from this directory. Only load these
   trusted local joblib files. Replay also checks every packaged SHA-256.
2. For a complete new kernel run, provide the original data at `data/TreeID/TreeID`
   or set `MMA3001_DATA_ROOT` / ignored `config/local.json`, open the main notebook
   and select Restart & Run All. Seasonal source filenames must be preserved.
   Use the recorded versions for the recorded selection fingerprints. A different
   software/device/feature configuration requires separate selection/output files.
3. The core ORB workflow works without PyTorch: set `MMA3001_SKIP_RESNET=1`.
   This is a separate run from this four-model export, so use a separate final
   lock/output (the notebook assigns these automatically when skipping ResNet).
4. `models/linear.joblib` and `models/svr.joblib` contain the full fitted pipeline,
   feature order and extraction configuration. Explicit imputer/scaler bundles
   are saved alongside them. `src.baseline.match_tree(query, gallery, bundle, root)`
   accepts RGB/D paths and ranks all supplied candidates; inspect `.head(5)`.
5. Main test metrics describe independently confirmed queries only. All sampled
   queries and the complete winter gallery are retained. Rejected/unresolved
   filename pairs are not assigned verified accuracy. Numeric regression errors
   for other candidates assume distinct filename IDs. No model is reselected on
   test results. ResNet results are supplementary to the ORB/HSV/D comparison.

The bundle excludes original images, example photographs, pretrained weights,
large development matrices and caches. Aggregate figures and numeric test
features/rankings are included. `manifest.json` excludes itself; its checksum
is saved in the workspace export_summary.json. No project report is generated.
"""
    (output/"RUN.md").write_text(instructions, encoding="utf-8")
    known.add("RUN.md")
    # Demonstrate saved image inference on the first validation ID, before export.
    # Scores must agree with the already-fitted main models across the gallery.
    from .main import portable_frame
    cases = portable_frame(preparation["cases"], root)
    val = cases[cases.split.eq("validation")].sort_values("tree_id")
    query = val.iloc[0]
    gallery = val[["tree_id", "winter_rgb", "winter_depth"]].rename(
        columns={"tree_id": "candidate_id", "winter_rgb": "rgb", "winter_depth": "depth"})
    inference_checks = []
    for name, info in model_map.items():
        if name.startswith("ResNet"):
            continue
        saved = joblib.load(output/info["model_file"])
        explicit_preprocessor = joblib.load(output/f"models/{'linear' if name == 'LinearRegression' else 'svr'}_preprocessor.joblib")
        feature_frame = final["features"][name][saved["feature_columns"]]
        np.testing.assert_allclose(explicit_preprocessor.transform(feature_frame),
                                   saved["pipeline"][:-1].transform(feature_frame), rtol=0, atol=0)
        pred = match_tree({"rgb": query.summer_rgb, "depth": query.summer_depth}, gallery, saved, root).sort_values("candidate_id")
        original = match_tree({"rgb": query.summer_rgb, "depth": query.summer_depth}, gallery, final["bundles"][name], root).sort_values("candidate_id")
        np.testing.assert_allclose(pred.score, original.score, rtol=1e-10, atol=1e-10)
        inference_checks.append({"model": name, "query": query.tree_id, "full_gallery": len(gallery), "passed": True})
    write("image_inference_checks.json", inference_checks)
    files = {p.relative_to(output).as_posix() for p in output.rglob("*")
             if p.is_file() and p.name != "manifest.json"
             and not {"__pycache__", ".ipynb_checkpoints"}.intersection(p.relative_to(output).parts)}
    if files != known:
        raise ValueError(f"Unexpected existing export files; use a fresh export_dir: {sorted(files-known)}")
    write_json(output/"manifest.json", {"format_version": 1,
        "files": {name: {"sha256": file_digest(output/name), "bytes": (output/name).stat().st_size} for name in sorted(known)}})
    verified = verify_export(output)
    small = root/settings["small_results_dir"]
    small.mkdir(parents=True, exist_ok=True)
    for name in ("test_comparison.csv", "test_metrics.json", "development_comparison.csv", "paired_model_differences.csv",
                 "svr_parameter_sensitivity.csv", "performance.csv", "memory_and_models.csv", "performance_scope.json", "run_summary.json", "experiment_lock.json"):
        shutil.copyfile(folder/name, small/name)
    for name in aggregate_figures:
        if (folder/"figures"/name).is_file():
            shutil.copyfile(folder/"figures"/name, small/name)
    summary = {"output_dir": str(output), "small_results_dir": str(small),
               "manifest_sha256": file_digest(output/"manifest.json"), **verified,
               "total_export_bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
               "saved_image_inference_checks": inference_checks,
               "exact_environment_saved": True, "original_data_packaged": False}
    write_json(folder/"export_summary.json", summary)
    write_json(small/"export_summary.json", {k: v for k, v in summary.items() if k not in {"output_dir", "small_results_dir"}})
    return summary


def _save_cases(cases, root, folder):
    """Write a portable case manifest for inclusion in an export.

    Args:
        cases (pandas.DataFrame): Preparation cases with seasonal paths.
        root (Path): Base for portable path conversion.
        folder (Path): Existing destination directory.

    Returns:
        Path: Written split_manifest.csv path.

    Raises:
        OSError: The CSV cannot be written.
    """
    from .main import portable_frame
    path = folder/"split_manifest.csv"
    portable_frame(cases, root).to_csv(path, index=False)
    return path


def main():
    """Verify an export from --verify, optionally using --checksums-only.

    Returns:
        None: Verification results are printed as JSON.

    Raises:
        SystemExit: Argument parsing fails or --help is requested.
        ValueError: Export integrity validation fails.
        AssertionError: Numeric replay differs from its reference.
        OSError: Export files cannot be read.
    """
    parser = argparse.ArgumentParser(description="Verify a trusted local stage-8 export and replay all scores/ranks")
    parser.add_argument("--verify", type=Path, required=True)
    parser.add_argument("--checksums-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify_export(args.verify, replay=not args.checksums_only), indent=2))


if __name__ == "__main__":
    main()
