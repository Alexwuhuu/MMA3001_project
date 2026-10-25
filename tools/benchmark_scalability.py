"""Measure development-gallery scaling without fitting or accessing test images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.appearance import extract_appearance, pair_appearance
from src.baseline import rank_candidates
from src.features import extract_image_features, extract_pair_features, preprocess_rgb, ImageFeatures
from src.utils import file_digest, read_depth, read_rgb, write_json

MODEL_PATHS = {
    "LinearRegression": "outputs/baseline_linear/models/linear_regression.joblib",
    "RBF-SVR": "outputs/svr_main/models/rbf_svr.joblib",
}
STAGES = ["query_image_seconds", "pair_features_seconds", "prediction_seconds",
          "ranking_seconds", "online_seconds", "first_query_seconds"]


def select_workload(cases, settings):
    """Freeze row-spread queries and nested, randomly ordered development galleries.

    Args:
        cases (pandas.DataFrame): Unique train/validation identities and seasonal paths.
        settings (dict): Seed, queries_per_split and split-keyed gallery_sizes.

    Returns:
        dict: Split-keyed query IDs and ordered gallery IDs; prefixes define sizes.

    Raises:
        ValueError: Test/duplicate identities, overlapping rows or invalid sizes/counts occur.
    """
    if (cases.empty or not set(cases.split).issubset({"train", "validation"})
            or cases.tree_id.duplicated().any()):
        raise ValueError("Need unique development identities, never test cases")
    if set(cases.loc[cases.split.eq("train"), "row"]) & set(cases.loc[cases.split.eq("validation"), "row"]):
        raise ValueError("Development orchard rows overlap")
    rng = np.random.default_rng(settings["seed"])
    result = {}
    for split, sizes in settings["gallery_sizes"].items():
        frame = cases[cases.split.eq(split)].sort_values("tree_id")
        count = settings["queries_per_split"]
        if (count < 2 or count > len(frame) or not sizes or sizes != sorted(set(sizes))
                or any(not isinstance(n, int) or n < 1 or n > len(frame) for n in sizes)):
            raise ValueError("Invalid query count or nested gallery sizes")
        rows = [list(rng.permutation(g.tree_id.tolist())) for _, g in frame.groupby("row", sort=True)]
        queries = []
        while len(queries) < count:
            for row in rows:
                if row and len(queries) < count:
                    queries.append(row.pop())
        result[split] = {"query_ids": queries, "gallery_ids": rng.permutation(frame.tree_id).tolist(),
                         "gallery_sizes": sizes}
    return result


def image_summary(root, row, season, settings):
    """Extract ORB/HSV/D directly, excluding content hashing and disk-cache writes.

    This mirrors production transforms and is checked against production summaries
    before timing. RGB decoding is shared by ORB and HSV within this benchmark.

    Args:
        root (Path): Base for seasonal image paths.
        row (pandas.Series): Seasonal RGB and optional depth path fields.
        season (str): summer or winter.
        settings (dict): Original fitted model extraction and histogram settings.

    Returns:
        tuple: ORB ImageFeatures and appearance dictionary in production units.

    Raises:
        ValueError: RGB cannot be read or preprocessing settings are invalid.
        cv2.error: An image cannot be processed.
    """
    original = read_rgb(root / row[f"{season}_rgb"])
    rgb = preprocess_rgb(original, settings["image_size"], settings["crop"])
    points, descriptors = cv2.ORB_create(nfeatures=settings["orb_nfeatures"], WTA_K=2).detectAndCompute(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None)
    xy = np.asarray([p.pt for p in points], dtype=np.float32).reshape(-1, 2)
    orb = ImageFeatures("scalability", xy, descriptors if descriptors is not None else np.empty((0, 32), np.uint8))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [settings["hue_bins"], settings["saturation_bins"]],
                       [0, 180, 0, 256]).ravel().astype(float)
    hist /= hist.sum()
    path = row[f"{season}_depth"]
    depth, _ = read_depth(str(root/path) if path else "", original.shape[:2])
    app = {"hsv": hist, "depth_hist": np.full(settings["depth_bins"], np.nan),
           "depth_mean": np.nan, "depth_std": np.nan, "depth_zero_fraction": np.nan,
           "missing_depth": float(depth is None), "depth_no_nonzero": 0.}
    if depth is not None:
        if settings["crop"] == "center":
            h, w = depth.shape
            side = min(h, w)
            depth = depth[(h-side)//2:(h-side)//2+side, (w-side)//2:(w-side)//2+side]
        depth = cv2.resize(depth, (settings["image_size"], settings["image_size"]), interpolation=cv2.INTER_NEAREST)
        valid = depth[depth != 0].astype(float)
        app.update(depth_zero_fraction=float(np.mean(depth == 0)), depth_no_nonzero=float(not len(valid)))
        if len(valid):
            dh, _ = np.histogram(valid, bins=settings["depth_bins"], range=(1, 256))
            app.update(depth_hist=dh/dh.sum(), depth_mean=float(valid.mean()), depth_std=float(valid.std()))
    return orb, app


def summarize(samples, keys, columns):
    """Report dispersion separately across repeats and across fixed query identities.

    Args:
        samples (pandas.DataFrame): Individual timing observations in seconds.
        keys (list[str]): Grouping columns, including query_id for within-query repeats.
        columns (list[str]): Numeric timing or array-storage fields to summarize.

    Returns:
        pandas.DataFrame: Long-form count, median, quartiles, min, max and sample std.
    """
    records = []
    for values, group in samples.groupby(keys, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        for column in columns:
            x = group[column].to_numpy(dtype=float)
            records.append({**dict(zip(keys, values)), "stage": column, "count": len(x),
                            "median": float(np.median(x)), "q25": float(np.quantile(x, .25)),
                            "q75": float(np.quantile(x, .75)), "min": float(x.min()),
                            "max": float(x.max()), "std": float(x.std(ddof=1)) if len(x)>1 else 0.})
    return pd.DataFrame(records)


def run_benchmark(root, cases, bundles, settings):
    """Measure cache construction and staged queries with fixed pretrained models.

    Each repetition rebuilds each gallery from images once, then reuses it for all
    fixed queries/models. Stage medians are not added: total summaries use measured
    per-observation sums. No accuracy metrics or model selection are performed.

    Args:
        root (str | Path): Project containing trusted models and development images.
        cases (pandas.DataFrame): Development-only manifest.
        bundles (dict): Model-keyed fitted pipelines with matching feature/settings schemas.
        settings (dict): Frozen workload, repetition and output settings.

    Returns:
        dict: Query timings, gallery-cache timings, summaries and protocol metadata.

    Raises:
        ValueError: Workload, repetitions or model schemas differ or include test cases.
        AssertionError: Repeated scores/ranks or benchmark/production extraction disagree.
        OSError: Images or reference-check caches cannot be read or written.
    """
    root = Path(root)
    workload = select_workload(cases, settings)
    repeats = settings["timing_repeats"]
    if repeats < 2 or not bundles:
        raise ValueError("Need models and at least two timing repetitions")
    first = next(iter(bundles.values()))
    cfg, columns = first["settings"], first["feature_columns"]
    extraction_keys = ["image_size", "crop", "orb_nfeatures", "ratio_threshold", "ransac_threshold_px",
                       "ransac_max_iters", "ransac_confidence", "seed", "hue_bins", "saturation_bins", "depth_bins",
                       "score_tie_atol", "score_tie_rtol"]
    for bundle in bundles.values():
        if (bundle["feature_columns"] != columns or
                any(bundle["settings"][k] != cfg[k] for k in extraction_keys)):
            raise ValueError("Models must share original feature/extraction settings")
    index = cases.set_index("tree_id")
    # A complete untimed pass warms image I/O and validates every accessed image.
    cv2.setNumThreads(1)
    reference_cache = root/settings["output_dir"]/"reference_check"
    image_paths = set()
    for split, spec in workload.items():
        for season, ids in (("summer", spec["query_ids"]), ("winter", spec["gallery_ids"][:max(spec["gallery_sizes"])])):
            for identity in ids:
                row = index.loc[identity]
                cv2.setRNGSeed(cfg["seed"])
                orb, app = image_summary(root, row, season, cfg)
                rgb = root/row[f"{season}_rgb"]
                depth = root/row[f"{season}_depth"] if row[f"{season}_depth"] else ""
                expected_orb = extract_image_features(rgb, cfg, reference_cache/"orb")
                expected_app = extract_appearance(rgb, depth, cfg, reference_cache/"appearance")
                np.testing.assert_array_equal(orb.xy, expected_orb.xy)
                np.testing.assert_array_equal(orb.descriptors, expected_orb.descriptors)
                for key in app:
                    np.testing.assert_allclose(app[key], expected_app[key], rtol=1e-12, atol=1e-12, equal_nan=True)
                image_paths.add(rgb)
                if depth and depth.is_file():
                    image_paths.add(depth)
        print(f"Validated/warmed {split} workload", flush=True)
    records, caches, references = [], [], {}
    rng = np.random.default_rng(settings["seed"])
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        jobs = [(s, n) for s, spec in workload.items() for n in spec["gallery_sizes"]]
        for repeat in range(repeats):
            for job_index in rng.permutation(len(jobs)):
                split, size = jobs[job_index]
                spec = workload[split]
                ids = spec["gallery_ids"][:size]
                start = perf_counter()
                gallery = [image_summary(root, index.loc[i], "winter", cfg) for i in ids]
                cache_seconds = perf_counter()-start
                array_bytes = sum(o.xy.nbytes+o.descriptors.nbytes+sum(np.asarray(v).nbytes for v in a.values())
                                  for o, a in gallery)
                caches.append({"split": split, "gallery_size": size, "repeat": repeat,
                               "cache_build_seconds": cache_seconds, "cache_array_bytes": array_bytes})
                for query_index in rng.permutation(len(spec["query_ids"])):
                    query_id = spec["query_ids"][query_index]
                    start = perf_counter()
                    query_orb, query_app = image_summary(root, index.loc[query_id], "summer", cfg)
                    query_seconds = perf_counter()-start
                    # Reset per query so preceding job order cannot change RANSAC draws.
                    cv2.setRNGSeed(cfg["seed"])
                    start = perf_counter()
                    frame = pd.DataFrame([{**extract_pair_features(query_orb, o, cfg),
                                           **pair_appearance(query_app, a)} for o, a in gallery])[columns]
                    pair_seconds = perf_counter()-start
                    names = list(bundles)
                    for model_index in rng.permutation(len(names)):
                        name = names[model_index]
                        start = perf_counter()
                        scores = bundles[name]["pipeline"].predict(frame)
                        prediction_seconds = perf_counter()-start
                        start = perf_counter()
                        ranked = rank_candidates(pd.DataFrame({"candidate_id": ids, "score": scores}),
                                                 cfg["score_tie_atol"], cfg["score_tie_rtol"])
                        ranking_seconds = perf_counter()-start
                        key = (split, size, query_id, name)
                        if key in references:
                            np.testing.assert_allclose(scores, references[key][0], rtol=1e-10, atol=1e-10)
                            np.testing.assert_array_equal(ranked[["candidate_id", "rank_min", "rank_max"]], references[key][1])
                        else:
                            references[key] = (scores.copy(), ranked[["candidate_id", "rank_min", "rank_max"]].to_numpy())
                        online = query_seconds+pair_seconds+prediction_seconds+ranking_seconds
                        records.append({"split": split, "model": name, "gallery_size": size,
                                        "query_id": query_id, "repeat": repeat,
                                        "query_image_seconds": query_seconds, "pair_features_seconds": pair_seconds,
                                        "prediction_seconds": prediction_seconds, "ranking_seconds": ranking_seconds,
                                        "online_seconds": online, "first_query_seconds": cache_seconds+online})
                print(f"Repeat {repeat+1}/{repeats}: {split}, gallery={size}", flush=True)
    raw, cache = pd.DataFrame(records), pd.DataFrame(caches)
    per_query = summarize(raw, ["split", "model", "gallery_size", "query_id"], STAGES)
    # Across-query IQR describes heterogeneous queries, not an independent-sample CI.
    summary = summarize(raw, ["split", "model", "gallery_size"], STAGES)
    cache_summary = summarize(cache, ["split", "gallery_size"], ["cache_build_seconds", "cache_array_bytes"])
    return {"timings": raw, "cache_timings": cache, "per_query_summary": per_query,
            "summary": summary, "cache_summary": cache_summary,
            "protocol": {"workload": workload, "settings": settings,
                         "extraction_settings": {k: cfg[k] for k in extraction_keys},
                         "image_sha256": {str(p.relative_to(root)): file_digest(p) for p in sorted(image_paths)},
                         "environment": {"python": sys.version, "platform": platform.platform(),
                                         "processor": platform.processor(), "opencv": cv2.__version__,
                                         "numpy": np.__version__, "pandas": pd.__version__, "threadpools": pools},
                         "checks": {"all_accessed_images_match_production": True,
                                    "repeated_scores_and_ranks_agree": True},
                         "scope": ["Development runtime only; no fitting, test images or retrieval accuracy.",
                                   "Nested seeded gallery prefixes; counterpart need not be present.",
                                   "OpenCV and BLAS single-threaded; shuffled sizes, queries and model order each repeat.",
                                   "Untimed production equivalence pass warms images; OS cache is uncontrolled, not cold disk.",
                                   "Cache build includes candidate RGB/D decode, crop, ORB, HSV/D and in-memory storage.",
                                   "Query image stage includes RGB/D decode and ORB/HSV/D; pair stage includes matching, RANSAC and DataFrame assembly.",
                                   "Prediction includes fitted imputation/scaling; ranking includes output-frame construction and tie handling.",
                                   "Online is sum of four measured stages; first_query adds that repetition's gallery build once.",
                                   "Image/pair extraction is shared between models; it represents either model used alone, not simultaneous throughput.",
                                   "Hashing, disk-cache writes, model loading, validation assertions and CSV I/O are outside timers.",
                                   "Cache array bytes exclude Python overhead and are not peak RSS.",
                                   "Summary quartiles pool fixed queries/repeats; per-query table isolates repeat variation. Neither is a confidence interval.",
                                   "Fixed trained models and observed gallery sizes do not measure training scalability or extrapolate to larger orchards."]}}


def save_results(result, root, settings):
    """Save raw timings, aggregate evidence and a stage/scaling figure.

    Args:
        result (dict): Tables and metadata returned by run_benchmark.
        root (Path): Project base.
        settings (dict): output_dir and small_results_dir.

    Returns:
        None: CSV, JSON and PNG artifacts are written to both destinations.

    Raises:
        OSError: An output cannot be written.
    """
    folders = [root/settings["output_dir"], root/settings["small_results_dir"]]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        for key in ["timings", "cache_timings", "per_query_summary", "summary", "cache_summary"]:
            result[key].to_csv(folder/f"{key}.csv", index=False)
        write_json(folder/"protocol.json", result["protocol"])
    fig, axes = plt.subplots(len(settings["gallery_sizes"]), 3,
                             figsize=(15, 4*len(settings["gallery_sizes"])), squeeze=False)
    for row, split in enumerate(settings["gallery_sizes"]):
        for col, stage in enumerate(["online_seconds", "prediction_seconds", "cache_build_seconds"]):
            table = result["cache_summary"] if col == 2 else result["summary"]
            table = table[(table.split == split) & (table.stage == stage)]
            groups = [("Gallery build", table)] if col == 2 else table.groupby("model")
            for label, group in groups:
                group = group.sort_values("gallery_size")
                axes[row, col].plot(group.gallery_size, group["median"], "o-", label=label)
                axes[row, col].fill_between(group.gallery_size, group.q25, group.q75, alpha=.2)
            axes[row, col].set(xlabel="Gallery candidates", ylabel="Seconds", title=f"{split}: {stage.removesuffix('_seconds')}")
            axes[row, col].legend(fontsize=8)
    fig.suptitle("Fixed development queries; median and IQR (not confidence intervals)")
    fig.tight_layout()
    for folder in folders:
        fig.savefig(folder/"scaling.png", dpi=140)
    plt.close(fig)


def main():
    """Run the separate, image-backed development scalability benchmark.

    Returns:
        None: Results and provenance are saved under outputs/scalability and res/scalability.

    Raises:
        OSError: Configuration, trusted models, images or output paths are inaccessible.
        ValueError: Saved input schemas or workload settings are invalid.
        AssertionError: Extraction equivalence or deterministic replay fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"config/scalability.json")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = ROOT/"outputs/data_preparation/split_manifest.csv"
    cases = pd.read_csv(manifest, keep_default_na=False)
    cases = cases[cases.split.isin(["train", "validation"])].copy()
    bundles = {name: joblib.load(ROOT/path) for name, path in MODEL_PATHS.items()}
    result = run_benchmark(ROOT, cases, bundles, settings)
    result["protocol"]["input_sha256"] = {str(args.config.relative_to(ROOT)): file_digest(args.config),
                                            str(manifest.relative_to(ROOT)): file_digest(manifest),
                                            **{p: file_digest(ROOT/p) for p in MODEL_PATHS.values()}}
    sources = ["tools/benchmark_scalability.py", "src/features.py", "src/appearance.py", "src/baseline.py", "src/utils.py"]
    result["protocol"]["source_sha256"] = {p: file_digest(ROOT/p) for p in sources}
    save_results(result, ROOT, settings)


if __name__ == "__main__":
    main()
