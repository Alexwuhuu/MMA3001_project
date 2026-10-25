"""Stages 7: development ablations, frozen independent evaluation and profiling.

Only the previously selected, train-fitted main models reach the frozen test.
Variant fits use the original training samples and fixed main SVR parameters;
their validation diagnostics cannot change the main model or test query list.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import cProfile
from io import StringIO
import json
from pathlib import Path
import pstats
import threading
from time import perf_counter

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

from .appearance import (APPEARANCE_COLUMNS, appearance_signature, extract_appearance,
                         pair_appearance)
from .baseline import make_pipeline, rank_candidates, regression_errors, retrieval_results
from .features import (FEATURE_COLUMNS, ImageFeatures, extract_image_features,
                       extract_pair_features, feature_signature, preprocess_rgb, stable_digest)
from .main import portable_frame
from .screening import _worker_setup, freeze_evaluation, load_screening_config
from .svr import _fit, _frame_hash, _inputs
from .utils import file_digest, read_rgb, write_json

KEYS = ["split", "query_id", "candidate_id"]
ALL_COLUMNS = FEATURE_COLUMNS + APPEARANCE_COLUMNS
GEOMETRY_COLUMNS = ["ransac_inliers", "ransac_inlier_ratio", "reprojection_rmse_px", "geometry_failed"]


def load_config(root):
    """Read the predeclared final-evaluation, resolution and repetition settings.

    Args:
        root (str | Path): Project containing config/final_experiments.json.

    Returns:
        dict: Final-experiment settings with the required timing and resolution protocol.

    Raises:
        OSError: Configuration cannot be read.
        ValueError: JSON or the predeclared resolution/repetition protocol is invalid.
    """
    cfg = json.loads((Path(root) / "config/final_experiments.json").read_text())
    if (cfg["timing_repeats"] != 5 or cfg["bootstrap_repeats"] < 100 or
            cfg["resolutions"] != [128, 256, 384] or cfg["reference_size"] != 256):
        raise ValueError("Use five timing repeats and the agreed 128/256/384 experiment")
    return cfg


def cross_pairs(queries, candidates, split):
    """Construct every query-candidate combination without using identity as a feature.

    Args:
        queries (sequence): Ordered query IDs.
        candidates (sequence): Ordered gallery IDs.
        split (str): Split label attached to all pairs.

    Returns:
        pandas.DataFrame: split/query_id/candidate_id rows in query-major Cartesian order.
    """
    return pd.DataFrame({"split": split, "query_id": np.repeat(list(queries), len(candidates)),
                         "candidate_id": np.tile(list(candidates), len(queries))})


def build_features(root, cases, pairs, settings, cache_dir):
    """Extract explicit within-split pairs with content/settings/source keyed caches.

    Requested galleries are checked before extraction. Missing descriptors or D
    remain represented; a cached query must reproduce its requested candidate list.

    Args:
        root (str | Path): Base for seasonal source paths.
        cases (pandas.DataFrame): Unique tree_id/split rows with seasonal RGB/D paths.
        pairs (pandas.DataFrame): Nonempty unique split/query_id/candidate_id requests.
        settings (dict): Combined ORB, appearance, worker and seed settings.
        cache_dir (str | Path): Writable parent cache directory.

    Returns:
        tuple[pandas.DataFrame, dict]: Requested metadata with all 21 numeric features, and
        image/pair counts, extraction times and cache namespace.

    Raises:
        ValueError: Pair IDs/splits/coverage, a cached gallery or numeric values are invalid.
        OSError: Source images or cache files are inaccessible.
    """
    root, cache_dir = Path(root), Path(cache_dir)
    cases = cases.set_index("tree_id")
    if not cases.index.is_unique:
        raise ValueError("Cases must have unique identities")
    if pairs.empty or pairs.duplicated(KEYS).any():
        raise ValueError("Need non-empty, unique image pairs")
    for field in ("query_id", "candidate_id"):
        if not set(pairs[field]).issubset(cases.index):
            raise ValueError("Pair identity is outside the supplied cases")
        if not np.array_equal(cases.loc[pairs[field], "split"].to_numpy(), pairs.split.to_numpy()):
            raise ValueError("Pair crosses split boundaries")
    namespace = cache_dir / stable_digest([feature_signature(settings), appearance_signature(settings),
                                          file_digest(Path(__file__))])[:20]
    namespace.mkdir(parents=True, exist_ok=True)
    paths = {(i, "summer") for i in pairs.query_id} | {(i, "winter") for i in pairs.candidate_id}
    images, hashes = {}, {}
    start = perf_counter()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(settings["seed"])
    for identity, season in sorted(paths):
        row = cases.loc[identity]
        rgb, depth = root / row[f"{season}_rgb"], row[f"{season}_depth"]
        depth = root / depth if depth else ""
        images[identity, season] = (
            extract_image_features(rgb, settings, namespace / "orb"),
            extract_appearance(rgb, depth, settings, namespace / "appearance"))
        hashes[identity, season] = [file_digest(rgb), file_digest(depth) if depth and depth.is_file() else "missing"]
    image_seconds = perf_counter() - start

    def one_query(item):
        (split, query_id), group = item
        ids = group.candidate_id.tolist()
        key = stable_digest([split, query_id, hashes[query_id, "summer"],
                             [(i, hashes[i, "winter"]) for i in ids]])
        path = namespace / f"pairs_{key}.csv"
        if path.is_file():
            frame = pd.read_csv(path, float_precision="round_trip")
            if frame.candidate_id.tolist() != ids or not frame.query_id.eq(query_id).all():
                raise ValueError("Invalid query-feature cache")
            return frame
        qorb, qapp = images[query_id, "summer"]
        records = []
        for candidate_id in ids:
            corb, capp = images[candidate_id, "winter"]
            records.append({"split": split, "query_id": query_id, "candidate_id": candidate_id,
                            **extract_pair_features(qorb, corb, settings), **pair_appearance(qapp, capp)})
        frame = pd.DataFrame(records)
        frame.to_csv(path, index=False)
        return frame

    groups = list(pairs.groupby(["split", "query_id"], sort=True))
    with ThreadPoolExecutor(max_workers=settings["workers"], initializer=_worker_setup,
                            initargs=(settings["seed"],)) as executor:
        frames = []
        for i, frame in enumerate(executor.map(one_query, groups), 1):
            frames.append(frame)
            if i % 100 == 0 or i == len(groups):
                print(f"Stage 7 feature queries: {i}/{len(groups)}", flush=True)
    result = pd.concat(frames, ignore_index=True)
    if len(result) != len(pairs) or np.isinf(result[ALL_COLUMNS].to_numpy()).any():
        raise ValueError("Missing pairs or infinite feature values")
    return result, {"images": len(images), "pairs": len(result), "image_seconds": image_seconds,
                    "total_seconds": perf_counter()-start, "namespace": namespace.name}


def bootstrap_interval(values, repeats=5000, seed=42):
    """Estimate a percentile 95% interval for a mean by resampling queries.

    Args:
        values (array-like): One finite scalar per query; an empty sequence is allowed.
        repeats (int): Positive number of bootstrap samples.
        seed (int): Reproducible NumPy resampling seed.

    Returns:
        list: Lower/upper 2.5/97.5 percentiles, or [None, None] for empty input.

    Raises:
        ValueError: Repetition count or numeric conversion is invalid.
    """
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [None, None]
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (repeats, len(values)))].mean(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def proportion_interval(values, repeats=5000, seed=42):
    """Use Wilson intervals for binary credit and query bootstrap for fractional ties.

    Args:
        values (array-like): Per-query Top-k credit in [0, 1]; empty input is allowed.
        repeats (int): Positive bootstrap repetition count for fractional outcomes.
        seed (int): Seed for fractional-credit resampling.

    Returns:
        dict: method, lower and upper for a 95% interval; empty input yields method
        "unavailable" and null bounds.

    Raises:
        ValueError: Numeric conversion or bootstrap repetition count is invalid.
    """
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"method": "unavailable", "lower": None, "upper": None}
    if not np.isin(values, [0, 1]).all():
        lower, upper = bootstrap_interval(values, repeats, seed)
        return {"method": "query_bootstrap_fractional_ties", "lower": lower, "upper": upper}
    n, p, z = len(values), float(values.mean()), 1.959963984540054
    denominator = 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return {"method": "Wilson_95", "lower": max(0., center-radius), "upper": min(1., center+radius)}


def verified_metrics(scored, audit, metric_settings, settings):
    """Retain every score while restricting verified truth to independently confirmed queries.

    Args:
        scored (pandas.DataFrame): Complete frozen-gallery query_id/candidate_id/score rows.
        audit (pandas.DataFrame): One tree_id/independent_status row per sampled query.
        metric_settings (dict): Score tie tolerances.
        settings (dict): bootstrap_repeats and seed.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame, dict]: Full rankings, per-query audit
        eligibility and filename/verified credit, and verified-subset metrics/intervals
        alongside explicitly labeled filename diagnostics.

    Raises:
        ValueError: Audit coverage/statuses or closed-set gallery scores/identities are invalid.
    """
    if (audit.tree_id.duplicated().any() or set(audit.tree_id) != set(scored.query_id) or
            not set(audit.independent_status).issubset({"confirmed", "rejected", "pending", "uncertain"})):
        raise ValueError("Audit must retain every frozen query exactly once")
    ranked, per_query, filename_metrics = retrieval_results(scored, metric_settings)
    per_query = per_query.merge(audit[["tree_id", "independent_status"]], left_on="query_id", right_on="tree_id",
                                validate="one_to_one").drop(columns="tree_id")
    # Keep unresolved queries visible while withholding unsupported verified accuracy.
    per_query["verified_eligible"] = per_query.independent_status.eq("confirmed")
    for source, target in [("top1", "verified_top1"), ("top5", "verified_top5"),
                           ("reciprocal_rank", "verified_reciprocal_rank")]:
        per_query[target] = per_query[source].where(per_query.verified_eligible)
    verified = per_query[per_query.verified_eligible]
    confirmed = set(verified.query_id)
    intervals = {key: proportion_interval(verified[key], settings["bootstrap_repeats"], settings["seed"])
                 for key in ("top1", "top5")}
    lower, upper = bootstrap_interval(verified.reciprocal_rank, settings["bootstrap_repeats"], settings["seed"])
    intervals["mrr"] = {"method": "query_bootstrap_95", "lower": lower, "upper": upper}
    metrics = {"queries_sampled": len(per_query), "queries_verified": len(verified),
               "gallery_size": filename_metrics["gallery_size"],
               "audit_counts": audit.independent_status.value_counts().to_dict(),
               "verified_retrieval": {"top1": float(verified.top1.mean()) if len(verified) else None,
                                      "top5": float(verified.top5.mean()) if len(verified) else None,
                                      "mrr": float(verified.reciprocal_rank.mean()) if len(verified) else None},
               "intervals_95": intervals,
               "verified_query_regression": regression_errors(scored[scored.query_id.isin(confirmed)]) if confirmed else None,
               "all_sample_filename_diagnostic": {"retrieval": filename_metrics, "regression": regression_errors(scored)},
               "scope": "Confirmed query subset only; other gallery IDs assumed distinct. Rejected/unresolved queries have no verified truth.",
               "uncertainty_limit": "Query intervals are conditional on this verified subset and fixed gallery; only four test rows, no orchard-population claim."}
    return ranked, per_query, metrics


def paired_comparison(left, right, settings):
    """Bootstrap query-paired SVR-minus-linear differences on shared verified queries.

    Args:
        left (pandas.DataFrame): Linear per-query results with verified_eligible and retrieval
            credit.
        right (pandas.DataFrame): SVR results with the same query/eligibility/metric schema.
        settings (dict): bootstrap_repeats and seed.

    Returns:
        pandas.DataFrame: Top-1, Top-5 and reciprocal-rank differences, sample count and paired
        95% bounds. No shared verified rows yields null estimates.

    Raises:
        pandas.errors.MergeError: Eligible query IDs are duplicated.
    """
    paired = left[left.verified_eligible].merge(right[right.verified_eligible], on="query_id", validate="one_to_one", suffixes=("_linear", "_svr"))
    rows = []
    for metric in ("top1", "top5", "reciprocal_rank"):
        delta = paired[f"{metric}_svr"]-paired[f"{metric}_linear"]
        lo, hi = bootstrap_interval(delta, settings["bootstrap_repeats"], settings["seed"])
        rows.append({"metric": metric, "verified_queries": len(paired),
                     "svr_minus_linear": float(delta.mean()) if len(delta) else None,
                     "lower_95": lo, "upper_95": hi, "method": "paired_query_bootstrap"})
    return pd.DataFrame(rows)


def development_experiments(root, cases, baseline, svr, settings, output):
    """Change one factor at a time using fixed main-SVR hyperparameters.

    Reference, D/RANSAC ablations, full-frame crop and 128/384 resolution controls
    use training/validation only. These diagnostics cannot revise the main selection.

    Args:
        root (Path): Project base for sources and caches.
        cases (pandas.DataFrame): Case manifest; test rows are filtered out.
        baseline (dict): Original samples, complete features and fitted baseline bundle.
        svr (dict): Locked main SVR result with selected parameters and solver settings.
        settings (dict): Final resolution, threshold, timing and cache settings.
        output (Path): Existing destination for development artifacts.

    Returns:
        tuple[pandas.DataFrame, dict]: Per-variant/model validation diagnostics and numeric
        feature/configuration fingerprints.

    Raises:
        ValueError: Development rows leak, shared samples are lost or a fixed-parameter fit does
            not converge.
        AssertionError: Reference scores differ from saved main-model scores.
        OSError: Variant artifacts cannot be accessed.
    """
    dev = cases[cases.split.isin(["train", "validation"])].copy()
    if set(dev.loc[dev.split.eq("train"), "row"]) & set(dev.loc[dev.split.eq("validation"), "row"]):
        raise ValueError("Development row leakage")
    sample = baseline["training_pairs"]
    columns = baseline["bundle"]["feature_columns"]
    base = pd.concat(baseline["rankings"], ignore_index=True)
    base = base[KEYS+columns]
    validation_ids = sorted(dev.loc[dev.split.eq("validation"), "tree_id"])
    requested = pd.concat([sample[KEYS], cross_pairs(validation_ids, validation_ids, "validation")], ignore_index=True)
    shared = sample[[*KEYS, "target", "sample_weight"]]
    params = svr["selection"]["parameters"]
    solver = svr["summary"]["settings"]
    variants = [("reference", 256, "center", columns),
                ("without_D", 256, "center", [c for c in columns if c not in APPEARANCE_COLUMNS[1:]]),
                ("without_RANSAC", 256, "center", [c for c in columns if c not in GEOMETRY_COLUMNS]),
                ("full_image", 256, "full", columns),
                ("center_128", 128, "center", columns), ("center_384", 384, "center", columns)]
    rows, fingerprints, extraction = [], {}, {}
    for name, size, crop, used in variants:
        cfg = {**baseline["bundle"]["settings"], "image_size": size, "crop": crop,
               "ransac_threshold_px": settings["reference_ransac_threshold_px"]*size/settings["reference_size"]}
        if crop == "center" and size == 256:
            frame = requested.merge(base, on=KEYS, validate="one_to_one")
            if len(frame) != len(requested):
                raise ValueError("Incomplete reference features")
        else:
            frame, extraction[name] = build_features(root, dev, requested, cfg, root/settings["cache_dir"])
        train = shared.merge(frame, on=KEYS, how="left", validate="one_to_one", indicator=True)
        if not train._merge.eq("both").all():
            raise ValueError("Variant lost a shared training sample")
        validation = frame[frame.split.eq("validation")].sort_values(["query_id", "candidate_id"])
        fingerprints[name] = {"feature_hash": _frame_hash(frame[KEYS+used].sort_values(KEYS)),
                              "columns": used, "size": size, "crop": crop,
                              "ransac_threshold_px": cfg["ransac_threshold_px"]}
        folder = output/"development"/name
        folder.mkdir(parents=True, exist_ok=True)
        train.drop(columns="_merge").to_csv(folder/"training_pairs.csv", index=False)
        for model_name in ("LinearRegression", "RBF-SVR"):
            durations = []
            for _ in range(settings["timing_repeats"]):
                if model_name == "LinearRegression":
                    start = perf_counter()
                    pipeline = make_pipeline().fit(train[used], train.target, regressor__sample_weight=train.sample_weight)
                    duration = perf_counter()-start
                else:
                    pipeline, duration, converged = _fit(params, solver, train[used], train.target, train.sample_weight)
                    if not converged:
                        raise ValueError("A fixed-parameter SVR variant failed to converge")
                durations.append(duration)
            predictions = []
            for _ in range(settings["timing_repeats"]):
                start = perf_counter()
                scores = pipeline.predict(validation[used])
                predictions.append(perf_counter()-start)
            scored = validation[KEYS].copy().assign(score=scores)
            ranked, queries, retrieval = retrieval_results(scored, cfg)
            errors = regression_errors(scored)
            train_errors = regression_errors(train[KEYS].assign(score=pipeline.predict(train[used])))
            ranked.to_csv(folder/f"{model_name}_validation_rankings.csv", index=False)
            queries.to_csv(folder/f"{model_name}_query_metrics.csv", index=False)
            rows.append({"variant": name, "model": model_name, "split": "validation",
                         "size": size, "crop": crop, "ransac_threshold_px": cfg["ransac_threshold_px"],
                         "features": len(used), **{k: retrieval[k] for k in ("queries", "gallery_size", "top1", "top5", "mrr")},
                         "train_balanced_mae": train_errors["balanced_class_mae"],
                         "validation_balanced_mae": errors["balanced_class_mae"],
                         "positive_mae": errors["positive_pairs"]["mae"], "negative_mae": errors["negative_pairs"]["mae"],
                         "positive_rmse": errors["positive_pairs"]["rmse"], "negative_rmse": errors["negative_pairs"]["rmse"],
                         "median_fit_seconds": float(np.median(durations)),
                         "median_cached_score_ms_per_query": float(1000*np.median(predictions)/len(validation_ids)),
                         "fit_seconds_repetitions": json.dumps(durations), "batch_seconds_repetitions": json.dumps(predictions),
                         "support_vectors": len(pipeline.named_steps["regressor"].support_) if model_name == "RBF-SVR" else 0})
            if name == "reference":
                reference = baseline["bundle"] if model_name == "LinearRegression" else svr["bundle"]
                np.testing.assert_allclose(scores, reference["pipeline"].predict(validation[used]), rtol=1e-10, atol=1e-10)
        pd.DataFrame(rows).to_csv(output/"development_comparison.csv", index=False)
        print(f"Stage 7 development variant complete: {name}", flush=True)
    write_json(output/"development_extraction.json", extraction)
    return pd.DataFrame(rows), fingerprints


def _uncached_orb(path, cfg):
    """Extract descriptors without hashing or disk caching for the timing experiment.

    Args:
        path (str | Path): RGB image path.
        cfg (dict): image_size, crop and orb_nfeatures.

    Returns:
        ImageFeatures: Resized pixel coordinates and descriptors with a benchmark-only hash
        marker.

    Raises:
        ValueError: RGB/preprocessing is invalid.
    """
    rgb = preprocess_rgb(read_rgb(path), cfg["image_size"], cfg["crop"])
    points, descriptors = cv2.ORB_create(nfeatures=cfg["orb_nfeatures"], WTA_K=2).detectAndCompute(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None)
    xy = np.asarray([p.pt for p in points], dtype=np.float32).reshape(-1, 2)
    return ImageFeatures("uncached_benchmark", xy, descriptors if descriptors is not None else np.empty((0, 32), np.uint8))


def benchmark(root, cases, bundles, settings, output):
    """Benchmark a fixed validation query and its entire winter gallery.

    Appearance summaries are shared across modes to isolate descriptor reuse.
    Timing includes correspondences, RANSAC, scoring and ranking; model loading,
    hashing and appearance extraction are excluded. OS file caches remain warm.

    Args:
        root (Path): Project base for seasonal images.
        cases (pandas.DataFrame): Manifest containing a nonempty validation split.
        bundles (dict): Fitted main models keyed by LinearRegression and RBF-SVR.
        settings (dict): timing_repeats and cache_dir.
        output (Path): Existing destination for timing/memory CSVs and profiles.

    Returns:
        pandas.DataFrame: Per-model/mode timings, descriptor storage and absolute process RSS
        sampled separately every 10 ms; array bytes are not peak process memory.

    Raises:
        AssertionError: Cached/batched predictions differ from the reference mode.
        ValueError: Images or predictions are invalid.
        OSError: Profiling/timing artifacts cannot be written.
    """
    val = cases[cases.split.eq("validation")].sort_values("tree_id")
    query = val.iloc[0]
    cfg = bundles["LinearRegression"]["settings"]
    paths = [root/query.summer_rgb, *[root/p for p in val.winter_rgb]]
    start = perf_counter()
    images = [_uncached_orb(p, cfg) for p in paths]
    cache_build = perf_counter()-start
    appearances = [extract_appearance(root/query.summer_rgb, root/query.summer_depth if query.summer_depth else "", cfg,
                                     root/settings["cache_dir"]/"benchmark_appearance")]
    appearances += [extract_appearance(root/r.winter_rgb, root/r.winter_depth if r.winter_depth else "", cfg,
                                       root/settings["cache_dir"]/"benchmark_appearance") for r in val.itertuples()]
    appearance_pairs = [pair_appearance(appearances[0], a) for a in appearances[1:]]
    descriptor_bytes = sum(i.xy.nbytes+i.descriptors.nbytes for i in images)
    records, memory = [], []
    process = psutil.Process()

    def query_once(bundle, cached, batch):
        values = []
        for j, candidate_path in enumerate(paths[1:], 1):
            q, c = (images[0], images[j]) if cached else (_uncached_orb(paths[0], cfg), _uncached_orb(candidate_path, cfg))
            values.append({**extract_pair_features(q, c, cfg), **appearance_pairs[j-1]})
        X = pd.DataFrame(values)[bundle["feature_columns"]]
        pipe = bundle["pipeline"]
        score = pipe.predict(X) if batch else np.concatenate([pipe.predict(X.iloc[j:j+1]) for j in range(len(X))])
        ranked = rank_candidates(pd.DataFrame({"candidate_id": val.tree_id.to_list(), "score": score}),
                                 cfg["score_tie_atol"], cfg["score_tie_rtol"])
        return score, ranked

    modes = [("uncached_descriptors_pairwise", False, False),
             ("cached_descriptors_pairwise", True, False), ("cached_descriptors_batch", True, True)]
    for name, bundle in bundles.items():
        reference, _ = query_once(bundle, True, True)
        for mode, cached, batch in modes:
            durations = []
            for _ in range(settings["timing_repeats"]):
                start = perf_counter()
                scores, _ = query_once(bundle, cached, batch)
                durations.append(perf_counter()-start)
                np.testing.assert_allclose(scores, reference, rtol=1e-10, atol=1e-10)
            # Keep RSS monitoring outside timed runs so it cannot distort latency.
            rss = [process.memory_info().rss]
            stop = threading.Event()
            def sample_rss():
                while not stop.wait(.01):
                    rss.append(process.memory_info().rss)
            monitor = threading.Thread(target=sample_rss, daemon=True)
            monitor.start()
            try:
                query_once(bundle, cached, batch)
                rss.append(process.memory_info().rss)
            finally:
                stop.set()
                monitor.join()
            records.append({"model": name, "mode": mode, "query_id": query.tree_id, "gallery_size": len(val),
                            "median_query_seconds": float(np.median(durations)), "seconds_repetitions": json.dumps(durations),
                            "timing_repeats": len(durations), "descriptor_cache_build_seconds": cache_build,
                            "descriptor_array_bytes": descriptor_bytes, "rss_before_bytes": rss[0],
                            "rss_peak_sampled_bytes": max(rss), "rss_after_bytes": rss[-1],
                            "rss_sample_interval_seconds": .01})
            print(f"Benchmark {name}, {mode}: {np.median(durations):.4f}s/query", flush=True)
        profiler = cProfile.Profile()
        profiler.runcall(query_once, bundle, True, True)
        profiler.dump_stats(str(output/f"profile_{name}.prof"))
        stream = StringIO()
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(30)
        (output/f"profile_{name}.txt").write_text(stream.getvalue(), encoding="utf-8")
        reg = bundle["pipeline"].named_steps["regressor"]
        memory.append({"model": name, "support_vectors": len(reg.support_) if hasattr(reg, "support_") else 0,
                       "support_vector_array_bytes": reg.support_vectors_.nbytes if hasattr(reg, "support_vectors_") else 0,
                       "serialized_bundle_bytes": len(__import__("pickle").dumps(bundle)),
                       "descriptor_array_bytes": descriptor_bytes,
                       "scope": "Sampled process RSS includes interpreter and other models; may miss sub-10ms peaks. Array bytes are not peak RAM."})
    frame = pd.DataFrame(records)
    frame.to_csv(output/"performance.csv", index=False)
    pd.DataFrame(memory).to_csv(output/"memory_and_models.csv", index=False)
    write_json(output/"performance_scope.json", {"query_selection": "first validation ID, complete validation gallery",
        "appearance": "precomputed and shared across modes", "descriptor_cache": "in-memory numeric arrays; build cost reported separately",
        "uncached": "decode/crop/ORB query and candidate anew for each pair; warm OS file cache",
        "timing": "perf_counter, five runs, median; includes matching/RANSAC/scoring/ranking; excludes model loading and appearance",
        "memory": "separate run, RSS sampled every 10ms, absolute shared process footprint"})
    return frame


def _lock_before_test(root, baseline, svr, evaluation, fingerprints, settings, resnet):
    """Freeze method choices and source/input hashes before test extraction.

    Args:
        root (Path): Project containing source files and the final lock.
        baseline (dict): Original query provenance and sample hash.
        svr (dict): Locked main selection.
        evaluation (dict): Frozen queries/gallery/protocol/review folder.
        fingerprints (dict): Development variant feature/configuration evidence.
        settings (dict): Final method settings and lock_file.
        resnet (dict | None): Optional previously locked supplementary selection.

    Returns:
        dict: Final input fingerprint and evidence, main model names and frozen sample/gallery
        sizes. Existing locks must match exactly.

    Raises:
        ValueError: Test identities overlap development or the existing final lock differs.
        OSError: Source/protocol/lock files are inaccessible.
    """
    development_ids = set(baseline["query_metrics"].query_id)
    test_ids = set(evaluation["gallery"].tree_id)
    if development_ids & test_ids:
        raise ValueError("Test identities overlap development data")
    frozen = {p.name: file_digest(p) for p in Path(evaluation["folder"]).iterdir() if p.suffix in {".csv", ".json"}}
    fingerprint = {"variants": fingerprints, "main_selection": svr["selection"],
                   "training_pair_identity_hash": baseline["summary"]["training_pair_identity_hash"],
                   "test_protocol_and_audit": frozen,
                   "settings": {k: v for k, v in settings.items() if k not in {"output_dir", "cache_dir", "lock_file", "export_dir", "small_results_dir"}},
                   "source_hashes": {p.name: file_digest(p) for p in sorted((root/"src").glob("*.py"))},
                   "supplementary_resnet_selection": resnet["selection"] if resnet else None}
    lock = {"input_fingerprint": stable_digest(fingerprint), "inputs": fingerprint,
            "main_models": ["LinearRegression", "RBF-SVR"], "variant_selection": "none; validation diagnostics with fixed main SVR parameters",
            "test_query_count": len(evaluation["queries"]), "test_gallery_count": len(evaluation["gallery"]),
            "refit_on_train_plus_validation": False, "frozen_before_test_features": True}
    path = root/settings["lock_file"]
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing != lock:
            raise ValueError("Final experiment lock changed; preserve this test result and use a new lock/output for an intentional experiment")
    else:
        write_json(path, lock)
    return lock


def validate_locked_models(baseline, svr, evaluation, resnet=None):
    """Verify shared samples, preprocessing and parameters against the recorded selections.

    Args:
        baseline (dict): Original baseline result accepted by src.svr._inputs.
        svr (dict): Main fitted bundle and original selection.
        evaluation (dict): Frozen evaluation with protocol file folder.
        resnet (dict | None): Optional supplementary result with locked SVR bundle.

    Returns:
        None: All main and requested supplementary invariants pass.

    Raises:
        ValueError: Samples, feature order, parameters, selection or protocol hashes changed.
        AssertionError: Main train-fitted imputer/scaler values disagree.
    """
    columns, sample, frames = _inputs(baseline)
    lock = svr["selection"]
    if (lock["training_pair_identity_hash"] != baseline["summary"]["training_pair_identity_hash"] or
            lock["feature_columns"] != columns or svr["bundle"]["feature_columns"] != columns or
            svr["bundle"]["selection"] != lock):
        raise ValueError("Main SVR model differs from its original selection")
    reg = svr["bundle"]["pipeline"].named_steps["regressor"]
    if any(getattr(reg, key) != value for key, value in lock["parameters"].items()):
        raise ValueError("Main SVR hyperparameters changed after validation selection")
    for step, attributes in [("imputer", ["statistics_"]), ("scaler", ["mean_", "scale_"])]:
        for attribute in attributes:
            np.testing.assert_allclose(getattr(baseline["bundle"]["pipeline"].named_steps[step], attribute),
                                       getattr(svr["bundle"]["pipeline"].named_steps[step], attribute), rtol=0, atol=0)
    hashes = {name: file_digest(evaluation["folder"]/name) for name in ("queries.csv", "gallery.csv", "protocol.json")}
    if hashes != baseline["summary"]["frozen_protocol_hashes"] or hashes != lock["frozen_protocol_hashes"]:
        raise ValueError("Final evaluation protocol differs from the selection protocol")
    if resnet:
        bundle = resnet["bundles"]["svr"]
        if (bundle["selection"] != resnet["selection"] or
                resnet["selection"]["training_pair_identity_hash"] != lock["training_pair_identity_hash"]):
            raise ValueError("Supplementary representation differs from its locked development model")
        for key, value in resnet["selection"]["parameters"].items():
            if getattr(bundle["pipeline"].named_steps["regressor"], key) != value:
                raise ValueError("Supplementary SVR hyperparameters changed")


def run_final(root, preparation, screening, baseline, svr, settings=None, resnet=None, render=True):
    """Run controlled development diagnostics, freeze choices and evaluate the independent test.

    Test scores never select or refit a model. Rejected/unresolved queries remain
    in rankings and counts but receive no verified accuracy; intervals describe the
    confirmed subset conditional on its fixed gallery.

    Args:
        root (str | Path): Project directory.
        preparation (dict): Completed preparation with all eligible cases.
        screening (dict): Screening result with frozen evaluation settings.
        baseline (dict): Completed stage-5 model, samples and development features.
        svr (dict): Completed locked stage-6 model and trials.
        settings (dict | None): Final-experiment settings; None loads defaults.
        resnet (dict | None): Previously selected neural extension, used only if
            include_resnet_test is true.
        render (bool): Whether to write final aggregate/example figures.

    Returns:
        dict: summary, metrics, comparison, development, performance, paired, per_queries,
        rankings, features, bundles, output_dir and lock.

    Raises:
        ValueError: Split/protocol/model/final-lock invariants fail or protected records change.
        AssertionError: Reference predictions, preprocessing or benchmark modes disagree.
        OSError: Images, caches or experiment artifacts cannot be accessed.
    """
    root = Path(root).resolve()
    settings = dict(settings or load_config(root))
    cases = portable_frame(preparation["cases"], root)
    evaluation = freeze_evaluation(cases, root, screening["summary"]["settings"])
    folder = root/settings["output_dir"]
    folder.mkdir(parents=True, exist_ok=True)
    if (set(cases.loc[cases.split.eq("test"), "row"]) & set(cases.loc[cases.split.ne("test"), "row"]) or
            set(evaluation["gallery"].tree_id) != set(cases.loc[cases.split.eq("test"), "tree_id"])):
        raise ValueError("Test rows must be disjoint and use the frozen complete gallery")
    protected = [evaluation["folder"]/name for name in ("queries.csv", "gallery.csv", "protocol.json", "independent_review.csv")]
    protected += [root/"config/pair_review.csv", root/svr["summary"]["settings"]["selection_file"]]
    protected = [p for p in protected if p.is_file()]
    before = {str(p): file_digest(p) for p in protected}
    validate_locked_models(baseline, svr, evaluation, resnet if settings["include_resnet_test"] else None)
    development, fingerprints = development_experiments(root, cases, baseline, svr, settings, folder)
    if not settings["include_resnet_test"]:
        resnet = None
    # Commit the protocol boundary before any test features or model scores exist.
    lock = _lock_before_test(root, baseline, svr, evaluation, fingerprints, settings, resnet)
    write_json(folder/"experiment_lock.json", lock)
    bundles = {"LinearRegression": baseline["bundle"], "RBF-SVR": svr["bundle"]}
    performance = benchmark(root, cases, bundles, settings, folder)
    queries, gallery = evaluation["queries"], evaluation["gallery"]
    pairs = cross_pairs(queries.tree_id.tolist(), gallery.tree_id.tolist(), "test")
    features, extraction = build_features(root, cases[cases.split.eq("test")], pairs,
                                          baseline["bundle"]["settings"], root/settings["cache_dir"])
    features.to_csv(folder/"test_pair_features.csv", index=False)
    evaluation["review"].to_csv(folder/"independent_review_normalized.csv", index=False)
    metrics, per_queries, rankings, comparison = {}, {}, {}, []
    model_features = {name: features for name in bundles}
    if resnet:
        from .deep_features import embed_paths, pair_matrix
        vectors, encoder_info = embed_paths([*queries.summer_rgb.tolist(), *gallery.winter_rgb.tolist()], root,
                                            resnet["bundles"]["svr"]["settings"])
        write_json(folder/"supplementary_test_encoder.json", encoder_info)
        query_paths = queries.set_index("tree_id").summer_rgb
        candidate_paths = gallery.set_index("tree_id").winter_rgb
        bundle = resnet["bundles"]["svr"]
        if encoder_info["weights_sha256"] != bundle["encoder_info"]["weights_sha256"]:
            raise ValueError("Supplementary test checkpoint differs from development")
        values = pair_matrix(np.stack([vectors[query_paths[q]] for q in pairs.query_id]),
                             np.stack([vectors[candidate_paths[c]] for c in pairs.candidate_id]), bundle["projection"])
        neural = pd.concat([pairs, pd.DataFrame(values, columns=bundle["feature_columns"])], axis=1)
        neural.to_csv(folder/"supplementary_test_pair_features.csv", index=False)
        for name, key in [("ResNet18+LinearRegression", "linear"), ("ResNet18+RBF-SVR", "svr")]:
            bundles[name], model_features[name] = resnet["bundles"][key], neural
    for name, bundle in bundles.items():
        frame = model_features[name]
        scored = frame[KEYS].copy().assign(score=bundle["pipeline"].predict(frame[bundle["feature_columns"]]))
        ranked, per_query, values = verified_metrics(scored, evaluation["review"], baseline["bundle"]["settings"], settings)
        metadata = queries[["tree_id", "row", "number"]]
        per_query = per_query.merge(metadata, left_on="query_id", right_on="tree_id", validate="one_to_one").drop(columns="tree_id")
        metrics[name], per_queries[name], rankings[name] = values, per_query, ranked
        ranked.to_csv(folder/f"{name}_test_rankings.csv", index=False)
        per_query.to_csv(folder/f"{name}_test_queries.csv", index=False)
        row = {"model": name, "role": "main" if name in {"LinearRegression", "RBF-SVR"} else "supplementary",
               "sampled": values["queries_sampled"], "verified": values["queries_verified"],
               "gallery_size": values["gallery_size"], **values["verified_retrieval"]}
        for metric, interval in values["intervals_95"].items():
            row[f"{metric}_lower_95"], row[f"{metric}_upper_95"] = interval["lower"], interval["upper"]
        errors = values["verified_query_regression"]
        for label in ("all_pairs", "positive_pairs", "negative_pairs"):
            for metric in ("mae", "rmse"):
                row[f"{label}_{metric}"] = errors[label][metric] if errors else None
        comparison.append(row)
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(folder/"test_comparison.csv", index=False)
    paired = paired_comparison(per_queries["LinearRegression"], per_queries["RBF-SVR"], settings)
    paired.to_csv(folder/"paired_model_differences.csv", index=False)
    if resnet:
        paired_comparison(per_queries["ResNet18+LinearRegression"], per_queries["ResNet18+RBF-SVR"], settings).to_csv(
            folder/"supplementary_paired_differences.csv", index=False)
    # Full validation search results provide parameter/error sensitivity without test reuse.
    svr["trials"].to_csv(folder/"svr_parameter_sensitivity.csv", index=False)
    write_json(folder/"test_metrics.json", metrics)
    after = {str(p): file_digest(p) for p in protected}
    if before != after:
        raise ValueError("A protected label/selection/protocol file changed")
    summary = {"settings": settings, "main_models": ["LinearRegression", "RBF-SVR"],
               "supplementary_models": [n for n in bundles if n.startswith("ResNet")],
               "test_queries_scored_per_model": len(queries), "test_gallery": len(gallery),
               "independent_review_counts": evaluation["review"].independent_status.value_counts().to_dict(),
               "verified_query_ids": sorted(evaluation["review"].loc[evaluation["review"].independent_status.eq("confirmed"), "tree_id"]),
               "test_extraction": extraction, "protected_files_unchanged": before == after,
               "protected_hashes": {str(Path(p).relative_to(root)) if Path(p).is_relative_to(root) else p: d for p, d in before.items()},
               "training_pair_identity_hash": baseline["summary"]["training_pair_identity_hash"],
               "variants_use_test": False, "variants_use_fixed_main_svr_parameters": True,
               "variant_interpretation": "Controlled validation diagnostics at fixed hyperparameters, not separately optimized models; full frame is resized to square.",
               "independent_accuracy_scope": "Verified subset only; rejected and uncertain queries retained without accuracy labels. Filename diagnostics do not validate the full sample.",
               "model_selection_after_test": False, "refitted_on_train_plus_validation": False,
               "uncertainty": "Wilson Top-k (bootstrap if fractional ties); query-bootstrap MRR and paired differences; conditional on verified queries/fixed gallery, not independent orchard rows."}
    write_json(folder/"run_summary.json", summary)
    result = {"summary": summary, "metrics": metrics, "comparison": comparison, "development": development,
              "performance": performance, "paired": paired, "per_queries": per_queries, "rankings": rankings,
              "features": model_features, "bundles": bundles, "output_dir": folder, "lock": lock}
    if render:
        render_figures(root, cases, result, svr["trials"])
    return result


def render_figures(root, cases, result, trials):
    """Render final aggregate comparisons, sensitivity/performance plots and fixed examples.

    Args:
        root (Path): Base for source image paths.
        cases (pandas.DataFrame): Eligible identity metadata.
        result (dict): Completed run_final result including output_dir.
        trials (pandas.DataFrame): Historical validation SVR search rows.

    Returns:
        None: PNG figures are written under the final output directory.

    Raises:
        ValueError: An example image is unreadable.
        OSError: Figures cannot be written.
    """
    folder = result["output_dir"]/"figures"
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    table = result["comparison"]
    for ax, metric in zip(axes, ("top1", "top5", "mrr")):
        selected = table[table.verified.gt(0)]
        values = selected[metric].to_numpy(dtype=float)
        lower, upper = selected[f"{metric}_lower_95"].to_numpy(dtype=float), selected[f"{metric}_upper_95"].to_numpy(dtype=float)
        ax.bar(np.arange(len(selected)), values, color=["steelblue", "darkorange", "lightsteelblue", "peachpuff"][:len(selected)])
        if len(selected):
            ax.errorbar(np.arange(len(selected)), values, yerr=np.maximum(0, np.array([values-lower, upper-values])), fmt="none", c="black", capsize=4)
        ax.set(xticks=np.arange(len(selected)), xticklabels=selected.model, ylabel=metric, ylim=(0, 1))
        ax.tick_params(axis="x", labelrotation=35, labelsize=7)
    fig.suptitle("Independent confirmed queries only; 95% conditional intervals")
    fig.tight_layout()
    fig.savefig(folder/"test_comparison.png", dpi=140)
    plt.close(fig)
    dev = result["development"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    names = list(dev.variant.unique())
    for j, model in enumerate(("LinearRegression", "RBF-SVR")):
        values = dev[dev.model.eq(model)].set_index("variant").loc[names]
        for ax, metric in zip(axes, ("top1", "top5", "mrr")):
            ax.bar(np.arange(len(names))+j*.38, values[metric], .38, label=model)
            ax.set(xticks=np.arange(len(names))+.19, xticklabels=names, ylabel=metric)
            ax.tick_params(axis="x", labelrotation=35, labelsize=8)
    axes[0].legend(fontsize=7)
    fig.suptitle("Validation ablations and sensitivity: fixed main SVR parameters")
    fig.tight_layout()
    fig.savefig(folder/"development_sensitivity.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for (gamma, epsilon), group in trials[trials.status.eq("ok")].groupby(["gamma", "epsilon"]):
        group = group.sort_values("C")
        label = f"g={gamma}, eps={epsilon}"
        axes[0].plot(group.C, group.train_balanced_class_mae, "o-", label=label)
        axes[1].plot(group.C, group.validation_balanced_class_mae, "o-", label=label)
        axes[2].plot(group.C, group.validation_mrr, "o-", label=label)
    for ax, title in zip(axes, ("Sampled train balanced MAE", "Full validation balanced MAE", "Validation MRR")):
        ax.set(xscale="log", xlabel="C", title=title)
    axes[2].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(folder/"parameter_error_sensitivity.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    timing = result["performance"]
    for j, model in enumerate(("LinearRegression", "RBF-SVR")):
        values = timing[timing.model.eq(model)]
        axes[0].bar(np.arange(3)+j*.38, values.median_query_seconds, .38, label=model)
        fit = dev[dev.variant.eq("reference") & dev.model.eq(model)].iloc[0]
        axes[1].bar(j, fit.median_fit_seconds, label=model)
    axes[0].set(xticks=np.arange(3)+.19, xticklabels=["Recompute ORB / pair", "Cache ORB / pair", "Cache ORB / batch"], ylabel="Median seconds / query", yscale="log")
    axes[0].legend(fontsize=7)
    axes[1].set(xticks=[0, 1], xticklabels=["Linear", "SVR"], ylabel="Median fit seconds", yscale="log")
    fig.suptitle("Five-repeat timing; query scope and cache build cost in performance_scope.json")
    fig.tight_layout()
    fig.savefig(folder/"performance.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name in zip(axes, ("LinearRegression", "RBF-SVR")):
        data = result["per_queries"][name]
        known = data[data.verified_eligible]
        unknown = data[~data.verified_eligible]
        ax.scatter(unknown.row.str[1:].astype(int), unknown.number, marker="x", c="gray", label="unverified/rejected")
        ax.scatter(known.row.str[1:].astype(int), known.number, c=known.top1, cmap="RdYlGn", vmin=0, vmax=1, label="confirmed Top-1 credit")
        ax.set(xlabel="R: row index", ylabel="N: tree number", title=name)
        ax.invert_yaxis()
        ax.legend(fontsize=6)
    fig.suptitle("Test ID index diagram (no metric coordinates); red=0, green=1")
    fig.tight_layout()
    fig.savefig(folder/"test_index.png", dpi=140)
    plt.close(fig)
    ids = cases.set_index("tree_id")
    data = result["per_queries"]["RBF-SVR"]
    chosen = data[data.verified_eligible & data.top1.lt(1)].sort_values("query_id").head(3)
    if len(chosen):
        fig, axes = plt.subplots(len(chosen), 3, figsize=(11, 3*len(chosen)), squeeze=False)
        for row_axes, record in zip(axes, chosen.itertuples()):
            paths = [(ids.loc[record.query_id].summer_rgb, f"{record.query_id} Summer"),
                     (ids.loc[record.query_id].winter_rgb, f"Confirmed pair; rank {record.rank_min}-{record.rank_max}"),
                     (ids.loc[record.top_candidate].winter_rgb, f"Displayed first: {record.top_candidate}")]
            for ax, (path, title) in zip(row_axes, paths):
                ax.imshow(read_rgb(root/path)); ax.set_title(title, fontsize=9); ax.axis("off")
        fig.suptitle("Main SVR errors: first three confirmed failures in ID order")
        fig.tight_layout()
        fig.savefig(folder/"error_cases.png", dpi=120)
        plt.close(fig)


def main():
    """Reload saved stages 1-6 and run final evaluation/export without new model selection.

    Returns:
        None: Configured final results/export are written and summaries printed.

    Raises:
        ValueError: Saved inputs or experiment locks fail validation.
        OSError: Required saved artifacts or outputs are inaccessible.
    """
    from .screening import load_screening_config
    root = Path(__file__).resolve().parents[1]
    cases = pd.read_csv(root/"outputs/data_preparation/split_manifest.csv", keep_default_na=False)
    baseline_folder = root/"outputs/baseline_linear"
    baseline = {"bundle": joblib.load(baseline_folder/"models/linear_regression.joblib"),
                "summary": json.loads((baseline_folder/"run_summary.json").read_text()),
                "training_pairs": pd.read_csv(baseline_folder/"training_pairs.csv", float_precision="round_trip"),
                "query_metrics": pd.read_csv(baseline_folder/"query_metrics.csv")}
    # Reuse saved stage-5 inputs so final evaluation cannot introduce a new selection.
    from .baseline import add_appearance
    pairs = pd.read_csv(root/"outputs/targeted_review/pair_features.csv", float_precision="round_trip")
    combined = add_appearance(cases, pairs, root, baseline["bundle"]["settings"])
    baseline["rankings"] = [combined[combined.split.eq(s)].copy() for s in ("train", "validation")]
    svr_folder = root/"outputs/svr_main"
    svr = {"bundle": joblib.load(svr_folder/"models/rbf_svr.joblib"),
           "summary": json.loads((svr_folder/"run_summary.json").read_text()),
           "selection": json.loads((root/"config/svr_selection.json").read_text()),
           "trials": pd.read_csv(svr_folder/"search_results.csv")}
    cfg = load_config(root)
    resnet = None
    if cfg["include_resnet_test"]:
        folder = root/"outputs/resnet_matching"
        resnet = {"selection": json.loads((root/"config/resnet_svr_selection.json").read_text()),
                  "bundles": {k: joblib.load(folder/f"models/{k}.joblib") for k in ("linear", "svr")}}
    final = run_final(root, {"cases": cases}, {"summary": {"settings": load_screening_config(root)}}, baseline, svr, cfg, resnet)
    print(final["comparison"].to_string(index=False))
    from .reproduce import export_run
    print(export_run(root, final, {"cases": cases}, baseline, svr, cfg))


if __name__ == "__main__":
    main()
