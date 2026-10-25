"""Opt-in ORB/CLAHE/heuristic-D and gallery-context features, train/val only.

The original src feature algorithms and their locked experiment results are preserved.
The optimized image inference API explicitly requires the complete gallery.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pandas as pd

from src.appearance import APPEARANCE_COLUMNS, hellinger, pair_appearance
from src.features import FEATURE_COLUMNS, ImageFeatures, preprocess_rgb, stable_digest
from src.screening import _worker_setup
from src.utils import file_digest, read_depth, read_rgb
from tools.matching_diagnostics import QUERY_ONLY, CANDIDATE_ONLY

KEYS = ["split", "query_id", "candidate_id"]
BASE_COLUMNS = FEATURE_COLUMNS+APPEARANCE_COLUMNS+["foreground_fraction_difference"]
SEASON_COLUMNS = ["season_hsv_residual_l2", "season_depth_mean_difference", "season_depth_std_difference",
                  "season_depth_zero_difference", "depth_rank_hellinger"]
LOWER_BETTER = {"hamming_mean", "hamming_median", "reprojection_rmse_px", "geometry_failed",
                *APPEARANCE_COLUMNS, "foreground_fraction_difference", *SEASON_COLUMNS}
RATIO_COLUMNS = ["good_matches", "good_match_ratio", "ransac_inliers", "ransac_inlier_ratio"]


def _save_npz(path, **values):
    """Publish a numeric cache atomically so interrupted writes cannot appear complete.

    Args:
        path (Path): Destination NPZ path in an existing directory.
        **values (numpy.ndarray): Named arrays passed to numpy.savez_compressed.

    Returns:
        None: A sibling pending file is replaced into the destination.

    Raises:
        OSError: The temporary file cannot be written or replaced.
    """
    temporary = path.with_name(path.stem+".pending.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def depth_rank_mask(depth, shape, settings):
    """Select a central D-intensity-rank band without assuming a physical near/far direction.

    Args:
        depth (numpy.ndarray | None): Same-size uint8 D intensities; zero denotes unavailable
            intensity.
        shape (tuple[int, int]): Required mask height and width.
        settings (dict): foreground_center_width, foreground_rank_band, foreground_min_fraction
            and foreground_max_fraction.

    Returns:
        tuple[numpy.ndarray, bool, numpy.ndarray]: Boolean mask, fallback flag, and per-pixel
        mid-CDF intensity ranks. Missing/poor D falls back to the full frame; unavailable
        ranks are NaN.

    Raises:
        ValueError: Array shapes cannot be combined.
    """
    full = np.ones(shape, dtype=bool)
    if depth is None or not np.any(depth):
        return full, True, np.full(shape, np.nan)
    histogram = np.bincount(depth[depth > 0], minlength=256).astype(float)
    cdf = (np.cumsum(histogram)-.5*histogram)/histogram.sum()
    ranks = np.where(depth > 0, cdf[depth], np.nan)
    h, w = shape
    seed = ranks[int(.30*h):int(.90*h), int(.45*w):max(int(.55*w), int(.45*w)+1)]
    seed = seed[np.isfinite(seed)]
    if len(seed) < 8:
        return full, True, ranks
    center = float(np.median(seed))
    horizontal = np.abs((np.arange(w)+.5)/w-.5) <= settings["foreground_center_width"]/2
    mask = np.isfinite(ranks) & (np.abs(ranks-center) <= settings["foreground_rank_band"]) & horizontal[None, :]
    # Closing fills tiny holes; intersect validity afterwards to retain nonzero D.
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    mask &= np.isfinite(ranks) & horizontal[None, :]
    fraction = float(mask.mean())
    if fraction < settings["foreground_min_fraction"] or fraction > settings["foreground_max_fraction"]:
        return full, True, ranks
    return mask, False, ranks


def image_signature(settings, preprocessing):
    """Fingerprint optimized preprocessing, image settings, OpenCV and source bytes.

    Args:
        settings (dict): ORB, crop/size, histogram, CLAHE and foreground-band settings.
        preprocessing (str): Declared original, clahe or clahe_depth mode.

    Returns:
        str: A 20-character cache namespace digest.

    Raises:
        KeyError: A required image setting is absent.
        OSError: Module source cannot be hashed.
    """
    keys = ["image_size", "crop", "orb_nfeatures", "hue_bins", "saturation_bins", "depth_bins",
            "clahe_clip_limit", "clahe_grid", "foreground_rank_band", "foreground_min_fraction",
            "foreground_max_fraction", "foreground_center_width"]
    return stable_digest({"settings": {k: settings[k] for k in keys}, "preprocessing": preprocessing,
                          "opencv": cv2.__version__, "source": file_digest(Path(__file__))})[:20]


def extract_image(rgb_path, depth_path, settings, preprocessing, cache):
    """Cache optimized ORB and appearance summaries while auditing foreground fallback.

    CLAHE changes ORB grayscale only. In clahe_depth mode, the heuristic mask also
    restricts appearance summaries. D intensities retain their original uint8 scale.

    Args:
        rgb_path (str | Path): Required readable RGB image.
        depth_path (str | Path | None): Optional compatible D image.
        settings (dict): Image/ORB/histogram/CLAHE/foreground settings.
        preprocessing (str): Declared original, clahe or clahe_depth mode.
        cache (str | Path): Writable numeric image-cache directory.

    Returns:
        tuple[ImageFeatures, dict]: Numeric keypoints/descriptors and HSV/D/rank histograms, D
        statistics, missing flags, foreground_fraction and foreground_fallback.

    Raises:
        ValueError: RGB or preprocessing size/crop is invalid.
        OSError: Source/cache bytes cannot be accessed.
    """
    rgb_path = Path(rgb_path)
    depth_path = Path(depth_path) if depth_path else None
    rgb_sha = file_digest(rgb_path)
    depth_sha = file_digest(depth_path) if depth_path and depth_path.is_file() else "missing"
    signature = image_signature(settings, preprocessing)
    key = stable_digest([signature, rgb_sha, depth_sha])
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache/f"{key}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            data = {name: saved[name].copy() for name in saved.files}
    else:
        original = read_rgb(rgb_path)
        rgb = preprocess_rgb(original, settings["image_size"], settings["crop"])
        depth, status = read_depth(str(depth_path) if depth_path else "", original.shape[:2])
        if depth is not None:
            if settings["crop"] == "center":
                h, w = depth.shape
                side = min(h, w)
                depth = depth[(h-side)//2:(h-side)//2+side, (w-side)//2:(w-side)//2+side]
            depth = cv2.resize(depth, (settings["image_size"], settings["image_size"]), interpolation=cv2.INTER_NEAREST)
        mask, fallback, ranks = depth_rank_mask(depth, rgb.shape[:2], settings)
        use_mask = preprocessing == "clahe_depth"
        if not use_mask:
            mask, fallback = np.ones(rgb.shape[:2], dtype=bool), False
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if preprocessing in {"clahe", "clahe_depth"}:
            gray = cv2.createCLAHE(clipLimit=settings["clahe_clip_limit"], tileGridSize=tuple(settings["clahe_grid"])).apply(gray)
        orb_mask = (mask.astype(np.uint8)*255) if use_mask else None
        points, descriptors = cv2.ORB_create(nfeatures=settings["orb_nfeatures"], WTA_K=2).detectAndCompute(gray, orb_mask)
        xy = np.asarray([p.pt for p in points], dtype=np.float32).reshape(-1, 2)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], mask.astype(np.uint8)*255,
                                 [settings["hue_bins"], settings["saturation_bins"]], [0, 180, 0, 256]).ravel().astype(float)
        histogram /= histogram.sum()
        data = {"xy": xy, "descriptors": descriptors if descriptors is not None else np.empty((0, 32), np.uint8),
                "hsv": histogram, "depth_hist": np.full(settings["depth_bins"], np.nan),
                "depth_rank_hist": np.full(settings["depth_bins"], np.nan),
                "depth_mean": np.nan, "depth_std": np.nan, "depth_zero_fraction": np.nan,
                "missing_depth": float(depth is None), "depth_no_nonzero": 0.,
                "foreground_fraction": float(mask.mean()), "foreground_fallback": float(fallback)}
        if depth is not None:
            region = depth[mask]
            values = region[region > 0].astype(float)
            data.update(depth_zero_fraction=float(np.mean(region == 0)), depth_no_nonzero=float(not len(values)))
            if len(values):
                dh = np.histogram(values, settings["depth_bins"], range=(1, 256))[0]
                rh = np.histogram(ranks[mask & (depth > 0)], settings["depth_bins"], range=(0, 1))[0]
                data.update(depth_hist=dh/dh.sum(), depth_rank_hist=rh/rh.sum(),
                            depth_mean=float(values.mean()), depth_std=float(values.std()))
        _save_npz(path, **data)
    image = ImageFeatures(key, data.pop("xy"), data.pop("descriptors"))
    return image, data


def pair_orb(query, candidate, ratios, settings):
    """Reuse one Hamming neighbour search across declared ratio/RANSAC filters.

    Args:
        query (ImageFeatures): Query keypoints/descriptors.
        candidate (ImageFeatures): Candidate keypoints/descriptors.
        ratios (sequence[float]): Ratio thresholds in desired output order.
        settings (dict): Affine-RANSAC pixel threshold, iteration limit and confidence.

    Returns:
        list[dict]: Twelve ORB/geometric values per ratio, retaining NaNs and
        missing/failed-geometry flags. Coordinates/errors use resized pixels.

    Raises:
        cv2.error: Descriptor or RANSAC options are incompatible.
    """
    nq, nc = len(query.descriptors), len(candidate.descriptors)
    neighbours = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(query.descriptors, candidate.descriptors, k=2) if nq and nc >= 2 else []
    outputs = []
    for ratio in ratios:
        values = dict.fromkeys(FEATURE_COLUMNS, 0.)
        values.update(query_keypoints=nq, candidate_keypoints=nc, hamming_mean=np.nan,
                      hamming_median=np.nan, reprojection_rmse_px=np.nan, geometry_failed=1.,
                      missing_query_descriptors=float(nq == 0), missing_candidate_descriptors=float(nc == 0))
        accepted = [p[0] for p in neighbours if len(p) == 2 and p[0].distance < ratio*p[1].distance]
        matches, targets = [], set()
        for match in sorted(accepted, key=lambda m: (m.distance, m.queryIdx, m.trainIdx)):
            if match.trainIdx not in targets:
                matches.append(match); targets.add(match.trainIdx)
        values.update(good_matches=len(matches), good_match_ratio=len(matches)/nq if nq else 0.)
        if matches:
            distances = np.array([m.distance for m in matches])
            values.update(hamming_mean=float(distances.mean()), hamming_median=float(np.median(distances)))
        if len(matches) >= 3:
            source = query.xy[[m.queryIdx for m in matches]].astype(float)
            target = candidate.xy[[m.trainIdx for m in matches]].astype(float)
            if np.linalg.matrix_rank(np.c_[source, np.ones(len(source))]) >= 3 and np.linalg.matrix_rank(np.c_[target, np.ones(len(target))]) >= 3:
                transform, mask = cv2.estimateAffine2D(source, target, method=cv2.RANSAC,
                    ransacReprojThreshold=settings["ransac_threshold_px"], maxIters=settings["ransac_max_iters"],
                    confidence=settings["ransac_confidence"], refineIters=10)
                if transform is not None and mask is not None and np.isfinite(transform).all() and mask.any():
                    inside = mask.ravel().astype(bool)
                    errors = np.linalg.norm(np.c_[source, np.ones(len(source))]@transform.T-target, axis=1)
                    values.update(ransac_inliers=int(inside.sum()), ransac_inlier_ratio=float(inside.mean()),
                                  reprojection_rmse_px=float(np.sqrt(np.mean(errors[inside]**2))), geometry_failed=0.)
        outputs.append(values)
    return outputs


def fit_season_calibration(records):
    """Fit separate seasonal appearance means/scales using training images only.

    Args:
        records (pandas.DataFrame): split, lowercase season and appearance-summary dictionaries.

    Returns:
        dict: summer/winter mean and scale arrays plus training image counts. Missing columns
        use zero mean/unit scale; near-constant scales become one.

    Raises:
        ValueError: Training records are empty or do not contain exactly both seasons.
    """
    training = records[records.split.eq("train")]
    if training.empty or set(training.season) != {"summer", "winter"}:
        raise ValueError("Need both training seasons for appearance calibration")
    calibration = {}
    for season, group in training.groupby("season"):
        vectors = np.stack([np.r_[a["hsv"], a["depth_mean"], a["depth_std"], a["depth_zero_fraction"]] for a in group.appearance])
        count = np.isfinite(vectors).sum(axis=0)
        mean = np.divide(np.nansum(vectors, axis=0), count, out=np.zeros(vectors.shape[1]), where=count > 0)
        variance = np.divide(np.nansum((vectors-mean)**2, axis=0), count, out=np.ones(vectors.shape[1]), where=count > 0)
        scale = np.sqrt(variance)
        scale[scale < 1e-6] = 1.
        calibration[season] = {"mean": mean, "scale": scale, "images": len(group)}
    return calibration


def season_vector(appearance, season, calibration):
    """Standardize HSV/D summaries using the saved training-season calibration.

    Args:
        appearance (dict): hsv, depth_mean, depth_std and depth_zero_fraction.
        season (str): "summer" or "winter" calibration key.
        calibration (dict): Training-fitted seasonal means/scales.

    Returns:
        numpy.ndarray: Standardized HSV entries followed by three D statistics; missing values
        propagate.

    Raises:
        KeyError: A season or required summary field is absent.
    """
    vector = np.r_[appearance["hsv"], appearance["depth_mean"], appearance["depth_std"], appearance["depth_zero_fraction"]]
    return (vector-calibration[season]["mean"])/calibration[season]["scale"]


def pair_season(query, candidate, calibration):
    """Compare seasonal residuals and nonzero-D percentile histograms.

    Args:
        query (dict): Summer appearance including depth_rank_hist.
        candidate (dict): Winter appearance including depth_rank_hist.
        calibration (dict): Saved training-only seasonal calibration.

    Returns:
        dict: Five SEASON_COLUMNS values: HSV residual RMS, three standardized D differences and
        rank-histogram Hellinger distance.

    Raises:
        KeyError: A required summary/calibration field is absent.
    """
    q, c = season_vector(query, "summer", calibration), season_vector(candidate, "winter", calibration)
    hsv = len(query["hsv"])
    return dict(zip(SEASON_COLUMNS, [float(np.sqrt(np.mean((q[:hsv]-c[:hsv])**2))),
        float(abs(q[-3]-c[-3])), float(abs(q[-2]-c[-2])), float(abs(q[-1]-c[-1])),
        hellinger(query["depth_rank_hist"], candidate["depth_rank_hist"])]))


def build_development(root, cases, settings, preprocessing):
    """Extract complete development galleries and train-only seasonal calibration.

    Test inputs are rejected before image access. Numeric caches preserve each full
    gallery and reuse the shared neighbour search for all declared ratio thresholds.

    Args:
        root (str | Path): Base for relative image/cache paths.
        cases (pandas.DataFrame): Unique train/validation identities, rows and seasonal RGB/D
            paths.
        settings (dict): Optimized image, ratio, RANSAC, seed, worker and cache settings.
        preprocessing (str): Declared image-preprocessing mode.

    Returns:
        dict: frames keyed by ratio string, calibration, audit, elapsed seconds, cached images
        and preprocessing name.

    Raises:
        ValueError: Cases contain test/duplicate/overlapping rows, seasonal training coverage is
            absent, or a cache gallery changed.
        OSError: Images or caches are inaccessible.
    """
    if (cases.empty or not set(cases.split).issubset({"train", "validation"}) or
            cases.tree_id.duplicated().any()):
        raise ValueError("Optimized feature construction accepts development identities only")
    if set(cases.loc[cases.split.eq("train"), "row"]) & set(cases.loc[cases.split.eq("validation"), "row"]):
        raise ValueError("Orchard rows cross development splits")
    root = Path(root)
    cache = root/settings["cache_dir"]/image_signature(settings, preprocessing)
    cache.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1); cv2.setRNGSeed(settings["seed"])
    images, records = {}, []
    started = perf_counter()
    for row in cases.sort_values("tree_id").itertuples():
        for season in ("summer", "winter"):
            depth = getattr(row, season+"_depth")
            image, appearance = extract_image(root/getattr(row, season+"_rgb"), root/depth if depth else "", settings, preprocessing, cache/"images")
            images[row.tree_id, season] = (image, appearance)
            records.append({"tree_id": row.tree_id, "row": row.row, "split": row.split, "season": season,
                            "appearance": appearance, "keypoints": len(image.xy), "image_hash": image.image_hash})
    records = pd.DataFrame(records)
    calibration = fit_season_calibration(records)
    calibration_hash = stable_digest({s: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in a.items()} for s, a in calibration.items()})
    groups = {s: g.sort_values("tree_id").tree_id.tolist() for s, g in cases.groupby("split")}
    gallery_hash = {s: stable_digest([(i, images[i, "winter"][0].image_hash) for i in ids]) for s, ids in groups.items()}
    namespace = cache/stable_digest([settings["ratios"], settings["ransac_threshold_px"], settings["ransac_max_iters"],
                                   settings["ransac_confidence"], calibration_hash])[:20]
    namespace.mkdir(exist_ok=True)
    columns = BASE_COLUMNS+SEASON_COLUMNS

    def one_query(record):
        split, query_id = record
        ids = groups[split]
        q, qa = images[query_id, "summer"]
        key = stable_digest([query_id, q.image_hash, gallery_hash[split]])
        path = namespace/f"{split}_{key}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as saved:
                if saved["candidates"].tolist() != ids:
                    raise ValueError("Cached gallery changed")
                return split, query_id, saved["features"].copy()
        matrix = np.empty((len(settings["ratios"]), len(ids), len(columns)))
        for j, identity in enumerate(ids):
            c, ca = images[identity, "winter"]
            orb = pair_orb(q, c, settings["ratios"], settings)
            shared = {**pair_appearance(qa, ca), **pair_season(qa, ca, calibration),
                      "foreground_fraction_difference": abs(float(qa["foreground_fraction"])-float(ca["foreground_fraction"]))}
            for r, values in enumerate(orb):
                values.update(shared)
                matrix[r, j] = [values[c] for c in columns]
        _save_npz(path, candidates=np.asarray(ids), features=matrix)
        return split, query_id, matrix

    queries = [(r.split, r.tree_id) for r in cases.sort_values(["split", "tree_id"]).itertuples()]
    frames = [[] for _ in settings["ratios"]]
    print(f"Preprocessing={preprocessing}: {len(images)} images; {len(queries)} full-gallery queries", flush=True)
    with ThreadPoolExecutor(max_workers=settings["workers"], initializer=_worker_setup, initargs=(settings["seed"],)) as executor:
        for i, (split, query_id, matrix) in enumerate(executor.map(one_query, queries), 1):
            ids = pd.DataFrame({"split": split, "query_id": query_id, "candidate_id": groups[split]})
            for ratio, lists in enumerate(frames):
                lists.append(pd.concat([ids, pd.DataFrame(matrix[ratio], columns=columns)], axis=1))
            if i % 25 == 0 or i == len(queries):
                print(f"{preprocessing}: {i}/{len(queries)} queries", flush=True)
    audits = records.drop(columns="appearance").copy()
    for field in ("foreground_fraction", "foreground_fallback", "depth_zero_fraction", "missing_depth"):
        audits[field] = records.appearance.map(lambda a: float(a[field]))
    return {"frames": {str(r): pd.concat(f, ignore_index=True) for r, f in zip(settings["ratios"], frames)},
            "calibration": calibration, "audit": audits, "seconds": perf_counter()-started,
            "images": images, "preprocessing": preprocessing}


def validate_complete_context(frame):
    """Require unique pairs and complete candidate coverage before gallery context.

    Args:
        frame (pandas.DataFrame): split/query_id/candidate_id pair rows for the full intended
            gallery.

    Returns:
        None: Every query has all candidates present in its split.

    Raises:
        ValueError: The frame is empty, pairs repeat or candidate coverage is incomplete.
    """
    if frame.empty or frame.duplicated(KEYS).any():
        raise ValueError("Query context needs non-empty unique complete galleries")
    for split, data in frame.groupby("split"):
        candidates = set(data.candidate_id)
        if not data.groupby("query_id").size().eq(len(candidates)).all():
            raise ValueError("Compute context on the full gallery, before sampling training pairs")


def representation_columns(name):
    """Resolve a declared feature representation and whether it uses gallery context.

    Args:
        name (str): raw_all, without_query, pair_only, query_relative or relative_season.

    Returns:
        tuple[list[str], bool]: Ordered raw feature names and whether percentile/z-score/ratio
        context is appended.

    Raises:
        ValueError: The representation name is unsupported.
    """
    if name == "raw_all":
        return list(BASE_COLUMNS), False
    if name == "without_query":
        return [c for c in BASE_COLUMNS if c not in QUERY_ONLY], False
    columns = [c for c in BASE_COLUMNS if c not in QUERY_ONLY+CANDIDATE_ONLY]
    if name == "pair_only":
        return columns, False
    if name == "relative_season":
        return columns+SEASON_COLUMNS, True
    if name == "query_relative":
        return columns, True
    raise ValueError(f"Unknown representation: {name}")


def context_features(frame, name, fills):
    """Build label-free gallery context using saved global missing-value fills.

    The full gallery must be transformed before sampled training pairs are selected.
    Changing inference candidates can change these features and the resulting scores.

    Args:
        frame (pandas.DataFrame): Complete galleries containing required base/season columns.
        name (str): Declared feature representation.
        fills (dict): Per-feature medians fitted on original sampled training pairs.

    Returns:
        tuple[pandas.DataFrame, list[str]]: Pair metadata plus numeric features, and exact
        feature order. Relative modes append oriented percentiles, clipped z-scores and
        bounded best/second-best ratios.

    Raises:
        ValueError: The gallery is incomplete or representation is unknown.
        KeyError: Required base/season columns are absent.
    """
    validate_complete_context(frame)
    columns, relative = representation_columns(name)
    result = frame[KEYS].copy()
    raw = frame[columns].astype(float).replace([np.inf, -np.inf], np.nan).fillna(fills)
    for column in columns:
        result[column] = raw[column]
    if relative:
        groups = raw.groupby([frame.split, frame.query_id], sort=False)
        count = groups[columns[0]].transform("count")
        for column in columns:
            sign = -1 if column in LOWER_BETTER else 1
            values = raw[column]*sign
            group = values.groupby([frame.split, frame.query_id], sort=False)
            ranks = group.rank(method="average", ascending=True)
            result[column+"__percentile"] = np.where(count > 1, (ranks-1)/(count-1), .5)
            std = group.transform("std", ddof=0)
            result[column+"__zscore"] = ((values-group.transform("mean"))/std.where(std > 1e-12)).fillna(0).clip(-8, 8)
        for column in RATIO_COLUMNS:
            values = raw[column].clip(lower=0)
            group = values.groupby([frame.split, frame.query_id], sort=False)
            maximum = group.transform("max")
            second = group.transform(lambda v: v.nlargest(min(2, len(v))).iloc[-1])
            result[column+"__over_best"] = np.divide(values, maximum, out=np.zeros(len(frame)), where=maximum > 0)
            denominator = second+maximum*1e-6
            # Cap ratios when the second-best evidence is near zero to limit instability.
            result[column+"__over_second"] = np.divide(values, denominator, out=np.zeros(len(frame)), where=denominator > 0).clip(0, 10)
    return result, [c for c in result if c not in KEYS]


def training_fills(frame, sample):
    """Learn missing-value medians on original sampled training pairs only.

    Args:
        frame (pandas.DataFrame): Complete development base/season feature table.
        sample (pandas.DataFrame): Original split/query_id/candidate_id sample keys.

    Returns:
        dict: Median fills for BASE_COLUMNS and SEASON_COLUMNS, using zero for all-missing
        columns.

    Raises:
        ValueError: An original training pair is absent or merge keys are duplicated.
    """
    # Fit global fills on sampled pairs, then apply them to the complete gallery context.
    train = frame[frame.split.eq("train")]
    selected = sample[KEYS].merge(train, on=KEYS, how="left", validate="one_to_one", indicator=True)
    if not selected._merge.eq("both").all():
        raise ValueError("Missing original training pair")
    return selected[BASE_COLUMNS+SEASON_COLUMNS].median().fillna(0).to_dict()


def match_tree_optimized(query, gallery, bundle, root):
    """Rank an image query with saved calibration and complete-gallery context.

    Args:
        query (dict): Required rgb path and optional depth path.
        gallery (pandas.DataFrame): Complete intended candidate_id/rgb gallery with unique IDs
            and optional depth.
        bundle (dict): Trusted optimized pipeline, settings, fills, calibration, preprocessing,
            ratio, representation and feature order.
        root (str | Path): Base for relative image/cache paths.

    Returns:
        pandas.DataFrame: Every candidate with continuous score and one-based display/tie ranks.
        Supplying fewer candidates changes the context and may change scores.

    Raises:
        ValueError: Gallery IDs, RGB data or saved feature order are invalid.
        OSError: Images or cache files are inaccessible.
    """
    from src.baseline import rank_candidates
    if gallery.empty or gallery.candidate_id.duplicated().any():
        raise ValueError("Need a complete gallery with unique candidate IDs")
    root = Path(root)
    cfg = bundle["settings"]
    cache = root/cfg["cache_dir"]/"inference"
    def image(row):
        depth = row.get("depth", "")
        return extract_image(root/row["rgb"], root/depth if depth else "", cfg, bundle["preprocessing"], cache)
    q, qa = image(query)
    rows = []
    for candidate in gallery.to_dict("records"):
        c, ca = image(candidate)
        orb = pair_orb(q, c, [bundle["ratio"]], cfg)[0]
        rows.append({"split": "inference", "query_id": "query", "candidate_id": candidate["candidate_id"], **orb,
                     **pair_appearance(qa, ca), **pair_season(qa, ca, bundle["calibration"]),
                     "foreground_fraction_difference": abs(float(qa["foreground_fraction"])-float(ca["foreground_fraction"]))})
    frame, columns = context_features(pd.DataFrame(rows), bundle["representation"], bundle["fills"])
    if columns != bundle["feature_columns"]:
        raise ValueError("Optimized inference feature order changed")
    scores = bundle["pipeline"].predict(frame[columns])
    return rank_candidates(frame[["candidate_id"]].assign(score=scores), cfg["score_tie_atol"], cfg["score_tie_rtol"])
