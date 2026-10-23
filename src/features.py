"""Reusable ORB and affine-RANSAC pair features; no learned prediction model."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .utils import file_digest, read_rgb

FEATURE_COLUMNS = [
    "query_keypoints", "candidate_keypoints", "good_matches", "good_match_ratio",
    "hamming_mean", "hamming_median", "ransac_inliers", "ransac_inlier_ratio",
    "reprojection_rmse_px", "missing_query_descriptors", "missing_candidate_descriptors", "geometry_failed",
]


def stable_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def feature_signature(settings: dict) -> str:
    """Invalidate cached descriptors/pair features on code, OpenCV or setting changes."""
    keys = ["image_size", "crop", "orb_nfeatures", "ratio_threshold", "ransac_threshold_px",
            "ransac_max_iters", "ransac_confidence", "seed"]
    return stable_digest({"settings": {k: settings[k] for k in keys}, "opencv": cv2.__version__,
                          "code_sha256": file_digest(Path(__file__))})[:20]


def preprocess_rgb(rgb: np.ndarray, size: int = 256, crop: str = "center") -> np.ndarray:
    """Center-square crop (480×480 for 640×480 RGB), then resize to size×size."""
    if size < 32 or crop not in {"center", "full"}:
        raise ValueError("size must be >= 32 and crop must be center/full")
    if crop == "center":
        height, width = rgb.shape[:2]
        side = min(height, width)
        top, left = (height - side) // 2, (width - side) // 2
        rgb = rgb[top:top + side, left:left + side]
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


@dataclass
class ImageFeatures:
    image_hash: str
    xy: np.ndarray
    descriptors: np.ndarray


def extract_image_features(path: str | Path, settings: dict, cache_dir: Path) -> ImageFeatures:
    """Cache numeric arrays only (no pickle); identity metadata never enters descriptors."""
    digest = file_digest(Path(path))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{feature_signature(settings)}_{digest}.npz"
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as stored:
            return ImageFeatures(digest, stored["xy"], stored["descriptors"])
    rgb = preprocess_rgb(read_rgb(path), settings["image_size"], settings["crop"])
    orb = cv2.ORB_create(nfeatures=settings["orb_nfeatures"], WTA_K=2)
    keypoints, descriptors = orb.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None)
    xy = np.asarray([k.pt for k in keypoints], dtype=np.float32).reshape(-1, 2)
    if descriptors is None:
        descriptors = np.empty((0, 32), dtype=np.uint8)
    np.savez_compressed(cache, xy=xy, descriptors=descriptors)
    return ImageFeatures(digest, xy, descriptors)


def extract_pair_features(query: ImageFeatures, candidate: ImageFeatures, settings: dict) -> dict:
    """Return fixed-order features, retaining missing descriptors/failed geometry.

    Hamming values are bit counts (0–256); coordinates/errors use resized pixels.
    Matches pass the forward 2-neighbour ratio test and a unique-target filter.
    NaN denotes unavailable distance/error, to be imputed on training data later.
    """
    nq, nc = len(query.descriptors), len(candidate.descriptors)
    result = dict.fromkeys(FEATURE_COLUMNS, 0.0)
    result.update(query_keypoints=nq, candidate_keypoints=nc,
                  hamming_mean=np.nan, hamming_median=np.nan, reprojection_rmse_px=np.nan,
                  missing_query_descriptors=float(nq == 0), missing_candidate_descriptors=float(nc == 0),
                  geometry_failed=1.0)
    if nq == 0 or nc < 2:
        return result
    neighbours = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(query.descriptors, candidate.descriptors, k=2)
    accepted = [pair[0] for pair in neighbours
                if len(pair) == 2 and pair[0].distance < settings["ratio_threshold"] * pair[1].distance]
    # Multiple source keypoints mapping to one target should not inflate evidence.
    matches, targets = [], set()
    for match in sorted(accepted, key=lambda m: (m.distance, m.queryIdx, m.trainIdx)):
        if match.trainIdx not in targets:
            matches.append(match)
            targets.add(match.trainIdx)
    result.update(good_matches=len(matches), good_match_ratio=len(matches) / nq)
    if matches:
        distances = np.array([m.distance for m in matches], dtype=np.float64)
        result.update(hamming_mean=float(distances.mean()), hamming_median=float(np.median(distances)))
    if len(matches) < 3:
        return result
    source = query.xy[[m.queryIdx for m in matches]].astype(np.float64)
    target = candidate.xy[[m.trainIdx for m in matches]].astype(np.float64)
    if (np.linalg.matrix_rank(np.c_[source, np.ones(len(source))]) < 3 or
            np.linalg.matrix_rank(np.c_[target, np.ones(len(target))]) < 3):
        return result
    transform, mask = cv2.estimateAffine2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=settings["ransac_threshold_px"],
        maxIters=settings["ransac_max_iters"], confidence=settings["ransac_confidence"], refineIters=10)
    if transform is None or mask is None or not np.isfinite(transform).all():
        return result
    inliers = mask.ravel().astype(bool)
    if not inliers.any():
        return result
    errors = np.linalg.norm(np.c_[source, np.ones(len(source))] @ transform.T - target, axis=1)
    result.update(ransac_inliers=int(inliers.sum()), ransac_inlier_ratio=float(inliers.mean()),
                  reprojection_rmse_px=float(np.sqrt(np.mean(errors[inliers] ** 2))), geometry_failed=0.0)
    return result


def evidence_scores(features, minimum_inliers: int = 6):
    """Heuristic for review ordering: inliers × inlier ratio, with a six-inlier floor.

    This is not a probability, identity label, regression model, or accuracy metric.
    The floor limits minimal affine fits being presented as strong evidence.
    """
    return np.where(features["ransac_inliers"] >= minimum_inliers,
                    features["ransac_inliers"] * features["ransac_inlier_ratio"], 0.0)
