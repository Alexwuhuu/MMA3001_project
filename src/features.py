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
    """Hash a canonical JSON representation for deterministic cache keys.

    Args:
        value (object): JSON-serializable settings or records; dictionary keys are sorted.

    Returns:
        str: SHA-256 hexadecimal digest of compact, UTF-8 JSON.

    Raises:
        TypeError: The value is not JSON-serializable.
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def feature_signature(settings: dict) -> str:
    """Namespace ORB caches by extraction settings, OpenCV version and source bytes.

    Args:
        settings (dict): image_size, crop, orb_nfeatures, ratio_threshold, RANSAC
            threshold/iterations/confidence and seed.

    Returns:
        str: First 20 hexadecimal characters of the configuration digest.

    Raises:
        KeyError: A required extraction setting is absent.
        OSError: This module cannot be read for hashing.
    """
    keys = ["image_size", "crop", "orb_nfeatures", "ratio_threshold", "ransac_threshold_px",
            "ransac_max_iters", "ransac_confidence", "seed"]
    return stable_digest({"settings": {k: settings[k] for k in keys}, "opencv": cv2.__version__,
                          "code_sha256": file_digest(Path(__file__))})[:20]


def preprocess_rgb(rgb: np.ndarray, size: int = 256, crop: str = "center") -> np.ndarray:
    """Crop and resize RGB consistently before descriptor or appearance extraction.

    Args:
        rgb (numpy.ndarray): Nonempty RGB image with shape (height, width, 3).
        size (int): Output height and width in pixels; must be at least 32.
        crop (str): "center" selects the largest centered square; "full" resizes the full frame.

    Returns:
        numpy.ndarray: Resized RGB image with shape (size, size, 3).

    Raises:
        ValueError: size is below 32 or crop is unsupported.
        cv2.error: The supplied image cannot be resized.
    """
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
    """Store numeric ORB observations without using identity metadata.

    Attributes:
        image_hash (str): Source-content digest, or extraction key for an optimized cache.
        xy (numpy.ndarray): Float32 keypoint positions with shape (N, 2), in resized-image
            pixels.
        descriptors (numpy.ndarray): uint8 ORB descriptors with shape (N, 32); an empty array
            represents no descriptors.
    """
    image_hash: str
    xy: np.ndarray
    descriptors: np.ndarray


def extract_image_features(path: str | Path, settings: dict, cache_dir: Path) -> ImageFeatures:
    """Load or compute ORB features in a content-addressed numeric cache.

    NPZ files are read with allow_pickle=False. Descriptor-free images remain valid
    observations with empty coordinate and descriptor arrays.

    Args:
        path (str | Path): RGB image to decode.
        settings (dict): Validated screening settings used by feature_signature.
        cache_dir (Path): Writable descriptor-cache directory, created if necessary.

    Returns:
        ImageFeatures: Image hash, resized keypoints and ORB descriptors.

    Raises:
        ValueError: RGB is unreadable or preprocessing settings are invalid.
        OSError: A source or cache file cannot be read or written.
        KeyError: A required extraction setting is absent.
    """
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
    """Measure descriptor and affine evidence while retaining failed geometry.

    Forward two-neighbour ratio filtering and unique-target matching avoid inflated
    evidence. Hamming distances use bit counts (0 to 256); reprojection errors use
    resized pixels. Unavailable distances/errors are NaN for later train-only filling.

    Args:
        query (ImageFeatures): Summer-query keypoints and descriptors.
        candidate (ImageFeatures): Winter-candidate keypoints and descriptors.
        settings (dict): Ratio threshold and affine-RANSAC threshold, maximum iterations and
            confidence.

    Returns:
        dict: The twelve FEATURE_COLUMNS values, including missing-descriptor and
        geometry_failed flags. Insufficient or degenerate correspondences return flags
        rather than dropping the pair.

    Raises:
        KeyError: A required matching setting is absent.
        cv2.error: Descriptors or RANSAC settings are incompatible with OpenCV.
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
    """Prioritize human review using inliers times inlier ratio above an evidence floor.

    The floor avoids presenting minimal affine fits as strong evidence. Scores
    provide review ordering only; they do not establish identities or probabilities.

    Args:
        features (pandas.DataFrame): Pair features containing ransac_inliers and
            ransac_inlier_ratio.
        minimum_inliers (int): Minimum inlier count receiving a nonzero score; defaults to six.

    Returns:
        numpy.ndarray: One float evidence score per feature row.

    Raises:
        KeyError: Required RANSAC columns are absent.
    """
    return np.where(features["ransac_inliers"] >= minimum_inliers,
                    features["ransac_inliers"] * features["ransac_inlier_ratio"], 0.0)
