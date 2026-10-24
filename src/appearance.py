"""Small, deterministic RGB/D summaries; D values are intensities, not metres."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from .features import preprocess_rgb, stable_digest
from .utils import file_digest, read_depth, read_rgb

APPEARANCE_COLUMNS = ["hsv_hellinger", "depth_hellinger", "depth_mean_difference",
                      "depth_std_difference", "depth_zero_fraction_difference",
                      "missing_query_depth", "missing_candidate_depth",
                      "query_depth_no_nonzero", "candidate_depth_no_nonzero"]


def appearance_signature(settings: dict) -> str:
    keys = ["image_size", "crop", "hue_bins", "saturation_bins", "depth_bins"]
    return stable_digest({"settings": {k: settings[k] for k in keys},
                          "opencv": cv2.__version__, "code": file_digest(Path(__file__))})[:20]


def extract_appearance(rgb_path, depth_path, settings: dict, cache_dir: Path) -> dict:
    """Cache per-image HSV and nonzero-D distributions using source content hashes.

    Missing, unreadable, unsupported, or mismatched D is flagged and retained.
    All-zero D has a known zero fraction but unavailable nonzero statistics.
    """
    rgb_path = Path(rgb_path)
    depth_path = Path(depth_path) if depth_path else None
    key = stable_digest([appearance_signature(settings), file_digest(rgb_path),
                         file_digest(depth_path) if depth_path and depth_path.is_file() else "missing"])
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as stored:
            return {name: stored[name].copy() for name in stored.files}
    original = read_rgb(rgb_path)
    rgb = preprocess_rgb(original, settings["image_size"], settings["crop"])
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [settings["hue_bins"], settings["saturation_bins"]],
                             [0, 180, 0, 256]).ravel().astype(float)
    histogram /= histogram.sum()
    depth, status = read_depth(str(depth_path) if depth_path else "", original.shape[:2])
    result = {"hsv": histogram, "depth_hist": np.full(settings["depth_bins"], np.nan),
              "depth_mean": np.nan, "depth_std": np.nan, "depth_zero_fraction": np.nan,
              "missing_depth": float(depth is None), "depth_no_nonzero": 0.0}
    if depth is not None:
        if settings["crop"] == "center":
            h, w = depth.shape
            side = min(h, w)
            depth = depth[(h-side)//2:(h-side)//2+side, (w-side)//2:(w-side)//2+side]
        size = settings["image_size"]
        depth = cv2.resize(depth, (size, size), interpolation=cv2.INTER_NEAREST)
        valid = depth[depth != 0].astype(float)
        result.update(depth_zero_fraction=float(np.mean(depth == 0)), depth_no_nonzero=float(not len(valid)))
        if len(valid):
            dh, _ = np.histogram(valid, bins=settings["depth_bins"], range=(1, 256))
            result.update(depth_hist=dh / dh.sum(), depth_mean=float(valid.mean()), depth_std=float(valid.std()))
    np.savez_compressed(path, **result)
    return result


def hellinger(left, right) -> float:
    """Hellinger distance for L1-normalized nonnegative histograms (range 0–1)."""
    return float(np.sqrt(np.sum((np.sqrt(left) - np.sqrt(right)) ** 2) / 2))


def pair_appearance(query: dict, candidate: dict) -> dict:
    return dict(zip(APPEARANCE_COLUMNS, [
        hellinger(query["hsv"], candidate["hsv"]),
        hellinger(query["depth_hist"], candidate["depth_hist"]),
        float(abs(query["depth_mean"] - candidate["depth_mean"])),
        float(abs(query["depth_std"] - candidate["depth_std"])),
        float(abs(query["depth_zero_fraction"] - candidate["depth_zero_fraction"])),
        float(query["missing_depth"]), float(candidate["missing_depth"]),
        float(query["depth_no_nonzero"]), float(candidate["depth_no_nonzero"])]))
