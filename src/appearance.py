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
    """Namespace appearance caches by histogram/preprocessing settings and source bytes.

    Args:
        settings (dict): image_size, crop, hue_bins, saturation_bins and depth_bins.

    Returns:
        str: First 20 hexadecimal characters of the settings/OpenCV/source digest.

    Raises:
        KeyError: A required setting is absent.
        OSError: This module cannot be hashed.
    """
    keys = ["image_size", "crop", "hue_bins", "saturation_bins", "depth_bins"]
    return stable_digest({"settings": {k: settings[k] for k in keys},
                          "opencv": cv2.__version__, "code": file_digest(Path(__file__))})[:20]


def extract_appearance(rgb_path, depth_path, settings: dict, cache_dir: Path) -> dict:
    """Cache HSV and nonzero-D summaries by source content.

    Missing, unreadable, unsupported or mismatched D is retained through flags and
    NaNs. All-zero D has zero fraction one and unavailable nonzero statistics. D
    uses nearest-neighbour resizing to preserve intensity values, never metres.

    Args:
        rgb_path (str | Path): Required readable RGB image.
        depth_path (str | Path | None): Optional D image matching the original RGB dimensions.
        settings (dict): Image size/crop and HSV/D histogram bin counts.
        cache_dir (Path): Writable NPZ cache directory, created when absent.

    Returns:
        dict: L1-normalized hsv and depth_hist arrays, D mean/std/zero fraction, missing_depth
        and depth_no_nonzero indicators. Cache hits may return zero-dimensional arrays for
        scalar entries.

    Raises:
        ValueError: RGB cannot be read or crop/size settings are invalid.
        OSError: RGB or cache files cannot be read or written.
        KeyError: A required setting is absent.
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
        # Nearest-neighbour sampling preserves the exported D-intensity categories.
        depth = cv2.resize(depth, (size, size), interpolation=cv2.INTER_NEAREST)
        valid = depth[depth != 0].astype(float)
        result.update(depth_zero_fraction=float(np.mean(depth == 0)), depth_no_nonzero=float(not len(valid)))
        if len(valid):
            dh, _ = np.histogram(valid, bins=settings["depth_bins"], range=(1, 256))
            result.update(depth_hist=dh / dh.sum(), depth_mean=float(valid.mean()), depth_std=float(valid.std()))
    np.savez_compressed(path, **result)
    return result


def hellinger(left, right) -> float:
    """Compare L1-normalized nonnegative histograms using Hellinger distance.

    Args:
        left (array-like): First normalized histogram.
        right (array-like): Compatible second normalized histogram.

    Returns:
        float: Distance from zero to one for valid distributions; NaN inputs propagate.

    Raises:
        ValueError: Histogram shapes cannot be broadcast together.
    """
    return float(np.sqrt(np.sum((np.sqrt(left) - np.sqrt(right)) ** 2) / 2))


def pair_appearance(query: dict, candidate: dict) -> dict:
    """Combine two image summaries into fixed-order appearance differences and flags.

    Args:
        query (dict): Summer summary returned by extract_appearance.
        candidate (dict): Winter summary returned by extract_appearance.

    Returns:
        dict: Nine APPEARANCE_COLUMNS values. Histogram distances are dimensionless; D mean/std
        differences use uint8 intensity units. Missing statistics remain NaN.

    Raises:
        KeyError: A required appearance summary field is absent.
    """
    return dict(zip(APPEARANCE_COLUMNS, [
        hellinger(query["hsv"], candidate["hsv"]),
        hellinger(query["depth_hist"], candidate["depth_hist"]),
        float(abs(query["depth_mean"] - candidate["depth_mean"])),
        float(abs(query["depth_std"] - candidate["depth_std"])),
        float(abs(query["depth_zero_fraction"] - candidate["depth_zero_fraction"])),
        float(query["missing_depth"]), float(candidate["missing_depth"]),
        float(query["depth_no_nonzero"]), float(candidate["depth_no_nonzero"])]))
