"""Read-only dataset inspection and identity-safe row splits for notebook parts 1–2.

R = orchard row, N = matched tree number, U = unmatched observation.
D intensities have no established metric scale. No model training is performed.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

FILE_PATTERN = re.compile(
    r"^(?P<row>R\d+)(?P<kind>[NU])(?P<number>\d+)_"
    r"(?P<season>Summer|Winter)_(?P<modality>RGB|D|GT)-(?P<suffix>.+)\.png$",
    re.IGNORECASE,
)



def write_json(path: Path, value: Any) -> None:
    """Write strict UTF-8 JSON, creating parent directories as needed.

    Args:
        path (Path): Destination JSON file.
        value (Any): JSON-serializable data; replace non-finite numbers with None.

    Returns:
        None: The destination is created or overwritten.

    Raises:
        ValueError: A value contains NaN or infinity.
        TypeError: A value cannot be serialized as JSON.
        OSError: The destination cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def file_digest(path: Path) -> str:
    """Hash a file in blocks without loading its complete contents into memory.

    Args:
        path (Path): Existing file to read in binary mode.

    Returns:
        str: Lowercase, 64-character SHA-256 hexadecimal digest.

    Raises:
        OSError: The file is missing or cannot be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_review_csv(path: str | Path) -> pd.DataFrame:
    """Read a review ledger without changing its original encoding or bytes.

    Args:
        path (str | Path): CSV encoded as UTF-8, UTF-8 with BOM, or legacy GB18030.

    Returns:
        pandas.DataFrame: Parsed records; empty fields remain empty strings.

    Raises:
        OSError: The source cannot be read.
        UnicodeDecodeError: Neither supported encoding can decode the source.
        pandas.errors.ParserError: The decoded CSV is malformed.
    """
    raw = Path(path).read_bytes()
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = raw.decode("gb18030")
    return pd.read_csv(io.StringIO(content), keep_default_na=False)


def parse_filename(name: str) -> dict | None:
    """Parse the leading TreeID identity rather than an optional historical suffix.

    Args:
        name (str): PNG basename using row, N/U number, season, modality and suffix.

    Returns:
        dict | None: Normalized row/kind/season/modality, integer number and raw suffix, or None
        for an unsupported name.

    Examples:
        >>> parse_filename("R04N01_Summer_RGB-old.png")["row"]
        'R04'
        >>> parse_filename("satellite.png") is None
        True
    """
    match = FILE_PATTERN.match(name)
    if match is None:
        return None
    result = match.groupdict()
    result.update(row=result["row"].upper(), kind=result["kind"].upper(),
                  number=int(result["number"]), season=result["season"].title(),
                  modality=result["modality"].upper())
    return result


def read_rgb(path: str | Path) -> np.ndarray:
    """Decode an image to an independent RGB array.

    Args:
        path (str | Path): Image file supported by Pillow.

    Returns:
        numpy.ndarray: Shape (height, width, 3), dtype uint8, RGB channel order.

    Raises:
        ValueError: The image is missing, corrupt, unreadable, or smaller than two pixels in
            either dimension.
    """
    try:
        with Image.open(path) as image:
            image.load()
            if image.width < 2 or image.height < 2:
                raise ValueError("image is too small")
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except Exception as exc:
        raise ValueError(f"Cannot read RGB image {path}: {exc}") from exc


def read_depth(path: str | Path | None, shape: tuple[int, int]) -> tuple[np.ndarray | None, str]:
    """Read a compatible D-intensity image and explain unavailable data.

    Zero intensity is retained; no conversion to physical distance is assumed.

    Args:
        path (str | Path | None): Optional D image; empty or missing paths are permitted.
        shape (tuple[int, int]): Required (height, width) matching the original RGB image.

    Returns:
        tuple[numpy.ndarray | None, str]: A uint8 2D array and "ok", or None with missing_depth,
        unsupported_depth_dtype, unsupported_coloured_depth, depth_shape_mismatch, or
        corrupt_depth. Three identical grayscale channels are accepted.
    """
    if not path or not Path(path).is_file():
        return None, "missing_depth"
    try:
        with Image.open(path) as image:
            image.load()
            array = np.asarray(image).copy()
        if array.dtype != np.uint8:
            return None, "unsupported_depth_dtype"
        if array.ndim == 3:
            if array.shape[2] != 3 or not (np.array_equal(array[..., 0], array[..., 1]) and
                                                         np.array_equal(array[..., 1], array[..., 2])):
                return None, "unsupported_coloured_depth"
            array = array[..., 0]
        if array.ndim != 2 or array.shape != shape:
            return None, "depth_shape_mismatch"
        return array, "ok"
    except Exception:
        return None, "corrupt_depth"


def index_dataset(root: str | Path) -> pd.DataFrame:
    """Audit TreeID observations, retaining invalid images and unmatched U records.

    One record represents a subset/identity/season. GT masks are inspected only for
    quality reporting and never establish a matched identity.

    Args:
        root (str | Path): Dataset directory recursively containing seasonal PNG files.

    Returns:
        pandas.DataFrame: Sorted observations with identity metadata, RGB/D/GT paths,
        dimensions, content hashes, validity flags and quality statuses.

    Raises:
        FileNotFoundError: The dataset directory does not exist.
        ValueError: No supported observations exist or a modality occurs twice for an
            observation.
        OSError: A source file cannot be read for hashing.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    records: dict[tuple, dict] = {}
    for path in sorted(root.rglob("*.png")):
        info = parse_filename(path.name)
        # Static resources such as satellite images do not represent seasonal observations.
        if info is None:
            continue
        subset = path.parent.relative_to(root).as_posix()
        key = (subset, info["row"], info["kind"], info["number"], info["season"])
        record = records.setdefault(key, {
            "subset": subset, "row": info["row"], "kind": info["kind"],
            "number": info["number"], "season": info["season"],
            "tree_id": f'{info["row"]}{info["kind"]}{info["number"]:02d}',
            "rgb_path": "", "depth_path": "", "gt_path": "", "special_suffix": False,
        })
        field = {"RGB": "rgb_path", "D": "depth_path", "GT": "gt_path"}[info["modality"]]
        if record[field]:
            raise ValueError(f"Duplicate {field} for {key}: {record[field]} and {path}")
        record[field] = str(path)
        record["special_suffix"] |= bool(re.search(r"[A-Za-z]", info["suffix"]))
    if not records:
        raise ValueError("No supported TreeID observations were found")
    for record in records.values():
        record.update(valid_rgb=False, rgb_status="missing_rgb", depth_status="not_checked",
                      width=0, height=0, rgb_sha256="", depth_sha256="",
                      depth_zero_fraction=np.nan, gt_status="not_provided",
                      gt_width=0, gt_height=0, gt_unique_values="")
        if record["gt_path"]:
            try:
                with Image.open(record["gt_path"]) as gt_image:
                    gt_image.load()
                    gt = np.asarray(gt_image)
                    record.update(gt_status="ok", gt_width=gt_image.width, gt_height=gt_image.height,
                                  gt_unique_values=",".join(map(str, np.unique(gt).tolist())))
            except Exception:
                record["gt_status"] = "corrupt_gt"
        if not record["rgb_path"]:
            continue
        try:
            rgb = read_rgb(record["rgb_path"])
            record.update(valid_rgb=True, rgb_status="ok", height=rgb.shape[0], width=rgb.shape[1],
                          rgb_sha256=file_digest(Path(record["rgb_path"])))
            depth, record["depth_status"] = read_depth(record["depth_path"], rgb.shape[:2])
            if depth is not None:
                record["depth_zero_fraction"] = float(np.mean(depth == 0))
            if record["depth_path"]:
                record["depth_sha256"] = file_digest(Path(record["depth_path"]))
        except ValueError:
            record["rgb_status"] = "corrupt_rgb"
    return pd.DataFrame(records.values()).sort_values(["subset", "row", "number", "season"]).reset_index(drop=True)


def matched_cases(index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build filename-based N pairs and record missing or corrupt seasonal exclusions.

    Args:
        index (pandas.DataFrame): Observation table from index_dataset.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]: Eligible cases with tree_id, row, number and
        seasonal RGB/D paths, plus exclusions with subset, tree_id and reason. U
        observations do not enter either N-pair table.

    Raises:
        ValueError: No valid N pairs remain or an N identity occurs in multiple subsets.
    """
    cases, excluded = [], []
    selected = index.loc[index["kind"].eq("N")]
    for (subset, tree_id), group in selected.groupby(["subset", "tree_id"], sort=True):
        seasons = {r["season"]: r for r in group.to_dict("records")}
        if set(seasons) != {"Summer", "Winter"} or not all(r["valid_rgb"] for r in seasons.values()):
            excluded.append({"subset": subset, "tree_id": tree_id, "reason": "missing_or_corrupt_season_rgb"})
            continue
        summer, winter = seasons["Summer"], seasons["Winter"]
        cases.append({"subset": subset, "tree_id": tree_id, "row": summer["row"], "number": summer["number"],
                      "summer_rgb": summer["rgb_path"], "summer_depth": summer["depth_path"],
                      "winter_rgb": winter["rgb_path"], "winter_depth": winter["depth_path"]})
    frame = pd.DataFrame(cases)
    if frame.empty:
        raise ValueError("No valid N-labelled Summer/Winter pairs")
    if frame["tree_id"].duplicated().any():
        raise ValueError("N identities occur in multiple subsets; select one proven matched subset")
    return frame.sort_values(["row", "number"]).reset_index(drop=True), pd.DataFrame(excluded, columns=["subset", "tree_id", "reason"])


def assign_splits(cases: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Assign whole orchard rows to deterministic 60/20/20 development and test splits.

    Args:
        cases (pandas.DataFrame): Eligible cases containing row, tree_id and summer_rgb.
        seed (int): Seed for the permutation of sorted orchard rows.

    Returns:
        pandas.DataFrame: A copy with split values train, validation or test; row and identity
        membership stays disjoint.

    Raises:
        ValueError: Fewer than five orchard rows are available.
        AssertionError: An internal split-isolation invariant fails.
    """
    rows = np.array(sorted(cases["row"].unique()))
    if len(rows) < 5:
        raise ValueError("At least five rows are needed for disjoint train/validation/test groups")
    rows = np.random.default_rng(seed).permutation(rows)
    n_train, n_val = int(0.6 * len(rows)), max(1, int(0.2 * len(rows)))
    mapping = {row: "train" if i < n_train else "validation" if i < n_train + n_val else "test"
               for i, row in enumerate(rows)}
    result = cases.copy()
    result["split"] = result["row"].map(mapping)
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        a, b = result[result["split"].eq(left)], result[result["split"].eq(right)]
        assert set(a["row"]).isdisjoint(b["row"])
        assert set(a["tree_id"]).isdisjoint(b["tree_id"])
        assert set(a["summer_rgb"]) .isdisjoint(b["summer_rgb"])
    return result
