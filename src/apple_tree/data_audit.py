

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

DATASET_FOLDERS = ("Dataset", "Dataset (With Summer GT)")
NAME_PATTERN = re.compile(
    r"^(?P<tree_id>R\d+[A-Z]\d+)_(?P<season>Summer|Winter)_"
    r"(?P<modality>RGB|D|GT)-(?P<capture>.+)\.png$"
)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def parse_filename(name: str) -> dict:

    match = NAME_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Unsupported image filename: {name}")
    result = match.groupdict()
    embedded = re.findall(r"R\d+[A-Z]\d+", result["capture"])
    result["embedded_tree_id"] = ";".join(embedded)
    result["needs_identity_review"] = bool(embedded)
    result["capture_format_unusual"] = not bool(
        re.fullmatch(r"\d{1,2}-\d{1,2}-\d{1,2}", result["capture"])
    )
    return result


def inspect_png(path: Path) -> dict:
    """Read basic PNG header information and a file hash."""
    raw = path.read_bytes()
    if len(raw) < 33 or raw[:8] != PNG_SIGNATURE or raw[12:16] != b"IHDR":
        raise ValueError("Invalid PNG header")
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    if width == 0 or height == 0:
        raise ValueError("Invalid image dimensions")
    return {
        "width": width, "height": height, "bit_depth": raw[24],
        "png_colour_type": raw[25], "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(), "png_integrity": "header_ok",
    }


def pair_records(records: list[dict]) -> tuple[list[dict], list[dict]]:

    groups = defaultdict(list)
    for record in records:
        groups[tuple(record[key] for key in ("dataset", "tree_id", "season", "capture"))].append(record)
    observations, issues = [], []
    for (dataset, tree_id, season, capture), group in sorted(groups.items()):
        by_modality = defaultdict(list)
        for record in group:
            by_modality[record["modality"]].append(record)
        row = {
            "dataset": dataset, "tree_id": tree_id, "physical_tree_id": "",
            "season": season, "capture_key": capture,
            "rgb_path": "", "depth_path": "", "gt_path": "",
            "embedded_tree_id": group[0]["embedded_tree_id"],
            "needs_identity_review": any(r["needs_identity_review"] for r in group),
            "annotation_alignment_verified": False, "metric_depth_verified": False,
            "split": "", "pairing_status": "ok",
        }
        for modality, column in (("RGB", "rgb_path"), ("D", "depth_path"), ("GT", "gt_path")):
            matches = by_modality[modality]
            if len(matches) == 1:
                row[column] = matches[0]["path"]
            if len(matches) > 1:
                row["pairing_status"] = "error"
                issues.append({"severity": "error", "code": "duplicate_modality", "path": matches[0]["path"], "detail": f"{modality}: {len(matches)} matches"})
            required = modality in ("RGB", "D") or (modality == "GT" and dataset == DATASET_FOLDERS[1] and season == "Summer")
            if required and not matches:
                row["pairing_status"] = "error"
                issues.append({"severity": "error", "code": "missing_modality", "path": group[0]["path"], "detail": f"Missing {modality} in observation {tree_id}/{season}/{capture}"})
        if row["needs_identity_review"]:
            issues.append({"severity": "review", "code": "embedded_tree_identifier", "path": group[0]["path"], "detail": f"Leading ID {tree_id}; embedded ID {row['embedded_tree_id']}; correspondence unverified"})
        elif any(r["capture_format_unusual"] for r in group):
            issues.append({"severity": "review", "code": "unusual_capture_suffix", "path": group[0]["path"], "detail": capture})
        observations.append(row)
    return observations, issues


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Write UTF-8 CSV with a stable header, including for an empty table."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit_dataset(project_root: Path, output_dir: Path) -> dict:

    project_root = project_root.resolve()
    data_root = project_root / "resources" / "TreeID"
    output_dir = output_dir.resolve()
    if output_dir == data_root.resolve() or data_root.resolve() in output_dir.parents:
        raise ValueError("Audit outputs must be outside the raw TreeID directory")
    for folder in DATASET_FOLDERS:
        if not (data_root / folder).is_dir():
            raise FileNotFoundError(f"Missing dataset folder: {data_root / folder}")
    records, issues, auxiliary = [], [], []
    for path in sorted(p for p in data_root.rglob("*") if p.is_file()):
        relative = path.relative_to(project_root).as_posix()
        folder = path.relative_to(data_root).parts[0]
        if folder not in DATASET_FOLDERS or path.suffix.lower() != ".png":
            auxiliary.append(relative)
            continue
        try:
            parsed = parse_filename(path.name)
        except ValueError as exc:
            issues.append({"severity": "error", "code": "unparsed_filename", "path": relative, "detail": str(exc)})
            continue
        record = {"path": relative, "dataset": folder, **parsed}
        try:
            record.update(inspect_png(path))
            expected = (256, 256, 8, 0) if parsed["modality"] == "GT" else (640, 480, 8, 2)
            actual = tuple(record[k] for k in ("width", "height", "bit_depth", "png_colour_type"))
            if actual != expected:
                issues.append({"severity": "review", "code": "unexpected_image_format", "path": relative, "detail": f"Observed {actual}; initial audit expected {expected}"})
        except (OSError, ValueError) as exc:
            record["png_integrity"] = "error"
            issues.append({"severity": "error", "code": "png_integrity", "path": relative, "detail": str(exc)})
        records.append(record)
    observations, pairing_issues = pair_records(records)
    issues.extend(pairing_issues)
    for row in observations:
        paired = [r for r in records if r["path"] in (row["rgb_path"], row["depth_path"])]
        shapes = {(r.get("width"), r.get("height")) for r in paired if r.get("png_integrity") == "header_ok"}
        if len(shapes) > 1:
            issues.append({"severity": "error", "code": "rgb_depth_shape_mismatch", "path": row["rgb_path"], "detail": str(sorted(shapes))})
    by_hash = defaultdict(list)
    for record in records:
        if record.get("sha256"):
            by_hash[record["sha256"]].append(record)
    duplicate_rows = [
        {"sha256": digest, "path": record["path"], "modality": record["modality"], "dataset": record["dataset"], "tree_id": record["tree_id"]}
        for digest, group in sorted(by_hash.items()) if len(group) > 1 for record in group
    ]
    reused_depth = {}
    for group in by_hash.values():
        depths = [record for record in group if record["modality"] == "D"]
        if len(depths) > 1:
            reused_depth.update({record["path"]: len(depths) for record in depths})
    review_rows = []
    for observation in observations:
        reasons = []
        if observation["needs_identity_review"]:
            reasons.append("embedded_tree_identifier")
        if observation["depth_path"] in reused_depth:
            reasons.append("reused_depth_file")
        if reasons:
            review_rows.append({**observation, "review_reasons": ";".join(reasons), "depth_duplicate_group_size": reused_depth.get(observation["depth_path"], 0)})
    folder_summary = {}
    for folder in DATASET_FOLDERS:
        rows = [r for r in observations if r["dataset"] == folder]
        files = [r for r in records if r["dataset"] == folder]
        folder_summary[folder] = {
            "filename_tree_ids": len({r["tree_id"] for r in rows}),
            "observations": len(rows), "png_files": len(files),
            "season_counts": dict(sorted(Counter(r["season"] for r in rows).items())),
            "modality_counts": dict(sorted(Counter(r["modality"] for r in files).items())),
            "identity_review_observations": sum(r["needs_identity_review"] for r in rows),
        }
    summary = {
        "audit_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "session_01_part_01_basic_file_checks",
        "files": len(records), "observations": len(observations),
        "summer_gt_masks": sum(r["modality"] == "GT" and r["season"] == "Summer" for r in records),
        "png_headers_checked": sum(r.get("png_integrity") == "header_ok" for r in records),
        "errors": sum(r["severity"] == "error" for r in issues),
        "review_issues": sum(r["severity"] == "review" for r in issues),
        "issue_counts": dict(sorted(Counter(r["code"] for r in issues).items())),
        "identity_review_observations": sum(r["needs_identity_review"] for r in observations),
        "exact_duplicate_groups": sum(len(group) > 1 for group in by_hash.values()),
        "exact_duplicate_files": len(duplicate_rows),
        "reused_depth_files": len(reused_depth),
        "observations_requiring_review": len(review_rows), "folders": folder_summary,
        "image_formats": dict(sorted(Counter(
            f"{r['modality']}|{r.get('width')}x{r.get('height')}|bit_depth={r.get('bit_depth')}|colour_type={r.get('png_colour_type')}"
            for r in records if r.get("png_integrity") == "header_ok"
        ).items())),
        "auxiliary_files": auxiliary,
        "unresolved": ["Depth encoding, units and calibration", "GT semantics and coordinate transform", "Physical identities and seasonal correspondence", "Independent 3D reference", "Dataset attribution and redistribution terms"],
        "checks": ["Full filename suffix pairing", "PNG signature, dimensions and colour type", "SHA-256 exact-file duplicate detection"],
        "limits": ["Header checks only; image pixels and compressed data are not decoded", "No depth, GT alignment or physical identity inference", "No train/validation/test split assignment"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    file_columns = ["path", "dataset", "tree_id", "season", "modality", "capture", "embedded_tree_id", "needs_identity_review", "capture_format_unusual", "width", "height", "bit_depth", "png_colour_type", "size_bytes", "sha256", "png_integrity"]
    observation_columns = ["dataset", "tree_id", "physical_tree_id", "season", "capture_key", "rgb_path", "depth_path", "gt_path", "embedded_tree_id", "needs_identity_review", "annotation_alignment_verified", "metric_depth_verified", "split", "pairing_status"]
    write_csv(output_dir / "files.csv", records, file_columns)
    write_csv(output_dir / "observations.csv", observations, observation_columns)
    write_csv(output_dir / "filename_anomalies.csv", [r for r in observations if r["needs_identity_review"]], observation_columns)
    write_csv(output_dir / "review_observations.csv", review_rows, observation_columns + ["review_reasons", "depth_duplicate_group_size"])
    write_csv(output_dir / "issues.csv", issues, ["severity", "code", "path", "detail"])
    write_csv(output_dir / "exact_duplicates.csv", duplicate_rows, ["sha256", "path", "modality", "dataset", "tree_id"])
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
