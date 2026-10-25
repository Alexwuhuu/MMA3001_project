"""Parts 1–2: dataset audit, review assets and reproducible row-level splits.

The root notebook is the primary interface. This module intentionally contains no
feature matching, regression fitting or model-selection code at this stage.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import assign_splits, file_digest, index_dataset, matched_cases, read_rgb, read_review_csv, write_json


def get_runtime_versions() -> dict[str, str | None]:
    """Record runtime dependency versions without requiring notebook controllers.

    Missing distribution metadata is provenance information. Missing runtime imports
    still fail normally when their modules are imported.

    Returns:
        dict[str, str | None]: Versions of numpy, pandas, Pillow and matplotlib, or None where
        package metadata is unavailable.
    """
    versions = {}
    for name in ("numpy", "pandas", "Pillow", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def load_config(project_root: str | Path, config_path: str = "config/experiment.json") -> dict:
    """Load defaults, then local overrides, then MMA3001_DATA_ROOT.

    Args:
        project_root (str | Path): Project directory used to resolve configuration paths.
        config_path (str): Defaults JSON path relative to project_root.

    Returns:
        dict: Merged experiment settings; only seed and review/page counts are validated here.

    Raises:
        OSError: A configuration file cannot be read.
        json.JSONDecodeError: A configuration file is malformed.
        ValueError: Seed or review/page counts are invalid.
        KeyError: A required configuration field is absent.
    """
    root = Path(project_root).resolve()
    config = json.loads((root / config_path).read_text(encoding="utf-8"))
    local = root / "config/local.json"
    if local.is_file():
        config.update(json.loads(local.read_text(encoding="utf-8")))
    if os.environ.get("MMA3001_DATA_ROOT"):
        config["data_root"] = os.environ["MMA3001_DATA_ROOT"]
    if not isinstance(config["seed"], int) or config["review_pairs_per_row"] < 1:
        raise ValueError("seed must be an integer and review_pairs_per_row must be positive")
    if config["contact_sheet_pairs_per_page"] < 1:
        raise ValueError("contact_sheet_pairs_per_page must be positive")
    return config


def portable_frame(frame: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    """Copy a table and make project-contained image paths portable.

    Args:
        frame (pandas.DataFrame): Table whose path columns end in _path, _rgb or _depth.
        project_root (Path): Absolute base directory for project-relative paths.

    Returns:
        pandas.DataFrame: A copy using POSIX path separators. External paths remain absolute;
        empty and non-string values are preserved.
    """
    frame = frame.copy()
    for column in frame.columns:
        if column.endswith("_path") or column.endswith("_rgb") or column.endswith("_depth"):
            def convert(value):
                if not isinstance(value, str) or not value:
                    return value
                try:
                    return Path(value).resolve().relative_to(project_root).as_posix()
                except ValueError:
                    return Path(value).as_posix()
            frame[column] = frame[column].map(convert)
    return frame


def prepare_review(cases: pd.DataFrame, project_root: Path, config: dict) -> pd.DataFrame:
    """Create or reconcile an editable ledger while preserving recorded human decisions.

    Required spot checks use review_pairs_per_row plus every identity in review_all_rows.
    Ledger rows propose filename-based N pairs, without automatically verifying them.

    Args:
        cases (pandas.DataFrame): Valid N pairs with identity and seasonal RGB metadata.
        project_root (Path): Project directory for the ledger.
        config (dict): Seed, review sampling settings and review_file destination.

    Returns:
        pandas.DataFrame: Review rows with required_review, human_status and human_notes. The
        CSV is written only when absent or its identity coverage changes.

    Raises:
        ValueError: Existing IDs, required columns or review statuses are invalid.
        OSError: The ledger cannot be read or written.
    """
    rng = np.random.default_rng(config["seed"])
    required = set()
    for row, group in cases.groupby("row", sort=True):
        count = min(config["review_pairs_per_row"], len(group))
        required.update(rng.choice(group["tree_id"].to_numpy(), count, replace=False).tolist())
        if row in config["review_all_rows"]:
            required.update(group["tree_id"])
    review = cases[["tree_id", "row", "number", "summer_rgb", "winter_rgb"]].copy()
    review["required_review"] = review["tree_id"].isin(required)
    review["human_status"] = "pending"
    review["human_notes"] = ""
    path = project_root / config["review_file"]
    needs_write = not path.is_file()
    if path.is_file():
        previous = read_review_csv(path)
        necessary = {"tree_id", "human_status", "human_notes"}
        if not necessary.issubset(previous.columns) or previous["tree_id"].duplicated().any():
            raise ValueError(f"Invalid human review ledger: {path}")
        unknown = set(previous["tree_id"]) - set(cases["tree_id"])
        if unknown:
            raise ValueError(f"Review ledger has identities outside the current N dataset: {sorted(unknown)[:5]}")
        allowed = {"pending", "confirmed", "rejected"}
        if not set(previous["human_status"]).issubset(allowed):
            raise ValueError("human_status must be pending, confirmed or rejected")
        old = previous.set_index("tree_id")
        for column in ["human_status", "human_notes"]:
            review[column] = review["tree_id"].map(old[column]).fillna(review[column])
        needs_write = set(previous["tree_id"]) != set(review["tree_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if needs_write:
        portable_frame(review, project_root).to_csv(path, index=False, encoding="utf-8-sig")
    return review


def validate_split(cases: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Check row, identity, path and exact RGB-content isolation between splits.

    Args:
        cases (pandas.DataFrame): Assigned cases with tree_id, row and split.
        index (pandas.DataFrame): Observation index with kind, RGB paths and content hashes.

    Returns:
        pandas.DataFrame: One overlap_count/passed record per split pair and isolation field.

    Raises:
        ValueError: Any isolation field overlaps between splits.
    """
    checks = []
    observations = index[index["kind"].eq("N") & index["tree_id"].isin(cases["tree_id"])].copy()
    observations["split"] = observations["tree_id"].map(cases.set_index("tree_id")["split"])
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        ca, cb = cases[cases["split"].eq(a)], cases[cases["split"].eq(b)]
        oa, ob = observations[observations["split"].eq(a)], observations[observations["split"].eq(b)]
        fields = {
            "row": (set(ca["row"]), set(cb["row"])),
            "tree_id": (set(ca["tree_id"]), set(cb["tree_id"])),
            "rgb_path": (set(oa["rgb_path"]), set(ob["rgb_path"])),
            "rgb_sha256": (set(oa["rgb_sha256"]) - {""}, set(ob["rgb_sha256"]) - {""}),
        }
        for field, (left, right) in fields.items():
            overlap = left & right
            checks.append({"split_a": a, "split_b": b, "field": field,
                           "overlap_count": len(overlap), "passed": not overlap})
    result = pd.DataFrame(checks)
    if not result["passed"].all():
        failures = result.loc[~result["passed"]].to_dict("records")
        raise ValueError(f"Data leakage detected; review source labels or duplicate images: {failures}")
    return result


def save_contact_sheets(review: pd.DataFrame, output: Path, per_page: int = 8,
                        *, all_pairs: bool = False) -> list[Path]:
    """Render original seasonal pairs with generation-time review statuses.

    Args:
        review (pandas.DataFrame): Review rows with readable summer_rgb/winter_rgb, tree_id,
            human_status and, for spot checks, required_review.
        output (Path): Parent output directory for review_sheets or review_sheets_all.
        per_page (int): Maximum pairs per PNG page; must be positive.
        all_pairs (bool): Whether to include all rows instead of only required spot checks.

    Returns:
        list[Path]: Written page paths in display order; empty when no rows are selected.

    Raises:
        ValueError: per_page is not positive or an RGB image is unreadable.
        OSError: A figure cannot be written.
    """
    if per_page < 1:
        raise ValueError("per_page must be positive")
    selected = (review if all_pairs else review[review["required_review"]]).reset_index(drop=True)
    folder = output / ("review_sheets_all" if all_pairs else "review_sheets")
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for start in range(0, len(selected), per_page):
        page = selected.iloc[start:start + per_page]
        fig, axes = plt.subplots(len(page), 2, figsize=(10, 2.9 * len(page)), squeeze=False)
        for i, (_, record) in enumerate(page.iterrows()):
            for j, season in enumerate(["summer", "winter"]):
                axes[i, j].imshow(read_rgb(record[f"{season}_rgb"]))
                axes[i, j].set_title(f'{record["tree_id"]} | {season.title()} | {record["human_status"]}', fontsize=10)
                axes[i, j].axis("off")
        fig.suptitle(
            f"Summer–winter candidate pairs | {start + 1}–{start + len(page)} of {len(selected)}"
            " | labels: human review snapshot", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        path = folder / f"review_{start // per_page + 1:02d}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(path)
    return paths


def save_overview_figures(index: pd.DataFrame, cases: pd.DataFrame, output: Path) -> list[Path]:
    """Plot observation counts, eligible pairs per row and assigned split sizes.

    Args:
        index (pandas.DataFrame): Audited observation table.
        cases (pandas.DataFrame): Eligible cases with row and split assignments.
        output (Path): Parent directory in which figures/ is created.

    Returns:
        list[Path]: A one-element list containing the combined overview PNG path.

    Raises:
        OSError: A figure cannot be written.
    """
    folder = output / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    counts = index.groupby(["kind", "season"]).size()
    counts.plot.bar(ax=axes[0], color=["#e79543", "#4378ab", "#e79543", "#4378ab"])
    axes[0].set(title="Observations by label and season", ylabel="RGB-D observations", xlabel="(N/U, season)")
    by_row = cases.groupby("row").size()
    by_row.plot.bar(ax=axes[1], color="#408a75")
    axes[1].set(title="Valid N pairs per orchard row", xlabel="Row", ylabel="Pairs")
    for name, color in [("train", "#4185b5"), ("validation", "#e5a23f"), ("test", "#b86379")]:
        group = cases[cases["split"].eq(name)]
        axes[2].scatter(group["row"].str[1:].astype(int), group["number"], s=10, label=name, color=color)
    axes[2].set(title="Index diagram (NOT metric coordinates)", xlabel="R: row index", ylabel="N: tree number")
    axes[2].invert_yaxis()
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    path = folder / "data_overview.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return [path]


def run_data_preparation(project_root: str | Path, config: dict | None = None,
                         render: bool = True, index: pd.DataFrame | None = None) -> dict:
    """Audit data, preserve review provenance and export row-isolated manifests.

    Row membership is assigned before human-rejected identities are removed. Source
    images are read-only; generated CSV paths are portable relative to the project.

    Args:
        project_root (str | Path): Project containing configuration and output directories.
        config (dict | None): Experiment settings, or None to load defaults and overrides.
        render (bool): Whether to create contact sheets and overview figures.
        index (pandas.DataFrame | None): Optional precomputed observation index; None scans the
            configured dataset.

    Returns:
        dict: index, cases, review, exclusions, leakage, summary, output_dir, contact_sheets and
        figures. cases includes split, human_verified and label_basis.

    Raises:
        ValueError: Dataset/review/split validation fails or required human review is
            incomplete. The latter check occurs after audit exports are written.
        OSError: Input data/configuration cannot be read or generated files cannot be written.
    """
    root = Path(project_root).resolve()
    config = dict(config or load_config(root))
    data = Path(config["data_root"])
    data = data if data.is_absolute() else root / data
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    print("Part 1: reading image headers, RGB content hashes and D/GT quality...", flush=True)
    if index is None:
        index = index_dataset(data)
    portable_frame(index, root).to_csv(output / "observation_index.csv", index=False)
    pairs, exclusions = matched_cases(index)
    if config.get("subset"):
        pairs = pairs[pairs["subset"].eq(config["subset"])].reset_index(drop=True)
        if pairs.empty:
            raise ValueError("Configured matched subset has no valid N pairs")
    print(f"Part 2: {len(pairs)} proposed N pairs; U observations stay unlabelled.", flush=True)
    # Freeze row membership before excluding any human-rejected identities.
    assigned = assign_splits(pairs, seed=config["seed"])
    review = prepare_review(pairs, root, config)
    rejected = set(review.loc[review["human_status"].eq("rejected"), "tree_id"])
    if rejected:
        rejected_rows = pairs[pairs["tree_id"].isin(rejected)][["subset", "tree_id"]].assign(reason="human_rejected")
        exclusions = pd.concat([exclusions, rejected_rows], ignore_index=True)
    cases = assigned[~assigned["tree_id"].isin(rejected)].reset_index(drop=True)
    cases = cases.merge(review[["tree_id", "human_status", "human_notes"]], on="tree_id", validate="one_to_one")
    cases["human_verified"] = cases["human_status"].eq("confirmed")
    cases["label_basis"] = np.where(cases["human_verified"], "human_confirmed", "filename_unverified")
    leakage = validate_split(cases, index)
    required = review[review["required_review"]]
    pending = int(required["human_status"].eq("pending").sum())
    versions = get_runtime_versions()
    summary = {
        "stage": "1_config_and_data_audit__2_label_review_and_row_split",
        "models_trained": False,
        "observations": len(index),
        "valid_rgb_observations": int(index["valid_rgb"].sum()),
        "proposed_n_pairs": len(review), "eligible_n_pairs": len(cases),
        "unmatched_u_observations": int(index["kind"].eq("U").sum()),
        "review_required": len(required), "review_pending": pending,
        "review_confirmed": int(required["human_status"].eq("confirmed").sum()),
        "review_rejected": int(review["human_status"].eq("rejected").sum()),
        "label_status": "filename_labels_with_partial_human_verification",
        "eligible_human_verified": int(cases["human_verified"].sum()),
        "eligible_filename_unverified": int((~cases["human_verified"]).sum()),
        "review_policy": "targeted_review; no automatic approvals or score-based exclusions",
        "split_rows": cases.groupby("split")["row"].nunique().to_dict(),
        "split_trees": cases.groupby("split").size().to_dict(),
        "leakage_checks_passed": bool(leakage["passed"].all()),
        "seed": config["seed"], "depth_metric_scale": "unknown",
        "source_paper": "https://arxiv.org/pdf/2010.06879",
        "config": config, "versions": versions,
    }
    portable_frame(cases, root).to_csv(output / "split_manifest.csv", index=False)
    exclusions.to_csv(output / "excluded_n_pairs.csv", index=False)
    leakage.to_csv(output / "leakage_checks.csv", index=False)
    index[index["special_suffix"]].pipe(portable_frame, root).to_csv(output / "special_filenames.csv", index=False)
    quality = index[(~index["valid_rgb"]) | index["depth_status"].ne("ok") | index["gt_status"].eq("corrupt_gt")]
    portable_frame(quality, root).to_csv(output / "quality_issues.csv", index=False)
    duplicate_rgb = index[index["rgb_sha256"].ne("") & index["rgb_sha256"].duplicated(keep=False)]
    portable_frame(duplicate_rgb, root).to_csv(output / "duplicate_rgb_content.csv", index=False)
    for split, group in cases.groupby("split"):
        for season in ["summer", "winter"]:
            manifest = group[["tree_id", "row", "number", f"{season}_rgb", f"{season}_depth",
                              "human_status", "human_verified", "label_basis"]].copy()
            name = "queries" if season == "summer" else "gallery"
            portable_frame(manifest, root).to_csv(output / f"{split}_{name}.csv", index=False)
    sheets = save_contact_sheets(review, output, config["contact_sheet_pairs_per_page"]) if render else []
    figures = save_overview_figures(index, cases, output) if render else []
    summary["split_manifest_sha256"] = file_digest(output / "split_manifest.csv")
    write_json(output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"config", "versions"}}, indent=2), flush=True)
    if pending:
        print(f"{pending} early spot-check pairs remain unresolved; targeted review retains their filename labels.", flush=True)
    if config["require_human_review"] and pending:
        raise ValueError("Required manual review is incomplete; edit the review ledger before continuing")
    return {"index": index, "cases": cases, "review": review, "exclusions": exclusions,
            "leakage": leakage, "summary": summary, "output_dir": output,
            "contact_sheets": sheets, "figures": figures}


def main():
    """Run the data-preparation CLI using --project-root, --config and --no-render.

    Returns:
        None: Audit artifacts are written to the configured output directory.

    Raises:
        SystemExit: Argument parsing fails or --help is requested.
        ValueError: Preparation fails its dataset, review or leakage checks.
        OSError: Inputs or output files are inaccessible.
    """
    parser = argparse.ArgumentParser(description="Run notebook data preparation parts 1–2 only")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="config/experiment.json")
    parser.add_argument("--no-render", action="store_true")
    arguments = parser.parse_args()
    config = load_config(arguments.project_root, arguments.config)
    run_data_preparation(arguments.project_root, config, render=not arguments.no_render)


if __name__ == "__main__":
    main()
