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

from .utils import assign_splits, file_digest, index_dataset, matched_cases, read_rgb, write_json


def get_runtime_versions() -> dict[str, str | None]:
    """Report computational dependencies without requiring notebook automation tools.

    VS Code can run the notebook in an existing kernel without nbformat/nbclient.
    A missing distribution record is provenance information, not a data error.
    Actual missing runtime imports are still raised normally when importing code.
    """
    versions = {}
    for name in ("numpy", "pandas", "Pillow", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def load_config(project_root: str | Path, config_path: str = "config/experiment.json") -> dict:
    """Load tracked defaults, optional ignored local overrides, then data-root env."""
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
    """Export project-relative image paths so the project can be moved intact."""
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
    """Create an editable review ledger, preserving all existing human decisions.

    Each row is a proposed filename-based N pair, not a verified human judgement.
    Two pairs/row plus every R04 pair are selected for mandatory spot checks.
    All 562 pairs remain editable. U observations are never added to this ledger.
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
    if path.is_file():
        previous = pd.read_csv(path, keep_default_na=False)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    portable_frame(review, project_root).to_csv(path, index=False)
    return review


def validate_split(cases: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Verify row, identity, image-path and exact RGB-content isolation."""
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


def save_contact_sheets(review: pd.DataFrame, output: Path, per_page: int = 8) -> list[Path]:
    """Save side-by-side RGB pairs for every mandatory review (no automatic approval)."""
    selected = review[review["required_review"]].reset_index(drop=True)
    folder = output / "review_sheets"
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
        fig.suptitle("Filename-based candidate pairs — HUMAN REVIEW PENDING unless confirmed", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        path = folder / f"review_{start // per_page + 1:02d}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(path)
    return paths


def save_overview_figures(index: pd.DataFrame, cases: pd.DataFrame, output: Path) -> list[Path]:
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
    """Run only notebook stages 1–2. Original images are never modified.

    Returns dataframes plus output paths. Persisted paths are project-relative.
    Human review decisions survive reruns; rejected pairs are explicitly excluded.
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
    review = prepare_review(pairs, root, config)
    rejected = set(review.loc[review["human_status"].eq("rejected"), "tree_id"])
    if rejected:
        rejected_rows = pairs[pairs["tree_id"].isin(rejected)][["subset", "tree_id"]].assign(reason="human_rejected")
        exclusions = pd.concat([exclusions, rejected_rows], ignore_index=True)
        pairs = pairs[~pairs["tree_id"].isin(rejected)].reset_index(drop=True)
    cases = assign_splits(pairs, seed=config["seed"])
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
        "label_status": "filename_based_pending_human_review" if pending else "required_review_completed",
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
            manifest = group[["tree_id", "row", "number", f"{season}_rgb", f"{season}_depth"]].copy()
            name = "queries" if season == "summer" else "gallery"
            portable_frame(manifest, root).to_csv(output / f"{split}_{name}.csv", index=False)
    sheets = save_contact_sheets(review, output, config["contact_sheet_pairs_per_page"]) if render else []
    figures = save_overview_figures(index, cases, output) if render else []
    summary["split_manifest_sha256"] = file_digest(output / "split_manifest.csv")
    write_json(output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"config", "versions"}}, indent=2), flush=True)
    if pending:
        print(f"Human review remains pending for {pending} required pairs. No human approvals are fabricated.", flush=True)
    if config["require_human_review"] and pending:
        raise ValueError("Required manual review is incomplete; edit the review ledger before continuing")
    return {"index": index, "cases": cases, "review": review, "exclusions": exclusions,
            "leakage": leakage, "summary": summary, "output_dir": output,
            "contact_sheets": sheets, "figures": figures}


def main():
    parser = argparse.ArgumentParser(description="Run notebook data preparation parts 1–2 only")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="config/experiment.json")
    parser.add_argument("--no-render", action="store_true")
    arguments = parser.parse_args()
    config = load_config(arguments.project_root, arguments.config)
    run_data_preparation(arguments.project_root, config, render=not arguments.no_render)


if __name__ == "__main__":
    main()
