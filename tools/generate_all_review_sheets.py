"""Generate a full review atlas from the ledger without modifying human labels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import io
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.main import load_config, save_contact_sheets
from src.utils import file_digest


def generate_atlas(root: Path, per_page: int | None = None) -> Path:
    """Render every ledger pair and a searchable page index while preserving ledger bytes.

    Args:
        root (Path): Project containing experiment settings and the review ledger.
        per_page (int | None): Positive pairs-per-page count; None uses experiment
            configuration.

    Returns:
        Path: review_sheets_all directory containing index.html, PNG pages, page_index.csv and
        atlas_manifest.json. Displayed statuses/notes are generation-time snapshots.

    Raises:
        ValueError: Page size, ledger columns, identities or human statuses are invalid.
        FileNotFoundError: A referenced seasonal image is absent.
        RuntimeError: Ledger bytes change during rendering.
        OSError: Ledger/assets cannot be read or written.
    """
    root = Path(root).resolve()
    config = load_config(root)
    per_page = config["contact_sheet_pairs_per_page"] if per_page is None else per_page
    if per_page < 1:
        raise ValueError("pairs per page must be positive")
    ledger = root / config["review_file"]
    before = file_digest(ledger)
    raw = ledger.read_bytes()
    try:
        content = raw.decode("utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        content = raw.decode("gb18030")
        encoding = "gb18030"
    review = pd.read_csv(io.StringIO(content), keep_default_na=False)
    necessary = {"tree_id", "row", "number", "summer_rgb", "winter_rgb", "human_status", "human_notes"}
    if not necessary.issubset(review.columns):
        raise ValueError(f"Missing review fields: {sorted(necessary - set(review.columns))}")
    if review.empty or review.tree_id.duplicated().any():
        raise ValueError("Review ledger must contain non-empty, unique tree IDs")
    if not review.tree_id.str.fullmatch(r"R\d+N\d+").all():
        raise ValueError("Only labelled N candidate pairs belong in this atlas")
    if not set(review.human_status).issubset({"pending", "confirmed", "rejected"}):
        raise ValueError("Unknown human review status")
    review = review.sort_values(["row", "number"], kind="stable").reset_index(drop=True)
    display_review = review.copy(deep=True)
    for column in ["summer_rgb", "winter_rgb"]:
        resolved = [p if p.is_absolute() else root / p for p in map(Path, review[column])]
        missing = [str(p) for p in resolved if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {column} images: {missing[:3]}")
        display_review[column] = [str(p.resolve()) for p in resolved]

    pages_count = (len(review) + per_page - 1) // per_page
    print(f"Rendering {len(review)} pairs, {per_page} per page, {pages_count} pages.", flush=True)
    print(f"Input CSV encoding: {encoding}; existing decisions are preserved.", flush=True)
    output = root / config["output_dir"]
    paths = save_contact_sheets(display_review, output, per_page, all_pairs=True)
    folder = output / "review_sheets_all"
    page_index = review.copy()
    page_index.insert(0, "page", page_index.index // per_page + 1)
    page_index.insert(1, "position_on_page", page_index.index % per_page + 1)
    page_index.insert(2, "sheet", [paths[i // per_page].name for i in range(len(review))])
    page_index.to_csv(folder / "page_index.csv", index=False, encoding="utf-8-sig")

    sections = []
    for page_number, page in page_index.groupby("page", sort=True):
        sheet = page.iloc[0]["sheet"]
        ids = ", ".join(page.tree_id)
        details = page[["tree_id", "human_status", "human_notes"]].to_html(index=False, escape=True)
        sections.append(
            f'<details id="page-{page_number}"><summary>Page {page_number:02d} — '
            f'{html.escape(ids)}</summary><p><a href="{sheet}">Open full-size PNG</a></p>'
            f'<img src="{sheet}" loading="lazy" alt="Review page {page_number}">'
            f'{details}</details>')
    document = '''<!doctype html>
<html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Apple-tree pair review atlas</title>
<style>body{font:16px/1.5 system-ui,sans-serif;max-width:1300px;margin:24px auto;padding:0 16px}
details{border:1px solid #bbb;border-radius:6px;padding:10px;margin:10px 0}
summary{cursor:pointer;font-weight:600}img{max-width:100%;height:auto}
table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #ccc;padding:6px;text-align:left}</style>
<h1>Complete Summer/Winter pair review atlas</h1>
<p>Summer RGB is on the left and Winter RGB on the right. Expand a page heading or use Ctrl+F to find a tree ID.
Statuses and original notes are generation-time snapshots. Record human decisions in config/pair_review.csv.</p>
'''
    document += f'<p>{len(review)} pairs, {len(paths)} pages, up to {per_page} pairs per page. '
    document += '<a href="page_index.csv">Download the tree-ID/page index (CSV UTF-8)</a></p>'
    document += "\n".join(sections) + "\n</html>\n"
    (folder / "index.html").write_text(document, encoding="utf-8")
    after = file_digest(ledger)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pairs": len(review), "pages": len(paths), "pairs_per_page": per_page,
        "ledger": ledger.relative_to(root).as_posix(),
        "ledger_encoding": encoding, "ledger_sha256_at_start": before,
        "ledger_unchanged_during_render": before == after,
        "status_counts": review.human_status.value_counts().to_dict(),
    }
    (folder / "atlas_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if before != after:
        raise RuntimeError("The review ledger changed during generation; rerun to update the snapshot")
    print(f"Atlas complete: {folder / 'index.html'}", flush=True)
    print(f"Coverage index: {folder / 'page_index.csv'}", flush=True)
    print(f"Ledger SHA-256 unchanged: {after}", flush=True)
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-per-page", type=int, help="Default: experiment.json setting")
    args = parser.parse_args()
    generate_atlas(ROOT, per_page=args.pairs_per_page)
