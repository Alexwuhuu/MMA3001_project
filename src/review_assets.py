"""Human review assets: candidate suggestions and a separate blind test packet."""
from __future__ import annotations

import html
from pathlib import Path

import matplotlib.pyplot as plt

from .main import save_contact_sheets
from .utils import read_rgb

STYLE = """body{font:16px/1.5 system-ui,sans-serif;max-width:1500px;margin:24px auto;padding:0 16px}
img{max-width:100%;height:auto}article{border:1px solid #ccc;border-radius:6px;padding:12px;margin:20px 0}
table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px;text-align:left}a{color:#145b9d}"""


def _document(title, body):
    return (f'<!doctype html><html lang="en"><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>{STYLE}</style><h1>{html.escape(title)}</h1>{body}</html>')


def render_targeted_assets(root, output, cases, selected, evaluation, settings):
    """Review original RGB views; scores never populate any human judgement fields."""
    directory = output / "review_batches"
    images = directory / "images"
    images.mkdir(parents=True, exist_ok=True)
    if not directory.resolve().is_relative_to(root.resolve()) or not images.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Generated review pages must remain inside the project output directory")
    # Remove only this renderer's obsolete files; never touch source images or ledgers.
    for parent, pattern in [(directory, "batch_*.html"), (images, "case_*.png")]:
        for obsolete in parent.glob(pattern):
            if obsolete.is_file() and obsolete.resolve().is_relative_to(parent.resolve()):
                obsolete.unlink()
    identities = cases.set_index("tree_id")
    batch_links = []
    for batch_number, batch in selected.groupby("batch", sort=True):
        articles = []
        for record in batch.to_dict("records"):
            query_id = record["query_id"]
            identity = identities.loc[query_id]
            columns = [(identity.summer_rgb, f"{query_id} | Summer query"),
                       (identity.winter_rgb, f"{query_id} | Original winter\n"
                        f"inliers={record['original_inliers']:.0f}, ratio={record['original_inlier_ratio']:.2f}, "
                        f"evidence={record['original_score']:.2f}")]
            for i in range(1, settings["alternatives"] + 1):
                key = f"alternative_{i}"
                if key in record and isinstance(record[key], str) and record[key]:
                    alternative = identities.loc[record[key]]
                    columns.append((alternative.winter_rgb,
                                    f"{record[key]} | Alternative {i}\n"
                                    f"inliers={record[key + '_inliers']:.0f}, "
                                    f"ratio={record[key + '_inlier_ratio']:.2f}, evidence={record[key + '_score']:.2f}"))
            fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 3.7), squeeze=False)
            for axis, (path, title) in zip(axes[0], columns):
                axis.imshow(read_rgb(root / path))
                axis.set_title(title, fontsize=9)
                axis.axis("off")
            fig.suptitle(f"Review priority {record['review_priority']} | {query_id} | "
                         f"{record['split']} | {record['human_status']} | suggestions require human verification", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.91))
            name = f"case_{record['review_priority']:03d}_{query_id}.png"
            fig.savefig(images / name, dpi=110)
            plt.close(fig)
            articles.append(
                f'<article id="{query_id}"><h2>{query_id} · Review priority {record["review_priority"]}</h2>'
                f'<p>Screening reasons: {html.escape(record["issue_tags"])}; original geometric rank range: '
                f'{record["original_rank_min"]}–{record["original_rank_max"]}/{record["gallery_size"]}。</p>'
                f'<p>Existing human status: {html.escape(record["human_status"])}; original notes: '
                f'{html.escape(str(record["human_notes"]))}</p>'
                f'<p>Geometric evidence status: {html.escape(record["evidence_status"])}; display basis: '
                f'{html.escape(record["alternative_display_basis"])}. Insufficient evidence cannot establish a mismatch or replacement identity.</p>'
                f'<a href="images/{name}"><img src="images/{name}" loading="lazy" alt="{query_id} suggestions"></a></article>')
        name = f"batch_{int(batch_number):02d}.html"
        instructions = ('<p><a href="../index.html">Back to index</a>. These are review suggestions, not ground truth. '
                        'The first two images show the original Summer/Winter pair; the others come from the full split gallery. '
                        'Repeated backgrounds can also produce inliers. Record human decisions in config/pair_review.csv; '
                        'a low score does not automatically exclude a pair.</p>')
        (directory / name).write_text(_document(f"Targeted review · Batch {batch_number} ({len(batch)} pairs)",
                                               instructions + "".join(articles)), encoding="utf-8")
        batch_links.append(f'<li><a href="review_batches/{name}">Batch {batch_number} · {len(batch)} pairs</a></li>')
        print(f"Rendered optional review batch {batch_number}: {len(batch)} cases", flush=True)

    # Independent images show neither screening scores nor alternative candidates.
    blind = evaluation["review"].copy().rename(columns={"independent_status": "human_status"})
    for column in ["summer_rgb", "winter_rgb"]:
        blind[column] = blind[column].map(lambda p: str(root / p))
    test_folder = output / "independent_test"
    test_pages = save_contact_sheets(blind, test_folder, per_page=8, all_pairs=True)
    page_links = "".join(f'<li><a href="review_sheets_all/{p.name}">Page {i}</a></li>' for i, p in enumerate(test_pages, 1))
    test_body = ('<p>These queries were randomly selected before automated screening scores were computed. '
                 'Review the original pairs independently and fill independent_status, reviewer, and human_notes '
                 'in config/evaluation/independent_review.csv.</p>'
                 '<p>States: pending / confirmed / rejected / uncertain. Preserve every sampled record. '
                 'Do not delete or replace cases because a future model fails to match them. '
                 'This packet contains no automated scores or alternative suggestions.</p>'
                 f'<p>Sampled pairs: {len(blind)}. Both future regression models share the complete '
                 f'winter test gallery of {len(evaluation["gallery"])} images.</p>'
                 f'<ul>{page_links}</ul>')
    (test_folder / "index.html").write_text(_document("Independent test query review", test_body), encoding="utf-8")
    body = ('<p>ORB/RANSAC schedules human review; it does not automatically confirm, reject, or re-pair identities. '
            'This queue uses training and validation rows only. Test queries have no screening scores.</p>'
            '<p>When all geometric scores are zero, preserve the complete geometric rank tie range. '
            'Weak descriptor correspondences select images for inspection, not reliable replacement identities. '
            'Cases without distinguishing evidence remain diagnostics and do not fill review batches automatically.</p>'
            f'<p><a href="independent_test/index.html">Independent test queries: original-pair atlas</a></p>'
            f'<ul>{"".join(batch_links)}</ul>'
            '<p>suspected_pairs.csv contains low-evidence diagnostics and an optional queue. '
            'review_batches.csv includes only filename anomalies or cases with distinguishing geometric evidence. '
            'Existing human records remain in config/pair_review.csv.</p>')
    index = output / "index.html"
    (index).write_text(_document("Targeted review and independent test sampling", body), encoding="utf-8")
    return {"index": index, "independent_test": test_folder / "index.html",
            "first_batch": directory / "batch_01.html" if batch_links else None}
