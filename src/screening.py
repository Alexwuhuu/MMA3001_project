"""Targeted label review using ORB/RANSAC; test sampling precedes all scoring."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pandas as pd

from .features import (FEATURE_COLUMNS, evidence_scores, extract_image_features,
                       extract_pair_features, feature_signature, stable_digest)
from .main import load_config, portable_frame, run_data_preparation
from .utils import file_digest, read_review_csv, write_json


def load_screening_config(root: Path) -> dict:
    settings = json.loads((Path(root) / "config/screening.json").read_text(encoding="utf-8"))
    if (not set(settings["screen_splits"]).issubset({"train", "validation"}) or
            not settings["screen_splits"] or settings["batch_size"] not in range(10, 21)):
        raise ValueError("Screen only train/validation; review batch size must be 10–20")
    if not 0 < settings["ratio_threshold"] < 1 or settings["workers"] < 1:
        raise ValueError("Invalid ratio threshold or worker count")
    return settings


def freeze_evaluation(cases: pd.DataFrame, root: Path, settings: dict) -> dict:
    """Uniformly sample test identities once, without looking at matching scores.

    The entire eligible test gallery is fixed for both future regression models.
    Reruns preserve query IDs and independent human decisions; changed test pools
    raise rather than silently resampling. Rejected/uncertain audit cases must be
    reported later, not replaced by easier cases after seeing model results.
    """
    folder = root / settings["evaluation_dir"]
    pool = portable_frame(cases[cases.split.eq("test")].sort_values("tree_id"), root)
    n = settings["evaluation_queries"]
    if n < 1 or n > len(pool):
        raise ValueError(f"Requested {n} independent queries, but only {len(pool)} test identities exist")
    qpath, gpath = folder / "queries.csv", folder / "gallery.csv"
    rpath, ppath = folder / "independent_review.csv", folder / "protocol.json"
    paths = [qpath, gpath, rpath, ppath]
    # Hashing images is an integrity check, not feature extraction or inspection.
    source = [{"tree_id": r.tree_id, "summer_sha256": file_digest(root / r.summer_rgb),
               "winter_sha256": file_digest(root / r.winter_rgb)} for r in pool.itertuples()]
    pool_signature = stable_digest(source)
    if ppath.is_file():
        if not all(p.is_file() for p in paths):
            raise ValueError("Incomplete frozen evaluation files; restore the existing protocol")
        protocol = json.loads(ppath.read_text(encoding="utf-8"))
        if (protocol["pool_signature"] != pool_signature or
                protocol["query_count"] != n or protocol["seed"] != settings["evaluation_seed"]):
            raise ValueError("Frozen test pool/settings changed; do not automatically resample the test queries")
        if file_digest(qpath) != protocol["queries_sha256"] or file_digest(gpath) != protocol["gallery_sha256"]:
            raise ValueError("Frozen test query/gallery files changed")
        queries, gallery = read_review_csv(qpath), read_review_csv(gpath)
    else:
        if any(p.exists() for p in paths):
            raise ValueError("Partial evaluation files exist; refusing to overwrite manual records")
        if (root / settings["output_dir"] / "screening_summary.json").exists():
            raise ValueError("Screening results already exist: an independent sample must be fixed before scoring")
        folder.mkdir(parents=True, exist_ok=True)
        indices = np.random.default_rng(settings["evaluation_seed"]).choice(len(pool), size=n, replace=False)
        queries = pool.iloc[indices][["tree_id", "row", "number", "summer_rgb", "winter_rgb"]].sort_values("tree_id")
        gallery = pool[["tree_id", "row", "number", "winter_rgb"]].copy()
        queries.to_csv(qpath, index=False, encoding="utf-8-sig")
        gallery.to_csv(gpath, index=False, encoding="utf-8-sig")
        independent = queries.copy()
        independent["independent_status"] = "pending"
        independent["reviewer"] = ""
        independent["human_notes"] = ""
        independent.to_csv(rpath, index=False, encoding="utf-8-sig")
        protocol = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sampling": "uniform_without_replacement_before_automated_scoring",
            "seed": settings["evaluation_seed"], "query_count": len(queries), "gallery_count": len(gallery),
            "test_rows": sorted(pool.row.unique().tolist()), "pool_signature": pool_signature,
            "queries_sha256": file_digest(qpath), "gallery_sha256": file_digest(gpath),
            "same_queries_and_full_gallery_for": ["LinearRegression", "RBF-SVR"],
            "selection_uses_scores": False, "test_screening_enabled": False,
            "uncertainty_plan": "Top-1/Top-5: 95% Wilson intervals; MRR: query bootstrap intervals",
            "manual_policy": "Review original pairs before model results. Keep all sampled decisions; do not replace failures.",
        }
        write_json(ppath, protocol)
    independent = read_review_csv(rpath)
    allowed = {"pending", "confirmed", "rejected", "uncertain"}
    if (independent.tree_id.duplicated().any() or set(independent.tree_id) != set(queries.tree_id) or
            not set(independent.independent_status).issubset(allowed)):
        raise ValueError("Independent review must retain the frozen identities and valid human statuses")
    return {"queries": queries, "gallery": gallery, "review": independent,
            "protocol": protocol, "folder": folder}


def _worker_setup(seed: int):
    cv2.setNumThreads(1)
    cv2.setRNGSeed(seed)


def build_pair_table(cases: pd.DataFrame, root: Path, settings: dict) -> tuple[pd.DataFrame, dict]:
    """Evaluate each selected query against its entire split gallery, with resume caches."""
    start = perf_counter()
    selected = cases[cases.split.isin(settings["screen_splits"])].sort_values("tree_id").copy()
    if selected.empty or selected.split.eq("test").any():
        raise ValueError("No eligible train/validation screening cases")
    namespace = root / settings["cache_dir"] / feature_signature(settings)
    image_cache, pair_cache = namespace / "images", namespace / "pairs"
    pair_cache.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(settings["seed"])
    images = {}
    paths = sorted(set(selected.summer_rgb) | set(selected.winter_rgb))
    for i, path in enumerate(paths, 1):
        images[path] = extract_image_features(root / path, settings, image_cache)
        if i % 100 == 0 or i == len(paths):
            print(f"Image features: {i}/{len(paths)}", flush=True)
    image_seconds = perf_counter() - start
    groups = {name: group.sort_values("tree_id").to_dict("records")
              for name, group in selected.groupby("split")}
    gallery_signatures = {name: stable_digest([(r["tree_id"], images[r["winter_rgb"]].image_hash) for r in group])
                          for name, group in groups.items()}

    def score_query(record):
        split, query_id = record["split"], record["tree_id"]
        query = images[record["summer_rgb"]]
        key = stable_digest([query_id, query.image_hash, gallery_signatures[split]])
        path = pair_cache / f"{split}_{key}.csv"
        if path.is_file():
            return pd.read_csv(path), True
        records = []
        for candidate in groups[split]:
            values = extract_pair_features(query, images[candidate["winter_rgb"]], settings)
            records.append({"split": split, "query_id": query_id, "candidate_id": candidate["tree_id"], **values})
        frame = pd.DataFrame(records)
        temp = path.with_suffix(".tmp")
        frame.to_csv(temp, index=False)
        temp.replace(path)
        return frame, False

    records = selected.to_dict("records")
    frames, cached = [], 0
    with ThreadPoolExecutor(max_workers=settings["workers"], initializer=_worker_setup,
                            initargs=(settings["seed"],)) as executor:
        for i, (frame, cache_hit) in enumerate(executor.map(score_query, records), 1):
            frames.append(frame)
            cached += int(cache_hit)
            if i % 20 == 0 or i == len(records):
                print(f"Full-gallery query comparisons: {i}/{len(records)} ({cached} reused)", flush=True)
    result = pd.concat(frames, ignore_index=True)
    result["evidence_score"] = evidence_scores(result, settings["min_screen_inliers"])
    stats = {"image_count": len(images), "query_count": len(records), "pair_count": len(result),
             "cached_queries": cached, "image_seconds": image_seconds,
             "comparison_seconds": perf_counter() - start - image_seconds,
             "feature_signature": feature_signature(settings)}
    return result, stats


def rank_review_cases(pair_table: pd.DataFrame, cases: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Describe label inconsistencies as review priorities, retaining all source labels."""
    identities = cases.set_index("tree_id")
    records = []
    for query_id, group in pair_table.groupby("query_id", sort=True):
        ordered = group.sort_values(["evidence_score", "good_matches", "hamming_mean", "candidate_id"],
                                    ascending=[False, False, True, True], na_position="last")
        original = ordered[ordered.candidate_id.eq(query_id)].iloc[0]
        alternatives = ordered[~ordered.candidate_id.eq(query_id)].head(settings["alternatives"])
        identity = identities.loc[query_id]
        best = alternatives.iloc[0] if not alternatives.empty else original
        # Geometry ties stay ties. Descriptor tie-breaks select display examples,
        # but must not turn all-zero geometric evidence into confident ranks.
        rank_min = 1 + int(ordered.evidence_score.gt(original.evidence_score).sum())
        rank_max = rank_min + int(ordered.evidence_score.eq(original.evidence_score).sum()) - 1
        geometry_available = bool(ordered.evidence_score.gt(0).any())
        anomaly = "RGB-R" in Path(identity.winter_rgb).name or "RGB-R" in Path(identity.summer_rgb).name
        tags = []
        if anomaly:
            tags.append("special_filename_suffix")
        if original.ransac_inliers < settings["min_screen_inliers"]:
            tags.append("weak_original_geometry")
        if best.evidence_score > original.evidence_score:
            tags.append("stronger_alternative_geometry")
        if rank_max > rank_min:
            tags.append("ranking_tie")
        record = {"query_id": query_id, "split": identity.split, "human_status": identity.human_status,
                  "human_verified": bool(identity.human_verified), "label_basis": identity.label_basis,
                  "human_notes": identity.human_notes, "numbering_anomaly": anomaly,
                  "issue_tags": ";".join(tags), "original_rank_min": rank_min, "original_rank_max": rank_max,
                  "original_score": original.evidence_score, "original_good_matches": original.good_matches,
                  "original_inliers": original.ransac_inliers, "original_inlier_ratio": original.ransac_inlier_ratio,
                  "alternative_margin": best.evidence_score - original.evidence_score,
                  "evidence_status": "geometry_available" if geometry_available else "insufficient_geometry",
                  "priority_supported": bool(anomaly or (geometry_available and
                                              (best.evidence_score > original.evidence_score or
                                               original.ransac_inliers < settings["min_screen_inliers"]))),
                  "alternative_display_basis": "geometry_then_descriptors" if geometry_available else "descriptors_only_weak",
                  "gallery_size": len(ordered)}
        for i, candidate in enumerate(alternatives.itertuples(), 1):
            record.update({f"alternative_{i}": candidate.candidate_id,
                           f"alternative_{i}_score": candidate.evidence_score,
                           f"alternative_{i}_inliers": candidate.ransac_inliers,
                           f"alternative_{i}_inlier_ratio": candidate.ransac_inlier_ratio})
        records.append(record)
    return pd.DataFrame(records)


def run_targeted_review(root: str | Path, preparation: dict | None = None,
                        settings: dict | None = None, render: bool = True) -> dict:
    """Freeze a blind audit, screen train/validation, and export human-review batches."""
    root = Path(root).resolve()
    settings = dict(settings or load_screening_config(root))
    if not set(settings["screen_splits"]).issubset({"train", "validation"}):
        raise ValueError("Test queries are reserved for independent verification, not score-based screening")
    ledger = root / load_config(root)["review_file"]
    before = file_digest(ledger) if ledger.exists() else None
    preparation = preparation or run_data_preparation(root, render=False)
    cases = portable_frame(preparation["cases"], root)
    output = root / settings["output_dir"]
    # This call intentionally precedes all descriptor extraction and scoring.
    evaluation = freeze_evaluation(cases, root, settings)
    print(f"Frozen independent test queries: {len(evaluation['queries'])}; full gallery: {len(evaluation['gallery'])}", flush=True)
    pairs, timings = build_pair_table(cases, root, settings)
    rankings = rank_review_cases(pairs, cases, settings)
    queue = rankings[rankings.human_status.eq("pending") & rankings.issue_tags.ne("")].copy()
    queue = queue.sort_values(["numbering_anomaly", "alternative_margin", "original_score", "original_rank_min", "query_id"],
                              ascending=[False, False, True, False, True]).reset_index(drop=True)
    queue.insert(0, "review_priority", np.arange(1, len(queue) + 1))
    # Do not manufacture a workload of 60 "suspects" when all scores are tied.
    selected = queue[queue.priority_supported].head(settings["max_review_cases"]).copy()
    selected.insert(1, "batch", np.arange(len(selected)) // settings["batch_size"] + 1)
    output.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output / "pair_features.csv", index=False)
    rankings.to_csv(output / "screening_all.csv", index=False, encoding="utf-8-sig")
    queue.to_csv(output / "suspected_pairs.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(output / "review_batches.csv", index=False, encoding="utf-8-sig")
    links = {}
    if render:
        from .review_assets import render_targeted_assets
        links = render_targeted_assets(root, output, cases, selected, evaluation, settings)
    after = file_digest(ledger)
    if before is not None and before != after:
        raise RuntimeError("Review ledger changed during screening; no automatic label updates are permitted")
    summary = {"models_trained": False, "automatic_labels_written": False,
               "eligible_pairs": len(cases),
               "excluded_human_rejected": int(preparation["exclusions"].reason.eq("human_rejected").sum()),
               "human_verified_pairs": int(cases.human_verified.sum()),
               "filename_unverified_pairs": int((~cases.human_verified).sum()),
               "screened_splits": settings["screen_splits"], "screened_queries": len(rankings),
               "queued_for_optional_review": len(queue), "rendered_review_cases": len(selected) if render else 0,
               "queries_with_geometric_evidence": int(rankings.evidence_status.eq("geometry_available").sum()),
               "queries_without_geometric_evidence": int(rankings.evidence_status.eq("insufficient_geometry").sum()),
               "screening_quality": ("geometric_evidence_available" if rankings.evidence_status.eq("geometry_available").any()
                                     else "insufficient_geometry_for_reliable_prioritisation"),
               "batch_size": settings["batch_size"], "independent_test_queries": len(evaluation["queries"]),
               "full_test_gallery": len(evaluation["gallery"]),
               "independent_review_status": evaluation["review"].independent_status.value_counts().to_dict(),
               "test_protocol_frozen_before_features": True, "test_pair_scores_computed": 0,
               "ledger_sha256": after, "ledger_unchanged": before is None or before == after,
               "settings": settings, "opencv_version": cv2.__version__, "timings": timings,
               "feature_columns": FEATURE_COLUMNS}
    write_json(output / "screening_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"settings", "feature_columns", "timings"}}, indent=2), flush=True)
    return {"cases": cases, "pair_features": pairs, "rankings": rankings, "queue": queue,
            "batches": selected, "evaluation": evaluation, "summary": summary, "links": links, "output_dir": output}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    run_targeted_review(Path(__file__).resolve().parents[1], render=not args.no_render)
