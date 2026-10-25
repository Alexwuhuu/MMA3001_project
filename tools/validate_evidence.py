"""Independently audit saved evaluation scores without fitting or opening images.

Uses standard-library arithmetic rather than the production ranking/metric code.
Run from the project root with ``python -m tools.validate_evidence``; only an
explicit --output path writes a new aggregate report.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist, mean


def read_rows(path):
    """Read a UTF-8 CSV without interpreting paths or loading binary artifacts.

    Args:
        path (str | Path): Existing table path.

    Returns:
        list[dict]: CSV rows, with values retained as strings.

    Raises:
        OSError: The table cannot be read.
    """
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def require_close(actual, expected, label):
    """Reject a non-finite or inconsistent saved numerical value.

    Args:
        actual (float): Recomputed reference value.
        expected (float): Saved value to check.
        label (str): Context for a failed comparison.

    Returns:
        None: Both values agree within absolute 1e-12 and relative 1e-10 tolerance.

    Raises:
        ValueError: A value is non-finite or disagrees with the reference.
    """
    if not (math.isfinite(actual) and math.isfinite(float(expected)) and
            math.isclose(actual, float(expected), rel_tol=1e-10, abs_tol=1e-12)):
        raise ValueError(f"Saved evidence mismatch: {label}")


def reference_query(rows, atol=1e-12, rtol=1e-10):
    """Recompute a query's tie-aware metrics and check every saved rank column.

    Args:
        rows (list[dict]): Unique candidate rows for one query, including its own ID.
        atol (float): Nonnegative absolute score tolerance.
        rtol (float): Nonnegative relative tolerance, anchored to each tie leader.

    Returns:
        dict: Independently recomputed top1, top5 and mrr values.

    Raises:
        ValueError: Scores, query/candidate identities, tolerances or saved ranks are invalid.
        KeyError: A required ranking column is absent.
    """
    if not rows or atol < 0 or rtol < 0 or not math.isfinite(atol + rtol):
        raise ValueError("Nonempty rankings and finite nonnegative tolerances required")
    queries = {r["query_id"] for r in rows}
    candidates = [r["candidate_id"] for r in rows]
    if len(queries) != 1 or len(set(candidates)) != len(candidates) or next(iter(queries)) not in candidates:
        raise ValueError("Each query needs unique candidates and exactly one original identity")
    ordered = sorted(rows, key=lambda row: (-float(row["score"]), row["candidate_id"]))
    if not all(math.isfinite(float(row["score"])) for row in ordered):
        raise ValueError("Scores must be finite")
    start, result = 0, {}
    while start < len(ordered):
        stop = start + 1
        leader = float(ordered[start]["score"])
        while stop < len(ordered) and abs(float(ordered[stop]["score"]) - leader) <= atol + rtol * abs(leader):
            stop += 1
        for index in range(start, stop):
            row = ordered[index]
            for column, value in {"display_rank": index + 1, "rank_min": start + 1, "rank_max": stop}.items():
                require_close(value, row[column], f"{row['query_id']} {column}")
            if row["candidate_id"] == row["query_id"]:
                positions = range(start + 1, stop + 1)
                result = {"top1": mean(int(p <= 1) for p in positions),
                          "top5": mean(int(p <= 5) for p in positions),
                          "mrr": mean(1 / p for p in positions)}
        start = stop
    return result


def audit_saved_evaluation(folder, query_ids, gallery_ids, atol=1e-12, rtol=1e-10):
    """Audit full-gallery saved rankings against frozen IDs, human states and metrics.

    Reads numeric CSV/JSON only. It neither validates human identity decisions
    afresh nor re-estimates bootstrap intervals or population-level accuracy.

    Args:
        folder (str | Path): Final experiment directory with test rankings and metrics.
        query_ids (sequence[str]): Frozen query IDs from the evaluation protocol.
        gallery_ids (sequence[str]): Complete frozen winter candidate IDs, at least two.
        atol (float): Absolute numerical tie tolerance used for the saved experiment.
        rtol (float): Relative numerical tie tolerance used for the saved experiment.

    Returns:
        dict: Aggregate audit counts, independently recomputed metrics, analytical
        references and source-file SHA-256 values; no image paths or reviewer notes.

    Raises:
        ValueError: IDs, audit states, coverage, saved ranks or metrics disagree.
        KeyError: A required field is absent.
        OSError: A required saved evidence file cannot be read.
    """
    folder = Path(folder)
    queries, gallery = set(query_ids), set(gallery_ids)
    if (not queries or len(gallery) < 2 or len(queries) != len(query_ids) or
            len(gallery) != len(gallery_ids) or not queries <= gallery):
        raise ValueError("Frozen IDs must be unique and closed-set, with queries and at least two candidates")
    paths = [folder / "test_metrics.json", folder / "independent_review_normalized.csv"]
    recorded = json.loads(paths[0].read_text(encoding="utf-8"))
    audit_rows = read_rows(paths[1])
    statuses = {r["tree_id"]: r["independent_status"] for r in audit_rows}
    if (len(statuses) != len(audit_rows) or set(statuses) != queries or
            not set(statuses.values()) <= {"confirmed", "rejected", "uncertain", "pending"}):
        raise ValueError("Normalized human audit must cover every frozen query exactly once")
    confirmed = {q for q in queries if statuses[q] == "confirmed"}
    counts = dict(Counter(statuses.values()))
    size = len(gallery)
    random = {"top1": 1 / size, "top5": min(5, size) / size,
              "mrr": mean(1 / p for p in range(1, size + 1))}
    report = {"queries": len(queries), "gallery_size": size, "audit_counts": counts,
              "uniform_ranking_reference": random,
              "constant_zero_reference": {"all_pairs_mae": 1 / size,
                  "all_pairs_rmse": 1 / math.sqrt(size), "positive_mae": 1.,
                  "negative_mae": 0., "balanced_class_mae": .5}, "models": {}}
    if not recorded:
        raise ValueError("No saved model metrics")
    for model, metrics in recorded.items():
        if Path(model).name != model or "/" in model or "\\" in model:
            raise ValueError("Model names must be plain filenames")
        path = folder / f"{model}_test_rankings.csv"
        paths.append(path)
        rows = read_rows(path)
        groups = defaultdict(list)
        for row in rows:
            if row["split"] != "test":
                raise ValueError("Only frozen test rankings are valid audit inputs")
            groups[row["query_id"]].append(row)
        if set(groups) != queries:
            raise ValueError("Rankings do not cover the frozen query set")
        per_query = {}
        for query, group in groups.items():
            if len(group) != size or {r["candidate_id"] for r in group} != gallery:
                raise ValueError("Incomplete or duplicated full gallery")
            per_query[query] = reference_query(group, atol, rtol)
        if (metrics["queries_sampled"] != len(queries) or metrics["gallery_size"] != size or
                metrics["queries_verified"] != len(confirmed) or metrics["audit_counts"] != counts):
            raise ValueError("Saved sample, gallery or audit counts disagree")
        verified = {key: mean(per_query[q][key] for q in sorted(confirmed)) if confirmed else None
                    for key in random}
        if not confirmed and metrics["verified_query_regression"] is not None:
            raise ValueError("Unverified cases cannot define verified regression error")
        for key in random:
            if confirmed:
                require_close(verified[key], metrics["verified_retrieval"][key], f"{model} verified {key}")
            elif metrics["verified_retrieval"][key] is not None:
                raise ValueError("Unverified cases cannot define verified accuracy")
            diagnostic = metrics["all_sample_filename_diagnostic"]["retrieval"]
            require_close(mean(per_query[q][key] for q in queries), diagnostic[key], f"{model} filename {key}")
            require_close(random[key], diagnostic["uniform_random_reference"][key], f"{model} random {key}")
        errors = {"positive_pairs": [], "negative_pairs": []}
        for row in rows:
            if row["query_id"] in confirmed:
                positive = row["query_id"] == row["candidate_id"]
                errors["positive_pairs" if positive else "negative_pairs"].append(float(row["score"]) - int(positive))
        errors["all_pairs"] = errors["positive_pairs"] + errors["negative_pairs"]
        recomputed_errors = {}
        for label, values in errors.items():
            if values:
                expected = metrics["verified_query_regression"][label]
                if expected["count"] != len(values):
                    raise ValueError("Saved regression sample count disagrees")
                recomputed_errors[label] = {"count": len(values), "mae": mean(abs(v) for v in values),
                                           "rmse": math.sqrt(mean(v * v for v in values))}
                for key in ("mae", "rmse"):
                    require_close(recomputed_errors[label][key], expected[key], f"{model} {label} {key}")
        if confirmed:
            balanced = mean(recomputed_errors[label]["mae"] for label in ("positive_pairs", "negative_pairs")
                            if label in recomputed_errors)
            require_close(balanced, metrics["verified_query_regression"]["balanced_class_mae"], f"{model} balanced MAE")
            z = NormalDist().inv_cdf(.975)
            for key in ("top1", "top5"):
                if all(per_query[q][key] in (0, 1) for q in confirmed):
                    n, p = len(confirmed), verified[key]
                    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
                    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
                    interval = metrics["intervals_95"][key]
                    if interval["method"] != "Wilson_95":
                        raise ValueError("Binary Top-k requires the recorded Wilson convention")
                    require_close(max(0., centre - radius), interval["lower"], f"{model} {key} lower")
                    require_close(min(1., centre + radius), interval["upper"], f"{model} {key} upper")
        report["models"][model] = {"pairs_checked": len(rows), "verified_queries": len(confirmed),
                                   "verified_retrieval": verified, "regression": recomputed_errors}
    report["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report["audit_implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["passed"] = True
    report["scope"] = ("Saved numeric evidence only; human decisions are inputs. No image inference, fitting, "
                       "selection or bootstrap re-estimation; no population accuracy guarantee.")
    return report


def main():
    """Audit an existing experiment and optionally save a new aggregate JSON report.

    Returns:
        None: Prints a report and writes only an explicitly requested output file.

    Raises:
        ValueError: An input is invalid or evidence is inconsistent.
        OSError: An input/output file cannot be accessed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("outputs/final_experiments"))
    parser.add_argument("--queries", type=Path, default=Path("config/evaluation/queries.csv"))
    parser.add_argument("--gallery", type=Path, default=Path("config/evaluation/gallery.csv"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_saved_evaluation(args.results, [r["tree_id"] for r in read_rows(args.queries)],
                                    [r["tree_id"] for r in read_rows(args.gallery)])
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
