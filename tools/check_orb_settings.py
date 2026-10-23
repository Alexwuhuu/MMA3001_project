"""Training-only diagnostic: distinguish extra correspondences from reliable identity evidence."""
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import extract_image_features, extract_pair_features, feature_signature
from src.screening import load_screening_config
from src.utils import write_json


def run_pilot(root, cases, settings=None):
    """Use up to 20 human-confirmed training pairs plus 10 filename controls each.

    Different N IDs are nominal negative controls, not independently established
    pairwise truth. These counts diagnose matching availability; they are not
    test accuracy, false-positive rates, or a fitted prediction model.
    """
    root = Path(root)
    settings = dict(settings or load_screening_config(root))
    train = cases[cases.split.eq("train")].copy()
    verified = train[train.human_verified.astype(str).str.lower().eq("true")]
    output = root / settings["output_dir"] / "pilot"
    output.mkdir(parents=True, exist_ok=True)
    if verified.empty:
        result = {"split": "train", "status": "no_human_confirmed_training_pairs", "results": [], "test_used": False}
        write_json(output / "pilot_summary.json", result)
        return result
    positive = verified.sample(n=min(20, len(verified)), random_state=42)
    rng = np.random.default_rng(42)
    pairs = [(q, q, True) for q in positive.to_dict("records")]
    for q in positive.to_dict("records"):
        controls = train[~train.tree_id.eq(q["tree_id"])]
        controls = controls.sample(n=min(10, len(controls)), random_state=int(rng.integers(1_000_000)))
        pairs.extend((q, c, False) for c in controls.to_dict("records"))
    configs = [("planned", 256, 500, 0.75), ("larger_085", 384, 1000, 0.85), ("larger_090", 384, 1000, 0.90)]
    summaries = []
    cv2.setNumThreads(1)
    for name, size, number, ratio in configs:
        current = dict(settings, image_size=size, orb_nfeatures=number,
                       ratio_threshold=ratio, ransac_threshold_px=3 * size / 256)
        started = perf_counter()
        cache = root / settings["cache_dir"] / "pilot" / feature_signature(current)
        paths = {q["summer_rgb"] for q, _, _ in pairs} | {c["winter_rgb"] for _, c, _ in pairs}
        features = {path: extract_image_features(root / path, current, cache) for path in sorted(paths)}
        values = []
        for q, c, is_original in pairs:
            numeric = extract_pair_features(features[q["summer_rgb"]], features[c["winter_rgb"]], current)
            values.append(dict(query_id=q["tree_id"], candidate_id=c["tree_id"], is_original=is_original, **numeric))
        frame = pd.DataFrame(values)
        frame.to_csv(output / f"{name}.csv", index=False)
        summaries.append({
            "setting": name, "image_size": size, "orb_nfeatures": number, "ratio_threshold": ratio,
            "ransac_threshold_px": current["ransac_threshold_px"], "originals": len(positive),
            "negative_controls": len(frame) - len(positive),
            "originals_with_6_inliers": int((frame.is_original & frame.ransac_inliers.ge(6)).sum()),
            "controls_with_6_inliers": int((~frame.is_original & frame.ransac_inliers.ge(6)).sum()),
            "original_median_good_matches": float(frame.loc[frame.is_original, "good_matches"].median()),
            "original_median_inliers": float(frame.loc[frame.is_original, "ransac_inliers"].median()),
            "seconds": perf_counter() - started,
        })
        print(f"Training-only ORB diagnostic: {summaries[-1]}", flush=True)
    result = {"split": "train", "selection": "up to 20 human-confirmed pairs; 10 different-filename controls per query",
              "results": summaries, "test_used": False, "changes_default_settings": False}
    write_json(output / "pilot_summary.json", result)
    return result


if __name__ == "__main__":
    cases = pd.read_csv(ROOT / "outputs/data_preparation/split_manifest.csv", keep_default_na=False)
    run_pilot(ROOT, cases)
