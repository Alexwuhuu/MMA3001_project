"""Small tests for the basic file-checking functions."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from apple_tree.data_audit import DATASET_FOLDERS, inspect_png, pair_records, parse_filename


def record(name):
    return {"dataset": DATASET_FOLDERS[0], "path": name, **parse_filename(name)}


class DataCheckTests(unittest.TestCase):
    def test_normal_filename(self):
        result = parse_filename("R01U01_Summer_RGB-12-40-11.png")
        self.assertEqual(result["tree_id"], "R01U01")
        self.assertEqual(result["modality"], "RGB")

    def test_embedded_identifier_is_flagged(self):
        result = parse_filename("R04N01_Winter_D-R04N05_RGB_12-17-20.png")
        self.assertTrue(result["needs_identity_review"])
        self.assertEqual(result["embedded_tree_id"], "R04N05")

    def test_missing_depth_is_reported(self):
        _, issues = pair_records([record("R01U01_Summer_RGB-1-2-3.png")])
        self.assertEqual(issues[0]["code"], "missing_modality")

    def test_complete_pair_is_accepted(self):
        rows, issues = pair_records([
            record("R01U01_Summer_RGB-1-2-3.png"),
            record("R01U01_Summer_D-1-2-3.png"),
        ])
        self.assertEqual(len(rows), 1)
        self.assertEqual(issues, [])

    def test_non_png_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "wrong.png"
            path.write_bytes(b"not an image")
            with self.assertRaises(ValueError):
                inspect_png(path)


if __name__ == "__main__":
    unittest.main()
