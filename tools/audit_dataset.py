"""Run Session 1 Part 1 from any working directory, without extra packages."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from apple_tree.data_audit import audit_dataset


def main() -> int:
    """Write audit evidence and return nonzero for file/pairing errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "manifests" / "session01_part01")
    args = parser.parse_args()
    summary = audit_dataset(args.project_root, args.output_dir)
    print(json.dumps(summary, indent=2))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
