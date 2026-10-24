"""Fetch the pinned official ImageNet checkpoint into the ignored project cache."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.deep_features import download_weights, verify_weights

if __name__ == "__main__":
    settings = json.loads((ROOT / "config/resnet.json").read_text())
    path = download_weights(ROOT, settings)
    print("Official weights:", path)
    print("SHA-256:", verify_weights(path, settings))
