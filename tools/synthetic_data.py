"""Small deterministic fixture for CI; it is not agricultural validation data."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def create_synthetic_dataset(root: Path, rows: int = 5, trees_per_row: int = 6) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    folder = root / "Dataset (With Summer GT)"
    folder.mkdir()
    for row in range(1, rows + 1):
        for number in range(1, trees_per_row + 1):
            for season, offset in [("Summer", 10000), ("Winter", 20000)]:
                rng = np.random.default_rng(offset + row * 100 + number)
                rgb = rng.integers(0, 160, (64, 96, 3), dtype=np.uint8)
                image = Image.fromarray(rgb)
                draw = ImageDraw.Draw(image)
                draw.line([(48, 63), (45, 34), (20 + number, 3)], fill=(220, 180, 130), width=4)
                draw.line([(45, 34), (70 + row, 3)], fill=(220, 180, 130), width=4)
                prefix = f"R{row:02d}N{number:02d}_{season}"
                image.save(folder / f"{prefix}_RGB-12-00-00.png")
                depth = rng.integers(20, 180, (64, 96), dtype=np.uint8)
                depth[:5] = 0
                Image.fromarray(np.repeat(depth[..., None], 3, axis=2)).save(folder / f"{prefix}_D-12-00-00.png")
                if season == "Summer":
                    mask = np.zeros((32, 32), dtype=np.uint8)
                    mask[10:30, 15:18] = 255
                    Image.fromarray(mask).save(folder / f"{prefix}_GT-12-00-00.png")
    # Two observations carrying the same U number do NOT establish their identity.
    unmatched = root / "Dataset"
    unmatched.mkdir()
    for season, seed in [("Summer", 91), ("Winter", 92)]:
        rng = np.random.default_rng(seed)
        image = rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)
        Image.fromarray(image).save(unmatched / f"R01U01_{season}_RGB-12-00-00.png")
        Image.fromarray(np.full((64, 96), 100, dtype=np.uint8)).save(unmatched / f"R01U01_{season}_D-12-00-00.png")
    return root
