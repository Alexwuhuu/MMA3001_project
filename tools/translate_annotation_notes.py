"""Translate existing annotation notes without changing any human decisions.

Original files are backed up under the ignored .cache directory before edits.
Only exact, reviewed translations are applied; unknown Chinese notes fail early.
"""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import html
import io
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
TRANSLATIONS = {
    "主干弯曲和低处分叉一致，确认同树": "Matching trunk curvature and lower branch junctions; same tree confirmed.",
    "夏季主干被完全遮挡，无法确认": "The summer trunk is fully occluded; cannot confirm the identity.",
    "主干结节,弯曲不同": "Different trunk knots and curvature.",
    "主干结节不同": "Different trunk knots.",
    "主干弯曲不同": "Different trunk curvature.",
    "枝干不同": "Different branch structure.",
    "支干不同": "Different secondary-branch structure.",
    "主干枝条形状一致": "Matching trunk and branch shapes.",
    "枝干被树叶遮挡, 无法识别": "Branches are occluded by foliage; cannot identify the tree.",
}
CJK = re.compile(r"[\u3400-\u9fff]")


def translate_notes(root: Path = ROOT) -> dict:
    """Translate only exact known annotation notes after backing up every changed file.

    Non-note fields are checked unchanged. Displayed CSV/HTML/notebook copies are
    updated too; this migration can change artifact bytes and should be run only
    when intentionally migrating annotations, never as a documentation-build step.

    Args:
        root (Path): Project with config/outputs ledgers and displayed notebook/HTML copies.

    Returns:
        dict: Change/backup/protected-file evidence, or changed_files=0 when no migration is
        needed.

    Raises:
        ValueError: An unknown Chinese note is encountered before writes.
        AssertionError: A non-note CSV field changes.
        OSError: Source/backup/output files are inaccessible.
    """
    root = root.resolve()
    planned, evidence = [], []
    for folder in (root / "config", root / "outputs"):
        for path in sorted(folder.rglob("*.csv")):
            raw = path.read_bytes()
            try:
                content = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                content = raw.decode("gb18030")
            rows = list(csv.reader(io.StringIO(content)))
            if not rows:
                continue
            columns = [i for i, name in enumerate(rows[0]) if "note" in name.lower()]
            original = [row[:] for row in rows]
            changed = 0
            for row in rows[1:]:
                for column in columns:
                    value = row[column]
                    if value in TRANSLATIONS:
                        row[column] = TRANSLATIONS[value]
                        changed += 1
                    elif CJK.search(value):
                        raise ValueError(f"Untranslated note in {path.relative_to(root)}: {value!r}")
            if not changed:
                continue
            for before, after in zip(original, rows):
                assert len(before) == len(after)
                assert all(before[i] == after[i] for i in range(len(before)) if i not in columns)
            buffer = io.StringIO(newline="")
            csv.writer(buffer, lineterminator="\n").writerows(rows)
            planned.append((path, buffer.getvalue().encode("utf-8-sig")))
            evidence.append({"file": path.relative_to(root).as_posix(), "translated_notes": changed,
                             "non_note_fields_unchanged": True})

    displayed = list((root / "outputs").rglob("*.html")) + list(root.glob("*.ipynb"))
    displayed += list((root / "outputs").glob("*.ipynb"))
    for path in sorted(displayed):
        content = path.read_text(encoding="utf-8")
        updated = content
        for before, after in TRANSLATIONS.items():
            updated = updated.replace(before, after).replace(html.escape(before), html.escape(after))
            # Some notebook serializers escape non-ASCII output strings.
            updated = updated.replace(json.dumps(before)[1:-1], json.dumps(after)[1:-1])
        if updated != content:
            planned.append((path, updated.encode("utf-8")))
            evidence.append({"file": path.relative_to(root).as_posix(), "displayed_notes_updated": True})
    if not planned:
        return {"changed_files": 0, "message": "All annotation notes are already in English."}
    backup = root / ".cache" / ("annotation_notes_original_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    for path, _ in planned:
        saved = backup / path.relative_to(root)
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, saved)
    for path, data in planned:
        path.write_bytes(data)
    report = {"changed_files": len(planned), "backup_directory": backup.relative_to(root).as_posix(),
              "identity_decisions_changed": False, "files": evidence,
              "frozen_file_sha256": {name: hashlib.sha256((root / "config/evaluation" / name).read_bytes()).hexdigest()
                                      for name in ("queries.csv", "gallery.csv", "protocol.json")}}
    (backup / "translation_map.json").write_text(json.dumps(TRANSLATIONS, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "outputs/annotation_note_translation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(translate_notes(), indent=2))
