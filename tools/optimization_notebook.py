"""Package the separate optimization notebook without changing locked model code."""
from pathlib import Path
import json

import nbformat

from tools.feature_optimization import (
    ROOT, export_optimization as _export_optimization, file_digest,
    run_optimization, verify_export, write_json,
)

NOTEBOOK_NAME = "apple_tree_matching_opt_attempt.ipynb"


def export_optimization(root, result, settings=None):
    """Extend the separate exporter with the named optimization notebook and entry point.

    Args:
        root (str | Path): Project containing NOTEBOOK_NAME and original artifacts.
        result (dict): Completed separate optimization result.
        settings (dict | None): Export settings; None delegates to the result settings.

    Returns:
        dict: Export path, updated manifest hash and verified checksum/model replays. Notebook
        code outputs are cleared in the export copy.

    Raises:
        FileNotFoundError: NOTEBOOK_NAME is absent.
        ValueError: Export integrity validation fails.
        AssertionError: Numeric replay differs.
        OSError: Notebook, instructions or manifest cannot be written.
    """
    root = Path(root)
    notebook = nbformat.read(root / NOTEBOOK_NAME, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    summary = _export_optimization(root, result, settings)
    output = Path(summary["export_dir"])
    nbformat.write(notebook, output / NOTEBOOK_NAME)
    instructions = output / "RUN.md"
    text = instructions.read_text(encoding="utf-8").replace(
        "then use the main notebook's optimization-only", "then use the optimization copy's")
    text = text.replace(
        "The primary single notebook also retains the previous experiments. This bundle",
        f"`{NOTEBOOK_NAME}` contains the new section 9 and retains stages 1-8 for context.\n"
        "`apple_tree_matching.ipynb` contains the original stages 1-8. The command above\n"
        "executes only configuration and section 9 in the optimization copy. This bundle")
    instructions.write_text(text, encoding="utf-8")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in (NOTEBOOK_NAME, "RUN.md"):
        path = output / name
        manifest["files"][name] = {"sha256": file_digest(path), "bytes": path.stat().st_size}
    write_json(manifest_path, manifest)
    summary = {"export_dir": str(output), "manifest_sha256": file_digest(manifest_path),
               **verify_export(output)}
    write_json(result["output_dir"] / "export_summary.json", summary)
    return summary


def main():
    """Run and export the separate optimization experiment with noninteractive plots.

    Returns:
        None: The comparison and export summary are printed.

    Raises:
        ValueError: Development inputs or lock validation fails.
        OSError: Required artifacts or the named notebook are inaccessible.
    """
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
    result = run_optimization(ROOT)
    print(result["comparison"].to_string(index=False))
    print(export_optimization(ROOT, result))


if __name__ == "__main__":
    main()
