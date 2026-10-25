"""Regression coverage for selecting and exporting the supplied notebook copy."""
import json
import os
from pathlib import Path
import sys

import nbformat

from src.reproduce import verify_export
from src.utils import file_digest, write_json
from tools import optimization_notebook, run_notebook

ROOT = Path(__file__).resolve().parents[1]
OPTIMIZATION_COPY = ROOT / "apple_tree_matching_opt_attempt.ipynb"


def test_optimization_only_default_executes_supplied_copy_without_historical_cells(tmp_path, monkeypatch):
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("Synthetic runner regression fixture"),
        nbformat.v4.new_code_cell("raise AssertionError('historical cell must stay unexecuted')"),
        nbformat.v4.new_code_cell("configured = 'configuration ready'"),
        nbformat.v4.new_code_cell("print(configured + ': optimization executed')",
                                 metadata={"tags": ["feature_optimization"]}),
        nbformat.v4.new_code_cell("raise AssertionError('test evaluation must stay unexecuted')"),
    ])
    source = tmp_path / OPTIMIZATION_COPY.name
    nbformat.write(notebook, source)
    before = source.read_bytes()
    output = tmp_path / "outputs/executed.ipynb"
    monkeypatch.setattr(run_notebook, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_notebook.py", "--optimization-only", "--output", str(output)])
    # Restore runner-managed environment values after this real kernel execution.
    for key in ("IPYTHONDIR", "JUPYTER_RUNTIME_DIR", "MPLCONFIGDIR", "PYTHONUTF8"):
        monkeypatch.setenv(key, os.environ.get(key, ""))

    run_notebook.main()

    executed = nbformat.read(output, as_version=4)
    assert executed.cells[2].execution_count == 1
    assert executed.cells[3].execution_count == 2
    assert executed.cells[3].outputs[0].text == "configuration ready: optimization executed\n"
    assert executed.cells[1].execution_count is None
    assert executed.cells[4].execution_count is None
    assert source.read_bytes() == before


def test_export_packages_supplied_copy_clears_outputs_and_verifies_manifest(tmp_path, monkeypatch):
    source = tmp_path / OPTIMIZATION_COPY.name
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        "print('saved result')", execution_count=7,
        outputs=[nbformat.v4.new_output("stream", name="stdout", text="saved result\n")],
    )])
    nbformat.write(notebook, source)
    before = source.read_bytes()
    output = tmp_path / "export"
    results = tmp_path / "results"
    output.mkdir()
    results.mkdir()

    def base_export(root, result, settings):
        instructions = output / "RUN.md"
        instructions.write_text(
            "then use the main notebook's optimization-only\n"
            "The primary single notebook also retains the previous experiments. This bundle",
            encoding="utf-8",
        )
        write_json(output / "replay.json", {"models": {}})
        write_json(output / "manifest.json", {"files": {
            name: {"sha256": file_digest(output / name), "bytes": (output / name).stat().st_size}
            for name in ("RUN.md", "replay.json")
        }})
        return {"export_dir": str(output)}

    monkeypatch.setattr(optimization_notebook, "_export_optimization", base_export)
    summary = optimization_notebook.export_optimization(tmp_path, {"output_dir": results})

    exported = nbformat.read(output / OPTIMIZATION_COPY.name, as_version=4)
    assert exported.cells[0].outputs == []
    assert exported.cells[0].execution_count is None
    manifest = json.loads((output / "manifest.json").read_text())
    assert OPTIMIZATION_COPY.name in manifest["files"]
    assert OPTIMIZATION_COPY.name in (output / "RUN.md").read_text()
    assert summary["checksummed_files"] == 3
    assert verify_export(output)["checksummed_files"] == 3
    assert source.read_bytes() == before
