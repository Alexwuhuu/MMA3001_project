"""Execute the notebook with the current Python, without global kernel installation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.synthetic_data import create_synthetic_dataset


def main():
    parser = argparse.ArgumentParser(description="Execute the notebook through numerical checks, linear baseline, and SVR selection")
    parser.add_argument("--synthetic", action="store_true", help="Use tiny synthetic CI data, never real data")
    parser.add_argument("--in-place", action="store_true", help="Save real-data outputs into the root notebook")
    parser.add_argument("--resnet-only", action="store_true",
                        help="Execute only the frozen-ResNet extension using existing stage-5 artifacts")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--kernel-python", type=Path,
                        help="Existing kernel Python; automation tools stay in this controller environment")
    arguments = parser.parse_args()
    if arguments.synthetic and arguments.in_place:
        parser.error("Synthetic results must not replace the real-data notebook")
    if arguments.synthetic and arguments.resnet_only:
        parser.error("Synthetic checks run the complete workflow")
    kernel_python = arguments.kernel_python or Path(sys.executable)
    if not kernel_python.is_file():
        parser.error(f"Kernel Python not found: {kernel_python}")
    notebook_path = ROOT / "apple_tree_matching.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    selected_indices = ([2] + [i for i, cell in enumerate(notebook.cells)
                              if "resnet_extension" in cell.metadata.get("tags", [])]) if arguments.resnet_only else None
    execution_notebook = (nbformat.v4.new_notebook(cells=[deepcopy(notebook.cells[i]) for i in selected_indices])
                          if selected_indices else notebook)
    output = arguments.output or (notebook_path if arguments.in_place else ROOT / "outputs" / (
        "synthetic_executed.ipynb" if arguments.synthetic else "data_preparation_executed.ipynb"))
    output.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / ".cache").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="notebook_", dir=ROOT / ".cache") as temporary:
        temp = Path(temporary)
        # Keep Jupyter/IPython state inside the project; do not alter user profiles.
        os.environ["IPYTHONDIR"] = str(temp / "ipython")
        os.environ["JUPYTER_RUNTIME_DIR"] = str(temp / "runtime")
        os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache" / "matplotlib")
        os.environ["PYTHONUTF8"] = "1"
        if arguments.synthetic:
            fixture = create_synthetic_dataset(temp / "synthetic_treeid")
            os.environ["MMA3001_DATA_ROOT"] = str(fixture)
            os.environ["MMA3001_SYNTHETIC"] = "1"
        specs = temp / "kernels"
        kernel_folder = specs / "mma3001-local"
        kernel_folder.mkdir(parents=True)
        kernel = {"argv": [str(kernel_python.resolve()), "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                  "display_name": "MMA3001 project Python", "language": "python"}
        (kernel_folder / "kernel.json").write_text(json.dumps(kernel), encoding="utf-8")
        manager = KernelManager(kernel_name="mma3001-local",
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(specs)], ensure_native_kernel=False))
        client = NotebookClient(execution_notebook, km=manager, timeout=900, allow_errors=False,
                                resources={"metadata": {"path": str(ROOT)}})
        def progress(cell, cell_index, **kwargs):
            if cell.cell_type == "code":
                print(f"Executing code cell {cell_index + 1}/{len(execution_notebook.cells)}", flush=True)
        client.on_cell_start = progress
        try:
            client.execute()
        finally:
            if selected_indices:
                for index, cell in zip(selected_indices, execution_notebook.cells):
                    notebook.cells[index] = cell
            notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
            nbformat.write(notebook, output)
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
    print(f"Executed notebook saved: {output}", flush=True)


if __name__ == "__main__":
    main()
