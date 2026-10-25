# Installation and first run

Use Python 3.12 from the project directory. The primary workflow is
`apple_tree_matching.ipynb`; reusable functions are documented in the
[API reference](_generated/index.rst).

## Install the experiment environment

```powershell
& 'D:\anacoda3\envs\MMA3001\python.exe' -m pip install -r config/requirements.txt
& 'D:\anacoda3\envs\MMA3001\python.exe' -m ipykernel install --sys-prefix --name mma3001 --display-name 'MMA3001 (Python 3.12)'
```

On another machine, substitute your Python 3.12 interpreter. The optional neural
extension also requires `config/requirements-resnet.txt` and the verified official
checkpoint downloaded by `python tools/download_resnet_weights.py`. See the root
README for CPU/CUDA installation commands. The ORB workflow can run with
`--skip-resnet` without installing PyTorch.

## Supply the original data

Place the dataset at `data/TreeID/TreeID`, or set `MMA3001_DATA_ROOT`. An ignored
`config/local.json` can also override `data_root`:

```json
{"data_root": "D:/datasets/TreeID/TreeID"}
```

Configuration precedence is tracked JSON, local JSON, then the data-root
environment variable. Keep the seasonal filenames intact. RGB images are
required; unsupported or missing D is recorded with flags and missing values.
The [data contract](data_contract.md) describes naming and table schemas.

## Choose an execution scope

| Task | Command from the project root | Result |
| --- | --- | --- |
| Audit and row split | `python -m src.main --no-render` | Preparation manifests and quality checks |
| Targeted review | `python -m src.screening` | Frozen audit and development review pages |
| Linear baseline | `python -m src.baseline --no-render` | Shared samples and full development rankings |
| Main workflow | `python tools/run_notebook.py --in-place` | Main notebook execution and saved outputs |
| Core synthetic smoke run | `python tools/run_notebook.py --synthetic --skip-resnet` | Separate synthetic executed notebook |
| Development diagnostics | `python -m tools.feature_optimization --diagnostics-only` | Existing validation feature/rank analysis |
| Tests | `python -m pytest -q` | Synthetic/unit/integration checks |

An interactive run uses the MMA3001 notebook kernel and **Restart Kernel and Run
All Cells**. Real runs need the original data. Tests and synthetic runs do not
establish agricultural matching accuracy.

**Historical experiment locks include source-file hashes.** Changes to comments
or docstrings change those hashes too. After this documentation revision, a real
rerun against an older lock can raise an input-fingerprint error. Preserve its
lock/results and use separate experiment paths as explained in
[maintenance](maintenance.md#source-hashes-and-historical-results).

The supplied optimization copy is `apple_tree_matching_opt_attempt.ipynb`.
The runner's historical default and `tools.optimization_notebook.NOTEBOOK_NAME`
still refer to `apple_tree_matching_optimization.ipynb`, which is absent in this
checkout. Select the existing file explicitly when using the runner:

```powershell
python tools/run_notebook.py --notebook apple_tree_matching_opt_attempt.ipynb --optimization-only --in-place
```

This command selects configuration and cells tagged `feature_optimization`.
Cells that call the separate notebook-export wrapper still require its configured
notebook name to exist. Use `python -m tools.feature_optimization` for the direct
optimization/export entry point. The documentation revision does not rename
notebooks or alter those historical runner/export defaults.

## Build and open the HTML documentation

Use a separate documentation environment so adding Sphinx does not change the
recorded experiment environment:

```powershell
python -m venv .cache/docs-env
& '.cache/docs-env/Scripts/python.exe' -m pip install -r config/requirements-docs.txt
& '.cache/docs-env/Scripts/python.exe' tools/build_docs.py
Start-Process 'doc/_build/html/index.html'
```

On Linux/macOS, use `.cache/docs-env/bin/python`. Building requires no images,
pretrained weights or fitted models. The command checks source documentation,
rebuilds every Sphinx page with warnings treated as errors, and verifies relative
links, local assets and fragments. Open `doc/_build/html/index.html` directly, or
serve `doc/_build/html` with `python -m http.server --directory doc/_build/html`.
The site includes navigation, search, API indices and readable source listings.

For a dependency-free source-docstring check:

```powershell
python tools/build_docs.py --check
```
