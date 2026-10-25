# MMA3001 — Cross-season Apple-tree Matching

Rank winter apple-tree images for a summer RGB query with optional D to associate observations across seasons.
**RBF-SVR is the main method; weighted Linear Regression is the baseline.**
Scores are continuous ranking values, not probabilities or automatic identity confirmations.
The main models retrieved no confirmed test identity in their first five positions (0/28).
This is a reproducible research prototype; reliable operational matching is not established.

## Start here

- [Main notebook](apple_tree_matching.ipynb): authoritative stages 1–8, executed outputs and conclusions.
- [First-run guide](doc/quickstart.md): installation, local paths and execution scopes.
- [Workflow and API](doc/workflow.md): saved-model inference, ties and failure handling.
- [Data contract](doc/data_contract.md): image/table formats, units and label provenance.
- [Experiment reference](doc/experiment_reference.md): methods, optional runs and recorded results.
- [Contributing](CONTRIBUTING.md): naming, branches, meaningful commits and review checks.

## Repository structure

```text
MMA3001_project/
├── apple_tree_matching.ipynb              # Main workflow and project entry point
├── apple_tree_matching_opt_attempt.ipynb  # Optional validation-only feature experiment
├── src/                                  # Reusable matching, evaluation and export code
├── tools/                                # Notebook runners, diagnostics and documentation tools
├── tests/                                # Unit and synthetic integration checks
├── config/                               # Settings, pinned requirements and selection locks
│   └── evaluation/                       # Frozen queries/gallery and independent human audit
├── doc/                                  # Setup, contracts, API and detailed experiment guides
├── res/                                  # Small versioned aggregate tables and figures
├── .github/                              # CI workflow and pull-request template
├── .gitattributes                        # Stable source/protocol line endings
├── .gitignore                            # Local data, environments and generated artifacts
├── CONTRIBUTING.md                       # Repository and change-management conventions
├── LICENSE                               # Code license; does not license the dataset
├── data/TreeID/TreeID/                    # User-supplied original data; ignored
├── outputs/                              # Generated models, rankings and exports; ignored
└── .cache/                               # Checkpoints and feature caches; ignored
```

Python files use `snake_case`; source, configuration, tests, guides and results have separate roles.

## Environment and dependencies

After cloning via GitHub's **Code** menu, use **Python 3.12** from the repository root; core stages need no GPU or PyTorch.
[requirements.txt](config/requirements.txt) pins the tested core environment;
[requirements.in](config/requirements.in) lists direct dependencies.

Create a local environment (PowerShell):

```powershell
python -m venv .venv
& ./.venv/Scripts/Activate.ps1
python -m pip install -r config/requirements.txt
python -m ipykernel install --sys-prefix --name mma3001 --display-name "MMA3001 (Python 3.12)"
```

Run the first command with your Python 3.12 interpreter. On Linux/macOS, activate with
`source .venv/bin/activate`; subsequent `python` commands are the same.

Optional frozen ResNet18/PCA features use [requirements-resnet.txt](config/requirements-resnet.txt).
Install the core requirements first, then the CPU stack and verified official checkpoint:

```powershell
python -m pip install -r config/requirements-resnet.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
python tools/download_resnet_weights.py
```

The [first-run guide](doc/quickstart.md#optional-resnet-environment) also documents CUDA installation.
Run All never downloads weights; the optional branch runs only when packages and weights exist.

## Supply and protect the data

The original TreeID images are supplied separately and are not included in Git.
Place the dataset at `data/TreeID/TreeID/`, containing `Dataset/`,
`Dataset (With Summer GT)/` and the numbering diagram `FANK_Sat.png`.
Alternatively, create the ignored `config/local.json`:

```json
{"data_root": "D:/datasets/TreeID/TreeID"}
```

`MMA3001_DATA_ROOT` overrides the configured path; keep original filenames intact.
RGB is required. D is optional uint8 intensity (0–255), with unknown physical units.
GT masks, paths, IDs, row numbers and timestamps never enter the numeric matching features.
R identifies an orchard row, N a nominal matched tree and U an unmatched observation.
U has no verified cross-season identity label and is excluded from supervised evaluation.

Human decisions are in `config/pair_review.csv`; the independent audit is in
`config/evaluation/independent_review.csv`. Automated scores never create confirmations.
Do not replace rejected/unresolved test queries or infer truth from a predicted match.
`data/`, `outputs/`, `.cache/`, `.venv/` and local path settings are ignored.
Executed notebooks can contain rendered source photographs; review permissions before sharing them.
The code license does not grant rights to the data; see [provenance and AI use](doc/ai_usage.md).

## Reproduce the workflow

Choose a route for the inputs available. Historical locks contain source, data and environment fingerprints; a new checkout/environment
may not match them. Preserve historical locks and use the
[separate-experiment procedure](doc/maintenance.md#source-hashes-and-historical-results)
for deliberate new runs. Do not edit a lock to bypass a mismatch.

### 1. Synthetic validation: no original images required

```powershell
python tools/run_notebook.py --synthetic --skip-resnet
python -m pytest -q
```

The runner writes `outputs/synthetic_executed.ipynb` without replacing the main notebook.
These checks exercise software behavior; they are not evidence of agricultural accuracy.
Optional neural tests require the additional packages/checkpoint and otherwise skip.

### 2. Original-image experiment

After supplying data and checking lock/environment compatibility, open the notebook:

```powershell
python -m jupyterlab apple_tree_matching.ipynb
```

Select the installed **MMA3001** kernel and **Restart Kernel and Run All Cells**.
For execution without a browser, choose one of these commands:

```powershell
python tools/run_notebook.py --skip-resnet --in-place
python tools/run_notebook.py --in-place
```

The first runs core stages; the second also enables ResNet when available.
Core-only final results use separate paths from the four-model experiment.
Both commands save executed outputs in the main notebook. Original images are read-only.

| Stage | Work performed |
| --- | --- |
| 1–2 | Audit observations, retain human provenance and split complete rows 12/4/4, seed 42 |
| 3 | Extract ORB/RANSAC features and prepare targeted development review |
| 4 | Verify affine least squares with independent known-transform probe points |
| 5 | Fit weighted Linear Regression on 3,355 sampled real-data pairs and 21 features |
| 6 | Select RBF-SVR on validation; separate parameter refinement and optional ResNet comparison |
| 7 | Lock choices, evaluate the frozen test and report controls, uncertainty and performance |
| 8 | Export models/source/numeric evidence and verify score/rank replay |

Preprocessing fits training samples only. Every query searches the complete split gallery.
The 40 frozen test queries retain 132 candidates each; only 28 confirmed pairs define verified accuracy.
Validation tuning does not establish new independent test performance.

### 3. Numeric replay: existing trusted export required

```powershell
python -m src.reproduce --verify outputs/reproducibility
```

The export is generated locally or supplied separately; it is not included in a Git clone.
Verification checks SHA-256 files, reloads models, and reproduces numeric scores and ranks.
For a core-only export, verify `outputs/reproducibility_core`; numeric replay needs no images or CNN weights.
Use the bundle's `RUN.md` and recorded versions; load only trusted joblib files.

## Find and interpret outputs

Existing notebook outputs appear below code cells; viewing them requires no rerun.
Generated files follow these locations (core-only final paths add `_core`):

| Location | Contents |
| --- | --- |
| `outputs/data_preparation/` | Observation index, split manifest, quality checks and review sheets |
| `outputs/targeted_review/index.html` | Original-pair review and development candidate pages |
| `outputs/baseline_linear/`, `outputs/svr_main/` | Main models, training samples and full development rankings |
| `outputs/svr_parameter_tuning/` | Separate validation-only parameter search and model replay |
| `outputs/resnet_matching/` | Optional frozen RGB representation experiment |
| `outputs/final_experiments/` | `test_comparison.csv`, `*_test_rankings.csv`, uncertainty and performance |
| `outputs/reproducibility/` | Trusted portable bundle with `RUN.md` and checksum manifest |
| [res/](res/README.md) | Small committed aggregate results available without private data |

Rank 1 has the largest score; numerical ties retain a rank interval.
Top-1/Top-5/MRR use independent confirmed pairs for verified test claims.

## Checks, documentation and commit history

```powershell
python tools/build_docs.py --check
git log --oneline --decorate
git diff --check
```

The [CI workflow](.github/workflows/ci.yml) runs tests, both synthetic notebook workflows,
and a separate documentation build; it uploads executed synthetic notebooks and HTML.
CI also enforces the README limit of 200 lines. It does not use private orchard images.
To build the HTML reference in a separate environment:

```powershell
python -m venv .cache/docs-env
& ./.cache/docs-env/Scripts/python.exe -m pip install -r config/requirements-docs.txt
& ./.cache/docs-env/Scripts/python.exe tools/build_docs.py
```

Open `doc/_build/html/index.html`; Linux/macOS uses `.cache/docs-env/bin/python`.
Meaningful existing commits cover auditing, review, baselines, SVR and evaluation.
[Contribution guidelines](CONTRIBUTING.md) record their hashes and define future branch/commit practices.
Project explanations are English; changes include validation and a commit description.
