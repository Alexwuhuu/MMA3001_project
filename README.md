# MMA3001 — Cross-season Apple-tree Matching

The main entry point is [`apple_tree_matching.ipynb`](apple_tree_matching.ipynb).
It currently implements **1. configuration and data auditing, 2. label records and
row-based splitting, and 3. ORB/RANSAC features and targeted review**.
Regression training is a later stage: **Linear Regression is the baseline and
RBF-SVR is the main regression method**. HOG is not used.

Project documentation, notebook explanations, generated page instructions, and
future explanatory material are maintained in **English**. Original human review
notes remain in the language in which the reviewer recorded them.

## Repository structure

```text
MMA3001_project/
├── .github/workflows/ci.yml     # Tests and a synthetic-data notebook check
├── .gitignore
├── README.md
├── LICENSE
├── pytest.ini                  # Private test directories under .cache/
├── apple_tree_matching.ipynb    # Main reproducible workflow
├── config/
│   ├── experiment.json         # Dataset paths, seed, initial review settings
│   ├── screening.json          # ORB/RANSAC, review batches, test sampling
│   ├── evaluation/             # Frozen queries, full gallery, independent audit
│   ├── requirements.in         # Direct dependencies
│   ├── requirements.txt        # Exact versions tested with Python 3.12
│   └── pair_review.csv         # Existing human decisions and original notes
├── doc/                        # Data contract and AI usage; not a project report
├── res/                        # Static resource information
├── src/
│   ├── __init__.py
│   ├── main.py                 # Audit, review sheets, portable exports
│   ├── features.py             # Reusable ORB/geometric features and caches
│   ├── screening.py            # Freeze test queries, screen train/validation
│   ├── review_assets.py        # Original pairs and alternative-candidate pages
│   └── utils.py                # Parsing, validation, pairing, row splits
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_main.py
│   └── test_screening.py
├── tools/
│   ├── run_notebook.py         # Execute with the selected interpreter
│   ├── generate_all_review_sheets.py
│   ├── check_orb_settings.py   # Training-only parameter diagnostic
│   └── synthetic_data.py       # CI fixtures, not agricultural validation data
├── data/TreeID/TreeID/          # Original dataset; ignored by Git
└── outputs/                    # Generated tables and atlases; ignored by Git
```

## Run on this machine: PowerShell

Use the existing **MMA3001 conda environment, Python 3.12**. To configure or
rebuild its packages and register its notebook kernel:

```powershell
cd D:\2026-S2\MMA3001\MMA3001_project
& 'D:\anacoda3\envs\MMA3001\python.exe' -m pip install -r config\requirements.txt
& 'D:\anacoda3\envs\MMA3001\python.exe' -m ipykernel install --sys-prefix --name mma3001 --display-name 'MMA3001 (Python 3.12)'
```

Open `apple_tree_matching.ipynb` in VS Code, choose **MMA3001** in the kernel
selector, and run **Restart Kernel and Run All Cells**. The interpreter must be
`D:\anacoda3\envs\MMA3001\python.exe`; the first code cell displays its actual path.
Alternatively, start JupyterLab from that environment:

```powershell
& 'D:\anacoda3\envs\MMA3001\python.exe' -m jupyterlab apple_tree_matching.ipynb
```

Execute the notebook without a browser, run data preparation alone, or run tests:

```powershell
& 'D:\anacoda3\envs\MMA3001\python.exe' tools\run_notebook.py --in-place
& 'D:\anacoda3\envs\MMA3001\python.exe' -m src.main
& 'D:\anacoda3\envs\MMA3001\python.exe' -m pytest -q
& 'D:\anacoda3\envs\MMA3001\python.exe' tools\run_notebook.py --synthetic
```

The project `.venv` is another option when it contains the current dependencies;
substitute `.\.venv\Scripts\python.exe` for the interpreter above.
`nbformat` and `nbclient` support automated notebook execution. Interactive data
preparation does not require them. Missing distribution metadata is recorded as
unavailable rather than interrupting otherwise valid computation.

On another machine, create a Python 3.12 environment and install
`config/requirements.txt`. Supply the original data separately. To override the
dataset location, create the ignored `config/local.json`:

```json
{"data_root": "D:/datasets/TreeID/TreeID"}
```

## Labels and human review

- **R** is an orchard row; **N** is a tree number used to establish candidate
  cross-season identities.
- **U** denotes unmatched observations. Equal U numbers do not establish a
  positive pair or justify unverified negative labels.
- N pairs initially follow filename prefixes. Two pairs per row and all R04
  pairs form the initial spot-check set.
- In `config/pair_review.csv`, `confirmed` means a human confirmed the identity;
  `rejected` means a human rejected it. `pending` means unverified or unresolved;
  record uncertainty and its reason in `human_notes`.
- Reruns preserve existing decisions and do not rewrite an existing review ledger.
- With seed 42, the 20 orchard rows are divided into 12/4/4 train/validation/test
  rows. Identities, rows, paths, and identical RGB content cannot cross splits.

The current policy is **targeted review**. Exclude explicitly rejected pairs and
retain the remaining filename labels. `human_verified` and `label_basis` separate
human-confirmed identities from `filename_unverified` labels. Full manual review
is optional; automated scores never change human decisions.

Run the targeted workflow directly, or use the main notebook:

```powershell
& 'D:\anacoda3\envs\MMA3001\python.exe' -m src.screening
```

Before extracting features or computing scores, the workflow uniformly samples
40 queries from the original eligible test rows. Their fixed IDs and complete
test winter gallery are saved under `config/evaluation/`. Reruns reuse these files
and preserve independent review decisions. A changed test pool raises an error
rather than silently resampling.

Each training/validation query is compared against its split's **entire winter
gallery**. Row numbers do not restrict candidates. ORB/RANSAC is a numerical
feature and review step, not an additional trained prediction model. Descriptor
distances, geometric inliers, and reprojection errors can be reused in regression.

Open `outputs/targeted_review/index.html`:

- `review_batches/`: up to 60 cases with a supported priority, at most 20 per
  batch; each page shows the original pair and three alternative images.
- `independent_test/`: the original pairs for 40 test queries, without automated
  scores or alternative suggestions.
- `pair_features.csv`: ORB/geometric features from full train/validation galleries.
- `suspected_pairs.csv`: optional review ordering and low-evidence diagnostics,
  not automatic mismatch labels.
- `screening_summary.json`: settings, cache signatures, counts, and timing.

Record independent test decisions in `config/evaluation/independent_review.csv`:
`pending`, `confirmed`, `rejected`, or `uncertain`. Retain every sampled record;
do not delete or replace queries because the final SVR predicts them incorrectly.
Both regression models will use the same fixed queries and complete test gallery.
Report total sample size, independently verified count, unresolved cases, and
uncertainty. Planned intervals are Wilson intervals for Top-1/Top-5 and query
bootstrap intervals for MRR. Results on the verified subset describe that subset,
not a certain accuracy for all observations.

Method references: [OpenCV feature matching](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html),
[2D affine RANSAC](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html), and
[VPR-Bench](https://link.springer.com/article/10.1007/s11263-021-01469-5).
Seasonal change, viewpoint, and repeated backgrounds affect the scores.
Candidate suggestions require human judgement.

When every geometric evidence score is zero, preserve the full tie range. Weak
descriptor correspondences do not establish reliable alternatives, and review
batches are not padded. The summary explicitly records insufficient evidence.

A training-only diagnostic uses up to 20 human-confirmed pairs plus 10 different
filename controls per query. It compares resolution/keypoints/ratio settings
256/500/0.75, 384/1000/0.85, and 384/1000/0.90, scaling the pixel threshold with
resolution. Controls rely on different filenames, not independent pairwise truth;
their counts diagnose correspondence availability rather than accuracy. Results
are saved under `outputs/targeted_review/pilot/`. The diagnostic neither changes
the defaults nor uses test data.

## Optional full review atlas

Generate side-by-side pages for all N candidate pairs while preserving decisions:

```powershell
& 'D:\anacoda3\envs\MMA3001\python.exe' tools\generate_all_review_sheets.py
```

Open `outputs/data_preparation/review_sheets_all/index.html`, expand a page, or
use Ctrl+F to find a tree ID. `page_index.csv` maps IDs to page numbers. At eight
pairs per page, 562 pairs produce 71 pages. The original spot-check pages remain
under `review_sheets/`. Statuses and notes are generation-time snapshots; rerun
the tool after updating the ledger if a refreshed atlas is needed.

See [`doc/data_contract.md`](doc/data_contract.md) for inputs and outputs and
[`doc/ai_usage.md`](doc/ai_usage.md) for AI assistance. Original images and generated
atlases are not automatically added to Git.

## Data source and scope

Dataset background: [Chen et al., Semantic Segmentation for Partially Occluded
Apple Trees Based on Deep Learning](https://arxiv.org/pdf/2010.06879).
The paper's 521 segmentation images differ in count from the local N subset, so
this project defines its own split. The physical encoding of the exported 8-bit D
images is unknown; the current workflow does not estimate metric 3D coordinates
or an orchard map in metres. The code license in `LICENSE` does not grant rights
to the original dataset or paper.

Changes are left in the working tree. No commit or push is performed automatically.
