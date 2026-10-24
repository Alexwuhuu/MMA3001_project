# MMA3001 — Cross-season Apple-tree Matching

The main entry point is [`apple_tree_matching.ipynb`](apple_tree_matching.ipynb).
It currently implements **1. configuration and data auditing, 2. label records and
row-based splitting, 3. ORB/RANSAC features and targeted review,
4. affine numerical verification, 5. a weighted Linear Regression baseline, and
6. validation-only RBF-SVR selection with a locked configuration**.
**RBF-SVR is the main regression method**. A frozen ResNet18/PCA feature
extension also compares Linear Regression and SVR. HOG is not used.

Project documentation, notebook explanations, generated page instructions, and
future explanatory material and annotation notes are maintained in **English**.
Existing Chinese notes were translated without changing identity decisions;
original files are backed up under the ignored `.cache/annotation_notes_original_*/`.

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
│   ├── baseline.json           # Negative sampling, histogram bins, timing
│   ├── svr.json                # RBF-SVR grid and solver settings
│   ├── svr_selection.json      # Original ORB selection and fingerprint
│   ├── resnet.json             # Frozen encoder, cache, PCA and device settings
│   ├── resnet_svr_selection.json # Separate neural-feature selection lock
│   ├── requirements-resnet.txt # Optional pinned PyTorch dependencies
│   ├── evaluation/             # Frozen queries, full gallery, independent audit
│   ├── requirements.in         # Direct dependencies
│   ├── requirements.txt        # Exact versions tested with Python 3.12
│   └── pair_review.csv         # Existing human decisions and English notes
├── doc/                        # Data contract and AI usage; not a project report
├── res/                        # Static resource information
├── src/
│   ├── __init__.py
│   ├── main.py                 # Audit, review sheets, portable exports
│   ├── features.py             # Reusable ORB/geometric features and caches
│   ├── appearance.py           # Cached color and grayscale-D features
│   ├── numerical.py            # Least squares, RANSAC and known-transform tests
│   ├── baseline.py             # Weighted linear regression and full ranking
│   ├── svr.py                  # Validation search and locked train-only SVR
│   ├── deep_features.py        # Frozen ResNet18, numeric cache, train-only PCA
│   ├── transfer.py             # Shared-pair linear/SVR feature experiment
│   ├── screening.py            # Freeze test queries, screen train/validation
│   ├── review_assets.py        # Original pairs and alternative-candidate pages
│   └── utils.py                # Parsing, validation, pairing, row splits
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_main.py
│   ├── test_screening.py
│   ├── test_baseline.py
│   ├── test_svr.py
│   └── test_transfer.py
├── tools/
│   ├── run_notebook.py         # Full execution or ResNet-only extension
│   ├── download_resnet_weights.py # Pinned official checkpoint, verified hash
│   ├── generate_all_review_sheets.py
│   ├── check_orb_settings.py   # Training-only parameter diagnostic
│   ├── translate_annotation_notes.py # Exact notes-only translations and backup
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

## Numerical verification and linear baseline

Stage 4 solves a six-parameter 2D affine least-squares problem with
`numpy.linalg.lstsq`, checks rank and condition, and compares all-point fitting
with RANSAC-selected inliers using the same fitting routine. A known transform,
100 fitting points, and 200 independent probe points check noiseless recovery
below 1e-6 px and sensitivity to Gaussian noise and incorrect correspondences.
Outputs: `outputs/numerical_verification/`.

Stage 5 samples one positive plus ten distinct negatives per training query,
preferring five same-row negatives. The real data supplies 3,355 training pairs.
Both future regression models reuse this sample table and inverse-frequency
weights. HSV histogram difference and grayscale D distribution differences join
the existing ORB/geometric features. D is never interpreted as metric distance.
Missing-value imputation and standardization fit only the sampled training pairs.

Linear Regression scores every training/validation query against the complete
winter gallery for its split. Row and tree IDs never enter numerical features.
Top-1, Top-5, MRR, and positive/negative MAE/RMSE are exported. Numerical ties get
expected credit over tied positions; IDs only order displayed ties. Training
scores describe resubstitution; validation scores mainly use filename labels.
They do not establish independent test accuracy. No test scores or SVR fits are
computed at this stage.

Run through the notebook, or execute the baseline directly:

```powershell
& 'D:\anacoda3\envs\MMA3001\python.exe' -m src.baseline
```

`outputs/baseline_linear/` contains the exact training pairs/features/weights,
full train and validation rankings, per-query metrics, regression errors,
coefficients, plots, configuration, dependency versions, and five-repeat median
fit/prediction timings. Batched scoring time excludes image processing and
ranking. `models/linear_regression.joblib` stores the fitted pipeline and feature
order; load only trusted local artifacts. `match_tree` accepts RGB/D paths and
ranks every supplied gallery candidate without requiring a ground-truth mask.

Independent audit entry spelling `confrimed` is normalized to `confirmed` in
memory. Reviewer notes explicitly saying an identity cannot be established map
pending entries to uncertain. The exported normalization audit records original
status and conversion reason without rewriting source status fields. All forty
sampled records and the full winter test gallery remain frozen for later final
evaluation, including rejected and unresolved audit cases.

## Stage 6: RBF-SVR selection

Continue through section 6 of `apple_tree_matching.ipynb`. No additional packages
are needed beyond the pinned environment. The SVR method uses the exact sampled
training pairs, feature values, order, and inverse-frequency weights from the
Linear Regression baseline. Each pipeline fits the imputer/scaler on those
training samples only; their learned statistics are checked against the baseline.

The RBF grid has 18 configurations: C={0.1,1,10}, gamma={scale,0.01,0.1}, and
epsilon={0.01,0.1}. Each configuration scores every validation query against all
validation winter candidates. Select highest Top-1, then highest MRR, then lowest
median amortized scoring time across five repetitions; trial ID resolves an exact
remaining tie. Non-converged configurations cannot win. Scores remain continuous
regression values, not probabilities. No independent test predictions are made.

`config/svr_selection.json` records the locked choice, selection metrics, input
fingerprint, shared sample hash, feature order, versions, and protocol hashes.
It is a small reproducibility artifact suitable for Git. An unchanged rerun
checks and reuses this choice, so timing noise cannot change the selected
parameters. Changed data, implementation, versions, or grid cause a clear error;
use a different `selection_file` for an intentional new experiment. Synthetic
notebook execution uses a temporary selection file.

`outputs/svr_main/` contains all 18 search results, the selected configuration,
metrics, full training/validation rankings, per-query provenance, support-vector
metadata, development comparison with Linear Regression, figures, and
`models/rbf_svr.joblib`. The saved bundle works with the same `match_tree` API.
Repeated fitting and cached-feature prediction timings exclude image extraction
and ranking. Support-vector array bytes describe that array rather than peak RAM.

The chosen SVR stays fitted on training samples only. Validation influenced
selection, so its scores are development results, mostly against filename labels.
Stage 7 retains the planned independent test comparison, uncertainty estimates,
ablations, resolution/crop sensitivity, profiling, and performance optimisation.
No test queries are scored during stage 6 and no test audit cases are replaced.

Method reference: [scikit-learn SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html).

The current real-data selection is C=10, gamma=0.01, epsilon=0.01 (all 18 trials
converged). Validation Top-1 is 3/112 (2.68%), Top-5 is 7/112 (6.25%), and MRR
is 0.06690; the linear baseline gives 2/112 (1.79%), 7/112 (6.25%), and 0.05803.
These weak development results do not establish reliable independent improvement.
The selected model has 3,336 support vectors out of 3,355 training pairs.

All 35 tests passed, and the complete notebook executed on both real and
synthetic data. Human records and frozen evaluation files were checked unchanged.
Python source checkouts use LF (`.gitattributes`) so Git line-ending conversion
does not change implementation fingerprints. No commit or push is automatic.

## Frozen ResNet18 feature extension

Notebook sections 6.4–6.7 use official ImageNet-1K ResNet18 V1 weights as a
fixed RGB feature extractor. The CNN is frozen in evaluation/inference mode;
no CNN weights or normalization statistics are learned from orchard images.
The original ORB/HSV/D experiments, their model files, and selection lock remain
available. This additional representation uses RGB only, so comparison with the
original RGB/D representation does not isolate the causal effect of the CNN.

The encoder center-square crops RGB, then applies the official resize256,
center-crop224, and ImageNet normalization. It returns 512-dimensional vectors,
which are L2-normalized. PCA32 fits on the 610 real training images only. Each
image pair has 65 features: absolute PCA differences, PCA products, and original
embedding cosine distance. Both regressors use exactly the original 3,355 sampled
pair IDs, filename-derived targets, and inverse-frequency weights. The new SVR
search uses the same 18 settings and validation selection order as before. No
test images are embedded or scored.

Install the additional pinned environment on this machine (NVIDIA CUDA 12.8):

```powershell
& 'D:\anacoda3\envs\MMA3001\python.exe' -m pip install -r config/requirements-resnet.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
& 'D:\anacoda3\envs\MMA3001\python.exe' tools/download_resnet_weights.py
```

For CPU-only use, substitute `https://download.pytorch.org/whl/cpu` for the CUDA
index. Core requirements remain in `config/requirements.txt`; install them first
in a new Python 3.12 environment. This optional file pins PyTorch 2.11.0 and
TorchVision 0.26.0; the package index supplies CPU/CUDA build suffixes. This
machine uses `2.11.0+cu128` and `0.26.0+cu128`. GPU availability is checked at
runtime, and `device=auto` supports CPU inference when CUDA is absent. The
additional tests skip when the optional PyTorch stack is not installed.

The 44.7 MB official checkpoint is stored under the ignored `.cache/pretrained/`.
The explicit downloader validates the official hash prefix; its full SHA-256 is
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
No download or random-weight fallback occurs inside image feature extraction.
Keep the checkpoint for offline runs or fetch it again with the utility.

Use Restart & Run All for the complete notebook. To execute the extension using
existing stage-5 inputs while preserving original experiment files:

```powershell
& 'D:\anacoda3\envs\MMA3001\python.exe' tools/run_notebook.py --resnet-only --in-place
```

`python -m src.transfer` is another entry point after data preparation and the
original baseline have produced their files. The new outputs are under
`outputs/resnet_matching/`: input hashes, exact pair features/weights, all grid
results, per-model rankings/errors, PCA and fitted model bundles, provenance,
timing, comparison figures, and a separate `config/resnet_svr_selection.json`.
Use `match_resnet` with a query RGB path and the complete candidate RGB gallery.
The existing `match_tree` API remains available for the original ORB models.

Image IDs, orchard rows, filenames, human notes, and GT masks are metadata only.
PCA, imputation, and scaling are fitted on training data; validation is used for
SVR selection. Both the small independent sample and full test gallery remain
unchanged. Frozen neural features are an empirical alternative, with no assumed
accuracy guarantee. Parameter changes require a separate experiment lock.

Sources: [TorchVision ResNet18](https://docs.pytorch.org/vision/0.26/models/generated/torchvision.models.resnet18.html),
[transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html),
[PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).

The extension has passed 40 tests and full synthetic notebook execution. The real
feature extraction and model fitting completed, and the corrected extension then
ran successfully in a fresh kernel, verifying cache and locked-selection reuse. All 31 protected original source/result/audit
files remained byte-identical. Real validation results are:

| Representation / regressor | Top-1 | Top-5 | MRR |
|---|---:|---:|---:|
| Original ORB/HSV/D + Linear Regression | 1.79% | 6.25% | 0.0580 |
| Original ORB/HSV/D + SVR | 2.68% | 6.25% | 0.0669 |
| Frozen ResNet18/PCA + Linear Regression | 1.79% | 11.61% | 0.0768 |
| Frozen ResNet18/PCA + SVR | 3.57% | 7.14% | 0.0834 |

The new SVR selection is C=0.1, gamma=scale, epsilon=0.1. It gains one Top-1
success over the original SVR on 112 validation queries, while the new linear
model has better Top-5. Overall ability remains weak. This development comparison
does not establish independent accuracy; the test sample remains unscored.

Exact selection fingerprints include hardware-dependent embeddings and software
versions. When reproducing with a CPU stack instead of the recorded CUDA stack,
use separate output and selection paths (for example `outputs/resnet_cpu/` and
`config/resnet_svr_selection_cpu.json`) to preserve the existing experiment lock.
