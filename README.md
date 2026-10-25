# MMA3001 — Cross-season Apple-tree Matching

The main entry point is [`apple_tree_matching.ipynb`](apple_tree_matching.ipynb).
It implements **1. configuration and data auditing, 2. label records and
row-based splitting, 3. ORB/RANSAC features and targeted review,
4. affine numerical verification, 5. a weighted Linear Regression baseline, and
6. validation-only RBF-SVR selection with a locked configuration,
7. independent evaluation, ablations, sensitivity and performance,
8. verified model/source/results export**.
**RBF-SVR is the main regression method**. A frozen ResNet18/PCA feature
extension also compares Linear Regression and SVR. HOG is not used.

Project documentation, notebook explanations, generated page instructions, and
future explanatory material and annotation notes are maintained in **English**.
Existing Chinese notes were translated without changing identity decisions;
original files are backed up under the ignored `.cache/annotation_notes_original_*/`.

## Software documentation

The [first-run guide](doc/quickstart.md), [workflow and API examples](doc/workflow.md),
[data contract](doc/data_contract.md), and [documentation conventions](doc/maintenance.md)
form the software reference. Source modules and tools use Google-style docstrings.
Sphinx generates signatures, parameter/return/exception details and source listings
directly from those files without importing the experiment stack.

```powershell
python -m venv .cache/docs-env
& '.cache/docs-env/Scripts/python.exe' -m pip install -r config/requirements-docs.txt
& '.cache/docs-env/Scripts/python.exe' tools/build_docs.py
Start-Process 'doc/_build/html/index.html'
```

The generated [HTML home page](doc/_build/html/index.html) is available after building.
`python tools/build_docs.py --check` checks source documentation without Sphinx.
The full build treats warnings as errors and verifies local links and fragments;
CI also uploads the complete HTML site. Generated pages are ignored by Git.

Experiment locks hash raw source bytes, so comment/docstring edits can invalidate
an old selection fingerprint. Existing experiment records are preserved; see
[historical-result maintenance](doc/maintenance.md#source-hashes-and-historical-results)
before starting a fresh real-data run.

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
│   ├── final_experiments.json  # Final experiments, timing, uncertainty and export
│   ├── final_experiments_lock.json # Choices frozen before test extraction
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
│   ├── final_experiments.py    # Controlled validation variants and frozen test
│   ├── reproduce.py            # Portable export, checksums and prediction replay
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

The historical RBF grid has 18 configurations: C={0.1,1,10}, gamma={scale,0.01,0.1}, and
epsilon={0.01,0.1}. Each configuration scores every validation query against all
validation winter candidates. Select highest Top-1, then highest MRR, then lowest
median amortized scoring time across five repetitions; trial ID resolves an exact
remaining tie. Non-converged configurations cannot win. Scores remain continuous
regression values, not probabilities. No independent test predictions are made.

This first search supplies the historical test model. Section 6.4 below adds the
new 36-configuration, MRR-first parameter revision on the same original features.

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
Stage 7 adds the independent test comparison, uncertainty estimates,
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

Main notebook sections 6.5–6.8 (6.4–6.7 in the feature-optimization copy) use official ImageNet-1K ResNet18 V1 weights as a
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
does not establish independent accuracy; the test sample remains unscored until
section 7.

Exact selection fingerprints include hardware-dependent embeddings and software
versions. When reproducing with a CPU stack instead of the recorded CUDA stack,
use separate output and selection paths (for example `outputs/resnet_cpu/` and
`config/resnet_svr_selection_cpu.json`) to preserve the existing experiment lock.

## Stages 7-8: final experiments and reproducible delivery

Restart & Run All now completes all eight stages. The optional ResNet section
runs when its packages and official checkpoint are present. To run the core
ORB/HSV/D workflow without PyTorch:

```powershell
& 'D:\anacoda3\envs\MMA3001\python.exe' tools/run_notebook.py --skip-resnet --in-place
```

The core-only mode has separate final lock, output, export and small-result paths
so it cannot overwrite the recorded four-model experiment. The default full run
retains the main Linear Regression / RBF-SVR comparison and separately labels the
previously authorized frozen-ResNet extension. Both model families keep their
original validation selections and remain fitted on training pairs only.

Section 7 first compares six development settings: reference 256 center crop,
without D, without RANSAC outputs and its failure flag, full-frame resize, and
128/384 center crops. Both regressors reuse exactly the original training pair
IDs, labels and inverse-frequency weights. All SVR variants use the main model's
locked hyperparameters to isolate the factor; they are not separately optimized.
Thresholds scale to 1.5/3/4.5 px. Train-only imputation/scaling is refitted for each
variant; validation uses its entire winter gallery. No variant scores test data.
The original 18 SVR trials supply train/validation parameter and error sensitivity.

Before extracting test features, `config/final_experiments_lock.json` records the
feature/configuration/source hashes, original selections and frozen protocol/audit
checksums. Changed inputs cause an error, requiring separate lock/output paths for
an intentional experiment. This never automatically replaces a query or model.
The fixed 40 queries each rank all 132 winter candidates, including rejected and
unresolved audit cases. Independently confirmed queries define the primary test
accuracy; other cases have no verified truth. All-sample filename metrics are
explicit diagnostics only. Positive/negative/all-pair MAE and RMSE accompany
Top-1, Top-5 and MRR. Other gallery IDs are assumed distinct by filename.

Top-k uses 95% Wilson intervals for binary outcomes; fractional tie credits use
query bootstrap. MRR and paired SVR-minus-linear differences use seed-42 query
bootstrap. These intervals condition on the confirmed subset and fixed gallery;
they do not infer accuracy for unresolved queries or the wider orchard population
from four test rows. No test results change model selection.

`outputs/final_experiments/` contains full test rankings, per-query audit status,
features, metrics/intervals, paired differences, validation ablation tables and
rankings, parameter diagnostics, error examples, ID index diagrams, profiles and
five-repeat timings. A fixed U query is a qualitative notebook demonstration only.
The first three confirmed SVR failures by ID form the error examples; this is an
inspection set, not an additional statistical sample.

Query benchmarks use the first validation ID and its complete gallery: recompute
ORB for each pair; reuse descriptors with pairwise prediction; reuse descriptors
with batch prediction. All share precomputed HSV/D summaries. Timings include
correspondences/RANSAC, pair features, scoring and ranking, and separately report
descriptor cache build cost. They exclude model loading and appearance extraction;
OS file caches stay warm. The median of five `perf_counter` runs is reported, with
all raw durations saved. Scores must agree across modes. `cProfile` identifies
cumulative-time bottlenecks in the cached batch path. A separate memory run samples
process RSS every 10 ms (including other loaded models, possibly missing short
peaks); array bytes, serialized model size and support-vector count are separate.

Section 8 creates `outputs/reproducibility/` with complete model pipelines,
explicit fitted preprocessors/PCA, feature schema/order, installed versions,
source, clean source notebook, configurations, human records, numeric test
features/rankings, metrics and aggregate figures. Original images, example
photographs, pretrained checkpoint and large caches remain outside the bundle.
`RUN.md` explains a new environment and original-image execution. Save/restore
inference is checked on a complete validation gallery; every exported model then
replays all numeric test scores and ranks. Each packaged file has a SHA-256 entry.

```powershell
& 'D:\anacoda3\envs\MMA3001\python.exe' -m src.reproduce --verify outputs/reproducibility
```

The image-free replay also works from a copied bundle with `--verify .`. Supplying
the original data and official checkpoint enables a complete new-kernel run.
The exact recorded versions/device are required to reuse existing selection
fingerprints; use new lock/output paths for a different environment. The pinned
core requirements already include psutil; it is now also a direct dependency.
`.gitattributes` preserves the recorded frozen CSV/JSON line endings across Git
checkouts so their byte hashes do not depend on the operating system's defaults.

For stage-7 execution using existing stages 1-6 artifacts and their locks:

```powershell
& 'D:\anacoda3\envs\MMA3001\python.exe' -m src.final_experiments
```

Small aggregate tables and figures live under `res/final_experiments/` for GitHub.
Model binaries, complete rankings, large exports, raw data and caches stay in
ignored directories. No report, commit, push, or data upload is automatic.

Recorded real-data run on 25 October 2026: the notebook executed all 28
code cells in a fresh kernel, with no errors. All 46 automated checks passed.
The 40-query audit has 28 confirmed, 7 rejected and 5 uncertain pairs. Every query
retains all 132 candidates; the table below uses only confirmed queries.

| Representation / regressor | Top-1 | Top-5 | MRR |
|---|---:|---:|---:|
| Main ORB/HSV/D + Linear Regression | 0/28 | 0/28 | 0.01644 |
| Main ORB/HSV/D + RBF-SVR | 0/28 | 0/28 | 0.02504 |
| Supplementary ResNet18/PCA + Linear Regression | 0/28 | 1/28 | 0.03790 |
| Supplementary ResNet18/PCA + RBF-SVR | 1/28 | 1/28 | 0.05762 |

The main models do not retrieve any confirmed identity in their first five
positions. SVR has a higher conditional MRR, but both main MRRs are below the
uniform-ranking reference for 132 candidates. These features do not establish
useful tree identification. Zero observed Top-k successes still has a Wilson
95% upper bound of 12.06%; it does not establish zero population accuracy or
model equivalence. The supplementary SVR's single Top-1 success has a wide
0.63%-17.71% interval and does not justify a general reliability claim.

At fixed SVR parameters, removing D reduces validation Top-1 from 3/112 to 1/112.
Removing RANSAC retains 3/112 and slightly changes MRR; no consistent contribution
is established here. Full-frame resize gives linear 4/112 and SVR 0/112 Top-1,
so the background/crop effect depends on the regressor and includes aspect-ratio
change. Center128/256/384 SVR Top-1 is 2/112, 3/112, 1/112; larger input does not
consistently improve retrieval. These observations do not trigger test retuning.

For the fixed 112-candidate validation query, the main SVR query median drops
from 2.001 s (recompute descriptors/pairwise scoring) to 0.184 s (descriptor reuse
and batch scoring), with 0.836 s descriptor-cache build cost reported separately.
Appearance summaries are shared and excluded in all benchmark modes. cProfile
attributes about 0.157 of the 0.196 s profiled query to OpenCV knnMatch; prediction
is about 0.014 s. The main SVR retains 3,336/3,355 support vectors. Absolute sampled
RSS is shared-process memory and is not per-model peak RAM.

Every exported model replays all 5,280 test pair scores and rankings. Reloaded
main models also reproduce image inference on all 112 validation candidates.
The independent standalone bundle verifier passed. Human ledgers, both selection
locks, and frozen queries/gallery/protocol remained byte-identical to the
pre-stage snapshot. Aggregate figures were visually inspected.

A final export-only correction excludes Python-generated bytecode/checkpoint
folders from repeated export inventory checks. The original pre-test lock and
ranking checksums are preserved in `initial_evaluation_lock.json` and
`export_revision.json` under the final output and bundle results. Only the export
source checksum changes in the current final lock; model choices, image features,
metric rules, human decisions and test rankings are unchanged. No test outcome
was used to select a new configuration.

## Parameter refinement in the original notebook (section 6.4)

[`apple_tree_matching.ipynb`](apple_tree_matching.ipynb) now includes a model-parameter
revision on the original 21 features and 3,355 training pairs. Linear Regression
remains the identical ordinary least-squares baseline. This revision changes only
RBF-SVR parameter selection: C={0.1,1,10}, gamma={scale,0.01,0.1}, and
epsilon={0.01,0.1,0.2,0.3}, giving 36 configurations. Validation MRR takes priority,
then Top-1, Top-5, median query scoring time over five repetitions, and trial ID.
Imputation/scaling and SVR fit training samples only. Failed fits cannot win.

```powershell
& 'D:\anacoda3\envs\MMA3001\python.exe' tools/run_notebook.py --tune-svr-only --in-place
```

This command uses the main notebook and existing train/validation artifacts. It
does not execute historical test cells or modify the feature-optimization copy.
Alternatively run `python -m tools.tune_svr_parameters`. A full Restart & Run All
of the main notebook includes the revision before the optional ResNet section.
The original `svr` object and test tables remain tied to `config/svr_selection.json`;
the new model is `svr_tuned` and is not assigned those historical test results.

`config/svr_tuning.json` defines the revised grid; `config/svr_tuning_selection.json`
locks its choice and input/code fingerprint. Unchanged runs verify and reuse the
saved experiment. With a matching lock but no local results folder, the grid is
rebuilt and its best validation metrics must reproduce the locked choice; timing
jitter cannot replace it. Changed inputs require a new selection/output path.
`outputs/svr_parameter_tuning/` contains models, separate imputer/scaler objects,
all trials, class-specific MAE/RMSE, support fractions, timings, complete rankings,
validation features and a SHA-256 manifest. Replay both regressors' validation
scores/ranks with `python -m src.reproduce --verify outputs/svr_parameter_tuning`.
`res/svr_parameter_tuning/` holds small tables/figures. The tuned bundle uses the
original `src.baseline.match_tree` interface. These repeatedly selected validation
results do not establish a new independent test accuracy.

Recorded real-data result: all 36 settings converged. The best remains C=10,
gamma=0.01, epsilon=0.01, with validation Top-1=3/112, Top-5=7/112 and MRR=0.06690.
Support-vector fraction remains 3,336/3,355 (99.43%). The expanded epsilon grid
and MRR-first rule did not improve retrieval over the original winner. Both
saved model rankings replayed exactly; this is a parameter-search result rather
than evidence of improved matching.

## Validation-only diagnosis and feature optimization (separate notebook)

The new stage lives in [`apple_tree_matching_opt_attempt.ipynb`](apple_tree_matching_opt_attempt.ipynb).
This copy retains stages 1-8 for context and adds section 9 with its saved outputs.
The original `apple_tree_matching.ipynb` contains stages 1-8 only.

The revised plan is [doc/optimization_plan.md](doc/optimization_plan.md). This
separate experiment responds to the weak original results. It preserves all old
source modules, model selections and test artifacts, and makes no new test
predictions. The original experiment and export remain historical records.

Run only the new notebook cells in a fresh kernel:

```powershell
& 'D:\anacoda3\envs\MMA3001\python.exe' tools/run_notebook.py --notebook apple_tree_matching_opt_attempt.ipynb --optimization-only --in-place
```

The runner defaults to the supplied optimization copy when `--optimization-only`
is used; the notebook-export wrapper packages the same file.
Alternatively run `python -m tools.optimization_notebook` for the experiment/export,
or `python -m tools.feature_optimization --diagnostics-only` for the low-cost
original validation analysis.
A full Restart & Run All of the copy also runs its
retained stages 1-8; the optimization-only command executes configuration and
section 9 without executing historical test cells this round.
The standalone commands use a noninteractive plotting backend on Windows.

For a full synthetic check of the copy, run `python tools/run_notebook.py
--notebook apple_tree_matching_opt_attempt.ipynb --synthetic --skip-resnet`.

`outputs/validation_diagnostics/` reports all 21 features' finite-value AUC,
missingness AUC, train-imputed AUC, missing coverage and complete-gallery MRR in
both directions; candidate Top-5 frequencies with fair tie treatment; and true
filename-pair rank distributions. On original validation results, Linear
Regression's largest hub appears in 42/112 Top-5 lists, and the top ten candidates
occupy 45.18% of Top-5 positions. SVR's largest hub appears in 14/112 and its top
ten occupy 19.82%. True-pair median ranks are 51.5 and 49.5. Candidate-keypoint
Spearman correlations with hub frequency are small (-0.138 and -0.059); candidate
attributes alone do not establish the cause. Original single-feature finite AUCs
are near 0.5, and default good-match count has higher MRR in the lower direction.
All these are validation/filename-label diagnostics, not independently confirmed
matching accuracy or evidence of a causal effect.

The new `config/optimization.json` uses 384 pixels, 1,000 ORB points, ratios
0.85/0.90 and RANSAC 4.5 px. It compares raw grayscale, CLAHE and CLAHE with a
central D-intensity-percentile mask. D's physical encoding direction is unknown:
the mask is a center-seeded intensity-band heuristic, not validated trunk
segmentation or metric proximity. Missing/unusable D/masks fall back to the full
image; audit tables and fixed training examples show coverage and fallbacks.
Masked settings also compute appearance summaries inside the selected region.
Separate Summer/Winter residual statistics fit training images only.

Each image setting compares raw columns as an ineligible control, dropping
query-only columns, dropping candidate-only columns too, query-relative ranks /
z-scores / best and second-best ratios, and seasonal residuals. The original
3,355 sampled training pairs, labels and weights are reused. Context is computed
against **all 305 training candidates before sampling**, and all 112 validation
candidates at evaluation. Global imputation/scaling fits training only. Relative
context is a label-free inference step using the intended gallery; changing the
gallery can change scores. It cannot be computed on only 11 sampled pairs.

A fixed pilot SVR (C=1, gamma=scale, epsilon=0.2) screens feature representations.
The winner then uses the 36-configuration C/gamma/epsilon grid, with epsilon
expanded to 0.2 and 0.3. Selection is **MRR first**, then Top-1, Top-5, scoring time
and trial ID. The new lock is `config/optimization_selection.json`, separate from
all original locks. Nonconverged fits cannot win. This staged search is not an
exhaustive joint optimization; repeated validation selection is development work.
No new independent test performance is claimed.

`outputs/feature_optimization/` stores the complete screening/search tables,
class errors, support-vector fractions, image/mask audits, selected train/validation
features, rankings, hubness, rank plots and saved linear/SVR bundles. Use
`tools.relative_features.match_tree_optimized(query, complete_gallery, bundle, root)`
for these bundles. The original `match_tree` function remains for original models.
Saved-model image inference must reproduce the complete validation ranking.

The new portable bundle is `outputs/optimization_reproducibility/`; its `RUN.md`
explains numeric replay and image-backed use. `python -m src.reproduce --verify
outputs/optimization_reproducibility` checks hashes and replays both new validation
rankings without images/weights. Small aggregate results are under
`res/feature_optimization/`. Raw images, foreground example photographs and large
caches stay outside the new export/small-result directories. No report is written
and no commit or push is automatic.

Recorded optimization results (112 validation queries, all 112 winter candidates):

| Model / shared representation | Top-1 | Top-5 | MRR |
|---|---:|---:|---:|
| Original Linear Regression | 2/112 | 7/112 | 0.05803 |
| Original RBF-SVR | 3/112 | 7/112 | 0.06690 |
| New selected-feature Linear Regression | 0/112 | 5/112 | 0.04449 |
| New selected-feature RBF-SVR | 4/112 | 7/112 | 0.07232 |

The selected new representation is raw grayscale, 384 pixels, 1,000 keypoints,
ratio 0.85, query-only columns removed. It has 18 columns, including a constant
foreground-fraction difference in this unmasked setting. Candidate-only columns
remain: removing them did not improve the pilot SVR. The chosen SVR is C=1,
gamma=scale, epsilon=0.2, with 2,619/3,355 support vectors (78.06%, versus 99.43%
originally). Enlarging epsilon lowers support fractions in many trials, but does
not monotonically improve MRR. All 36 configurations converged.

CLAHE, the current mask and the relative/seasonal representations did not beat
this pilot SVR representation. Query-relative features helped a particular
CLAHE/0.85 linear control (MRR 0.0846), so their effect is model-dependent. Both
final comparators still use the same selected feature matrix/weights; the linear
control is not separately optimized to maximize its own score. The staged result
is not the best possible feature/grid combination over an exhaustive search.

The original seasonal D observation is confirmed descriptively: validation zero
fraction averages 0.120 in Summer and 0.548 in Winter before masking. The heuristic
keeps about 30.4% of Summer pixels and 14.7% of Winter pixels (no fallback on these
inputs). These are different seasonal masks, not evidence of aligned trunk
segmentation. Their current use did not improve SVR validation ranking.

Hubness is not fixed by the selected changes: maximum candidate Top-5 appearances
increase from 14 to 17 queries for SVR and 42 to 48 for linear regression. The MRR
gain and one additional Top-1 success remain small; the new linear model worsens.
The feature changes therefore do not establish broadly reliable tree matching.
All new results are repeatedly selected validation diagnostics.

All 54 tests passed, including rendered synthetic end-to-end optimization and
forbidden-test-image guards. The complete 68-cell synthetic notebook ran through
section 9. Saved main/new regression models and their complete-gallery numerical
rankings were reloaded successfully. The new export replays 12,544 validation pair
scores/ranks for each of the two optimized models. All 128 protected historical
source/model/label/result files remained byte-identical. No historical test
prediction was repeated in the real optimization workflow. Figure layout and
foreground examples were inspected. No report, commit or push was produced.
