# Experiment reference and recorded results

The main notebook, `apple_tree_matching.ipynb`, is the authoritative workflow.
This guide collects details previously spread through the root README. Values
below describe the recorded real-data experiments on 25 October 2026, not new
runs performed while editing documentation. Synthetic results test implementation
and must not be substituted for these observations.

## Data and label boundaries

The local data contains 562 proposed N summer/winter pairs. Existing explicit
human rejections exclude 13, leaving 549 eligible identities: 305 training, 112
validation and 132 test. The seed-42 row split is 12/4/4, assigned before human
exclusions. Identity, row, path and identical RGB content cannot cross splits.

R is the orchard row; N identifies a nominal tree correspondence. Parse identity
from the filename prefix, including corrected R04 names with older suffix IDs.
U observations do not establish cross-season correspondence, even when their
numbers coincide. GT segmentation masks remain audit metadata.

The initial review set consists of two pairs per row plus all R04 pairs. Later
targeted review is optional; it does not require every nominal pair to be verified.
`confirmed`, `rejected` and `pending` in `config/pair_review.csv` describe human
records. Remaining filename labels retain their unverified provenance. Model
scores never automatically change decisions or re-pair images.

Before scoring, freeze 40 test queries and the complete 132-image winter gallery
under `config/evaluation/`. The separate independent audit retains all 40 entries:
28 confirmed, 7 rejected and 5 uncertain. Only confirmed queries define verified
test accuracy. Whole-sample filename metrics are separately labelled diagnostics.
See the [data contract](data_contract.md) for schemas and normalization rules.

## Main methods: stages 3–6

- Centre-square crop RGB and resize to 256 pixels. Extract up to 500 ORB keypoints,
  use Hamming matching and a 0.75 ratio test, and retain unique target matches.
- Fit a full 2D affine transform with RANSAC, a 3-pixel resized-coordinate threshold,
  2,000 iterations and confidence 0.99. Missing descriptors and failed geometry
  remain represented by flags and unavailable statistics, rather than removed pairs.
- Verify affine least squares with a known transform, 100 fitting points and 200
  independent probe points. Test noiseless recovery below 1e-6 pixels and sensitivity
  to noise and incorrect correspondences. This does not validate tree identity.
- Combine 12 ORB/geometric columns with nine HSV/D columns. D is uint8 intensity
  with unknown physical units; unavailable D remains flagged. HOG is not used.
- Sample one positive and ten distinct nominal negatives per training query,
  preferring five same-row negatives. The resulting 3,355 pairs and inverse-frequency
  weights are identical for Linear Regression and RBF-SVR.
- Fit median imputation, scaling and each regressor on sampled training pairs only.
  IDs, rows, paths, dates, GT and human decisions do not enter the numeric features.
- The original 18-setting SVR grid uses C={0.1,1,10}, gamma={scale,0.01,0.1} and
  epsilon={0.01,0.1}. Select validation Top-1, then MRR, then repeated scoring time.
  Non-converged fits cannot win. The selected main parameters are 10/0.01/0.01.

Each query searches its complete split gallery. Numeric ties receive expected
Top-k/reciprocal-rank credit over tied positions; IDs only order display ties.
The score is unbounded and is not a calibrated probability. See the
[workflow guide](workflow.md) for `match_tree` and the [contract](data_contract.md)
for feature units, NaNs and input/output columns.

## Recorded independent results: stages 7–8

All 40 queries retain full rankings; the following table uses 28 confirmed queries.

| Representation / regressor | Top-1 | Top-5 | MRR |
| --- | ---: | ---: | ---: |
| Main ORB/HSV/D + Linear Regression | 0/28 | 0/28 | 0.01644 |
| Main ORB/HSV/D + RBF-SVR | 0/28 | 0/28 | 0.02504 |
| Supplementary ResNet18/PCA + Linear Regression | 0/28 | 1/28 | 0.03790 |
| Supplementary ResNet18/PCA + RBF-SVR | 1/28 | 1/28 | 0.05762 |

Versioned evidence is in `res/final_experiments/test_comparison.csv` and
`test_metrics.json`; full local rankings are in `outputs/final_experiments/`.
The main representation does not deliver useful tree identification. Zero observed
successes have a Wilson 95% upper bound of 12.06%, not proof of zero population
accuracy or model equivalence. The supplementary SVR's single success has a wide
0.63%–17.71% interval and does not establish reliable identification.

Top-k uses Wilson intervals for binary outcomes and bootstrap for fractional ties.
MRR and paired differences use seed-42 query bootstrap. These intervals condition
on the confirmed subset and fixed gallery, not an orchard population inferred
from four rows. Test outcomes never change selection, queries or human decisions.

Validation controls compare the reference, omission of D/RANSAC, full-frame resize
and centre crops at 128/384 pixels. SVR parameters stay fixed. Larger images do
not consistently improve retrieval; crop/background effects differ by regressor.
These controls do not select a new model using test performance.

For one 112-candidate validation query, the main SVR's recorded median drops from
2.001 seconds with repeated descriptor extraction/pairwise scoring to 0.184 seconds
with descriptor reuse/batching; descriptor-cache construction costs 0.836 seconds.
Appearance summaries are precomputed and excluded in every mode. Sampled absolute
RSS includes other loaded models and can miss short peaks; it is not isolated
per-model peak RAM. See notebook section 7.2 for full benchmark scope.

## Optional development experiments

These commands run from the repository root and require existing stage-5 artifacts.
Historical fingerprint compatibility still applies; see
[maintenance](maintenance.md#source-hashes-and-historical-results).

| Experiment | Command | Output and interpretation |
| --- | --- | --- |
| Main-feature parameter refinement | `python tools/run_notebook.py --tune-svr-only --in-place` | Main section 6.4; `outputs/svr_parameter_tuning/`; validation only |
| Frozen RGB representation | `python tools/run_notebook.py --resnet-only --in-place` | Main sections 6.5–6.8; `outputs/resnet_matching/`; needs packages/checkpoint |
| Alternative features | `python tools/run_notebook.py --optimization-only --in-place` | Separate `apple_tree_matching_opt_attempt.ipynb`; `outputs/feature_optimization/`; validation only |

The 36-setting refinement expands epsilon to {0.01,0.1,0.2,0.3} and selects MRR
before Top-1/Top-5/time. All settings converge; the winner remains C=10,
gamma=0.01, epsilon=0.01, with MRR 0.06690 and 3,336/3,355 support vectors.
It gives no retrieval improvement on the original representation. Its separate
lock is `config/svr_tuning_selection.json`; historical test results do not evaluate
a newly tuned model, even if parameters happen to coincide.

The optional ResNet18 IMAGENET1K_V1 encoder is frozen. L2-normalized 512-dimensional
embeddings feed PCA32 fitted only on 610 training images, then 65 pair features.
The regressors reuse the original sampled IDs, labels and weights. RGB-only inputs
omit D, so the representation comparison does not isolate the CNN's causal effect.
Weights remain external to fitted bundles; CPU/CUDA differences can change fingerprints.
Install instructions are in the [first-run guide](quickstart.md#optional-resnet-environment).

The separate feature experiment selects a 384-pixel, 1,000-keypoint, ratio-0.85
representation with query-only columns removed. Its chosen SVR uses C=1,
gamma=scale, epsilon=0.2, with 2,619/3,355 support vectors. Original/new validation
SVR Top-1 is 3/112 versus 4/112 and MRR 0.06690 versus 0.07232. The new linear
comparison worsens. This small repeatedly selected validation gain is not new
independent test evidence. See the [optimization plan](optimization_plan.md).

## Human review utilities

Run `python -m src.screening` for development review and score-free independent
original-pair pages under `outputs/targeted_review/index.html`. Alternative candidates
are suggestions, not verified identities. Zero geometric evidence retains ties;
review batches are not padded with unsupported alternatives.

Run `python tools/generate_all_review_sheets.py` for the optional full N atlas at
`outputs/data_preparation/review_sheets_all/index.html`. The original 562 pairs form
71 pages at eight pairs per page. `page_index.csv` maps IDs to pages. Decisions are
generation-time snapshots; refresh pages after intentional ledger updates.

The training-only ORB diagnostic compares 256/500/0.75, 384/1000/0.85 and
384/1000/0.90 resolution/keypoint/ratio settings. More inliers do not establish
better identity accuracy; nominal negative controls are not independently verified.

## Output and replay boundaries

Stage 8 exports model pipelines and explicit preprocessors, ordered feature schemas,
versions, source/configuration, human records, numeric evidence and a SHA-256 manifest.
Original image files, pretrained checkpoints and large caches stay outside the bundle.
The clean source notebook in an export differs from the executed workspace notebook.
Inspect any rendered source photographs before sharing notebooks or other artifacts.

With an existing trusted bundle, `python -m src.reproduce --verify outputs/reproducibility`
checks file integrity and reproduces every model's numeric scores and rank intervals.
The parameter and feature revisions have separate replay bundles under
`outputs/svr_parameter_tuning/` and `outputs/optimization_reproducibility/`.
Their `RUN.md` files describe the correct source, models and environment.

Numeric replay does not reopen original images. Image-backed reproduction needs
the original data, recorded software and optional checkpoint. A clean clone contains
small aggregate evidence, not the ignored model/export directories. Use the
[maintenance procedure](maintenance.md#source-hashes-and-historical-results) for
separate new experiments rather than weakening historical hash checks.
