# Data contract: preparation and targeted review

## Original observations

The dataset root contains `Dataset/`, `Dataset (With Summer GT)/`, and
`FANK_Sat.png`. Supported names are `RxxNyy_Season_Modality-suffix.png` and
`RxxUyy_Season_Modality-suffix.png`, where `Season` is Summer/Winter and `Modality`
is RGB/D/GT. Some R04 suffixes contain an older filename; identity is parsed from
the prefix only.

- **RGB:** readable PNG, typically 640×480. Record actual dimensions and SHA-256.
- **D:** uint8 grayscale, or three identical grayscale channels, with the same
  dimensions as RGB. Intensities are 0–255 and physical units are unknown. Zero
  intensity fraction is a statistic, not a calibrated distance measurement.
- **GT:** a 2D segmentation mask. Record dimensions and unique values. It does not
  enter identity matching features or determine the row split.

Missing or corrupt RGB observations remain in the audit table; corresponding N
pairs receive an exclusion reason. Missing or unsupported D does not exclude an
otherwise valid RGB pair. Multiple files for the same identity/season/modality
raise an error rather than selecting one arbitrarily. U observations do not enter
labelled N pairs or their splits.

## Preparation outputs

| File | Meaning |
|---|---|
| observation_index.csv | One observation per subset/ID/season, with readability, dimensions, hashes, and D/GT status |
| config/pair_review.csv | All N candidate pairs and human decisions; required_review marks the initial spot-check set |
| split_manifest.csv | Eligible N identities, Summer/Winter paths, fixed split, and label provenance |
| *_queries.csv / *_gallery.csv | Summer queries and winter galleries for each split; IDs are supervision/evaluation metadata |
| leakage_checks.csv | Pairwise split intersections for rows, identities, paths, and RGB content hashes |
| quality_issues.csv | Missing, unreadable, or unsupported observations |
| special_filenames.csv | Names with additional letters in the suffix, including corrected R04 names |
| duplicate_rgb_content.csv | Identical RGB content, including possible N/U overlap |
| excluded_n_pairs.csv | Input-quality and human-rejection exclusions with reasons |
| summary.json | Counts, review status, row split, versions, and split-manifest hash |
| review_sheets/ | Side-by-side original RGB images for the initial spot-check set |

Paths are saved relative to the project root when possible. Moving the project
with its data preserves those paths. R/N index plots have no metric coordinate
meaning. The pair ledger accepts `pending`, `confirmed`, and `rejected` only;
automated checks never become human confirmations.

Existing ledgers support UTF-8 CSV with or without a BOM and legacy GB18030 CSV.
Reading does not rewrite the original file. Filename identity remains the default;
only explicitly human-rejected identities are excluded from labelled queries and
galleries. `human_verified=True` comes only from `confirmed`. Other retained labels
have `label_basis=filename_unverified`.

Row membership is assigned before human exclusions. Removing an entire row must
not reassign other rows to different splits.

## ORB and geometric features

Current screening uses RGB only. Center-square crop, then resize to 256×256.
D and GT do not enter screening features. Filename, row, and tree number are
used only for indexing, grouping, and comparison with the original identity label.

ORB descriptors contain 32 bytes. Hamming distance is a bit count in 0–256.
Geometric coordinates and errors use resized-image pixels; the default affine
RANSAC threshold is 3 px. Fixed-order features include keypoint counts, accepted
matches and ratio, descriptor distances, inlier count and ratio, reprojection
RMSE, and missing/failed-estimation flags.

Pairs with missing descriptors, insufficient matches, or degenerate geometry are
retained. Unavailable distances and errors are NaN, not zero. Later regression
imputation and scaling must be fitted on training data only.

Caches depend on image content hashes, processing settings, implementation hash,
and OpenCV version. Each train/validation query is compared against its complete
split gallery. Independent test queries are not scored during screening.

The evidence heuristic is `inlier_count × inlier_ratio`, set to zero below six
inliers by default. It schedules review rather than establishing truth. When
geometric evidence is absent, extra descriptor correspondences do not prove that
an alternative candidate is correct. Zero geometric scores retain their full tie
range, and unsupported cases do not fill review batches automatically.

## Frozen independent evaluation

`config/evaluation/` is created before automated scoring:

| File | Meaning |
|---|---|
| queries.csv | Forty queries sampled uniformly from the original eligible test rows with seed 42; never replaced based on scores |
| gallery.csv | Complete eligible winter test gallery shared by both future regression models |
| independent_review.csv | Human decisions, reviewer, and notes after viewing original pairs only |
| protocol.json | Sampling policy, fixed file hashes, image integrity signature, and future evaluation conventions |

Independent audit states are `pending`, `confirmed`, `rejected`, and `uncertain`,
stored separately from the original pair ledger. Audit entries do not inherit
existing confirmations or automated screening results.

Rejected and unresolved cases remain in the sampling audit. Do not replace them
with easier queries to maintain a desired confirmed sample count. Fix human
decisions before final evaluation; model errors cannot justify sample deletion.
Report the sampled count, independently verifiable count, unresolved cases, and
uncertainty. Unverified records are not independently established identity truth.

## Documentation language

Explanatory documentation, generated interface text, and annotation notes are
English. Existing Chinese notes were translated with original files backed up
under `.cache/annotation_notes_original_*/`; non-note fields and identity
decisions were checked unchanged. Multilingual encoding-test fixtures remain
test data, separate from annotations.

## Affine verification and regression features

`fit_affine_lstsq(source, target)` accepts matching finite N×2 float64 pixel
coordinates, N>=3 with non-collinear source points. It returns a 2×3 affine
matrix and rank/condition/residual diagnostics. Invalid shape, non-finite values,
and rank deficiency raise `ValueError`. Numerical experiments test the fitted
transform on separate probe points, never used for fitting or RANSAC selection.
Synthetic coordinates and errors are in pixels, not orchard metres.

`extract_appearance` accepts a valid RGB path and optional D path. HSV H16×S8
histograms and nonzero D16-bin histograms are L1-normalized. Pair distances use
Hellinger distance (0–1). D mean/std differences use uint8 grayscale units;
zero-fraction differences are dimensionless. D uses the same center crop and
nearest-neighbor resizing to avoid fabricating intermediate intensity values.
Missing/unsupported D has missing flags and NaN statistics. All-zero D has
zero fraction 1 and a no-nonzero flag; its nonzero distribution is unavailable.

The fixed 21-feature baseline combines 12 ORB/geometric columns with 9 color/D
columns. Identity, row, paths, filenames, dates, seasons, GT masks, review notes,
and status fields are metadata only. All-missing feature columns are retained by
the training-fitted imputer. Scores are unbounded continuous regression values,
not probabilities. Targets and balanced sample weights come only from sampled
training pairs; no validation/test values fit preprocessing.

`training_pairs.csv` records exactly the samples, features, weights, and label
provenance to reuse for SVR. `train_rankings.csv` and `validation_rankings.csv`
record every gallery candidate and score, displayed rank, and numerical tie
range. Per-query Top-k and MRR average over tied positions. `metrics.json` reports
class-specific errors as well as overall values. The query-level export retains
source label provenance. `run_summary.json` records settings, dependencies,
protocol hashes, all-missing features, weighted design rank, feature-matrix bytes,
and timing scope. Memory bytes cover the numeric matrix, not process peak RAM.

The independent audit is read separately. A documented spelling alias
`confrimed` maps to confirmed; an identified reviewer plus an explicit inability
to identify/confirm note maps pending to uncertain. Original source status is
exported alongside the effective status and normalization reason. It is not
written back. No rejected audit case is removed from the frozen sample/gallery.
Stage 5 computes zero independent test scores.

## RBF-SVR inputs, selection, and outputs

`run_svr(root, baseline, settings)` accepts the unmodified in-memory result from
`run_baseline` and `config/svr.json`. It rejects test feature tables, overlapping
train/validation identities or orchard rows, partial galleries, metadata feature
columns, changed sampled targets/weights, and disagreement between sampled and
full-gallery training feature values. Existing NaNs are handled by the same
training-fitted imputation as the baseline. Both models share all numeric inputs.

The default grid enumerates 18 RBF configurations. C must be positive, gamma must
be positive or `scale`, and epsilon must be nonnegative. C acts with the shared
sample weights; epsilon has target-score units. Resolved gamma is exported after
training preprocessing. Fit convergence is checked and failed configurations are
ineligible for selection. Top-1, MRR, and median amortized cached-feature scoring
time form the ordered selection criteria; trial ID is the final exact tie-break.
Validation uses every candidate in the validation split, without row restriction.

`search_results.csv` includes every configuration, fit status/time, Top-1/Top-5/MRR,
class-specific sampled-training and full-validation MAE/RMSE, balanced-class MAE,
resolved gamma, support-vector count, iterations, and repeated prediction times.
Training and validation error aggregates have different class frequencies; their
class-specific errors should accompany overall errors. Stage 6 creates no test metrics.

`config/svr_selection.json` is the frozen small selection record. It contains a
fingerprint of feature tables, shared training samples, code, settings, and
versions. Repeated runs reuse the choice and check its best validation metrics;
changed fingerprints raise an error before fitting or replacing that lock. A new
intentional experiment requires a separate selection file. The lock is copied to
`outputs/svr_main/selected_config.json`. Synthetic runs use temporary paths.

`metrics.json`, full split rankings, `query_metrics.csv`, and
`development_comparison.csv` follow the baseline's label and tie conventions.
`support_vectors.csv` maps each support vector to a sampled training pair, target,
weight, and dual coefficient. `run_summary.json` records convergence, timing,
storage, selected settings, versions, fingerprints, and saved-model checksum.
Support-vector array bytes exclude other process/solver memory. The configured
kernel cache is a limit rather than a measured peak memory value.

`models/rbf_svr.joblib` stores the train-only fitted pipeline with the exact feature
order and extraction settings for `match_tree`. Validation rows are never merged
into fitting after selection. All 40 audit records and the 132-candidate winter
test gallery remain fixed and unscored until the final evaluation stage.

## Frozen neural-feature extension

`src.deep_features.embed_paths(paths, root, settings)` accepts explicit RGB image
paths and returns 512-D L2-normalized ResNet18 feature vectors. The production
path requires the verified official IMAGENET1K_V1 checkpoint. Missing/corrupt
weights or RGB inputs raise rather than switching to random weights or deleting
queries. Preprocessing is a center-square crop followed by official weight
transforms (resize256, crop224, ImageNet normalization). Inference uses a frozen
network in evaluation mode; no orchard CNN fitting or batch-normalization updates
occur. Cache keys include source/weight/code hashes, software, device, batch size,
and CPU thread settings. Numeric caches use `allow_pickle=False`.

`fit_projection` learns a PCA basis from training images only, pooling Summer and
Winter training images. The fixed default is 32 components; validation images are
transformed without refitting. Pair features are 32 absolute PCA-coordinate
differences, 32 PCA-coordinate products, and cosine distance (0–2) between original
normalized embeddings. All are dimensionless, with no metric 3D interpretation.
No D intensity, tree ID, row, path, note, season label, or GT mask enters this
65-column numerical matrix. Seasons only identify query versus gallery images.

`run_transfer` validates the original sample identity/label/weight hash, disjoint
orchard rows, and complete galleries. Both regressors use identical features and
training-only preprocessing; the original 18-setting SVR search is retained.
Only train/validation RGB paths reach the encoder. No test images are embedded,
scored, or used for PCA/selection. The selected SVR stays fitted on training pairs.
A distinct selection file freezes the new representation's configuration.

`outputs/resnet_matching/embedding_manifest.csv` records image metadata and hashes.
`training_pairs.csv` freezes the exact original pairs/targets/weights and 65 new
features. `search_results.csv` preserves all configurations and diagnostic errors.
`linear/` and `svr/` contain complete train/validation rankings, query provenance,
and class-specific metrics. `development_comparison.csv` retains original and new
results. Original D features are omitted in this extension, so the comparison is
not a controlled isolation of the CNN's effect. Synthetic execution reads only
its own synthetic comparison, never real experiment results.

`models/linear.joblib` and `models/svr.joblib` hold the fitted regressor pipeline,
PCA, exact feature order, and verified encoder reference. The large checkpoint is
cached separately; preserve it for offline inference. `match_resnet(query_rgb,
gallery, bundle, root)` accepts candidate_id/rgb columns and returns every ranked
candidate with score and numerical tie ranges. Labels are unnecessary at inference.
Its known-query predictions are checked against the saved full-gallery scores.

The extension-only notebook runner executes configuration and cells tagged
`resnet_extension` in a fresh kernel, then updates those cells in the main notebook.
It uses existing stage-5 inputs and preserves original experiment files. A normal
Restart & Run All executes the entire workflow. Additional dependency versions and
the explicit checkpoint download are documented in README and requirements-resnet.

## Stage 7: final evaluation contract

`src.final_experiments.run_final` takes the completed preparation, screening,
baseline and locked SVR results. Its preflight validates exact sampled features,
weights, preprocessing, model parameters and original protocol hashes. The six
controlled variants use training/validation only; the unchanged main SVR settings
apply to every variant. The reference must reproduce saved validation scores.
Feature caches key on image content, ORB/appearance settings and code. Missing D,
descriptors and geometry remain represented, never silently deleting a pair.

Before test feature extraction, `final_experiments_lock.json` records all choices
and input hashes. Each frozen query uses the entire frozen gallery. The source
human ledgers, query/gallery/protocol and selections remain byte-identical.
The optional ResNet test branch uses its previously locked PCA/model without fit.

`test_pair_features.csv` has split/query/candidate metadata and the 21 fixed
numerical features. `supplementary_test_pair_features.csv` holds the 65 neural
pair features when enabled. Metadata columns never reach pipeline.predict.
Each `*_test_rankings.csv` retains all candidates for every sampled query with
scores, display rank and tolerance-aware rank ranges. `*_test_queries.csv` keeps
the independent human status for all sampled IDs. `verified_eligible` is true only
for confirmed pairs; verified metric columns are null for all other statuses.
Top-k and reciprocal-rank columns without the verified prefix describe filename
positions, including unverified records, and are not independent truth.

`test_metrics.json` separates confirmed-subset retrieval/regression/95% intervals
from `all_sample_filename_diagnostic`. Rejected/uncertain/pending cases remain in
sample counts and rankings. Primary regression errors use confirmed queries only;
other gallery IDs are assumed distinct filenames. Wilson intervals require binary
Top-k credit; fractional ties use bootstrap instead. MRR and paired differences
bootstrap queries with seed 42. These are conditional on the verified subset and
fixed gallery, not orchard-row population intervals. No model is tuned on test.

`development_comparison.csv` reports reference, D/RANSAC ablations, crop and size
controls, positive/negative errors, support counts and raw/median fit/score times.
`svr_parameter_sensitivity.csv` preserves all original 18 development trials.
`performance.csv`, `performance_scope.json`, and `memory_and_models.csv` distinguish
cache build, warm query timing, sampled absolute RSS and array storage. `.prof`
and text summaries profile cumulative calls in the cached batch query path.

## Stage 8: export and verification contract

`src.reproduce.export_run` copies source/configuration, a clean source notebook,
all fitted models and explicit preprocessors, environment versions, human
records, numeric test features/rankings and aggregate figures to an ignored
portable directory. Raw data, example photographs, pretrained weights and large
development caches are excluded. The source workspace notebook keeps outputs.
`RUN.md` gives environment and execution instructions; `feature_schema.json`
fixes feature order and preprocessing; `replay.json` maps each model to its feature
and reference ranking tables. `manifest.json` lists SHA-256 and bytes for every
packaged file except itself. Its own digest is in local `export_summary.json`.
Unexpected old export files cause an error instead of silently retaining them.

`verify_export` checks paths/hashes before loading trusted joblib files, replays
all scores from numeric features, and checks rank ranges/display order. The
original-image check separately reloads both main models and compares full-gallery
validation predictions. A copied bundle can replay without data or CNN weights;
image extraction needs the user-supplied original images and optional checkpoint.
Small aggregate results under res/ are suitable for version control. A different
software/feature/device environment needs separate selection/output paths.

## Original-feature model-parameter revision (main notebook section 6.4)

`tools/tune_svr_parameters.py` validates/reuses the baseline's exact numeric
features, training pairs and inverse-frequency weights. It accepts complete
train/validation galleries only. The revised 36-setting grid and MRR-first order
are in `config/svr_tuning.json`, with a separate `config/svr_tuning_selection.json`
lock and `outputs/svr_parameter_tuning/` artifacts. Imputation/scaling fit training
samples only and are checked against the baseline. All trial errors distinguish
positive/negative pairs; batch prediction and final fitting repeat five times.

The main notebook stores the revised model as `svr_tuned`; its original `svr`
continues to define historical sections 7-8. No test features or results enter
this revision. The saved validation feature table, two model bundles, reference
rankings and manifest support numeric replay with `src.reproduce.verify_export`.
The optimization-copy notebook remains separate and is not edited by this run.

## Validation-only optimization contract (section 9 in the separate copy)

`apple_tree_matching_opt_attempt.ipynb` contains the validation-only revision and
its saved outputs. `apple_tree_matching.ipynb` retains the original stages 1-8.
Use `tools/run_notebook.py --notebook apple_tree_matching_opt_attempt.ipynb
--optimization-only --in-place` to select this checkout's copy. The historical
runner default and notebook-export wrapper still name the absent
`apple_tree_matching_optimization.ipynb`; see the [first-run guide](quickstart.md).

The original src/ executable implementation and its locked model/test artifacts
are preserved. Comments/docstrings were expanded for the software reference;
raw source hashes consequently change (see [maintenance](maintenance.md)).
The opt-in revision is in tools/matching_diagnostics.py,
tools/relative_features.py and tools/feature_optimization.py. Its plan is
`doc/optimization_plan.md`, its settings `config/optimization.json`, and its
selection `config/optimization_selection.json`. No test features/predictions are
accepted by the new feature constructor or used in selection.

Original validation diagnostics report finite-only AUC with class-specific
coverage and missingness, train-imputed AUC and all-gallery single-feature MRR
for both directions. Full rankings distribute expected Top-k credit over boundary
ties; displayed ties still sort IDs solely for inspection. Hub counts and rank
histograms/CDFs describe filename identities, not new human confirmations.

Image features key on RGB/D content, settings, OpenCV and implementation hashes.
Numeric NPZ caches are written atomically and loaded with allow_pickle=False.
One Hamming kNN computation serves both 0.85/0.90 ratio filters, each retaining the
original unique-target filter and affine-RANSAC definition. CLAHE affects ORB
grayscale only. Masked settings also restrict appearance summaries to a central
D-intensity-percentile band; zero/unavailable/poor masks fall back to full image
and are audited. D intensity direction is unknown, and these masks have no
verified segmentation/physical-distance interpretation.

The five raw/screening representations are explicit. Eligible models remove
`query_keypoints`, `missing_query_descriptors`, `missing_query_depth`, and
`query_depth_no_nonzero`. The pair-only and relative representations additionally
remove the four pure candidate columns. A pairwise foreground-fraction difference
is added. Seasonal residual features use per-season means/scales fitted on the
two seasons of training images only; percentile-D histograms use each image's
nonzero distribution without a fitted depth scale.

Missing fills fit the original sampled training pairs. The full-gallery context
then appends direction-oriented percentile ranks/z-scores, and nonnegative match
count/rate ratios to the maximum/second maximum (bounded at 10 for a near-zero
second maximum). Constant columns have zero z-score and neutral percentile. The
full training gallery is processed before selecting the original sampled pair
rows. Inference context uses the entire intended candidate gallery, so adding or
removing candidates can change scores. No candidate labels enter this transform.
These global fills and the pipeline imputer/scaler are training-only fits.

The screening pilot is C=1/gamma=scale/epsilon=0.2. Only eligible pilot SVR results
select an image/representation by MRR first. A separate 36-fit grid selects
C/gamma/epsilon on that representation using MRR, then Top-1, Top-5, scoring time
and trial ID. The grid stores five raw timing repetitions and all class errors /
support-vector fractions. Raw all-column controls are descriptive and cannot win
final selection. This staged design is not an exhaustive joint search.

The new selection fingerprint includes raw development image hashes, original
pair IDs/targets/weights, source/versions and the complete predeclared settings.
A changed fingerprint requires a separate lock/output. An unchanged run verifies
artifact hashes and reuses saved fitted models, rather than repeating selection
with noisy timing. Historical model, label and result hashes are checked around
execution. No test image inference occurs in this new workflow.

`match_tree_optimized` takes query RGB/D, a complete candidate_id/rgb/depth gallery,
a trusted saved optimized bundle and a root. Bundles contain global fills,
train-fitted season statistics, image settings, feature order and the fitted
imputer/scaler/regressor. The legacy matching API remains for the old models.
Every saved new model must reproduce complete validation image inference.

The separate optimization export includes the new source/configuration/clean
notebook, models, numerical validation features and rankings, metrics and aggregate
figures. It excludes original images, foreground example photographs and large
caches. The unchanged generic checksum/score/rank replay verifies both optimized
models. Small results go under res/feature_optimization/. The historical export
and test results are preserved; new validation scores are development evidence.
