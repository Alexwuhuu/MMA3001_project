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
class-specific errors should accompany overall errors. No test metrics exist.

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
