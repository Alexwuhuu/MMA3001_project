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

Explanatory documentation and generated interface text are English. Human review
notes and multilingual encoding-test fixtures remain verbatim source evidence.
