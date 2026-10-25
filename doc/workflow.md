# Workflow and API usage

## Stages and data boundaries

| Stage | Entry point | Contract |
| --- | --- | --- |
| 1–2: audit and labels | `src.main.run_data_preparation` | Index N/U observations, preserve human review, split complete orchard rows |
| 3: geometric screening | `src.screening.run_targeted_review` | Freeze test queries before scoring; screen full train/validation galleries |
| 4: numerical checks | `src.numerical.run_numerical_verification` | Fit affine models and measure error on independent synthetic probes |
| 5: baseline | `src.baseline.run_baseline` | Sample training pairs and fit weighted Linear Regression |
| 6: main selection | `src.svr.run_svr` | Reuse exact samples/features/weights; select RBF-SVR on validation |
| Optional RGB extension | `src.transfer.run_transfer` | Frozen ResNet18, training-only PCA and a separate selection lock |
| 7: independent evaluation | `src.final_experiments.run_final` | Fixed-model diagnostics, pre-test lock, frozen-gallery evaluation and profiling |
| 8: portable export | `src.reproduce.export_run` | Package source/models/numeric evidence and verify prediction replay |
| Separate refinement | `tools.tune_svr_parameters.run_tuning` | Original features, separate MRR-first parameter search |
| Separate optimization | `tools.feature_optimization.run_optimization` | Pilot image/representation screen followed by MRR-first SVR search |

Training samples, imputation and scaling stay training-only. Validation selection
does not merge validation rows into fitting. Test results do not change model
choices, independent review decisions or the frozen query list.

## Inputs, outputs and units

Case tables use `tree_id`, `row`, `split`, seasonal RGB/D paths and review
provenance. Pair tables use `split`, `query_id` and `candidate_id` plus a fixed
numeric feature order. Metadata columns do not enter the regressor.

| Quantity | Meaning and missing-value behavior |
| --- | --- |
| Keypoint coordinates | Pixels after crop/resize, with shape `(N, 2)` |
| ORB descriptors | uint8 arrays with shape `(N, 32)`; no descriptors means empty arrays |
| Hamming distance | Bit count from 0 to 256; unavailable values remain NaN |
| Reprojection RMSE | Resized-image pixels; unavailable geometry has NaN error and a failure flag |
| HSV/D histogram distances | Dimensionless Hellinger distances, 0 to 1 for valid normalized histograms |
| D mean/std | Exported uint8 intensity units with an unknown physical scale |
| ResNet embedding | Unit-normalized 512-dimensional vector |
| PCA pair features | Dimensionless absolute differences/products plus embedding cosine distance |
| Model score | Unbounded continuous regression value; larger scores rank first |

Missing D is retained with flags. All-zero D has known zero fraction one and an
unavailable nonzero distribution. Median imputation fits the original sampled
training rows, including preservation of all-missing feature columns.

## Minimal numerical example

This example needs no dataset or model files:

```python
import numpy as np
from src.numerical import fit_affine_lstsq, transform_points

source = np.array([[0., 0.], [1., 0.], [0., 1.]])
target = source + [2., 3.]
matrix, diagnostics = fit_affine_lstsq(source, target)
assert np.allclose(transform_points(source, matrix), target)
assert diagnostics["rank"] == 3
```

Invalid coordinate shapes, non-finite values and rank-deficient source geometry
raise `ValueError`. RANSAC uses the same least-squares solver on selected inliers.

## Rank a query with a saved main model

```python
from pathlib import Path
import joblib
import pandas as pd
from src.baseline import match_tree

root = Path.cwd()
bundle = joblib.load(root / "outputs/svr_main/models/rbf_svr.joblib")
gallery = pd.DataFrame({
    "candidate_id": ["candidate_1", "candidate_2"],
    "rgb": ["my_images/winter_1.png", "my_images/winter_2.png"],
    "depth": ["", ""],
})
ranked = match_tree({"rgb": "my_images/summer.png", "depth": ""}, gallery, bundle, root)
print(ranked[["candidate_id", "score", "rank_min", "rank_max", "display_rank"]].head(5))
```

Supply your own readable paths and all intended gallery candidates. Load only
trusted joblib artifacts. The same API supports the fitted main Linear Regression
bundle. RGB failures raise; missing/unsupported D is retained. Candidate IDs name
output rows and resolve display order within ties; they supply no numeric features.

`src.transfer.match_resnet` instead accepts an RGB query, `candidate_id`/`rgb`
gallery and a saved neural bundle with PCA. It verifies the encoder checkpoint
against the fitted experiment.

`tools.relative_features.match_tree_optimized` accepts the RGB/D query/gallery
schema with a saved optimized bundle. Its percentile/z-score/ratio features depend
on the **entire intended gallery**. Adding/removing candidates can change scores;
do not construct context from the sampled training-pair table.

## Ties, truth and evaluation

`rank_min` and `rank_max` describe numerical tie positions using the configured
absolute/relative tolerance against each tie-group leader. `display_rank` is
one-based and deterministic. Top-k credit and reciprocal rank average over every
position in the tie, preventing candidate-ID ordering from determining accuracy.

Development positives mean matching filename IDs. Other IDs supply nominal
negative targets; this does not independently verify every negative pair. Only
`confirmed` audit queries enter verified test metrics. Rejected, uncertain and
pending queries remain in full rankings and counts. Filename diagnostics for all
sampled queries are labeled separately. Wilson intervals cover binary Top-k;
fractional ties and MRR use query bootstrap. Intervals describe the verified subset
and fixed gallery, not an orchard-population estimate.

## Review and export pages

`outputs/targeted_review/index.html` is a human-review interface. Its alternatives
are suggestions, while `independent_test/index.html` contains original pairs
without scores. `review_sheets_all/index.html` shows ledger statuses/notes at
generation time; rerender it after intentional ledger updates.

These review pages differ from this Sphinx software reference. Rendering them
does not confirm/reject identities. Record review decisions in the designated
CSV ledgers and retain frozen independent queries.

For an existing trusted portable export:

```powershell
python -m src.reproduce --verify outputs/reproducibility
python -m src.reproduce --verify outputs/reproducibility --checksums-only
```

The first command verifies manifest hashes before loading models and replaying
scores/tie/display ranks from numeric features. The second checks file integrity
without model loading. Raw image inference separately requires the original
images and, for ResNet, the verified checkpoint.

## Failure handling

| Failure | Next action |
| --- | --- |
| Missing/corrupt RGB | Restore the source image; do not silently remove a frozen query |
| Missing/unsupported D | Inspect exported quality/missingness flags; the pair remains present |
| Changed input fingerprint | Preserve historical artifacts and choose separate experiment paths |
| Incomplete frozen protocol | Restore original queries/gallery/protocol/ledger together |
| No converged SVR trial | Review saved trial statuses and solver/grid settings before a separate intentional run |
| Export checksum/replay mismatch | Restore the corresponding trusted bundle; do not bypass validation |
| Missing optimization notebook | Use an explicit existing `--notebook` path; see the first-run guide |

Exact parameters, return dictionaries, side effects and exceptions are described
per callable in the [source-derived API reference](_generated/index.rst).
