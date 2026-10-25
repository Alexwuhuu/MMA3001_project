# AI usage record: implementation stage

## 2026-10-25: CI notebook-path and runner compatibility repair

Codex reproduced the failing CI notebook command: it referenced the absent
`apple_tree_matching_optimization.ipynb` instead of the supplied
`apple_tree_matching_opt_attempt.ipynb`. The workflow, optimization-only runner
default and optimization notebook exporter now use the supplied filename.
README, first-run instructions and the data contract describe the repaired paths.

Both jobs use Ubuntu 24.04 and v6 of checkout, setup-python and upload-artifact.
Notebook artifacts are uploaded only when a matching executed file exists;
artifact uploads fail explicitly if an expected file disappears before upload.
Regression tests execute the default optimization-only selection in a real kernel
and verify clean notebook packaging plus updated manifest hashes.

Local verification: 58 tests passed. The exact core-plus-optimization synthetic
CI command completed all 32 code cells; the full main synthetic command completed
all 31 code cells with the optional ResNet extension enabled. Both generated
notebooks contain no cell errors. Workflow YAML/notebook paths and the Sphinx
build with 63 HTML pages passed validation. Hosted Actions execution still
requires committing/pushing the repair; no commit or push was performed.

## 2026-10-25: inline comments, docstrings and HTML reference

Codex revised production-module and tool docstrings to a consistent Google style,
including parameter meanings/shapes/units, return structures, principal validation
and I/O exceptions, and representative usage. Inline comments explain intent at
split, tie, missing-data, cache-publication and experiment-lock boundaries.

Sphinx now builds the first-run/workflow/data-contract/maintenance guides and an
API reference derived from source ASTs and docstrings. It does not import or run
the experiment stack. The documentation command checks signature/docstring
coverage, fails on Sphinx warnings and verifies generated relative links, assets
and fragments. A separate CI job uploads the HTML site. Documentation dependencies
are isolated from the pinned experiment requirements.

Verification: all 56 project tests passed; six doctest examples passed; the HTML
build completed without warnings and 63 generated pages passed local link checks.
Parsed ASTs, after removing docstrings, matched the pre-edit production/tool
snapshot. Existing configuration/result/notebook bytes remained unchanged.
Raw source hashes still change for comments/docstrings, so existing experiment
locks are preserved and the maintenance guide explains separate intentional reruns.
The currently supplied optimization notebook filename is documented explicitly;
historical entries below retain their original chronology. No real-data fitting,
test rescoring, annotation migration, commit, push or upload was performed.

Dates below use Australia/Sydney.

## 2026-10-22: data preparation

Codex assisted with planning and implementing the notebook, filename parsing,
image-quality audit, candidate pair ledger, orchard-row splits, leakage checks,
review sheets, portable exports, synthetic tests, and GitHub CI configuration.

The user specified the repository structure, Linear Regression baseline,
course-taught SVR as the later main regression method, removal of HOG, and an
initial implementation scope limited to stages 1–2. The user instructed Codex not
to commit or push.

Sources were the user-provided PNG data, the user's R/N/U definitions,
`FANK_Sat.png`, and the [dataset paper](https://arxiv.org/pdf/2010.06879).
The paper's deep networks were not treated as a project requirement.

Synthetic checks cover special names, U isolation, corrupt inputs, missing D,
preservation of human decisions, explicit rejection, deterministic splits, and
identical RGB content crossing splits. The complete notebook was also executed.
Validation status must follow actual command results and notebook outputs.

Actual reviewers enter identity decisions in `config/pair_review.csv`. Neither
AI nor automated checks fabricate approvals. Unverified records retain pending
status. No regression model was trained and no orchard retrieval accuracy was
claimed. Codex did not commit or push changes.

## 2026-10-23: review atlases and targeted screening

At the user's request, Codex added a reproducible full N-pair atlas, an HTML
browser index, and a UTF-8 page lookup. Generation reads existing decisions and
notes for display while preserving original images and review-ledger bytes.

The user then chose targeted review instead of full manual verification. Codex
implemented ORB/affine-RANSAC features, full-gallery comparisons within train and
validation splits, candidate-comparison pages, and optional batches of up to 20.

Before automated scoring, 40 independent test queries and the complete test
gallery were fixed. Independent audit states start as pending and must be entered
by a human. The program does not produce identity truth or delete cases because
a final model predicts them incorrectly.

Only the existing 13 human-rejected pairs were excluded. Retained identities
explicitly distinguish filename labels from human confirmations. No additional
trained prediction model or HOG was used; original images and human decisions
were preserved.

Additional checks cover known affine correspondences with outliers, missing and
degenerate inputs, stable sampling, cache reuse, zero-evidence ties, and absence
of automatic relabelling. Real-data validation establishes execution and
diagnostic output, not identity accuracy.

Default settings provided insufficient geometric evidence on the real
train/validation data. Full tie ranges were preserved, weak descriptor ordering
was not presented as reliable mismatch ranking, and review batches were not
padded. A separate parameter diagnostic uses training data only and does not
change defaults or inspect test scores.

## 2026-10-24: English documentation

At the user's request, Codex translated project Markdown, notebook explanations,
and generated review-page instructions into English. English is the default for
future explanatory content. At this point original human notes were kept verbatim;
the subsequent notes-only translation is recorded below. Chinese encoding-test
strings remain test data. No human identity decisions or frozen
query/gallery files are changed by translation.

## 2026-10-24: annotation notes in English

At the user's explicit request, Codex translated all 99 non-empty source annotation
notes (59 pair-ledger entries and 40 independent-review entries) and their CSV,
HTML, and notebook display copies into English. The original files were backed
up under `.cache/annotation_notes_original_*/`. Non-note CSV fields, identity
decisions, IDs, image paths, and the frozen query/gallery/protocol files were
verified unchanged. Existing status spelling is handled separately at read time.

## 2026-10-24: numerical verification and linear baseline

Following the user's instruction to continue after independent human review,
Codex added affine least-squares verification and the weighted Linear Regression
baseline to the main notebook. Color and grayscale-D summaries complement the
existing cached ORB/geometric features. Source filename labels and human review
provenance remain distinct. No HOG or SVR training was introduced at this stage.

The baseline uses training-only negative sampling, imputation, standardization,
and inverse-frequency weights. Full training/validation galleries support
tie-aware retrieval metrics and class-specific regression errors. Model bundles,
rankings, exact sampled pairs, coefficients, configuration, versions, and repeated
timings are exported. Independent test queries are not scored or used to tune
features. Rejected and unresolved audit records remain in the frozen sample.

The spelling `confrimed` is recognized as an input alias for confirmed. Pending
entries explicitly marked by a reviewer as impossible to identify are represented
as uncertain in memory. Normalized outputs retain original status and reasons;
source status fields are not overwritten and no human confirmation is invented.

All 26 automated tests passed. Checks cover known affine transforms and separate
probe points, missing/all-zero D, cache changes, training-only preprocessing,
sample isolation, tied rankings, protocol integrity, and persisted-model inference.
The complete 33-cell notebook ran successfully on both synthetic and real data.
Noiseless probe RMSE was 2.20e-13 px. At 25% outliers and 0.5 px noise, median
probe RMSE was 27.42 px for all-point fitting and 0.178 px after RANSAC selection.
These verify synthetic numerical behavior, not real tree identity.

The real baseline used 3,355 sampled training pairs and 21 features. Validation
used 112 queries against all 112 winter candidates: Top-1 2/112 (1.79%), Top-5
7/112 (6.25%), and MRR 0.0580. Performance is weak and does not establish a useful
robot-localization system or verified test accuracy. All 40 sampled test records,
the 132-image gallery, and protocol hashes remain fixed; zero test queries were
scored. RBF-SVR selection and final comparisons remain later stages.

Dependency consistency was checked in the MMA3001 environment. CI was updated
for the new synthetic workflow but has not been run on GitHub. No commit or push
was performed.

## 2026-10-24: RBF-SVR main regression method

At the user's request to implement stage 6, Codex added the course-taught RBF-SVR
main method using the exact Linear Regression training samples, feature values,
weights, and preprocessing. The predefined 18-configuration search uses complete
validation galleries and selects by Top-1, MRR, and repeated scoring time. No
extra prediction model or HOG features are introduced.

A small tracked selection lock records parameters and evidence. Repeated runs
reuse it after integrity and metric checks; changed inputs require a separate
experiment lock. The chosen model remains fitted on training samples only.
Search records, model bundles, support vectors, rankings, diagnostics, and a
baseline development comparison are exported through the same main notebook.

The independent test sample, gallery, and human records are preserved and never
scored for parameter selection. Validation affects model selection, so these
results are development diagnostics rather than independent accuracy estimates.
Implementation checks and executed results are recorded after validation below.
No commit or push is performed.

Stage 6 verification completed: all 35 automated tests passed, including shared
input and row/gallery isolation, selection priority, saved-model image inference,
and repeated-run selection-lock reuse. The full 41-cell notebook (19 code cells)
executed successfully on both synthetic and real data. All 18 real grid fits
converged. The selected parameters were C=10, gamma=0.01, epsilon=0.01.

On the 112-query/112-candidate validation task, SVR Top-1 was 3/112 (2.68%),
Top-5 was 7/112 (6.25%), and MRR was 0.06690. The linear baseline achieved
2/112 (1.79%), 7/112 (6.25%), and 0.05803 respectively. A one-query difference
on a selection set is not evidence of reliable independent improvement. Overall
matching ability remains weak. The selected model used 3,336 of 3,355 training
pairs as support vectors. Median fitting time was about 0.246 s; cached-feature
scoring of the complete validation gallery took about 12.5 ms amortized per
query, excluding extraction and ranking. These timings are machine-dependent.

Source pair reviews, independent audit records, and frozen query/gallery/protocol
files were checked against a pre-stage snapshot and remained byte-identical.
The saved regressors share the exact training-pair hash, feature order, and fitted
preprocessing statistics. Zero independent test queries were scored. GitHub CI
was updated for stage 6 but has not been run remotely. Python source line endings
are fixed to LF in `.gitattributes` to preserve source fingerprints across Git
checkouts on different platforms. No commit or push was performed.

## 2026-10-24: frozen pretrained image features

The user explicitly authorized the proposed pipeline: Summer/Winter images →
pretrained ResNet18 → pair-feature differences → Linear Regression/SVR → rankings.
Codex implemented this as an additional notebook experiment with separate output
and selection paths. The original ORB/HSV/D methods and human records are retained.

The external pretrained checkpoint is TorchVision ResNet18 IMAGENET1K_V1, trained
on ImageNet-1K for generic object classification. Its official download is
https://download.pytorch.org/models/resnet18-f37072fd.pth and its full SHA-256 is
f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec.
It is not treated as an orchard-specific or independently validated identity model.
The CNN remains frozen; only train-image PCA and the two regressors are fitted.

The implementation uses 512-D normalized embeddings, train-only PCA32, absolute
coordinate differences/products, and cosine distance. Both regressors reuse the
original sample IDs, targets, and weights. The new representation uses RGB only;
comparison with original RGB/D features does not isolate the CNN alone. All model
choices use train/validation data, and independent test images remain unembedded.

Codex configured the optional PyTorch 2.11.0+cu128 / TorchVision 0.26.0+cu128 stack
in MMA3001, verified CUDA computation, downloaded the official weights, and added
pinned optional dependencies, cache/provenance checks, saved inference, and
synthetic tests. The new section has an extension-only execution option, allowing
original real-data result files to remain unchanged. Test and run outcomes are
recorded below after verification. No commit or push is performed.

Verification completed: all 40 tests passed. Tests cover frozen parameters and
normalization state, image-content caches, train-only PCA, pair symmetry, shared
samples/weights, test-image isolation, lock reuse, and persisted-model inference.
Unit fixtures with synthetic vectors/random architecture weights verify behavior
only; real experimental metrics use the verified official pretrained checkpoint.
The complete 49-cell synthetic notebook ran successfully. Real feature extraction and model fitting completed in the first fresh kernel,
followed by a display-example missing-import error. After correcting that import,
the complete extension ran successfully in a new kernel with 834 cached embeddings
and the same locked configuration. Both saved regressors and the demonstration
predictions were checked against complete-gallery scores.

The encoder processed 834 train/validation images on the RTX 5070 Laptop GPU.
PCA fitted 610 training images; 32 components retained about 81.27% of training
embedding variance. All 18 SVR configurations converged. Selection chose C=0.1,
gamma=scale, epsilon=0.1. Validation results (112 queries, 112 candidates) were:

| Representation and regressor | Top-1 | Top-5 | MRR |
|---|---:|---:|---:|
| Original ORB/HSV/D + Linear Regression | 2/112 (1.79%) | 7/112 (6.25%) | 0.05803 |
| Original ORB/HSV/D + SVR | 3/112 (2.68%) | 7/112 (6.25%) | 0.06690 |
| Frozen ResNet18/PCA + Linear Regression | 2/112 (1.79%) | 13/112 (11.61%) | 0.07677 |
| Frozen ResNet18/PCA + SVR | 4/112 (3.57%) | 8/112 (7.14%) | 0.08341 |

The new SVR gains one Top-1 success over the original SVR; matching remains weak.
The new linear model has better Top-5 on validation. These selection-set results
are not independent accuracy estimates or proof of a reliable improvement.
All 31 protected original source/result/audit files remained byte-identical,
including original model bundles and frozen evaluation records. Test images
embedded/scored remained zero. No commit or push was performed. CI was updated
for optional CPU dependencies and explicit weight download but not run remotely.

## 2026-10-25: final experiments and reproducible exports

The user authorized stages 7 and 8 of the agreed Linear Regression baseline /
RBF-SVR main-method plan. Codex added controlled validation ablations for D and
RANSAC, crop and 128/256/384 sensitivity, parameter/error plots, fixed-model
independent evaluation, conditional uncertainty and paired comparisons, a
qualitative U example, cProfile, five-repeat perf_counter timings and separately
sampled RSS/model storage measurements. The existing frozen-ResNet extension is
retained as a separately labelled supplementary test comparison.

All original sampled training IDs/weights are reused. Variants keep main SVR
hyperparameters fixed and never score test data. Main models remain train-only;
no post-test selection occurs. A final source/configuration/development lock is
written before test feature extraction. All 40 frozen queries and 132 candidates
are preserved. Confirmed queries alone define verified accuracy; rejected and
unresolved records have no inferred identity. Whole-sample filename positions
are diagnostic only. Query intervals condition on the confirmed subset/fixed
gallery and do not infer wider accuracy from four orchard rows.

The portable export contains source, configuration, clean notebook, fitted
pipelines and explicit preprocessors/PCA, versions, human records, numeric test
features/ranks, aggregate results and checksums. It excludes raw images, example
photographs, checkpoint and large caches. Exported predictions/ranks and saved
image inference are checked. Small aggregate results are prepared under res/;
no report is written. AI does not create human identity confirmations.

Initial verification: the previous 40 checks and five new substantive checks
passed. The full core eight-stage synthetic notebook and the optional-ResNet
synthetic notebook executed successfully. Real-data execution and final results
are recorded below after their actual completion. No commit or push is performed.

Real execution completed in a fresh kernel: all 28 code cells of the 59-cell
notebook ran without errors. All 46 tests passed. The complete core and optional
synthetic executions succeeded; the core was rerun after final preflight checks
were added. Aggregate and error figures were visually inspected.

The frozen sample has 28 confirmed, 7 rejected and 5 uncertain pairs. The main
linear/SVR Top-1 and Top-5 are both 0/28; MRR is 0.01644/0.02504. Supplementary
ResNet linear/SVR Top-1 is 0/28 and 1/28, Top-5 both 1/28, MRR 0.03790/0.05762.
Results remain weak; a higher conditional MRR does not establish useful matching.
Wilson zero-success upper bounds and all paired/individual intervals are retained.
Conditional query intervals are not orchard-row population confidence intervals.

At fixed main SVR settings, D removal lowers validation Top-1 from 3/112 to 1/112;
RANSAC removal preserves 3/112. Full-frame and resolution effects vary by method,
and no test result changes the selected original models. The fixed main SVR query
benchmark improves from about 2.001 s to 0.184 s, excluding shared appearance
extraction and reporting 0.836 s cache build separately. OpenCV descriptor matching
is the cached-batch profile bottleneck. The main SVR uses 3,336 support vectors.

The standalone exported source verifier replayed 5,280 scores/ranks for each of
four model bundles. Both saved main models also reproduced complete validation
image inference. All seven protected source ledgers/protocol/selection files
matched their pre-stage hashes. No report, commit, push or data upload occurred.

Post-execution export fix: a standalone replay creates Python __pycache__ files
in the copied source directory. The exporter originally treated those generated
runtime files as unexpected on a subsequent export. Codex excluded bytecode and
notebook checkpoint folders from the inventory comparison; numerical code,
features, models, sampling, metric rules and rankings did not change. The original
pre-test lock and ranking hashes are preserved in the final output as
initial_evaluation_lock.json and export_revision.json. Only the export-module
checksum was revised in the current lock, with the reason recorded. The revised
export is replayed again and checked against the original ranking hashes.

## 2026-10-25: validation diagnostics and feature revision

The user authorized validation-only single-feature AUC/MRR, candidate hubness and
true-pair rank diagnosis, stronger ORB parameters, query-relative features,
foreground/seasonal appearance controls and a revised MRR-first selection with
larger epsilon values. Codex wrote doc/optimization_plan.md and separate config,
source tools, notebook section 9 and selection/output/export paths. Original src
modules and all historical model-selection/test artifacts are preserved.

Each training query uses the full training gallery for relative context before
the original positive/ten-negative samples are selected. Training/validation
remain row-disjoint. Global missing fills, appearance season statistics and
standardization fit training only. Validation candidate distributions are
label-free inference context. Query-constant attributes are excluded from eligible
new models, but a raw-column control measures their removal. Candidate-only
features receive separate ablation/hubness checks. No HOG or additional learned
prediction algorithm is introduced.

The D mask deliberately avoids assuming an unknown grayscale near/far encoding.
It is a central-seed percentile band, with fallback flags/audits and training-only
illustrations. It is not verified trunk segmentation or a depth map in metres.
CLAHE, masks and seasonal residuals are empirical feature controls. The user is
not promised an improvement. MRR is the new primary selection measure, with
36 C/gamma/epsilon configurations after fixed-pilot representation screening.
Repeated validation use does not establish new independent test accuracy.

Original diagnostics show linear/SVR largest candidate Top-5 frequency 42/112 and
14/112, top-ten shares 45.18% and 19.82%, and median true-pair ranks 51.5 and 49.5.
Feature diagnostics report both directions and missing coverage rather than
silently discarding failed pairs. No human identity confirmation is fabricated.

Eight new checks initially passed, including a synthetic end-to-end run that
reuses sample IDs/weights, rejects test-image access, reloads image inference,
reuses the new selection lock and rejects changed inputs. Rendering was added to
the end-to-end check after the first synthetic notebook exposed a missing cv2
import in mask-illustration code; that import was corrected. A Windows Tk/worker
conflict in the standalone command was fixed using a noninteractive plotting
backend. Numeric caches are saved atomically. No model choice was changed in
response to these implementation fixes. Final execution outcomes follow below.

The completed real revision screened 30 image/representation settings with both
regressors, then all 36 SVR parameter settings converged. The selected raw
384/1000/0.85 representation excludes query-only columns and retains candidate
columns; relative/seasonal or CLAHE/masked variants did not improve the pilot
SVR. Selection chose C=1, gamma=scale, epsilon=0.2. New SVR validation Top-1 is
4/112 versus original 3/112, Top-5 remains 7/112, and MRR is 0.07232 versus 0.06690.
The shared-feature linear model worsens to 0/112, 5/112 and 0.04449. A different
relative-feature linear control reaches MRR 0.0846, but it is not the SVR-selected
representation. Results are not uniformly improved.

Support vectors fall to 2,619/3,355 (78.06%). Larger epsilon does not guarantee
better ranking. Validation D zero fractions average 0.120 Summer / 0.548 Winter;
mask coverage is 30.4% / 14.7%, with no fallback for this data. Mask examples reveal
that the heuristic is not a verified correspondence of trunk regions across
seasons. The selected model does not use CLAHE or this mask. New SVR largest hub
frequency is 17/112 (original 14/112); hubness is not resolved.

All 54 tests passed. The 68-cell core synthetic notebook executed through the
new revision successfully after fixing headless CLI plotting and a missing mask
preview import. Both optimized models reloaded complete validation image inference
and replayed all 12,544 numerical validation pair scores/ranks. The separate new
export passed checksum/ranking verification. All 128 protected original files
remained byte-identical, including original models, selections, human records and
historical test results. No real test images were scored by this revision. No
report, commit, push or upload occurred. The optimization-only fresh-kernel
notebook run subsequently passed: all four new code cells executed without
errors, reusing the new selection lock. The independent copied-bundle verifier
checked 81 packaged files and reproduced both full validation rankings. Updated
source notes and aggregate figures were included in the final export snapshot.

## Notebook separation (25 October 2026)

At the user's request, section 9 and its saved diagnostic/optimization outputs
were moved into the new `apple_tree_matching_optimization.ipynb` copy. The copy
retains stages 1-8 for context; the original `apple_tree_matching.ipynb` now
contains stages 1-8 only. The optimization-only runner selects the new copy.
The export adapter includes both notebook sources and updated run instructions
without modifying locked feature/model implementation or selection files.

## Model-parameter refinement in the original notebook (25 October 2026)

The user requested model-parameter optimization inside the original notebook.
Section 6.4 now refines RBF-SVR on the original 21-feature representation and
shared 3,355 training examples/weights. It expands epsilon to 0.2/0.3 (36 total
configurations) and selects validation MRR first. Linear Regression remains the
same baseline. This is distinct from the feature revision in the optimization
copy. Original source/model locks and historical test predictions are retained;
the new `svr_tuned` model and validation artifacts use separate paths. AI does
not add human identity confirmations or use independent test scores for tuning.

All 36 real-data fits converged. MRR-first selection retained the original
C=10, gamma=0.01, epsilon=0.01; validation Top-1/Top-5/MRR remain
3/112, 7/112, 0.06690. Support count remains 3,336/3,355. No improvement
is claimed from this expanded grid. Eleven targeted SVR/tuning tests passed,
including class-error checks, failed-fit exclusion, leakage rejection,
locked-input rejection and model/ranking replay.

A cold-start check also covers a tracked selection lock with no local output
directory: the declared grid is rebuilt, the locked validation metrics are
verified, and its choice is retained. The initial tuning lock was backed up in
ignored `.cache/` while this rebuild path was completed; the same real-data grid
was rerun under the final implementation fingerprint. No numerical fitting or
feature definition was changed by that cache-rebuild fix.

Final verification passed on the completed implementation: the main notebook
executed the configuration and three parameter-refinement cells in a fresh
kernel, and its complete 66-cell synthetic core workflow also passed. The
18-file numeric manifest replayed 12,544 scores/ranks for each regressor.
All original code/test outputs, the optimization copy and 128 protected
historical files remained unchanged. No report, commit, push or upload occurred.
