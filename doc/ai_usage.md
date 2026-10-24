# AI usage record: implementation stage

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
