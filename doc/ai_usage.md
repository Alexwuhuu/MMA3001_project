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
