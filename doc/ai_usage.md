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
future explanatory content. Original human notes remain verbatim, and Chinese
encoding-test strings remain test data. No human identity decisions or frozen
query/gallery files are changed by translation.
