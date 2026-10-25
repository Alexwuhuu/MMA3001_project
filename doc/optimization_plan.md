# Validation-only feature optimization plan (25 October 2026)

The user authorized diagnostics and a new development experiment after the first
final evaluation. Preserve the original methods, model-selection locks, human
records and historical test results. This revision uses training/validation only;
it neither extracts test features nor produces a new independent test claim.

1. Diagnose the original 21 numerical features on the full validation gallery:
   finite-value positive/negative AUC, missingness AUC, train-median-imputed AUC,
   and single-feature Top-1/Top-5/MRR in both directions. Report missing coverage,
   query-constant features and their uninformative within-query rankings. The
   feature labels are predominantly filename identities, not new human decisions.
2. Count each candidate's displayed and tie-adjusted Top-5/Top-1 appearances for
   both original regressors. Export concentration, candidate-attribute
   correlations, and true-pair rank histograms/CDFs. IDs/rows annotate figures only.
3. Compare ORB at 384 pixels, 1,000 keypoints and ratio 0.85/0.90. Scale RANSAC to
   4.5 px. Compare raw grayscale, CLAHE, and CLAHE plus a center-seeded D-rank mask.
   Reuse k-nearest matches across the two ratios. Keep missing/failed pairs.
4. The D mask is an explicitly heuristic central depth-intensity band, not metric
   depth or validated trunk segmentation. The physical direction of grayscale D
   is unknown, so do not assume larger values mean nearer. Use each image's
   nonzero intensity percentile ranks and a central seed; use the full image with
   a recorded fallback when D/seed/mask is unusable. Masked variants compute HSV/D
   summaries in the selected region. Show masks and their coverage/fallbacks.
5. Within each image setting compare: all original pair columns as a control;
   remove query-only columns; remove query- and candidate-only columns; add
   within-query percentile ranks/z-scores and best/second-best ratios; then add
   train-fitted, season-specific appearance residuals and D-rank histogram distance.
   Query-only columns cannot rank candidates alone or in a linear combination;
   in an RBF model they can condition scores, so measure their removal. Candidate
   attributes may cause hubs or encode useful matchability: measure, do not assume.
6. Compute training context using the **entire training winter gallery**, then
   select the original 1-positive/10-negative pair rows and weights. Validation
   context uses its complete gallery. No sampled 11-pair context is allowed.
   Imputation, global season statistics and the model scaler fit training only.
   Label-free per-query gallery statistics are inference context, not a refit.
   Monotonic relative transformations do not improve single-feature rankings by
   themselves; their potential benefit is combining signals in regression.
7. Screen the six image settings x five representations with Linear Regression
   and the predeclared pilot SVR C=1, gamma=scale, epsilon=0.2. All original-query
   columns are ineligible for the new final selection. Choose image/representation
   by pilot SVR validation **MRR first**, then Top-1, Top-5, scoring time, trial ID.
   This staged search controls computation; it is not an exhaustive joint search.
8. On that representation search C={0.1,1,10}, gamma={scale,0.01,0.1}, and
   epsilon={0.01,0.1,0.2,0.3}: 36 fits, **MRR-first** selection with the same tie
   order. Record class errors, support-vector fractions and five-repeat timing.
   High support-vector fraction is a diagnostic, not proof of underfitting or a
   guarantee that a larger epsilon improves ranking.
9. Save a distinct `config/optimization_selection.json`, train-fitted linear/SVR
   bundles and all validation rankings/diagnostics. Reload both models and repeat
   full-gallery inference. Keep this experiment in the separate
   `apple_tree_matching_opt_attempt.ipynb` copy, retaining stages 1-8 as context.
   The original notebook contains stages 1-8 only. Select that explicit filename
   with the optimization-only runner (see [first-run notes](quickstart.md)) so
   historical test cells need not execute. Export a
   separate reproducibility bundle and small summaries. Existing locked inputs
   cannot be overwritten in response to a new result.

Repeated validation selection makes these development results optimistic. No
claim of a fresh independent test result is made. Keep the course-method contrast
Linear Regression versus RBF-SVR, no HOG, no new learned prediction algorithm.

Sources: [OpenCV CLAHE](https://docs.opencv.org/5.0/main_modules/imgproc_hist.html),
[ORB](https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html),
[AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html),
[SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html).
