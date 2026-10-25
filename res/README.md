# Versioned aggregate results and resources

Keep small, shareable aggregate results here so a Git clone provides inspectable
evidence without original data or fitted model binaries.

| Directory | Contents | Evaluation scope |
| --- | --- | --- |
| `final_experiments/` | Comparison tables, intervals, controls and performance figures | Historical confirmed-subset test and separate development controls |
| `svr_parameter_tuning/` | Search results, selected configuration and diagnostic plots | Validation-only refinement on original features |
| `feature_optimization/` | Screening, diagnostics and selected representation comparison | Repeatedly selected validation results |
| `validation/` | Independent arithmetic audit of saved scores and analytical references | Existing frozen evaluation only; no new inference or selection |

The main workflow remains `apple_tree_matching.ipynb`. See the
[experiment reference](../doc/experiment_reference.md) for interpretation and
the [workflow guide](../doc/workflow.md) for complete local output locations.
Full rankings, model binaries, review photographs and portable bundles belong in
ignored `outputs/`; large feature caches and checkpoints belong in `.cache/`.
Do not substitute these aggregate summaries for a numeric replay bundle.

## Original resources

The original orchard numbering diagram is
`data/TreeID/TreeID/FANK_Sat.png`: R increases to the right and N downward.
Original resources remain with the dataset rather than being copied into the
code repository.

Generated review sheets and the R/N index diagram are under
`outputs/data_preparation/`. Targeted-review pages and independent test sheets
are under `outputs/targeted_review/`. These generated review resources are ignored
by Git; the selected aggregate tables/plots listed above are intentionally versioned.
