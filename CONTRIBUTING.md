# Contributing

The main project entry point is `apple_tree_matching.ipynb`. Keep its problem
definition, methods and conclusions consistent with implementation and results.
Use English for project explanations, documentation and annotation notes.

## Directory responsibilities and naming

| Location | Responsibility |
| --- | --- |
| `src/` | Reusable production functions and matching/evaluation interfaces |
| `tools/` | Execution, diagnostics, downloads and documentation entry points |
| `tests/` | Behavioral unit tests and synthetic integration checks |
| `config/` | Experiment settings, dependencies, provenance and frozen protocols |
| `doc/` | Setup, detailed workflows, contracts and maintenance instructions |
| `res/` | Small aggregate results suitable for version control |
| `data/`, `outputs/`, `.cache/` | Ignored original inputs and generated/local artifacts |

Use descriptive `snake_case` names for Python modules, functions, variables and
experiment output folders. Test modules use `test_<subject>.py`; configuration
names should identify the experiment they control. Keep entry-point filenames
and existing result paths stable: stored locks, notebooks and exports refer to them.
Move files only with their imports, links, configuration and replay references.

Source APIs use Google-style docstrings; see [maintenance](doc/maintenance.md).
The README is an orientation and runnable quickstart, limited to **200 lines**.
Put detailed explanations in `doc/` and link them from the README. Do not duplicate
long experiment logs or present an old test count as a current verification result.

## Branches and focused commits

Use a short branch such as `codex/repository-documentation` for a coherent change.
For example, create it with `git switch -c codex/repository-documentation` before
editing. This is a convention for future work, not a claim about historical branches.
Keep unrelated changes separate and review the diff before staging specific files.

Use an imperative, descriptive subject; a type prefix is encouraged for new commits:

```text
docs: clarify repository setup and reproducibility routes

Explain the environment, data prerequisites and generated output locations.
Separate synthetic verification from historical score/rank replay.

Validation: documentation links and README line count checked.
```

For implementation changes, explain the behavior changed and why. Record relevant
validation and any remaining limitation. Avoid subjects such as `update`, `fix`, or
`final version` without identifying the change. Include a proposed commit description
in the delivery note after each editing task. A description does not itself create
a commit or authorize a push.

## Existing development evidence

The repository already contains focused commits documenting successive stages:

| Commit | Recorded subject |
| --- | --- |
| `3e98563` | implement dataset auditing and row-based splits |
| `0cfa75b` | add targeted pair review and frozen test sampling |
| `ca60593` | add numerical verification and linear regression baseline |
| `4d8cf7b` | add SVR tuning and frozen ResNet18 matching experiments |
| `49541f3` | add final evaluation, validation tuning and API documentation |
| `32a2b0e` | correct notebook paths and update CI |
| `6cbb5ce` | clarify engineering problem and input/output specifications |

Use `git log --oneline --decorate` to inspect the actual checkout and
`git show <commit>` to examine the corresponding evidence. Preserve the original
history; do not manufacture or rewrite commits to satisfy an assessment criterion.

## Data, provenance and reproducibility

Keep original images, local paths, environments, checkpoints, model binaries,
full rankings and large caches in their ignored locations. Review notebook
outputs for embedded source photographs before sharing; ignoring `data/` alone
does not remove embedded images. Code licensing does not license the dataset.

Human records and frozen queries/galleries/protocols are evidence. Automated
scores never confirm identities. Do not resample test failures, infer a new truth
from rankings, or change an audit to improve metrics. Document intentional human
corrections separately with their evidence and preserve the original records.

Locks hash source bytes, data, settings and versions. Use a trusted historical
export for numeric replay. A changed fingerprint requires separate development
settings/output/selection paths, as described in the
[maintenance guide](doc/maintenance.md#source-hashes-and-historical-results).
Do not delete a lock, remove its checks or edit a hash to force compatibility.
New validation-only models cannot inherit historical test performance claims.

## Review and validation

1. Inspect `git diff` and the paths reported by `git status --short`.
2. Run `git diff --check` and `python tools/build_docs.py --check`.
3. For guide edits, verify relative links, commands, the README line limit and
   consistency with the actual CLI. Build Sphinx when guide pages change.
4. For executable changes, run relevant tests and the applicable synthetic notebook
   workflow. Broaden testing when the change or a failure warrants it.
5. Confirm that private/generated artifacts and unintended label/lock changes are
   absent. Use `git check-ignore <path>` to inspect the existing ignore policy.
6. Summarize the problem, resulting behavior and verification in the commit or PR.

Use the pull-request template to make validation and data/protocol effects visible.
CI exercises synthetic data and optional neural code, checks the README limit,
and builds the software reference. It does not establish real-tree matching accuracy.
