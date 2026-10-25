# Documentation conventions and maintenance

## One docstring style

Production modules and tools use Google-style docstrings. A callable starts with
an action-oriented summary and explains its purpose or non-obvious constraints.
`Args` follows signature order and describes meanings, units, shapes, defaults
and allowed missing values. `Returns` describes the real object structure and
uses `None` for side-effect-only functions. `Raises` documents relevant validation,
I/O and delegated failures; do not invent exceptions for an infallible helper.
Classes describe their stored `Attributes`. Examples show realistic use without
requiring a full model-selection run.

```python
def fit_affine_lstsq(source, target):
    """Fit an affine transform from corresponding pixel coordinates.

    Args:
        source: Finite coordinates with shape (N, 2), N >= 3.
        target: Corresponding coordinates with the same shape.

    Returns:
        A (2, 3) matrix and rank/condition/RMSE diagnostics.

    Raises:
        ValueError: Shapes, finiteness or source-design rank are invalid.
    """
```

This excerpt demonstrates the convention; the actual function documentation is
generated from `src/numerical.py`.

Inline comments explain intent: why a tie is averaged, why row membership is
frozen before exclusions, why D uses nearest-neighbour interpolation, or why a
cache is published atomically. Avoid narrating assignments, loops or library calls.
Keep explanatory text in English, following the repository convention.

## Keep source and HTML in sync

Run these commands after changing source or documentation:

```powershell
python tools/build_docs.py --check
python tools/build_docs.py
```

The first command uses only the standard library. It checks module docstrings,
top-level function parameter names/order, return sections, explicitly raised
exception names and annotated class attributes in `src/` and `tools/`. This is a
structural check: a reviewer must still confirm descriptions accurately match
behavior and document meaningful propagated exceptions. Short local closures and
tests are outside the published top-level API coverage count.

The build performs a full Sphinx warning-as-error rebuild, then checks generated
relative links/assets/fragments. `doc/_ext/source_api.py` parses source ASTs and
uses Sphinx Napoleon to translate Google docstrings. It reads signatures,
docstrings and source listings from the same file and publishes its SHA-256 on
each module page. It does not import or execute experiment modules, so no dataset,
checkpoint, fitted model or numerical stack is needed for documentation.

`doc/_generated/` and `doc/_build/` are generated and ignored by Git. Edit source
docstrings or handwritten guide pages, then rebuild. Share the whole
`doc/_build/html/` directory to retain styles, navigation and search. Do not edit
HTML or generated reStructuredText directly.

Build from the repository checkout. Historical portable experiment exporters copy
the Markdown guides but do not package the Sphinx configuration/extensions or the
generated HTML site. Share the software-reference HTML directory separately from
those experiment replay bundles.

The independent CI documentation job installs only
`config/requirements-docs.txt`, runs the same build, and uploads the HTML directory.
It catches parameter drift, missing docs, Sphinx warnings and broken internal links
without executing experiments. External links are not fetched by this offline
link check.

## Source hashes and historical results

The feature caches and selection locks intentionally include **raw source-byte
hashes**. Comments, whitespace and docstrings therefore change fingerprints even
when executable behavior is identical. This documentation revision retains that
strict provenance rule and preserves existing lock/result files.

For a historical numeric replay, use the existing trusted portable bundle and
its recorded files. A hash mismatch on a fresh image-backed run must not be
"fixed" by editing a saved lock or deleting human records. For an intentional new
development run, copy the relevant settings in memory and assign distinct
`selection_file`, `output_dir`, `cache_dir`, `small_results_dir` and `export_dir`
where supported. The final stage uses `lock_file`. The original and neural SVR
branches each need their own selection path. Keep the independently frozen
test protocol and human ledgers intact; new development documentation is not a
reason to resample or rerun independent evaluation.

For example, prepare a separate refinement configuration:

```python
from pathlib import Path
from tools.tune_svr_parameters import load_config, load_baseline, run_tuning

root = Path.cwd()
settings = load_config(root)
settings.update(
    selection_file="config/svr_tuning_documented_selection.json",
    output_dir="outputs/svr_tuning_documented",
    small_results_dir="res/svr_tuning_documented",
)
result = run_tuning(root, load_baseline(root), settings=settings)
```

This example intentionally launches a new training/validation experiment using
existing original stage-5 inputs. It is not part of the documentation build and
does not score test queries.

## Contributor verification

Before delivery, review the changed docstrings against the code and run relevant
project tests. For a comments/docstrings-only change, compare parsed ASTs after
removing docstring expressions to establish that executable code is unchanged.
Keep notebooks, original inputs, labels, locks and experiment results unchanged.

The documentation tooling follows the official
[Sphinx build guide](https://www.sphinx-doc.org/en/master/usage/quickstart.html)
and [Napoleon Google-docstring reference](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html).
