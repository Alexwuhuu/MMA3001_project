# Independent validation evidence

`evidence_audit.json` records the standard-library audit of saved numeric results
from the existing 40-query, 132-candidate real-data experiment. It checks 5,280
ranking rows for each of four models against the frozen query/gallery IDs and
normalized human audit. Only 28 confirmed queries define verified metrics.

It independently recalculates ranks/ties, retrieval metrics, regression errors,
binary Top-k Wilson intervals and analytical uniform/constant-zero references.
Hashes identify the CSV/JSON evidence actually checked. No images, model fitting,
new predictions or identity confirmations are involved. Bootstrap intervals and
human decisions are not independently re-established by this report.

Recreate it from existing local evidence with:

```powershell
python -m tools.validate_evidence --output outputs/validation_audit/final_experiments.json
```

See main notebook sections 7.1 and 7.4–7.5 and the
[experiment reference](../../doc/experiment_reference.md#independent-saved-evidence-audit).
Passing this arithmetic check does not establish reliable tree identification.
