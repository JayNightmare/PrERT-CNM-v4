# Phase 2 Implementation Runbook

This runbook executes Phase 2 in an isolated output folder so Phase 1 code and artifacts remain unchanged.

## Scope

- Build metric specifications from Phase 1 controls.
- Generate synthetic observations for normal, stressed, and adversarial scenarios.
- Optionally map public breach datasets to canonical fields.
- Produce baseline user/system/organization scores and composite risk summaries.

## Inputs

Required:

- `artifacts/phase-1/controls_all.jsonl`

Optional:

- Public breach dataset as `.csv` or `.jsonl` (for example ENISA/PRC extracts)

## Command

Run with default inputs and outputs:

```bash
PYTHONPATH=src python3 scripts/run_phase2_metrics.py
```

Run with optional public dataset mapping:

```bash
PYTHONPATH=src python3 scripts/run_phase2_metrics.py \
  --public-input path/to/public_breach_data.csv
```

Run with custom output directory:

```bash
PYTHONPATH=src python3 scripts/run_phase2_metrics.py \
  --output-dir artifacts/phase-2
```

## Outputs

Written to `artifacts/phase-2/`:

- `metric_specs.jsonl`
- `synthetic_events.jsonl`
- `public_data_mapped.jsonl`
- `baseline_scores.jsonl`
- `phase2_manifest.json`
- `synthetic_data_dictionary.md`

## Quality Checks

- `phase2_manifest.json.coverage_summary.mapped_controls == total_controls`
- `phase2_manifest.json.coverage_summary.missing_controls` is empty.
- `baseline_scores.jsonl` values for `compliance_score` and `risk_score` stay within [0, 1].
- `public_data_mapped.jsonl` rows with missing required fields are flagged in `dq_missing_required_fields`.

## Notes

- This pipeline is intentionally phase-scoped.
- No files under `artifacts/phase-1/` are modified.
- Phase 1 source modules are not imported for mutation; they are only read as input artifacts.

---

## Navigation

[⬅ Back](06-phase1-implementation-runbook.md) | [Next ⮕](08-phase2-technical-documentation.md)
