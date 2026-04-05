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
- OPP-115 raw corpus available at `data/raw/OPP-115` (reference corpus)

## Using OPP-115

- The raw OPP-115 corpus is present in this repo under `data/raw/OPP-115`.
- Raw OPP-115 files are not directly consumable by `--public-input`, which expects a flat `.csv` or `.jsonl` table.
- Use the OPP-115 preprocessing command below to create canonical flat exports with required fields (`event_date`, `sector`, `records_affected`).
- If you do not have that processed export yet, run Phase 2 without `--public-input` and use OPP-115 only as a reference corpus.

## Commands

Create processed OPP-115 flat files (default: consolidation threshold 0.75):

```bash
PYTHONPATH=src python3 scripts/process_opp115_for_phase2.py
```

Create processed OPP-115 files from the raw annotation set instead of consolidated annotations:

```bash
PYTHONPATH=src python3 scripts/process_opp115_for_phase2.py \
  --input-set annotations
```

Equivalent package command:

```bash
prert-opp115
```

Run Phase 2 using OPP-115 as a reference corpus only (no `--public-input`):

```bash
PYTHONPATH=src python3 scripts/run_phase2_metrics.py
```

Run Phase 2 with a processed OPP-115 CSV export:

```bash
PYTHONPATH=src python3 scripts/run_phase2_metrics.py \
  --public-input data/processed/opp115_public_mapping.csv
```

Run Phase 2 with a processed OPP-115 JSONL export:

```bash
PYTHONPATH=src python3 scripts/run_phase2_metrics.py \
  --public-input data/processed/opp115_public_mapping.jsonl
```

Run Phase 2 with processed OPP-115 input and a custom output folder:

```bash
PYTHONPATH=src python3 scripts/run_phase2_metrics.py \
  --public-input data/processed/opp115_public_mapping.csv \
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
