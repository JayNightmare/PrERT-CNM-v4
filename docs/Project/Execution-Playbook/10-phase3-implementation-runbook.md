# Phase 3 Implementation Runbook

This runbook executes the Phase 3 baseline classifier pipeline in an isolated output folder.

## Scope

- Build a clause-level labeled dataset from OPP-115 annotations or a provided labeled JSONL.
- Create deterministic train, validation, and test splits with policy-level leakage protection.
- Train a baseline text classifier for user/system/organization labels.
- Evaluate held-out metrics and write reproducible Phase 3 artifacts.

## Current Baseline Snapshot

From the latest `artifacts/phase-3/` run:

- Source: `opp115::consolidation-0.75`
- Total rows: 5000 (train 3872, validation 437, test 691)
- Labels: user, system, organization
- Vocabulary size: 3088
- Validation: accuracy 0.885584, macro F1 0.69043
- Test: accuracy 0.716353, macro F1 0.573119
- Policy overlap checks: train/validation 0, train/test 0, validation/test 0

## Visualisations

- [Phase 3 Baseline Visual Dashboard](11-phase3-visual-dashboard.md) - Dataset mix, split profile, held-out quality indicators, and test confusion flow.

## Inputs

Default source:

- `data/raw/OPP-115`

Optional source:

- Labeled JSONL with fields: `text`, `label`, `policy_uid` (optional: `example_id`, `category`)

## Commands

Run Phase 3 baseline using OPP-115 default input set:

```bash
PYTHONPATH=src python3 scripts/run_phase3_baseline.py
```

Equivalent package command:

```bash
prert-phase3
```

Run with a custom labeled JSONL dataset:

```bash
PYTHONPATH=src python3 scripts/run_phase3_baseline.py \
  --labeled-input-path data/processed/phase3_labeled.jsonl \
  --output-dir artifacts/phase-3
```

Run a bounded sample for quick iteration:

```bash
PYTHONPATH=src python3 scripts/run_phase3_baseline.py \
  --max-rows 5000 \
  --seed 42
```

## Outputs

Written to `artifacts/phase-3/`:

- `training_dataset.jsonl`
- `validation_dataset.jsonl`
- `test_dataset.jsonl`
- `dataset_manifest.json`
- `classifier_checkpoint/model.json`
- `classifier_metrics.json`
- `classifier_metrics.jsonl`
- `validation_predictions.jsonl`
- `test_predictions.jsonl`
- `model_card.md`
- `scoring_spec.md`
- `prototype_demo.md`
- `phase3_manifest.json`

## Quality Checks

- `dataset_manifest.json.policy_overlap.* == 0`
- `classifier_metrics.json.validation.macro_f1` and `classifier_metrics.json.test.macro_f1` are in [0, 1]
- `phase3_manifest.json` includes input config, split counts, and output file references

## Verification

Run pipeline tests:

```bash
PYTHONPATH=src pytest -q tests/test_phase3_pipeline.py
```

Run the cross-phase regression check used in this workspace:

```bash
PYTHONPATH=src pytest -q tests/test_phase2_pipeline.py tests/test_phase3_pipeline.py
```

## Notes

- This baseline intentionally focuses on reproducible held-out classification metrics.
- Bayesian scoring and API integration are deferred to the next Phase 3 increment.

---

## Navigation

[⬅ Back](09-phase1-phase2-progress-dashboard.md) | [Next ⮕](11-phase3-visual-dashboard.md)
