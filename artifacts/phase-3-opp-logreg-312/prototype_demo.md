# Phase 3 Baseline Prototype Demo

Run the baseline pipeline:

```bash
prert phase3
```

Run with a custom labeled dataset:

```bash
prert phase3   --labeled-input-path path/to/labeled_dataset.jsonl   --output-dir artifacts/phase-3
```

Run with auxiliary training data while keeping held-out evaluation on the primary dataset:

```bash
prert phase3     --opp115-root data/raw/OPP-115     --auxiliary-labeled-input-path path/to/auxiliary_dataset.jsonl     --output-dir artifacts/phase-3-aux
```

Inspect outputs:

- artifacts/phase-3/phase3_manifest.json
- artifacts/phase-3/classifier_metrics.json
- artifacts/phase-3/calibration_test.json
- artifacts/phase-3/threshold_sweep_test.json
- artifacts/phase-3/bootstrap_ci_test.json
- artifacts/phase-3/model_card.md
