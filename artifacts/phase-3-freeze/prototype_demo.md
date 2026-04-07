# Phase 3 Baseline Prototype Demo

Run the baseline pipeline:

```bash
PYTHONPATH=src python scripts/run_phase3_baseline.py
```

Run with a custom labeled dataset:

```bash
PYTHONPATH=src python scripts/run_phase3_baseline.py   --labeled-input-path path/to/labeled_dataset.jsonl   --output-dir artifacts/phase-3
```

Inspect outputs:

- artifacts/phase-3/phase3_manifest.json
- artifacts/phase-3/classifier_metrics.json
- artifacts/phase-3/model_card.md
