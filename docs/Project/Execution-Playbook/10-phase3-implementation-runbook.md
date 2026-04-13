# Phase 3 Implementation Runbook

This runbook executes the Phase 3 classifier-and-risk pipeline in an isolated output folder.

## Scope

- Build a clause-level labeled dataset from OPP-115 annotations or a provided labeled JSONL.
- Build a clause-level labeled dataset from OPP-115 annotations, normalized Polisis JSONL/CSV files, or a provided labeled JSONL.
- Create deterministic train, validation, and test splits with policy-level leakage protection.
- Train a text classifier backend for user/system/organization labels (`naive_bayes`, `logreg_tfidf`, or `privacybert`).
- Evaluate held-out classifier metrics and emit Bayesian posterior risk outputs.
- Emit calibration reports (reliability bins, ECE, Brier), threshold sweeps, and bootstrap confidence intervals.
- Append canonical run history metadata for dated trend snapshot generation.
- Write reproducible Phase 3 artifacts and manifest-level provenance.

## Current Snapshot (Comparable Baseline vs Upgraded)

From the latest full-data comparable runs:

- Baseline run: `artifacts/phase-3-nb/` using `multinomial_naive_bayes`
- Upgraded run: `artifacts/phase-3-logreg/` using `logreg_tfidf`
- Source: `opp115::consolidation-0.75`
- Total rows: 19720 (train 15645, validation 1803, test 2272)
- Labels: user, system, organization
- Baseline validation/test macro F1: 0.625907 / 0.640117
- Upgraded validation/test macro F1: 0.760958 / 0.780633
- Baseline validation/test accuracy: 0.816417 / 0.814701
- Upgraded validation/test accuracy: 0.889628 / 0.893046
- Policy overlap checks: train/validation 0, train/test 0, validation/test 0

## Visualisations

- [Phase 3 Visual Dashboard](11-phase3-visual-dashboard.md) - Dataset mix, split profile, baseline-vs-upgraded comparisons, and confusion flow.

## Inputs

Default source:

- `data/raw/OPP-115`

Optional source:

- Labeled JSONL with fields: `text`, `label`, `policy_uid` (optional: `example_id`, `category`)
- Normalized Polisis folder (default profile: `data/raw/Polisis/normalized`) with `.jsonl` and/or `.csv` files.

Normalized Polisis row contract:

- Required: `text`
- Preferred for harmonization: `category`
- Optional: `label` (`user|system|organization`), `policy_uid`, `example_id`
- Unknown categories are skipped by the Polisis harmonization path.

## Commands

Run Phase 3 baseline using OPP-115 default input set:

```bash
PYTHONPATH=src python scripts/run_phase3_baseline.py
```

Equivalent package command:

```bash
prert-phase3
```

Run with a custom labeled JSONL dataset:

```bash
PYTHONPATH=src python scripts/run_phase3_baseline.py \
  --labeled-input-path data/processed/phase3_labeled.jsonl \
  --output-dir artifacts/phase-3
```

Run with normalized Polisis source files:

```bash
PYTHONPATH=src python scripts/run_phase3_baseline.py \
  --polisis-root data/raw/Polisis \
  --polisis-input-set normalized \
  --output-dir artifacts/phase-3
```

Run a bounded sample for quick iteration:

```bash
PYTHONPATH=src python scripts/run_phase3_baseline.py \
  --max-rows 5000 \
  --seed 42
```

Run with explicit run metadata and measurement controls:

```bash
PYTHONPATH=src python scripts/run_phase3_baseline.py \
  --run-id phase3-2026-04-07-a \
  --calibration-bins 10 \
  --bootstrap-resamples 1000 \
  --output-dir artifacts/phase-3-nb
```

Run with the upgraded TF-IDF + weighted logistic regression model:

```bash
PYTHONPATH=src python scripts/run_phase3_baseline.py \
  --model-type logreg_tfidf \
  --max-features 20000 \
  --ngram-max 2 \
  --max-iter 1000 \
  --output-dir artifacts/phase-3-logreg
```

Run with the PrivacyBERT backend scaffold:

```bash
PYTHONPATH=src python scripts/run_phase3_baseline.py \
  --model-type privacybert \
  --privacybert-model-name bert-base-uncased \
  --privacybert-epochs 2 \
  --privacybert-batch-size 8 \
  --privacybert-learning-rate 5e-5 \
  --privacybert-max-length 256 \
  --output-dir artifacts/phase-3-privacybert
```

Run with custom Bayesian priors (enabled by default):

```bash
PYTHONPATH=src python scripts/run_phase3_baseline.py \
  --model-type logreg_tfidf \
  --bayesian-priors-path configs/phase3_bayesian_priors.json \
  --bayesian-top-k 5 \
  --output-dir artifacts/phase-3
```

Disable Bayesian scoring output (benchmark/diagnostic only):

```bash
PYTHONPATH=src python scripts/run_phase3_baseline.py \
  --model-type logreg_tfidf \
  --disable-bayesian-scoring \
  --output-dir artifacts/phase-3-no-bayes
```

Run a comparable full-data Naive Bayes baseline for side-by-side benchmarking:

```bash
PYTHONPATH=src python scripts/run_phase3_baseline.py \
  --model-type naive_bayes \
  --output-dir artifacts/phase-3-nb
```

Run a proposal-aligned Phase 3 acceptance freeze (PrivacyBERT + Bayesian-primary checks):

```bash
PYTHONPATH=src python scripts/run_phase3_acceptance_freeze.py \
  --model-type privacybert \
  --strict \
  --output-dir artifacts/phase-3-freeze
```

Run acceptance freeze with Polisis advisory reporting (non-blocking for current milestone):

```bash
PYTHONPATH=src python scripts/run_phase3_acceptance_freeze.py \
  --polisis-root data/raw/Polisis \
  --polisis-input-set normalized \
  --output-dir artifacts/phase-3-freeze
```

## Outputs

Written to the selected `--output-dir` (for example `artifacts/phase-3/`, `artifacts/phase-3-nb/`, or `artifacts/phase-3-logreg/`):

- `training_dataset.jsonl`
- `validation_dataset.jsonl`
- `test_dataset.jsonl`
- `dataset_manifest.json`
- `classifier_checkpoint/model.json` (naive_bayes)
- `classifier_checkpoint/model.pkl` (logreg_tfidf)
- `classifier_checkpoint/privacybert/` (privacybert)
- `classifier_metrics.json`
- `classifier_metrics.jsonl`
- `validation_predictions.jsonl`
- `test_predictions.jsonl`
- `calibration_validation.json`
- `calibration_test.json`
- `threshold_sweep_validation.json`
- `threshold_sweep_test.json`
- `bootstrap_ci_validation.json`
- `bootstrap_ci_test.json`
- `bayesian_risk_validation.json` (when Bayesian scoring enabled)
- `bayesian_risk_test.json` (when Bayesian scoring enabled)
- `model_card.md`
- `scoring_spec.md`
- `prototype_demo.md`
- `phase3_manifest.json`
- `artifacts/phase3_run_history.jsonl` (canonical run-history index)
- `phase3_acceptance_report.json` (acceptance-freeze runs)
- `phase3_acceptance_report.md` (acceptance-freeze runs)

## Quality Checks

- `dataset_manifest.json.policy_overlap.* == 0`
- `classifier_metrics.json.validation.macro_f1` and `classifier_metrics.json.test.macro_f1` are in [0, 1]
- `classifier_metrics.json.bayesian.enabled == true` for Bayesian-primary runs
- `classifier_metrics.json.bayesian.primary_score` is in [0, 1] when enabled
- `calibration_test.json.overall.ece` and `calibration_test.json.overall.brier` are in [0, 1]
- `threshold_sweep_test.json.by_label.*[].precision|recall|f1` are in [0, 1]
- `bootstrap_ci_test.json.metrics.*.interval_95.lower <= upper`
- `phase3_manifest.json` includes input config, split counts, and output file references
- `phase3_manifest.json.primary_metric_surface` is `bayesian_posterior` for default runs
- `phase3_manifest.json.execution_metadata.run_id` and `executed_at` are populated

## Verification

Run pipeline tests:

```bash
PYTHONPATH=src pytest -q tests/test_phase3_pipeline.py tests/test_phase3_analytics.py
```

Run the cross-phase regression check used in this workspace:

```bash
PYTHONPATH=src pytest -q tests/test_phase2_pipeline.py tests/test_phase3_pipeline.py
```

Regenerate dashboard figures (Figure 5-17):

```bash
PYTHONPATH=src python scripts/generate_phase3_dashboard_figures.py
```

## Notes

- Bayesian posterior scoring remains enabled by default and is tracked alongside classifier metrics.
- Calibration, threshold-sensitivity, bootstrap confidence intervals, and run-history indexing are now emitted in Phase 3 outputs.
- Dashboard generation now includes Figure 13-17 when comparable model artifacts are present.

---

## Navigation

[⬅ Back](09-phase1-phase2-progress-dashboard.md) | [Next ⮕](11-phase3-visual-dashboard.md)
