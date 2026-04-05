# Phase 3 Implementation Runbook

This runbook executes the Phase 3 classifier-and-risk pipeline in an isolated output folder.

## Scope

- Build a clause-level labeled dataset from OPP-115 annotations or a provided labeled JSONL.
- Create deterministic train, validation, and test splits with policy-level leakage protection.
- Train a text classifier backend for user/system/organization labels (`naive_bayes`, `logreg_tfidf`, or `privacybert`).
- Evaluate held-out classifier metrics and emit Bayesian posterior risk outputs.
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

Run with the upgraded TF-IDF + weighted logistic regression model:

```bash
PYTHONPATH=src python3 scripts/run_phase3_baseline.py \
  --model-type logreg_tfidf \
  --max-features 20000 \
  --ngram-max 2 \
  --max-iter 1000 \
  --output-dir artifacts/phase-3-logreg
```

Run with the PrivacyBERT backend scaffold:

```bash
PYTHONPATH=src python3 scripts/run_phase3_baseline.py \
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
PYTHONPATH=src python3 scripts/run_phase3_baseline.py \
  --model-type logreg_tfidf \
  --bayesian-priors-path configs/phase3_bayesian_priors.json \
  --bayesian-top-k 5 \
  --output-dir artifacts/phase-3
```

Disable Bayesian scoring output (benchmark/diagnostic only):

```bash
PYTHONPATH=src python3 scripts/run_phase3_baseline.py \
  --model-type logreg_tfidf \
  --disable-bayesian-scoring \
  --output-dir artifacts/phase-3-no-bayes
```

Run a comparable full-data Naive Bayes baseline for side-by-side benchmarking:

```bash
PYTHONPATH=src python3 scripts/run_phase3_baseline.py \
  --model-type naive_bayes \
  --output-dir artifacts/phase-3-nb
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
- `bayesian_risk_validation.json` (when Bayesian scoring enabled)
- `bayesian_risk_test.json` (when Bayesian scoring enabled)
- `model_card.md`
- `scoring_spec.md`
- `prototype_demo.md`
- `phase3_manifest.json`

## Quality Checks

- `dataset_manifest.json.policy_overlap.* == 0`
- `classifier_metrics.json.validation.macro_f1` and `classifier_metrics.json.test.macro_f1` are in [0, 1]
- `classifier_metrics.json.bayesian.enabled == true` for Bayesian-primary runs
- `classifier_metrics.json.bayesian.primary_score` is in [0, 1] when enabled
- `phase3_manifest.json` includes input config, split counts, and output file references
- `phase3_manifest.json.primary_metric_surface` is `bayesian_posterior` for default runs

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

- Bayesian posterior scoring is now integrated as a first implementation slice and is enabled by default.
- Classifier held-out metrics remain available for benchmark comparison and regression checks.
- Full PrivacyBERT benchmark hardening, Bayesian calibration governance, and API endpoints remain next-increment work.

---

## Navigation

[⬅ Back](09-phase1-phase2-progress-dashboard.md) | [Next ⮕](11-phase3-visual-dashboard.md)
