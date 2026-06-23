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
- Auxiliary labeled JSONL with the same row contract, appended to the training split only when `--auxiliary-labeled-input-path` is used
- Normalized Polisis folder (default profile: `data/raw/Polisis/normalized`) with `.jsonl` and/or `.csv` files.

Normalized Polisis row contract:

- Required: `text`
- Preferred for harmonization: `category`
- Optional: `label` (`user|system|organization`), `policy_uid`, `example_id`
- Unknown categories are skipped by the Polisis harmonization path.

## Commands

Canonical command map: [12-command-reference.md](12-command-reference.md)

Run Phase 3 baseline using OPP-115 default input set:

```bash
prert phase3
```

Run with a custom labeled JSONL dataset:

```bash
prert phase3 \
  --labeled-input-path data/processed/phase3_labeled.jsonl \
  --output-dir artifacts/phase-3
```

Prepare a conservative APP-350 auxiliary dataset from the raw zip archive:

```bash
prert app350 \
  --input-path data/raw/APP-350_v1.1.zip \
  --output-jsonl data/processed/app350_phase3_auxiliary.jsonl \
  --output-manifest data/processed/app350_phase3_auxiliary_manifest.json
```

Run with training-only auxiliary data while keeping validation and test anchored to OPP-115:

```bash
prert phase3 \
  --opp115-root data/raw/OPP-115 \
  --auxiliary-labeled-input-path data/processed/app350_phase3_auxiliary.jsonl \
  --output-dir artifacts/phase-3-opp-app350-nb
```

Run with normalized Polisis source files:

```bash
prert phase3 \
  --polisis-root data/raw/Polisis \
  --polisis-input-set normalized \
  --output-dir artifacts/phase-3
```

Run a bounded sample for quick iteration:

```bash
prert phase3 \
  --max-rows 5000 \
  --seed 42
```

Run with explicit run metadata and measurement controls:

```bash
prert phase3 \
  --run-id phase3-2026-04-07-a \
  --calibration-bins 10 \
  --bootstrap-resamples 1000 \
  --output-dir artifacts/phase-3-nb
```

Run with the upgraded TF-IDF + weighted logistic regression model:

```bash
prert phase3 \
  --model-type logreg_tfidf \
  --max-features 20000 \
  --ngram-max 2 \
  --max-iter 1000 \
  --output-dir artifacts/phase-3-logreg
```

Run with the PrivBERT-backed transformer path:

```bash
prert phase3 \
  --model-type privacybert \
  --privacybert-model-name mukund/privbert \
  --privacybert-epochs 2 \
  --privacybert-batch-size 8 \
  --privacybert-learning-rate 5e-5 \
  --privacybert-max-length 256 \
  --output-dir artifacts/phase-3-privacybert
```

Run with custom Bayesian priors (enabled by default):

```bash
prert phase3 \
  --model-type logreg_tfidf \
  --bayesian-priors-path configs/phase3_bayesian_priors.json \
  --bayesian-top-k 5 \
  --output-dir artifacts/phase-3
```

Disable Bayesian scoring output (benchmark/diagnostic only):

```bash
prert phase3 \
  --model-type logreg_tfidf \
  --disable-bayesian-scoring \
  --output-dir artifacts/phase-3-no-bayes
```

Run a comparable full-data Naive Bayes baseline for side-by-side benchmarking:

```bash
prert phase3 \
  --model-type naive_bayes \
  --output-dir artifacts/phase-3-nb
```

Run a proposal-aligned Phase 3 acceptance freeze (PrivacyBERT + Bayesian-primary checks):

```bash
prert phase3-freeze \
  --model-type privacybert \
  --strict \
  --output-dir artifacts/phase-3-freeze
```

Run acceptance freeze with Polisis advisory reporting (non-blocking for current milestone):

```bash
prert phase3-freeze \
  --polisis-root data/raw/Polisis \
  --polisis-input-set normalized \
  --output-dir artifacts/phase-3-freeze
```

Current benchmark caveat:

- The checked-in `artifacts/phase-3-*` folders are historical summary artefacts and do not yet contain the full modern `phase3_manifest.json` bundle expected by the current Phase 4 validator.
- Regenerate Phase 3 outputs with the current CLI before using `prert phase4` for a live comparison run.

## Outputs

Written to the selected `--output-dir` (for example `artifacts/phase-3/`, `artifacts/phase-3-nb/`, or `artifacts/phase-3-logreg/`):

- `training_dataset.jsonl`
- `validation_dataset.jsonl`
- `test_dataset.jsonl`
- `dataset_manifest.json`
- `dataset_manifest.json.training_sources`
- `dataset_manifest.json.primary_anchor`
- `dataset_manifest.json.auxiliary`
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
- `phase3_manifest.json.dataset_manifest.primary_anchor` remains OPP-held-out when auxiliary training data is used
- `phase3_manifest.json.dataset_manifest.auxiliary.enabled` reflects whether train-only auxiliary rows were appended
- `phase3_manifest.json.primary_metric_surface` is `bayesian_posterior` for default runs
- `phase3_manifest.json.execution_metadata.run_id` and `executed_at` are populated

## Verification

Run pipeline tests:

```bash
python -m pytest -q tests/test_phase3_pipeline.py tests/test_phase3_analytics.py
```

Run the cross-phase regression check used in this workspace:

```bash
python -m pytest -q tests/test_phase2_pipeline.py tests/test_phase3_pipeline.py
```

Regenerate dashboard figures (Figure 5-17):

```bash
PYTHONPATH=src python scripts/generate_phase3_dashboard_figures.py
```

## Notes

- Bayesian posterior scoring remains enabled by default and is tracked alongside classifier metrics.
- Calibration, threshold-sensitivity, bootstrap confidence intervals, and run-history indexing are now emitted in Phase 3 outputs.
- The current APP-350 auxiliary path is intentionally conservative: sentence-level `PERFORMED` annotations only, synthetic-source policies excluded by default, and ambiguous multi-label sentences dropped rather than forced into the task label space.
- Dashboard generation now includes Figure 13-17 when comparable model artifacts are present.

---

## Navigation

[⬅ Back](09-phase1-phase2-progress-dashboard.md) | [Next ⮕](11-phase3-visual-dashboard.md)
