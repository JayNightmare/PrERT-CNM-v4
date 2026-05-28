# Reproducibility Guide

This document explains how to reproduce the committed PrERT results using the `prert` CLI rather than the legacy wrapper scripts.

## Scope

This guide covers the workflow used to produce or validate the main committed result directories:

- `artifacts/phase-2/`
- `artifacts/phase-3-nb/`
- `artifacts/phase-3-logreg/`
- `artifacts/phase-3-no-bayes/`
- `artifacts/phase-3-privacybert/`
- `artifacts/phase-3-freeze/`
- `artifacts/phase-4/`

For the canonical command surface, see [Project/Execution-Playbook/12-command-reference.md](./Project/Execution-Playbook/12-command-reference.md).

## Prerequisites

Before running the full rebuild path, make sure the following are in place:

- Python 3.11 to 3.14
- The package installed from the repository root
- A `.env` file created from `.env.example` if you want to run the full end-to-end setup checks and Chroma migration steps
- Regulation source DOCX files under `docs/Standards/Regulations/`
- The OPP-115 corpus extracted under `data/raw/OPP-115`, with `documentation/`, `annotations/`, and `consolidation/` present under that root
- The APP-350 archive downloaded to `data/raw/APP-350_v1.1.zip` if you want to reproduce the auxiliary-data pilot

Install the project:

```bash
python -m pip install -e .
```

Install development dependencies if you also want to run tests locally:

```bash
python -m pip install -e .[dev]
```

## Current Committed-State Caveat

The checked-in `artifacts/phase-3-*` folders currently contain summary markdown files, but they do not include the full modern Phase 3 manifest bundle expected by the current `prert phase4` validator.

In particular, the committed benchmark folders do not currently include `phase3_manifest.json`, so a live `prert phase4` revalidation against those checked-in folders will fail until the Phase 3 artefacts are regenerated with the current pipeline.

If you only want to inspect the historical benchmark evidence that is already committed, review these files directly:

- `artifacts/phase-3-freeze/model_card.md`
- `artifacts/phase-3-freeze/phase3_acceptance_report.md`
- `artifacts/phase-4/phase4_validation_report.md`

For the fastest validator-ready benchmark path that already works in the current repo state, use:

- `deployment/demo-assets/phase-3-freeze/`
- `deployment/demo-assets/phase-3-nb/`

## Fastest Live Validation Path

The fastest honest way to exercise the current validator is:

1. Use `deployment/demo-assets/phase-3-freeze` as the committed validator-ready baseline bundle, or regenerate a modern Phase 3 bundle yourself.
2. Run `prert phase4` against that baseline and any regenerated comparison folders.

Example:

```bash
prert phase4 \
  --baseline-dir deployment/demo-assets/phase-3-freeze \
  --comparison-dirs deployment/demo-assets/phase-3-nb \
  --output-dir artifacts/phase-4-demo-check
```

## Full Rebuild Path

Use this path if you want to rebuild the main results from the raw inputs rather than only validating the committed artefacts.

### 1. Configure the environment

Create `.env` from `.env.example` and fill in the Chroma values if you plan to run `prert doctor` and the Phase 1 migration step.

### 2. Run the preflight check

```bash
prert doctor
```

Note: `prert doctor` checks the full end-to-end workspace, including the Chroma environment variables and the regulation source files. If you are only rebuilding the modelling outputs, the Chroma migration itself is optional, but the doctor command is still the cleanest workspace check.

### 3. Extract Phase 1 controls and chunks

```bash
prert extract \
  --chunk \
  --output-dir artifacts/phase-1
```

Expected Phase 1 outputs include:

- `artifacts/phase-1/controls_all.jsonl`
- `artifacts/phase-1/chunks_all.jsonl`

### 4. Optionally run Chroma migration

This step is part of the full pipeline, but it is not required to rebuild the Phase 2 to Phase 4 modelling results.

```bash
prert migrate \
  --input-dir artifacts/phase-1
```

### 5. Rebuild the Phase 2 baseline artefacts

```bash
prert phase2
```

Expected Phase 2 outputs include:

- `artifacts/phase-2/metric_specs.jsonl`
- `artifacts/phase-2/synthetic_events.jsonl`
- `artifacts/phase-2/baseline_scores.jsonl`
- `artifacts/phase-2/phase2_manifest.json`

If you also want the flattened OPP-115 mapping artefacts used by the Phase 2 public-data path, run:

```bash
prert opp115
```

### 6. Rebuild the Phase 3 comparison runs

Keep `--seed` and `--random-state` aligned so the classifier training and split logic stay reproducible.

#### Naive Bayes baseline

```bash
prert phase3 \
  --model-type naive_bayes \
  --opp115-root data/raw/OPP-115 \
  --seed 42 \
  --random-state 42 \
  --output-dir artifacts/phase-3-nb
```

#### Logistic regression baseline

```bash
prert phase3 \
  --model-type logreg_tfidf \
  --opp115-root data/raw/OPP-115 \
  --max-features 20000 \
  --ngram-max 2 \
  --max-iter 1000 \
  --seed 42 \
  --random-state 42 \
  --output-dir artifacts/phase-3-logreg
```

#### Logistic regression without Bayesian scoring

```bash
prert phase3 \
  --model-type logreg_tfidf \
  --opp115-root data/raw/OPP-115 \
  --max-features 20000 \
  --ngram-max 2 \
  --max-iter 1000 \
  --disable-bayesian-scoring \
  --seed 42 \
  --random-state 42 \
  --output-dir artifacts/phase-3-no-bayes
```

#### PrivBERT comparison run

Before running the transformer path, set `HF_TOKEN` in the repository-root `.env`. The Phase 3 CLI now auto-loads that file before contacting Hugging Face.

The `mukund/privbert` checkpoint is loaded into a sequence-classification head, so the one-time Transformers report about `lm_head.*` being unexpected and `classifier.*` weights being newly initialized is expected transfer-learning behavior.

```bash
prert phase3 \
  --model-type privacybert \
  --opp115-root data/raw/OPP-115 \
  --privacybert-model-name mukund/privbert \
  --privacybert-epochs 2 \
  --privacybert-batch-size 8 \
  --privacybert-learning-rate 5e-5 \
  --privacybert-max-length 256 \
  --seed 42 \
  --random-state 42 \
  --output-dir artifacts/phase-3-privacybert
```

#### Acceptance-freeze run used as the main Phase 4 baseline

```bash
prert phase3-freeze \
  --model-type privacybert \
  --opp115-root data/raw/OPP-115 \
  --privacybert-model-name mukund/privbert \
  --seed 42 \
  --random-state 42 \
  --strict \
  --output-dir artifacts/phase-3-freeze
```

The most important Phase 3 artefacts to check after each run are:

- `classifier_metrics.json`
- `model_card.md`
- `phase3_manifest.json`
- `bayesian_risk_test.json` for Bayesian-enabled runs
- `phase3_acceptance_report.md` for the freeze run

### 6a. Build the APP-350 auxiliary dataset for controlled augmentation

The current conservative APP-350 path uses sentence-level `PERFORMED` annotations only, skips policies marked as synthetic by default, and drops ambiguous multi-label sentences rather than forcing them into the `user/system/organization` task.

```bash
prert app350 \
  --input-path data/raw/APP-350_v1.1.zip \
  --output-jsonl data/processed/app350_phase3_auxiliary.jsonl \
  --output-manifest data/processed/app350_phase3_auxiliary_manifest.json
```

The manifest records retained-label counts, retained practices, dropped-sentence reasons, and whether synthetic APP-350 policies were excluded.

### 6b. Run the controlled OPP-115 plus APP-350 auxiliary pilot

This keeps OPP-115 as the held-out evaluation anchor and appends the APP-350 JSONL to the training split only.

```bash
prert phase3 \
  --model-type naive_bayes \
  --opp115-root data/raw/OPP-115 \
  --auxiliary-labeled-input-path data/processed/app350_phase3_auxiliary.jsonl \
  --seed 42 \
  --random-state 42 \
  --output-dir artifacts/phase-3-opp-app350-nb
```

For the 2026-05-25 controlled pilot captured in this workspace, the auxiliary APP-350 manifest retained 3384 rows after excluding 142 synthetic-source policies and dropping ambiguous or unmapped sentence cases.

### 7. Rebuild the Phase 4 validation report

Once the Phase 3 artefacts exist, regenerate the cross-run comparison report:

```bash
prert phase4 \
  --baseline-dir artifacts/phase-3-freeze \
  --comparison-dirs artifacts/phase-3-nb artifacts/phase-3-logreg artifacts/phase-3-no-bayes artifacts/phase-3-privacybert \
  --output-dir artifacts/phase-4 \
  --require-bayesian \
  --strict
```

For the current validator-ready benchmark path plus the APP-350 pilot in this repo state, use:

```bash
prert phase4 \
  --baseline-dir deployment/demo-assets/phase-3-freeze \
  --comparison-dirs artifacts/phase-3-opp-nb artifacts/phase-3-opp-app350-nb \
  --output-dir artifacts/phase-4-opp-app350-compare
```

The main outputs are:

- `artifacts/phase-4/phase4_validation_report.json`
- `artifacts/phase-4/phase4_validation_report.md`
- `artifacts/phase-4/phase4_leaderboard.jsonl`

## What to Check After Rebuilding

A rebuild should be treated as successful when the following artefacts exist and remain internally coherent:

- `artifacts/phase-2/phase2_manifest.json`
- `artifacts/phase-3-freeze/phase3_acceptance_report.md`
- `artifacts/phase-3-freeze/model_card.md`
- `artifacts/phase-4/phase4_validation_report.md`
- `artifacts/phase-4/phase4_leaderboard.jsonl`

For the Phase 3 runs in particular, verify that:

- `phase3_manifest.json` exists in each output directory
- policy overlap remains zero across train, validation, and test splits
- the configured checkpoint recorded in the manifest matches the intended run configuration
- Bayesian outputs are present when Bayesian scoring was enabled

## Notes

- This guide intentionally uses the `prert` command surface so the reproduction path stays close to the supported CLI entry points.
- `prert phase4` validates and compares Phase 3 artefacts; it does not retrain models.
- `prert app350` is the current CLI path for preparing the conservative APP-350 auxiliary corpus used by the train-only augmentation experiment.
- The current default Phase 3 privacy-model checkpoint is `mukund/privbert`, but it is still passed explicitly above so the documented workflow remains stable even if defaults change later.
- Chroma migration is part of the end-to-end pipeline, but it is not required for regenerating the committed Phase 2 to Phase 4 benchmark artefacts.
- As of 2026-05-25, a live recheck confirmed that the current validator expects `phase3_manifest.json` in each Phase 3 artefact directory, so the older summary-only committed benchmark folders should be treated as historical evidence rather than directly re-runnable validator inputs.

> [!NOTE]
> Use this command for full training and evaluation runs. It takes roughly 3 hours (depending on hardware).
>
> ```bash
> mkdir -p artifacts/phase-3-opp-privacybert-312 && HF_HUB_DISABLE_SYMLINKS_WARNING=1 .venv/Scripts/prert.exe phase3 --model-type privacybert --privacybert-model-name mukund/privbert --opp115-root data/raw/OPP-115 --privacybert-epochs 4 --privacybert-batch-size 8 --privacybert-max-length 256 --seed 42 --random-state 42 --output-dir artifacts/phase-3-opp-privacybert-312 --run-id opp-privacybert-py312-overnight-20260526 2>&1 | tee artifacts/phase-3-opp-privacybert-312/run.log
> ```
