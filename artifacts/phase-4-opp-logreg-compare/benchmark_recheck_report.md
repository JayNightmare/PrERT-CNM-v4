# Modelling Environment And Benchmark Recheck

Date: 2026-05-26

## Environment Summary

- Workspace environment: `.venv/Scripts/python.exe`
- Python version: `3.14.5`
- Torch build: `2.12.0+cpu`
- CUDA available: `false`
- Project package installed in environment: `prert-cnm-v4==0.1.0`

Key package status inside `.venv`:

- `pytest==9.0.3`
- `scikit-learn==1.8.0`
- `torch==2.12.0`
- `transformers==5.9.0`
- `datasets==4.8.5`
- `python-docx==1.2.0`
- `accelerate==1.13.0`
- `PyYAML==6.0.3`

The bare system interpreter at `C:/Users/KU79173/AppData/Local/Python/pythoncore-3.14-64/python.exe` remained unsuitable for modelling and validation because it did not have `pytest`, `scikit-learn`, `torch`, `transformers`, `datasets`, or `python-docx` installed.

## Commands Run

```bash
.venv/Scripts/python.exe -m pytest -q tests/test_cli_meta.py tests/test_phase3_pipeline.py tests/test_phase4_validation.py
.venv/Scripts/python.exe -m pytest -q tests/test_phase3_app350.py tests/test_import_boundaries.py tests/test_phase3_pipeline.py tests/test_cli_meta.py
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -m prert.cli.main phase3 --model-type logreg_tfidf --opp115-root data/raw/OPP-115 --seed 42 --random-state 42 --max-features 20000 --ngram-max 2 --max-iter 1000 --output-dir artifacts/phase-3-opp-logreg --run-id opp-logreg-20260526
.venv/Scripts/python.exe -m prert.cli.main phase4 --baseline-dir deployment/demo-assets/phase-3-freeze --comparison-dirs artifacts/phase-3-opp-logreg --output-dir artifacts/phase-4-opp-logreg-compare
```

PrivacyBERT capability was also checked directly:

```bash
.venv/Scripts/python.exe -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('mukund/privbert'); AutoModelForSequenceClassification.from_pretrained('mukund/privbert', num_labels=3); print('privbert-load-ok')"
```

That load succeeded, which means the current blocker is no longer missing packages or model access. The practical limitation is that the environment is CPU-only, so a full comparable PrivacyBERT rerun is likely to be substantially slower than the TF-IDF logistic-regression path.

## Artefact Locations

- Regenerated stronger benchmark bundle: `artifacts/phase-3-opp-logreg/`
- Validator comparison report: `artifacts/phase-4-opp-logreg-compare/phase4_validation_report.md`
- OPP-only Naive Bayes rerun: `artifacts/phase-3-opp-nb/`
- OPP plus APP-350 auxiliary rerun: `artifacts/phase-3-opp-app350-nb/`
- APP-350 auxiliary manifest: `data/processed/app350_phase3_auxiliary_manifest.json`
- Validator-ready freeze baseline used for comparison: `deployment/demo-assets/phase-3-freeze/`

## Benchmark Outcome

Regenerated `logreg_tfidf` metrics on OPP-115:

- Validation macro F1: `0.807784`
- Test macro F1: `0.737192`
- Validation accuracy: `0.900794`
- Test accuracy: `0.874191`
- Bayesian primary score: `0.817274`

Comparison against the validator-ready freeze baseline:

- Freeze test macro F1: `0.894166`
- Freeze test accuracy: `0.957306`
- Delta test macro F1: `-0.156974`

This regenerated stronger benchmark is materially better than the fresh Naive Bayes rerun, but it is still far below the historical freeze summary.

It is also below the older checked-in historical `phase-3-logreg` summary, which reported test macro F1 `0.779024` and test accuracy `0.892606`.

## Auxiliary-Data Decision

APP-350 should be treated as provisionally not useful for this label space.

Evidence:

- The conservative APP-350 normalisation retained `3384` rows.
- Retained label mix was `organization=1544`, `system=1840`, `user=0`.
- The controlled OPP plus APP-350 run degraded every key test metric relative to the OPP-only rerun.

If auxiliary-data work continues, the next candidate should be Polisis rather than APP-350, because the existing repo harmonisation already maps Polisis categories across all three task labels, including the missing `user` surface. If a credible Polisis-derived corpus cannot be obtained with clean provenance and licensing, auxiliary-data expansion should be paused rather than extended with weak-fitting sources.
