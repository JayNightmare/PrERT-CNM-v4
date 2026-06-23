# Python 3.12 Modelling Environment And Benchmark Recheck

Date: 2026-05-26

## Environment Summary

- Workspace environment: `.venv/Scripts/python.exe`
- Python version: `3.12.0`
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

Runtime checks completed successfully in the Python 3.12 environment.

- Focused validation slice: `17 passed`
- `pip check`: no broken requirements found

One non-blocking runtime quirk remains: `multiprocess.resource_tracker` raises an ignored shutdown exception on interpreter exit under this environment. It did not prevent test execution or artefact generation, but it should not be mistaken for a benchmark failure.

## Commands Run

```bash
.venv/Scripts/python.exe --version
.venv/Scripts/python.exe -c "import pytest, sklearn, torch, transformers, datasets, docx, accelerate, yaml, json, sys; print(json.dumps({'python': sys.version, 'pytest': pytest.__version__, 'scikit_learn': sklearn.__version__, 'torch': torch.__version__, 'transformers': transformers.__version__, 'datasets': datasets.__version__, 'python_docx': docx.__version__, 'accelerate': accelerate.__version__, 'pyyaml': yaml.__version__, 'cuda_available': bool(torch.cuda.is_available()), 'cuda_device_count': int(torch.cuda.device_count()), 'cuda_version': torch.version.cuda}, indent=2))"
.venv/Scripts/python.exe -m pytest -q tests/test_phase3_pipeline.py tests/test_phase3_acceptance.py tests/test_phase4_validation.py
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -m prert.cli.main phase3 --model-type logreg_tfidf --opp115-root data/raw/OPP-115 --seed 42 --random-state 42 --max-features 20000 --ngram-max 2 --max-iter 1000 --output-dir artifacts/phase-3-opp-logreg-312 --run-id opp-logreg-py312-20260526
.venv/Scripts/python.exe -m prert.cli.main phase4 --baseline-dir deployment/demo-assets/phase-3-freeze --comparison-dirs artifacts/phase-3-opp-logreg-312 --output-dir artifacts/phase-4-opp-logreg-312-compare
HF_HUB_DISABLE_SYMLINKS_WARNING=1 .venv/Scripts/python.exe -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('mukund/privbert'); AutoModelForSequenceClassification.from_pretrained('mukund/privbert', num_labels=3); print('privbert-load-ok')"
HF_HUB_DISABLE_SYMLINKS_WARNING=1 .venv/Scripts/python.exe -m prert.cli.main phase3 --model-type privacybert --privacybert-model-name mukund/privbert --opp115-root data/raw/OPP-115 --max-rows 500 --privacybert-epochs 1 --privacybert-batch-size 4 --privacybert-max-length 128 --seed 42 --random-state 42 --output-dir artifacts/phase-3-opp-privacybert-smoke-312 --run-id opp-privacybert-smoke-py312-20260526
```

## Artefact Locations

- Regenerated Python-3.12 logreg bundle: `artifacts/phase-3-opp-logreg-312/`
- Python-3.12 validator comparison report: `artifacts/phase-4-opp-logreg-312-compare/phase4_validation_report.md`
- PrivacyBERT smoke bundle: `artifacts/phase-3-opp-privacybert-smoke-312/`
- Validator-ready freeze baseline used for comparison: `deployment/demo-assets/phase-3-freeze/`
- APP-350 auxiliary manifest retained for decision evidence: `data/processed/app350_phase3_auxiliary_manifest.json`

## Benchmark Outcome

Regenerated `logreg_tfidf` metrics on OPP-115 under Python 3.12:

- Validation macro F1: `0.807784`
- Test macro F1: `0.737192`
- Validation accuracy: `0.900794`
- Test accuracy: `0.874191`
- Bayesian primary score: `0.817274`

Comparison against the validator-ready freeze baseline:

- Freeze test macro F1: `0.894166`
- Freeze test accuracy: `0.957306`
- Delta test macro F1: `-0.156974`

This Python 3.12 rerun reproduces the same weaker modern logreg result already seen in the earlier Python 3.14-based virtual environment. That means the benchmark gap is not explained by the interpreter version.

It also remains below the checked-in historical `phase-3-logreg` summary-only artefact, which reported test macro F1 `0.779024` and test accuracy `0.892606`.

## PrivacyBERT Practicality

The Python 3.12 environment can load `mukund/privbert` and execute the full transformer training path.

Evidence:

- Direct checkpoint load succeeded (`privbert-load-ok`)
- A bounded smoke run completed successfully on CPU
- Smoke-run training time was approximately `63.78` seconds for `209` training rows and one epoch

Smoke-run metrics are not publication-grade and are not comparable to the freeze baseline because the run was intentionally bounded:

- Validation macro F1: `0.321569`
- Test macro F1: `0.301552`
- Test accuracy: `0.825911`

Conclusion: PrivacyBERT is runnable in the current environment, but a full comparable CPU-only rerun remains materially more expensive than the TF-IDF logistic-regression path. It should be treated as feasible but not yet practical enough to replace the logreg rerun as the main evidence path without additional runtime budget.

## Auxiliary-Data Decision

APP-350 should remain paused for this label space.

Evidence:

- The conservative APP-350 normalisation retained `3384` rows
- Retained label mix was `organization=1544`, `system=1840`, `user=0`
- The earlier controlled OPP plus APP-350 run degraded key test metrics relative to the OPP-only rerun

Polisis remains the only plausible next auxiliary direction already supported by the codebase, but no local `data/raw/Polisis` corpus is currently present.

## Bottom Line

The Python 3.12 environment is now properly runnable and reproducible.

The strongest practical fresh benchmark under that environment remains `logreg_tfidf`, and it is still materially below the validator-ready freeze baseline. Auxiliary-data work should not continue on APP-350. PrivacyBERT now looks like an environment-and-runtime question rather than a packaging blocker, but a full comparable CPU-only run still needs explicit time budget before it should be treated as the main benchmark path.
