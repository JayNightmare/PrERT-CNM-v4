# Recommended Tech Stack

This stack is selected to align with the proposal goals: standards mapping, NLP clause classification, probabilistic risk scoring, and validation.

## Decision Principles

- Prefer mature, well-documented libraries over experimental ones.
- Keep the stack small to reduce integration risk in a 4-month timeline.
- Choose tools that support reproducibility, traceability, and explainability.

## Core Stack (Recommended)

## Language and Runtime

- Python 3.11

Why:

- Best ecosystem fit for NLP, probabilistic modeling, and data analysis.
- Broad support for Hugging Face and scientific computing.

## NLP and Model Training

- `transformers` (Hugging Face)
- `datasets`
- `tokenizers`
- `accelerate`
- PyTorch

Why:

- Direct support for PrivacyBERT-style transformer fine-tuning workflows.
- Efficient training/inference pipelines and reproducible configs.

## Risk Scoring and Probabilistic Modeling

- PyMC (preferred) or scikit-learn probabilistic models (fallback)
- NumPy and SciPy

Why:

- PyMC supports interpretable Bayesian model construction.
- Allows uncertainty-aware risk outputs, matching proposal direction.

## Data Engineering and Analysis

- Pandas
- DuckDB (local analytics over CSV/Parquet)
- Pydantic for schema validation

Why:

- Rapid experimentation and strong data manipulation support.
- DuckDB simplifies ad hoc analytical queries without infrastructure overhead.

## API and Prototype Service Layer

- FastAPI
- Uvicorn

Why:

- Quick path to expose classification + risk scoring endpoints.
- Generates OpenAPI docs automatically for demo/review.

## Experiment Tracking and Reproducibility

- MLflow (experiment tracking)
- DVC (dataset/model versioning) or lightweight Git LFS fallback

Why:

- Prevents confusion across multiple model/metric iterations.
- Improves traceability for final validation and reporting.

## Testing and Quality

- pytest
- mypy (optional but recommended)
- ruff (linting/formatting)

Why:

- Ensures repeatable correctness checks and cleaner code quality.

## Documentation and Knowledge Management

- Markdown docs in repo
- Mermaid diagrams for architecture/flow

Why:

- Low overhead and easy collaboration/review.

## Reference Project Structure (Recommended)

```text
src/
  standards/
    mapper.py
    traceability.py
  data/
    loaders.py
    synthetic.py
    validators.py
  nlp/
    train_privacybert.py
    infer_privacybert.py
  risk/
    bayesian_model.py
    scorer.py
  api/
    app.py
  eval/
    benchmark.py
    reports.py
tests/
configs/
artifacts/
```

## Alternative Choices

- If Bayesian stack feels heavy, start with logistic/ordinal probabilistic baseline in scikit-learn, then migrate to PyMC.
- If API is out of scope for timeline, run prototype as CLI notebooks + scripted pipeline.
- If MLflow/DVC overhead is too high, enforce strict run naming and artifact folder conventions as fallback.

## Minimum Viable Stack (If Time-Constrained)

- Python 3.11
- PyTorch + transformers + datasets
- Pandas + NumPy
- PyMC (or scikit-learn fallback)
- pytest

This minimum still supports full proposal goals while reducing operational complexity.

---

## Navigation

[⬅ Back](../../README.md) | [Next ⮕](02-phase-1-standards-mapping.md)
