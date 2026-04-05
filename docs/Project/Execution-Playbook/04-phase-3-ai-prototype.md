# Phase 3 Detailed Plan: AI Prototype (PrivacyBERT + Bayesian Scoring)

Timeline target: Month 3

## Status Update

Phase 3 implementation now includes a deterministic multinomial Naive Bayes baseline and an upgraded TF-IDF + weighted logistic regression path (see `10-phase3-implementation-runbook.md` and `11-phase3-visual-dashboard.md`).

This document remains the target plan for the next increment: PrivacyBERT-oriented modeling, Bayesian risk scoring, explainability hardening, and service-level integration.

## Goal

Build an integrated prototype that classifies privacy clauses and converts them into quantitative risk scores.

## Scope

In scope:

- Privacy clause classification fine-tuning/inference workflow.
- Bayesian/probabilistic risk scoring model.
- Integration pipeline connecting NLP outputs to risk computation.

Out of scope:

- Full production deployment hardening.
- Large-scale MLOps automation.

## Inputs

- OPP-115 dataset.
- Polisis datasets.
- Phase 2 metric definitions and data schemas.

## Steps to Complete

1. Prepare training and evaluation datasets

- Consolidate labels and map them to Phase 1/2 indicator taxonomy.
- Create train/validation/test splits with leakage checks.

2. Fine-tune clause classifier

- Train PrivacyBERT-based classifier with documented hyperparameters.
- Track model runs and store checkpoints.
- Evaluate macro F1, per-class precision/recall, and calibration quality.

3. Build probabilistic risk engine

- Define Bayesian model structure:
     - priors from standards expectations,
     - likelihood from observed indicator evidence,
     - posterior risk score output.
- Add uncertainty outputs (credible intervals).

4. Integrate NLP + risk modules

- Build adapter translating clause probabilities to indicator evidence.
- Feed evidence into probabilistic scoring.
- Output user/system/org risk scores with explanation payload.

5. Implement explainability layer

- Provide top contributing clauses/indicators per score.
- Surface uncertainty/confidence markers.

6. Build prototype interface

- Preferred: FastAPI endpoints for batch and single-item scoring.
- Fallback: notebook + CLI pipeline for demo.

7. Freeze Phase 3 baseline

- Package model card, scoring spec, and integration test evidence.

## Deliverables

- Trained clause classifier checkpoint(s).
- Bayesian scoring module.
- Integrated prototype pipeline.
- Model card and scoring documentation.

## Recommended Acceptance Checks

- End-to-end run completes from policy text to final risk score.
- Classification metrics meet predefined minimum thresholds.
- Risk scores include uncertainty and explanation fields.
- Repeated runs with fixed seeds are reproducible.

## Risks and Mitigations

- Risk: Label mismatch across OPP-115 and Polisis.
     - Mitigation: Create explicit label harmonization table and unit tests.
- Risk: Bayesian model instability.
     - Mitigation: Start with simpler priors and gradual complexity.
- Risk: Weak explainability.
     - Mitigation: Enforce mandatory explanation outputs in API schema.

## Recommended Week-by-Week Breakdown

Week 1:

- Dataset prep, harmonization, and baseline model run.

Week 2:

- Fine-tuning and classifier evaluation refinement.

Week 3:

- Bayesian scoring implementation and calibration.

Week 4:

- Integration, explainability pass, and Phase 3 freeze.

---

## Navigation

[⬅ Back](03-phase-2-metrics-and-data.md) | [Next ⮕](05-phase-4-validation-and-reporting.md)
