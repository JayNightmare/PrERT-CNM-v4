# Phase 3 Baseline Scoring Specification

## Output Schema

- actual_label: ground-truth label from held-out set.
- predicted_label: model prediction in {user, system, organization}.
- confidence: predicted class probability.

## Metrics

- accuracy
- macro_precision
- macro_recall
- macro_f1
- per-class precision/recall/f1/support

## Bayesian Risk Outputs

- bayesian_risk_validation.json
- bayesian_risk_test.json

Each Bayesian output includes:

- per-level posterior alpha/beta
- posterior mean risk and interval bounds
- top contributing clauses for each level


## Constraints

- All metric values are in [0, 1] except support counts.
- Dataset splits are deterministic for a fixed seed.
