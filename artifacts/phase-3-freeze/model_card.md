# Phase 3 Baseline Model Card

## Model

- Type: privacybert
- Labels: user, system, organization
- Training rows: 15645
- Vocabulary size: 0

## Dataset Source

- opp115::consolidation-0.75

## Held-Out Metrics

Validation:

- Accuracy: 0.948419
- Macro precision: 0.887308
- Macro recall: 0.817498
- Macro F1: 0.849021

Test:

- Accuracy: 0.957306
- Macro precision: 0.878063
- Macro recall: 0.911619
- Macro F1: 0.894166

## Notes

- Classifier metrics are retained for diagnostics and benchmark comparison.
- Bayesian posterior risk outputs are emitted when Bayesian scoring is enabled.
