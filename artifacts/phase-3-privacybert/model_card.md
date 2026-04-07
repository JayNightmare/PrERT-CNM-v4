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

- Accuracy: 0.951192
- Macro precision: 0.911817
- Macro recall: 0.840546
- Macro F1: 0.872441

Test:

- Accuracy: 0.955986
- Macro precision: 0.864696
- Macro recall: 0.923179
- Macro F1: 0.891999

## Notes

- Classifier metrics are retained for diagnostics and benchmark comparison.
- Bayesian posterior risk outputs are emitted when Bayesian scoring is enabled.
