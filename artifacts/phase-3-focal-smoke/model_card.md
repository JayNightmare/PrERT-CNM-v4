# Phase 3 Baseline Model Card

## Model

- Type: privacybert
- Backbone checkpoint: mukund/privbert
- Labels: user, system, organization
- Training rows: 15484
- Vocabulary size: 0

## Dataset Source

- opp115::consolidation-0.75

## Held-Out Metrics

Validation:

- Accuracy: 0.972789
- Macro precision: 0.964103
- Macro recall: 0.906417
- Macro F1: 0.932979

Test:

- Accuracy: 0.963188
- Macro precision: 0.919562
- Macro recall: 0.85856
- Macro F1: 0.886777

## Notes

- Classifier metrics are retained for diagnostics and benchmark comparison.
- Bayesian posterior risk outputs are emitted when Bayesian scoring is enabled.
