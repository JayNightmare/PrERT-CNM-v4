# Phase 3 Baseline Model Card

## Model

- Type: privacybert
- Backbone checkpoint: mukund/privbert
- Labels: user, system, organization
- Training rows: 138
- Vocabulary size: 0

## Dataset Source

- opp115::consolidation-0.75

## Held-Out Metrics

Validation:

- Accuracy: 0.0
- Macro precision: 0.0
- Macro recall: 0.0
- Macro F1: 0.0

Test:

- Accuracy: 0.839506
- Macro precision: 0.279835
- Macro recall: 0.333333
- Macro F1: 0.304251

## Notes

- Classifier metrics are retained for diagnostics and benchmark comparison.
- Bayesian posterior risk outputs are emitted when Bayesian scoring is enabled.
