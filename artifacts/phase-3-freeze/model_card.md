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
- Macro precision: 0.946422
- Macro recall: 0.924544
- Macro F1: 0.935227

Test:

- Accuracy: 0.953883
- Macro precision: 0.864226
- Macro recall: 0.873372
- Macro F1: 0.868631

## Notes

- Classifier metrics are retained for diagnostics and benchmark comparison.
- Bayesian posterior risk outputs are emitted when Bayesian scoring is enabled.
