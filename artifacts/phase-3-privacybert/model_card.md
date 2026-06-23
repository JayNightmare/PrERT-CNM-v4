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

- Accuracy: 0.968254
- Macro precision: 0.933376
- Macro recall: 0.918171
- Macro F1: 0.92562

Test:

- Accuracy: 0.956311
- Macro precision: 0.880388
- Macro recall: 0.858978
- Macro F1: 0.869379

## Notes

- Classifier metrics are retained for diagnostics and benchmark comparison.
- Bayesian posterior risk outputs are emitted when Bayesian scoring is enabled.
