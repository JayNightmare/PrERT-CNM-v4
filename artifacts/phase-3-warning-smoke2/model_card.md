# Phase 3 Baseline Model Card

## Model

- Type: privacybert
- Backbone checkpoint: mukund/privbert
- Labels: user, system, organization
- Training rows: 209
- Vocabulary size: 0

## Dataset Source

- opp115::consolidation-0.75

## Held-Out Metrics

Validation:

- Accuracy: 0.931818
- Macro precision: 0.310606
- Macro recall: 0.333333
- Macro F1: 0.321569

Test:

- Accuracy: 0.825911
- Macro precision: 0.275304
- Macro recall: 0.333333
- Macro F1: 0.301552

## Notes

- Classifier metrics are retained for diagnostics and benchmark comparison.
- Bayesian posterior risk outputs are emitted when Bayesian scoring is enabled.
