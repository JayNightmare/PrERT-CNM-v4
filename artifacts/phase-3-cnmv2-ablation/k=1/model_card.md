# Phase 3 Baseline Model Card

## Model

- Type: cnmv2_privacybert
- Backbone checkpoint: mukund/privbert
- Labels: user, system, organization
- Training rows: 15484
- Vocabulary size: 0

## Dataset Source

- opp115::consolidation-0.75

## Held-Out Metrics

Validation:

- Accuracy: 0.968821
- Macro precision: 0.923602
- Macro recall: 0.941059
- Macro F1: 0.931969

Test:

- Accuracy: 0.957524
- Macro precision: 0.868696
- Macro recall: 0.892072
- Macro F1: 0.879488

## Notes

- Classifier metrics are retained for diagnostics and benchmark comparison.
- Bayesian posterior risk outputs are emitted when Bayesian scoring is enabled.
