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

- Accuracy: 0.971088
- Macro precision: 0.926247
- Macro recall: 0.94198
- Macro F1: 0.933547

Test:

- Accuracy: 0.956715
- Macro precision: 0.866726
- Macro recall: 0.891755
- Macro F1: 0.878496

## Notes

- Classifier metrics are retained for diagnostics and benchmark comparison.
- Bayesian posterior risk outputs are emitted when Bayesian scoring is enabled.
