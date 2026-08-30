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
- Macro precision: 0.926684
- Macro recall: 0.939666
- Macro F1: 0.933083

Test:

- Accuracy: 0.96157
- Macro precision: 0.885981
- Macro recall: 0.879355
- Macro F1: 0.882618

## Notes

- Classifier metrics are retained for diagnostics and benchmark comparison.
- Bayesian posterior risk outputs are emitted when Bayesian scoring is enabled.
