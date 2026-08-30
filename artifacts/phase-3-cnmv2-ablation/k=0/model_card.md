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

- Accuracy: 0.820862
- Macro precision: 0.273621
- Macro recall: 0.333333
- Macro F1: 0.30054

Test:

- Accuracy: 0.848301
- Macro precision: 0.282767
- Macro recall: 0.333333
- Macro F1: 0.305975

## Notes

- Classifier metrics are retained for diagnostics and benchmark comparison.
- Bayesian posterior risk outputs are emitted when Bayesian scoring is enabled.
