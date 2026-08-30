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

- Accuracy: 0.966553
- Macro precision: 0.910034
- Macro recall: 0.932232
- Macro F1: 0.919952

Test:

- Accuracy: 0.947411
- Macro precision: 0.829861
- Macro recall: 0.875618
- Macro F1: 0.846736

## Notes

- Classifier metrics are retained for diagnostics and benchmark comparison.
- Bayesian posterior risk outputs are emitted when Bayesian scoring is enabled.
