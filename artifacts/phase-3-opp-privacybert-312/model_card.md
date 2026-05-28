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

- Accuracy: 0.971655
- Macro precision: 0.954342
- Macro recall: 0.916177
- Macro F1: 0.934455

Test:

- Accuracy: 0.958333
- Macro precision: 0.885005
- Macro recall: 0.865288
- Macro F1: 0.874909

## Notes

- Classifier metrics are retained for diagnostics and benchmark comparison.
- Bayesian posterior risk outputs are emitted when Bayesian scoring is enabled.
