# Phase 3 Baseline Model Card

## Model

- Type: multinomial_naive_bayes
- Backbone checkpoint: n/a
- Labels: user, system, organization
- Training rows: 15484
- Vocabulary size: 5799

## Dataset Source

- opp115::consolidation-0.75

## Held-Out Metrics

Validation:

- Accuracy: 0.841837
- Macro precision: 0.650851
- Macro recall: 0.824158
- Macro F1: 0.703748

Test:

- Accuracy: 0.820793
- Macro precision: 0.595123
- Macro recall: 0.764596
- Macro F1: 0.649229

## Notes

- Classifier metrics are retained for diagnostics and benchmark comparison.
- Bayesian posterior risk outputs are emitted when Bayesian scoring is enabled.
