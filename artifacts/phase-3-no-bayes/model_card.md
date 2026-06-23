# Phase 3 Baseline Model Card

## Model

- Type: logreg_tfidf
- Backbone checkpoint: n/a
- Labels: user, system, organization
- Training rows: 15484
- Vocabulary size: 20000

## Dataset Source

- opp115::consolidation-0.75

## Held-Out Metrics

Validation:

- Accuracy: 0.900794
- Macro precision: 0.765363
- Macro recall: 0.864938
- Macro F1: 0.807784

Test:

- Accuracy: 0.874191
- Macro precision: 0.686578
- Macro recall: 0.814132
- Macro F1: 0.737192

## Notes

- Classifier metrics are retained for diagnostics and benchmark comparison.
- Bayesian posterior risk outputs are emitted when Bayesian scoring is enabled.
