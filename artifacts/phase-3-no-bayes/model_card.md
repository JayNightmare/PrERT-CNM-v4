# Phase 3 Baseline Model Card

## Model

- Type: logreg_tfidf
- Labels: user, system, organization
- Training rows: 15645
- Vocabulary size: 20000

## Dataset Source

- opp115::consolidation-0.75

## Held-Out Metrics

Validation:

- Accuracy: 0.889628
- Macro precision: 0.723574
- Macro recall: 0.806726
- Macro F1: 0.75797

Test:

- Accuracy: 0.892606
- Macro precision: 0.715532
- Macro recall: 0.88346
- Macro F1: 0.779024

## Notes

- Classifier metrics are retained for diagnostics and benchmark comparison.
- Bayesian posterior risk outputs are emitted when Bayesian scoring is enabled.
