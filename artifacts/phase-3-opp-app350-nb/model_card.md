# Phase 3 Baseline Model Card

## Model

- Type: multinomial_naive_bayes
- Backbone checkpoint: n/a
- Labels: user, system, organization
- Training rows: 18868
- Vocabulary size: 6719

## Dataset Source

- opp115::consolidation-0.75 + auxiliary::app350_phase3_auxiliary.jsonl

## Held-Out Metrics

Validation:

- Accuracy: 0.731293
- Macro precision: 0.562943
- Macro recall: 0.731961
- Macro F1: 0.567686

Test:

- Accuracy: 0.754045
- Macro precision: 0.532825
- Macro recall: 0.713974
- Macro F1: 0.562583

## Notes

- Classifier metrics are retained for diagnostics and benchmark comparison.
- Bayesian posterior risk outputs are emitted when Bayesian scoring is enabled.
