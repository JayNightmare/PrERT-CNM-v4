# Phase 3 Baseline Model Card

## Model

- Type: multinomial_naive_bayes
- Labels: user, system, organization
- Training rows: 3872
- Vocabulary size: 3088

## Dataset Source

- opp115::consolidation-0.75

## Held-Out Metrics

Validation:

- Accuracy: 0.885584
- Macro precision: 0.665276
- Macro recall: 0.732948
- Macro F1: 0.69043

Test:

- Accuracy: 0.716353
- Macro precision: 0.530286
- Macro recall: 0.729766
- Macro F1: 0.573119

## Notes

- This is a deterministic baseline for Phase 3 acceptance and reproducibility.
- Bayesian scoring integration is intentionally deferred to the next increment.
