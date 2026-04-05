# Phase 3 Baseline Model Card

## Model

- Type: multinomial_naive_bayes
- Labels: user, system, organization
- Training rows: 15645
- Vocabulary size: 5851

## Dataset Source

- opp115::consolidation-0.75

## Held-Out Metrics

Validation:

- Accuracy: 0.816417
- Macro precision: 0.587961
- Macro recall: 0.761608
- Macro F1: 0.625907

Test:

- Accuracy: 0.814701
- Macro precision: 0.586193
- Macro recall: 0.758442
- Macro F1: 0.640117

## Notes

- This is a deterministic baseline for Phase 3 acceptance and reproducibility.
- Bayesian scoring integration is intentionally deferred to the next increment.
