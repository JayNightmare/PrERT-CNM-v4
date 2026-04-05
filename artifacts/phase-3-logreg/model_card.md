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
- Macro precision: 0.722643
- Macro recall: 0.816358
- Macro F1: 0.760958

Test:

- Accuracy: 0.893046
- Macro precision: 0.717148
- Macro recall: 0.884646
- Macro F1: 0.780633

## Notes

- This is a deterministic baseline for Phase 3 acceptance and reproducibility.
- Bayesian scoring integration is intentionally deferred to the next increment.
