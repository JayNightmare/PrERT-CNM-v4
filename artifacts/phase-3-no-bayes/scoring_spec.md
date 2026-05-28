# Phase 3 Baseline Scoring Specification

## Output Schema

- actual_label: ground-truth label from held-out set.
- predicted_label: model prediction in {user, system, organization}.
- confidence: predicted class probability.
- probabilities: per-class probability map for calibration and threshold analysis.

## Metrics

- accuracy
- macro_precision
- macro_recall
- macro_f1
- per-class precision/recall/f1/support
- calibration_ece and calibration_brier
- threshold_sweep precision/recall/f1 by threshold
- bootstrap confidence intervals for key held-out metrics


## Constraints

- All metric values are in [0, 1] except support counts.
- Dataset splits are deterministic for a fixed seed.
