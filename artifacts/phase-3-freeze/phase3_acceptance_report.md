# Phase 3 Acceptance Freeze Report

- Passed: True

## Checks

| Check | Passed | Details |
| --- | --- | --- |
| policy_leakage_protection | yes | {"train_validation": 0, "train_test": 0, "validation_test": 0} |
| classifier_metrics_in_range | yes | {"validation_macro_f1": 0.849021, "test_macro_f1": 0.894166, "validation_accuracy": 0.948419, "test_accuracy": 0.957306} |
| privacybert_model_required | yes | {"model_type": "privacybert"} |
| bayesian_primary_surface | yes | {"primary_metric_surface": "bayesian_posterior"} |
| bayesian_primary_score_in_range | yes | {"bayesian_primary_score": 0.987538} |
| expected_artifacts_present | yes | {"present": ["dataset_manifest", "classifier_metrics", "phase3_manifest", "validation_predictions", "test_predictions", "bayesian_validation", "bayesian_test"], "missing": []} |
| bayesian_outputs_have_evidence | yes | {"has_evidence": true, "total_evidence": 2272, "contributors": 15} |
