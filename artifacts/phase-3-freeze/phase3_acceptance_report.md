# Phase 3 Acceptance Freeze Report

- Passed: True

## Checks

| Check | Required | Passed | Details |
| --- | --- | --- | --- |
| policy_leakage_protection | yes | yes | {"train_validation": 0, "train_test": 0, "validation_test": 0} |
| classifier_metrics_in_range | yes | yes | {"validation_macro_f1": 0.935227, "test_macro_f1": 0.868631, "validation_accuracy": 0.972789, "test_accuracy": 0.953883} |
| privacybert_model_required | yes | yes | {"model_type": "privacybert"} |
| polisis_source_advisory | advisory | no | {"has_polisis": false, "dataset_source": "opp115::consolidation-0.75", "polisis_root": "", "polisis_source_dir": "", "labeled_input_path": ""} |
| bayesian_primary_surface | yes | yes | {"primary_metric_surface": "bayesian_posterior"} |
| bayesian_primary_score_in_range | yes | yes | {"bayesian_primary_score": 0.900649} |
| expected_artifacts_present | yes | yes | {"present": ["dataset_manifest", "classifier_metrics", "phase3_manifest", "validation_predictions", "test_predictions", "bayesian_validation", "bayesian_test"], "missing": []} |
| bayesian_outputs_have_evidence | yes | yes | {"has_evidence": true, "total_evidence": 2472, "contributors": 15} |
