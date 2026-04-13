# Phase 4 Validation Report

- Baseline artifact: artifacts\phase-3-freeze
- Baseline passed: True

## Baseline Metrics

- Test macro F1: 0.894166
- Test accuracy: 0.957306
- Bayesian primary score: 0.987538
- Calibration test ECE: None

## Baseline Checks

| Check                            | Required | Passed | Details                                                                                                                                                                         |
| -------------------------------- | -------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| phase3_manifest_present          | yes      | yes    | {"phase": "phase-3"}                                                                                                                                                            |
| expected_artifacts_present       | yes      | yes    | {"present": ["dataset_manifest", "classifier_metrics", "validation_predictions", "test_predictions"], "missing": []}                                                            |
| policy_leakage_protection        | yes      | yes    | {"train_validation": 0, "train_test": 0, "validation_test": 0}                                                                                                                  |
| core_metrics_in_range            | yes      | yes    | {"validation_macro_f1": 0.849021, "test_macro_f1": 0.894166, "validation_accuracy": 0.948419, "test_accuracy": 0.957306}                                                        |
| prediction_counts_match_manifest | yes      | yes    | {"validation_expected": 1803, "validation_actual": 1803, "test_expected": 2272, "test_actual": 2272}                                                                            |
| prediction_row_schema            | yes      | yes    | {"rows_checked": 4075, "missing_field_rows": [], "invalid_label_rows": 0}                                                                                                       |
| prediction_probability_mass      | advisory | yes    | {"rows_with_probabilities": 0, "invalid_rows": 0, "rows_without_probabilities": 4075, "max_probability_sum_delta": 0.0, "reason": "probabilities_not_available_in_predictions"} |
| manifest_timestamp_utc           | advisory | no     | {"executed_at": ""}                                                                                                                                                             |
| bayesian_evidence_available      | advisory | yes    | {"total_evidence": 2272, "contributors": 15}                                                                                                                                    |
| polisis_source_advisory          | advisory | no     | {"source": "opp115::consolidation-0.75"}                                                                                                                                        |
| class_balance_distribution       | advisory | no     | {"fractions": {"user": 0.12003, "system": 0.038134, "organization": 0.841836}, "minimum_fraction_threshold": 0.05}                                                              |

## Leaderboard

| Rank | Run                 | Passed | Test Macro F1 | Test Accuracy | Delta F1 vs Baseline |
| ---- | ------------------- | ------ | ------------- | ------------- | -------------------- |
| 1    | phase-3-freeze      | yes    | 0.894166      | 0.957306      | 0.0                  |
| 2    | phase-3-privacybert | yes    | 0.891999      | 0.955986      | -0.002167            |
| 3    | phase-3-logreg      | yes    | 0.779024      | 0.892606      | -0.115142            |
| 4    | phase-3-nb          | yes    | 0.640117      | 0.814701      | -0.254049            |
