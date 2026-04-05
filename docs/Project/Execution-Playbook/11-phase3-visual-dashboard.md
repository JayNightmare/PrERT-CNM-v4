# Phase 3 Baseline Visual Dashboard

Snapshot date: 2026-04-05

## Purpose

Provide a visual and tabular status view of dataset shape and held-out classifier quality for the implemented Phase 3 baseline.

## Data Sources

- `artifacts/phase-3/dataset_manifest.json`
- `artifacts/phase-3/classifier_metrics.json`
- `artifacts/phase-3/classifier_metrics.jsonl`

## Executive Summary

| Area    | Metric                      |                                            Current Value |
| ------- | --------------------------- | -------------------------------------------------------: |
| Dataset | Total rows                  |                                                     5000 |
| Dataset | Class mix (org/system/user) | 83.88% / 4.12% / 12.00% (0.838800 / 0.041200 / 0.120000) |
| Split   | Train / Validation / Test   | 77.44% / 8.74% / 13.82% (0.774400 / 0.087400 / 0.138200) |
| Model   | Validation accuracy         |                                        88.56% (0.885584) |
| Model   | Validation macro F1         |                                        69.04% (0.690430) |
| Model   | Test accuracy               |                                        71.64% (0.716353) |
| Model   | Test macro F1               |                                        57.31% (0.573119) |
| Leakage | Policy overlap (all pairs)  |                                                        0 |

## Figure Table

| Figure ID | Figure Preview                                            | Key Takeaway                                                                                 |
| --------- | --------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Fig 5     | ![Fig 5](figures/fig-05-phase3-class-distribution.png)    | Organization clauses dominate baseline data volume, indicating label imbalance risk.         |
| Fig 6     | ![Fig 6](figures/fig-06-phase3-split-sizes.png)           | Split sizing is train-heavy as expected, while preserving held-out validation and test sets. |
| Fig 7     | ![Fig 7](figures/fig-07-phase3-test-confusion-matrix.png) | Most errors route from organization to user/system, while user->system confusion is minimal. |

## Fig 5. Dataset Class Distribution

![Figure 5: Phase 3 Dataset Class Distribution](figures/fig-05-phase3-class-distribution.png)

What this means:

- The baseline is strongly organization-heavy.
- Per-class metrics should be reviewed alongside macro metrics to avoid majority-class overconfidence.

## Fig 6. Split Size Profile

![Figure 6: Phase 3 Split Sizes](figures/fig-06-phase3-split-sizes.png)

What this means:

- The split is suitable for baseline training with separate held-out validation and test checks.
- `dataset_manifest.json.policy_overlap.* == 0` confirms policy-level leakage protection.

## Fig 7. Test-Set Confusion Matrix

![Figure 7: Phase 3 Test Confusion Matrix](figures/fig-07-phase3-test-confusion-matrix.png)

What this means:

- Correct organization predictions are high in absolute count (420), but many organization clauses shift into user/system.
- User recall is moderate on test (66.18% (0.661765)), while user precision remains low (27.78% (0.277778)).
- System recall is strong on test (81.08% (0.810811)), with lower precision (37.97% (0.379747)).

## Held-Out Quality Indicator Tables

### Table A. Aggregate Held-Out Metrics

| Split      | Rows |          Accuracy |   Macro Precision |      Macro Recall |          Macro F1 |
| ---------- | ---: | ----------------: | ----------------: | ----------------: | ----------------: |
| Validation |  437 | 88.56% (0.885584) | 66.53% (0.665276) | 73.29% (0.732948) | 69.04% (0.690430) |
| Test       |  691 | 71.64% (0.716353) | 53.03% (0.530286) | 72.98% (0.729766) | 57.31% (0.573119) |

### Table B. Test Per-Class Metrics

| Label        |         Precision |            Recall |                F1 | Support |
| ------------ | ----------------: | ----------------: | ----------------: | ------: |
| user         | 27.78% (0.277778) | 66.18% (0.661765) | 39.13% (0.391304) |      68 |
| system       | 37.97% (0.379747) | 81.08% (0.810811) | 51.72% (0.517241) |      37 |
| organization | 93.33% (0.933333) | 71.67% (0.716724) | 81.08% (0.810811) |     586 |

## Next Measurement Targets

1. Add probability calibration visuals (reliability curve and expected calibration error).
2. Add threshold-sensitivity analysis for class-specific operating points.
3. Add trend snapshots across dated Phase 3 runs to monitor drift and stability.

Regeneration command:

```bash
PYTHONPATH=src python3 scripts/generate_phase3_dashboard_figures.py
```

---

## Navigation

[⬅ Back](10-phase3-implementation-runbook.md) | [Next ⮕](README.md)
