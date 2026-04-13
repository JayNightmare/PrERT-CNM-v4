"""Cross-run comparison helpers for Phase 4 validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


def compare_validation_reports(
    baseline: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    baseline_metrics = _as_dict(_as_dict(baseline.get("summary")).get("metrics"))

    comparisons: List[Dict[str, Any]] = []
    leaderboard: List[Dict[str, Any]] = []

    baseline_row = {
        "artifact_dir": str(baseline.get("artifact_dir", "")),
        "name": Path(str(baseline.get("artifact_dir", ""))).name,
        "is_baseline": True,
        "validation_passed": bool(_as_dict(baseline.get("validation")).get("passed", False)),
        "test_macro_f1": _to_optional_float(baseline_metrics.get("test_macro_f1")),
        "test_accuracy": _to_optional_float(baseline_metrics.get("test_accuracy")),
        "bayesian_primary_score": _to_optional_float(baseline_metrics.get("bayesian_primary_score")),
        "calibration_test_ece": _to_optional_float(baseline_metrics.get("calibration_test_ece")),
    }
    leaderboard.append(baseline_row)

    for report in candidates:
        metrics = _as_dict(_as_dict(report.get("summary")).get("metrics"))
        artifact_dir = str(report.get("artifact_dir", ""))
        row = {
            "artifact_dir": artifact_dir,
            "name": Path(artifact_dir).name,
            "is_baseline": False,
            "validation_passed": bool(_as_dict(report.get("validation")).get("passed", False)),
            "test_macro_f1": _to_optional_float(metrics.get("test_macro_f1")),
            "test_accuracy": _to_optional_float(metrics.get("test_accuracy")),
            "bayesian_primary_score": _to_optional_float(metrics.get("bayesian_primary_score")),
            "calibration_test_ece": _to_optional_float(metrics.get("calibration_test_ece")),
        }
        leaderboard.append(row)

        comparisons.append(
            {
                "artifact_dir": artifact_dir,
                "deltas": {
                    "test_macro_f1": _delta(row.get("test_macro_f1"), baseline_row.get("test_macro_f1")),
                    "test_accuracy": _delta(row.get("test_accuracy"), baseline_row.get("test_accuracy")),
                    "bayesian_primary_score": _delta(
                        row.get("bayesian_primary_score"), baseline_row.get("bayesian_primary_score")
                    ),
                    "calibration_test_ece": _delta(
                        row.get("calibration_test_ece"), baseline_row.get("calibration_test_ece")
                    ),
                },
            }
        )

    ranked = sorted(
        leaderboard,
        key=lambda row: (
            _rank_float(row.get("test_macro_f1")),
            _rank_float(row.get("test_accuracy")),
        ),
        reverse=True,
    )

    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank

    return {
        "baseline": baseline_row,
        "comparisons": comparisons,
        "leaderboard": ranked,
    }


def _rank_float(value: Any) -> float:
    maybe = _to_optional_float(value)
    if maybe is None:
        return -1.0
    return maybe


def _delta(current: Any, baseline: Any) -> float | None:
    current_float = _to_optional_float(current)
    baseline_float = _to_optional_float(baseline)
    if current_float is None or baseline_float is None:
        return None
    return round(float(current_float - baseline_float), 6)


def _to_optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}
