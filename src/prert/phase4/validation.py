"""Artifact-based Phase 4 validation checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from prert.phase4.io import (
    load_optional_json,
    load_optional_jsonl_rows,
    load_phase3_manifest,
    resolve_output_path,
)


LABELS: tuple[str, str, str] = ("user", "system", "organization")


def evaluate_phase4_validation(
    artifact_dir: Path,
    manifest: Optional[Mapping[str, Any]] = None,
    require_bayesian: bool = False,
    polisis_advisory: bool = True,
    ece_threshold: float = 0.20,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    loaded_manifest: Mapping[str, Any]
    if manifest is None:
        try:
            loaded_manifest = load_phase3_manifest(artifact_dir)
        except FileNotFoundError as exc:
            _add_check(
                checks,
                "phase3_manifest_present",
                False,
                {"reason": str(exc), "path": str(artifact_dir / "phase3_manifest.json")},
            )
            return {
                "phase": "phase-4",
                "artifact_dir": str(artifact_dir),
                "validation": {
                    "passed": False,
                    "required": {"bayesian": require_bayesian},
                    "advisory": {"polisis": bool(polisis_advisory and not require_bayesian)},
                    "checks": checks,
                },
                "summary": {
                    "source": "",
                    "model_type": "",
                    "metrics": {},
                    "rows": {},
                },
            }
    else:
        loaded_manifest = manifest

    _add_check(
        checks,
        "phase3_manifest_present",
        str(loaded_manifest.get("phase", "")).strip().lower() == "phase-3",
        {"phase": loaded_manifest.get("phase")},
    )

    dataset_manifest = _as_dict(loaded_manifest.get("dataset_manifest"))
    inputs = _as_dict(loaded_manifest.get("inputs"))
    output_files = _as_dict(loaded_manifest.get("output_files"))

    dataset_manifest_path = resolve_output_path(
        artifact_dir,
        output_files,
        key="dataset_manifest",
        fallback="dataset_manifest.json",
    )
    classifier_metrics_path = resolve_output_path(
        artifact_dir,
        output_files,
        key="classifier_metrics",
        fallback="classifier_metrics.json",
    )
    validation_predictions_path = resolve_output_path(
        artifact_dir,
        output_files,
        key="validation_predictions",
        fallback="validation_predictions.jsonl",
    )
    test_predictions_path = resolve_output_path(
        artifact_dir,
        output_files,
        key="test_predictions",
        fallback="test_predictions.jsonl",
    )

    expected = {
        "dataset_manifest": dataset_manifest_path,
        "classifier_metrics": classifier_metrics_path,
        "validation_predictions": validation_predictions_path,
        "test_predictions": test_predictions_path,
    }

    if require_bayesian:
        expected["bayesian_test"] = resolve_output_path(
            artifact_dir,
            output_files,
            key="bayesian_test",
            fallback="bayesian_risk_test.json",
        )

    missing = [key for key, path in expected.items() if not path.exists()]
    present = [key for key, path in expected.items() if path.exists()]
    _add_check(
        checks,
        "expected_artifacts_present",
        len(missing) == 0,
        {"present": present, "missing": missing},
    )

    dataset_manifest_file = load_optional_json(dataset_manifest_path) or {}
    classifier_metrics = load_optional_json(classifier_metrics_path) or {}

    merged_dataset = dict(dataset_manifest_file)
    for key, value in dataset_manifest.items():
        merged_dataset.setdefault(key, value)

    merged_metrics = _derive_metrics(
        manifest_metrics=_as_dict(loaded_manifest.get("metrics")),
        classifier_metrics=classifier_metrics,
    )

    overlap = _as_dict(merged_dataset.get("policy_overlap"))
    _add_check(
        checks,
        "policy_leakage_protection",
        all(int(overlap.get(key, 1)) == 0 for key in ("train_validation", "train_test", "validation_test")),
        {
            "train_validation": int(overlap.get("train_validation", -1)),
            "train_test": int(overlap.get("train_test", -1)),
            "validation_test": int(overlap.get("validation_test", -1)),
        },
    )

    _add_check(
        checks,
        "core_metrics_in_range",
        _in_unit_interval(merged_metrics.get("validation_macro_f1"))
        and _in_unit_interval(merged_metrics.get("test_macro_f1"))
        and _in_unit_interval(merged_metrics.get("validation_accuracy"))
        and _in_unit_interval(merged_metrics.get("test_accuracy")),
        {
            "validation_macro_f1": merged_metrics.get("validation_macro_f1"),
            "test_macro_f1": merged_metrics.get("test_macro_f1"),
            "validation_accuracy": merged_metrics.get("validation_accuracy"),
            "test_accuracy": merged_metrics.get("test_accuracy"),
        },
    )

    validation_rows = load_optional_jsonl_rows(validation_predictions_path) or []
    test_rows = load_optional_jsonl_rows(test_predictions_path) or []

    validation_target = int(_as_dict(_as_dict(merged_dataset.get("splits")).get("validation")).get("rows", -1))
    test_target = int(_as_dict(_as_dict(merged_dataset.get("splits")).get("test")).get("rows", -1))
    count_pass = True
    if validation_target >= 0:
        count_pass = count_pass and validation_target == len(validation_rows)
    if test_target >= 0:
        count_pass = count_pass and test_target == len(test_rows)

    _add_check(
        checks,
        "prediction_counts_match_manifest",
        count_pass,
        {
            "validation_expected": validation_target,
            "validation_actual": len(validation_rows),
            "test_expected": test_target,
            "test_actual": len(test_rows),
        },
    )

    schema_ok, schema_details = _validate_prediction_rows(validation_rows, test_rows)
    _add_check(checks, "prediction_row_schema", schema_ok, schema_details)

    prob_ok, prob_details = _validate_probability_mass(validation_rows + test_rows)
    _add_check(checks, "prediction_probability_mass", prob_ok, prob_details, required=False)

    executed_at = str(_as_dict(loaded_manifest.get("execution_metadata")).get("executed_at", ""))
    _add_check(
        checks,
        "manifest_timestamp_utc",
        bool(executed_at) and executed_at.endswith("Z"),
        {"executed_at": executed_at},
        required=False,
    )

    calibration_test_path = resolve_output_path(
        artifact_dir,
        output_files,
        key="calibration_test",
        fallback="calibration_test.json",
    )
    calibration_test = load_optional_json(calibration_test_path)
    ece = _resolve_ece(merged_metrics, calibration_test)
    if ece is not None:
        _add_check(
            checks,
            "calibration_ece_target",
            ece <= float(ece_threshold),
            {"ece": ece, "threshold": float(ece_threshold)},
            required=False,
        )

    bootstrap_path = resolve_output_path(
        artifact_dir,
        output_files,
        key="bootstrap_ci_test",
        fallback="bootstrap_ci_test.json",
    )
    bootstrap_payload = load_optional_json(bootstrap_path)
    if bootstrap_payload is not None:
        boot_ok, boot_details = _validate_bootstrap_intervals(bootstrap_payload)
        _add_check(checks, "bootstrap_intervals_valid", boot_ok, boot_details, required=False)

    threshold_path = resolve_output_path(
        artifact_dir,
        output_files,
        key="threshold_sweep_test",
        fallback="threshold_sweep_test.json",
    )
    threshold_payload = load_optional_json(threshold_path)
    if threshold_payload is not None:
        threshold_ok, threshold_details = _validate_threshold_sweep(threshold_payload)
        _add_check(checks, "threshold_sweep_valid", threshold_ok, threshold_details, required=False)

    bayesian_test_path = resolve_output_path(
        artifact_dir,
        output_files,
        key="bayesian_test",
        fallback="bayesian_risk_test.json",
    )
    bayesian_payload = load_optional_json(bayesian_test_path)
    if require_bayesian:
        bayes_ok, bayes_details = _validate_bayesian_evidence(bayesian_payload)
        _add_check(checks, "bayesian_evidence_available", bayes_ok, bayes_details)
    elif bayesian_payload is not None:
        bayes_ok, bayes_details = _validate_bayesian_evidence(bayesian_payload)
        _add_check(checks, "bayesian_evidence_available", bayes_ok, bayes_details, required=False)

    if polisis_advisory:
        source = _resolve_dataset_source(loaded_manifest, merged_dataset)
        has_polisis = "polisis" in source.lower()
        _add_check(
            checks,
            "polisis_source_advisory",
            has_polisis,
            {"source": source},
            required=False,
        )

    imbalance_ok, imbalance_details = _check_class_balance(merged_dataset)
    _add_check(checks, "class_balance_distribution", imbalance_ok, imbalance_details, required=False)

    required_checks = [check for check in checks if bool(check.get("required", True))]
    passed = all(bool(check["passed"]) for check in required_checks)

    summary = {
        "source": _resolve_dataset_source(loaded_manifest, merged_dataset),
        "model_type": str(inputs.get("model_type", loaded_manifest.get("model_summary", {}).get("model_type", ""))),
        "metrics": {
            "validation_macro_f1": merged_metrics.get("validation_macro_f1"),
            "test_macro_f1": merged_metrics.get("test_macro_f1"),
            "validation_accuracy": merged_metrics.get("validation_accuracy"),
            "test_accuracy": merged_metrics.get("test_accuracy"),
            "bayesian_primary_score": merged_metrics.get("bayesian_primary_score"),
            "calibration_test_ece": ece,
        },
        "rows": {
            "validation_predictions": len(validation_rows),
            "test_predictions": len(test_rows),
        },
    }

    return {
        "phase": "phase-4",
        "artifact_dir": str(artifact_dir),
        "validation": {
            "passed": passed,
            "required": {
                "bayesian": require_bayesian,
            },
            "advisory": {
                "polisis": bool(polisis_advisory),
            },
            "checks": checks,
        },
        "summary": summary,
    }


def _derive_metrics(manifest_metrics: Mapping[str, Any], classifier_metrics: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    validation = _as_dict(classifier_metrics.get("validation"))
    test = _as_dict(classifier_metrics.get("test"))
    bayesian = _as_dict(classifier_metrics.get("bayesian"))

    return {
        "validation_macro_f1": _to_optional_float(
            manifest_metrics.get("validation_macro_f1", validation.get("macro_f1"))
        ),
        "test_macro_f1": _to_optional_float(manifest_metrics.get("test_macro_f1", test.get("macro_f1"))),
        "validation_accuracy": _to_optional_float(
            manifest_metrics.get("validation_accuracy", validation.get("accuracy"))
        ),
        "test_accuracy": _to_optional_float(manifest_metrics.get("test_accuracy", test.get("accuracy"))),
        "bayesian_primary_score": _to_optional_float(
            manifest_metrics.get("bayesian_primary_score", bayesian.get("primary_score"))
        ),
        "calibration_test_ece": _to_optional_float(_as_dict(classifier_metrics.get("measurement_targets")).get("calibration", {}).get("test_ece")),
    }


def _resolve_ece(metrics: Mapping[str, Optional[float]], calibration_payload: Optional[Mapping[str, Any]]) -> Optional[float]:
    metric_ece = metrics.get("calibration_test_ece")
    if metric_ece is not None:
        return float(metric_ece)
    if calibration_payload is None:
        return None
    overall = _as_dict(calibration_payload.get("overall"))
    return _to_optional_float(overall.get("ece"))


def _validate_prediction_rows(
    validation_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
) -> tuple[bool, Dict[str, Any]]:
    required_fields = {"example_id", "policy_uid", "actual_label", "predicted_label", "confidence"}

    checked = 0
    invalid_examples: List[str] = []
    invalid_labels = 0

    for row in [*validation_rows, *test_rows]:
        checked += 1
        missing = [name for name in required_fields if name not in row]
        if missing:
            invalid_examples.append(str(row.get("example_id", f"row-{checked}")))
            continue

        actual = str(row.get("actual_label", "")).strip().lower()
        predicted = str(row.get("predicted_label", "")).strip().lower()
        if actual not in LABELS or predicted not in LABELS:
            invalid_labels += 1

    return len(invalid_examples) == 0 and invalid_labels == 0, {
        "rows_checked": checked,
        "missing_field_rows": invalid_examples[:10],
        "invalid_label_rows": invalid_labels,
    }


def _validate_probability_mass(rows: Sequence[Mapping[str, Any]]) -> tuple[bool, Dict[str, Any]]:
    inspected = 0
    invalid = 0
    missing_probabilities = 0
    max_delta = 0.0

    for row in rows:
        probabilities = row.get("probabilities")
        if not isinstance(probabilities, dict):
            missing_probabilities += 1
            continue

        values: List[float] = []
        for label in LABELS:
            value = probabilities.get(label)
            if value is None:
                continue
            maybe_float = _to_optional_float(value)
            if maybe_float is None or maybe_float < 0.0 or maybe_float > 1.0:
                invalid += 1
                maybe_float = None
            if maybe_float is not None:
                values.append(maybe_float)

        if not values:
            invalid += 1
            continue

        inspected += 1
        delta = abs(sum(values) - 1.0)
        max_delta = max(max_delta, delta)
        if delta > 0.01:
            invalid += 1

    if inspected == 0:
        return True, {
            "rows_with_probabilities": inspected,
            "invalid_rows": invalid,
            "rows_without_probabilities": missing_probabilities,
            "max_probability_sum_delta": round(float(max_delta), 6),
            "reason": "probabilities_not_available_in_predictions",
        }

    passed = invalid == 0
    return passed, {
        "rows_with_probabilities": inspected,
        "invalid_rows": invalid,
        "rows_without_probabilities": missing_probabilities,
        "max_probability_sum_delta": round(float(max_delta), 6),
    }


def _validate_bootstrap_intervals(payload: Mapping[str, Any]) -> tuple[bool, Dict[str, Any]]:
    metrics = _as_dict(payload.get("metrics"))
    checked = 0
    invalid = 0

    for metric_name, metric_payload in metrics.items():
        metric_dict = _as_dict(metric_payload)
        interval = _as_dict(metric_dict.get("interval_95"))
        lower = _to_optional_float(interval.get("lower"))
        upper = _to_optional_float(interval.get("upper"))
        center = _to_optional_float(metric_dict.get("mean"))
        if lower is None or upper is None:
            continue
        checked += 1
        if lower > upper:
            invalid += 1
            continue
        if center is not None and not (lower <= center <= upper):
            invalid += 1
            continue
        if lower < 0.0 or upper > 1.0:
            invalid += 1
            continue

    return invalid == 0, {
        "metrics_checked": checked,
        "invalid_intervals": invalid,
    }


def _validate_threshold_sweep(payload: Mapping[str, Any]) -> tuple[bool, Dict[str, Any]]:
    by_label = _as_dict(payload.get("by_label"))
    series_count = 0
    invalid_points = 0

    for _, points in by_label.items():
        if not isinstance(points, list):
            continue
        for point in points:
            if not isinstance(point, dict):
                invalid_points += 1
                continue
            series_count += 1
            for key in ("precision", "recall", "f1"):
                value = _to_optional_float(point.get(key))
                if value is None or value < 0.0 or value > 1.0:
                    invalid_points += 1

    return invalid_points == 0 and series_count > 0, {
        "points_checked": series_count,
        "invalid_points": invalid_points,
    }


def _validate_bayesian_evidence(payload: Optional[Mapping[str, Any]]) -> tuple[bool, Dict[str, Any]]:
    if payload is None:
        return False, {"reason": "bayesian_payload_missing"}

    levels = _as_dict(payload.get("levels"))
    evidence_count = 0
    contributor_count = 0
    for level_payload in levels.values():
        level_dict = _as_dict(level_payload)
        evidence_count += int(level_dict.get("evidence_count", 0))
        contributors = level_dict.get("top_contributors", [])
        if isinstance(contributors, list):
            contributor_count += len(contributors)

    return evidence_count > 0 and contributor_count > 0, {
        "total_evidence": evidence_count,
        "contributors": contributor_count,
    }


def _check_class_balance(dataset_manifest: Mapping[str, Any]) -> tuple[bool, Dict[str, Any]]:
    distribution = _as_dict(dataset_manifest.get("class_distribution"))
    total = sum(int(value) for value in distribution.values())
    if total <= 0:
        return False, {"reason": "class_distribution_missing"}

    fractions = {
        label: round(float(int(distribution.get(label, 0)) / total), 6)
        for label in LABELS
    }
    min_fraction = min(fractions.values())

    return min_fraction >= 0.05, {
        "fractions": fractions,
        "minimum_fraction_threshold": 0.05,
    }


def _resolve_dataset_source(manifest: Mapping[str, Any], dataset_manifest: Mapping[str, Any]) -> str:
    source = str(dataset_manifest.get("source", "")).strip()
    if source:
        return source

    inputs = _as_dict(manifest.get("inputs"))
    labeled = str(inputs.get("labeled_input_path", "")).strip()
    if labeled:
        return f"labeled::{Path(labeled).name}"

    polisis = str(inputs.get("polisis_root", "")).strip()
    if polisis:
        profile = str(inputs.get("polisis_input_set", "normalized")).strip() or "normalized"
        return f"polisis::{profile}"

    input_set = str(inputs.get("input_set", "")).strip() or "unknown"
    return f"opp115::{input_set}"


def _in_unit_interval(value: Any) -> bool:
    numeric = _to_optional_float(value)
    if numeric is None:
        return False
    return 0.0 <= numeric <= 1.0


def _to_optional_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _add_check(
    checks: List[Dict[str, Any]],
    name: str,
    passed: bool,
    details: Mapping[str, Any],
    required: bool = True,
) -> None:
    checks.append(
        {
            "name": name,
            "required": bool(required),
            "passed": bool(passed),
            "details": dict(details),
        }
    )


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}
