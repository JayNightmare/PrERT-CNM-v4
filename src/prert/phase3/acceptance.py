"""Phase 3 acceptance-freeze evaluation helpers.

This module converts a completed Phase 3 artifact directory into a proposal-oriented
acceptance report so teams can gate transition into Phase 4 validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def evaluate_phase3_acceptance(
    output_dir: Path,
    manifest: Mapping[str, Any],
    require_privacybert: bool = True,
    require_bayesian: bool = True,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    inputs = _as_dict(manifest.get("inputs"))
    metrics = _as_dict(manifest.get("metrics"))
    dataset = _as_dict(manifest.get("dataset_manifest"))
    overlap = _as_dict(dataset.get("policy_overlap"))
    output_files = _as_dict(manifest.get("output_files"))

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
        "classifier_metrics_in_range",
        _in_unit_interval(metrics.get("validation_macro_f1"))
        and _in_unit_interval(metrics.get("test_macro_f1"))
        and _in_unit_interval(metrics.get("validation_accuracy"))
        and _in_unit_interval(metrics.get("test_accuracy")),
        {
            "validation_macro_f1": metrics.get("validation_macro_f1"),
            "test_macro_f1": metrics.get("test_macro_f1"),
            "validation_accuracy": metrics.get("validation_accuracy"),
            "test_accuracy": metrics.get("test_accuracy"),
        },
    )

    if require_privacybert:
        _add_check(
            checks,
            "privacybert_model_required",
            str(inputs.get("model_type", "")).strip().lower() == "privacybert",
            {"model_type": inputs.get("model_type")},
        )

    if require_bayesian:
        _add_check(
            checks,
            "bayesian_primary_surface",
            str(manifest.get("primary_metric_surface", "")).strip().lower() == "bayesian_posterior",
            {"primary_metric_surface": manifest.get("primary_metric_surface")},
        )
        _add_check(
            checks,
            "bayesian_primary_score_in_range",
            _in_unit_interval(metrics.get("bayesian_primary_score")),
            {"bayesian_primary_score": metrics.get("bayesian_primary_score")},
        )

    files_ok, file_details = _check_expected_files(output_dir=output_dir, output_files=output_files)
    _add_check(checks, "expected_artifacts_present", files_ok, file_details)

    bayesian_details = _load_bayesian_details(output_dir=output_dir, output_files=output_files)
    if require_bayesian:
        _add_check(
            checks,
            "bayesian_outputs_have_evidence",
            bayesian_details["has_evidence"],
            bayesian_details,
        )

    passed = all(bool(check["passed"]) for check in checks)
    return {
        "phase": "phase-3",
        "acceptance": {
            "passed": passed,
            "required": {
                "privacybert": require_privacybert,
                "bayesian": require_bayesian,
            },
            "checks": checks,
        },
    }


def write_phase3_acceptance_report(output_dir: Path, report: Mapping[str, Any]) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "phase3_acceptance_report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_path = output_dir / "phase3_acceptance_report.md"
    md_path.write_text(_render_markdown(report), encoding="utf-8")

    return {
        "json": str(json_path),
        "markdown": str(md_path),
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    acceptance = _as_dict(report.get("acceptance"))
    checks = acceptance.get("checks", [])

    lines = [
        "# Phase 3 Acceptance Freeze Report",
        "",
        f"- Passed: {acceptance.get('passed')}",
        "",
        "## Checks",
        "",
        "| Check | Passed | Details |",
        "| --- | --- | --- |",
    ]

    for check in checks:
        name = str(check.get("name", ""))
        passed = "yes" if check.get("passed") else "no"
        details = json.dumps(check.get("details", {}), ensure_ascii=False)
        lines.append(f"| {name} | {passed} | {details} |")

    lines.append("")
    return "\n".join(lines)


def _check_expected_files(output_dir: Path, output_files: Mapping[str, Any]) -> tuple[bool, Dict[str, Any]]:
    required_keys = [
        "dataset_manifest",
        "classifier_metrics",
        "phase3_manifest",  # not in output_files map, checked below
        "validation_predictions",
        "test_predictions",
    ]

    missing: List[str] = []
    present: List[str] = []

    for key in required_keys:
        if key == "phase3_manifest":
            candidate = output_dir / "phase3_manifest.json"
        else:
            rel = str(output_files.get(key, "")).strip()
            candidate = output_dir / rel if rel else None

        if candidate is None or not candidate.exists():
            missing.append(key)
        else:
            present.append(key)

    for key in ("bayesian_validation", "bayesian_test"):
        rel = str(output_files.get(key, "")).strip()
        if not rel:
            continue
        candidate = output_dir / rel
        if not candidate.exists():
            missing.append(key)
        else:
            present.append(key)

    return len(missing) == 0, {"present": present, "missing": missing}


def _load_bayesian_details(output_dir: Path, output_files: Mapping[str, Any]) -> Dict[str, Any]:
    test_rel = str(output_files.get("bayesian_test", "")).strip()
    if not test_rel:
        return {"has_evidence": False, "reason": "bayesian_test_output_not_declared"}

    test_path = output_dir / test_rel
    if not test_path.exists():
        return {"has_evidence": False, "reason": "bayesian_test_output_missing", "path": str(test_path)}

    payload = json.loads(test_path.read_text(encoding="utf-8"))
    levels = _as_dict(payload.get("levels"))

    total_evidence = 0
    contributors = 0
    for level_payload in levels.values():
        level_dict = _as_dict(level_payload)
        total_evidence += int(level_dict.get("evidence_count", 0))
        contributors += len(level_dict.get("top_contributors", []))

    return {
        "has_evidence": total_evidence > 0 and contributors > 0,
        "total_evidence": total_evidence,
        "contributors": contributors,
    }


def _in_unit_interval(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return 0.0 <= numeric <= 1.0


def _add_check(checks: List[Dict[str, Any]], name: str, passed: bool, details: Mapping[str, Any]) -> None:
    checks.append({
        "name": name,
        "passed": bool(passed),
        "details": dict(details),
    })


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}
