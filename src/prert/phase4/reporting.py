"""Report writers for Phase 4 validation outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from prert.phase3.io import write_json, write_jsonl


def write_phase4_validation_outputs(output_dir: Path, payload: Mapping[str, Any]) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "phase4_validation_report.json"
    md_path = output_dir / "phase4_validation_report.md"
    leaderboard_path = output_dir / "phase4_leaderboard.jsonl"

    write_json(json_path, dict(payload))
    md_path.write_text(_render_markdown(payload), encoding="utf-8")

    comparison_summary = _as_dict(payload.get("comparison_summary"))
    leaderboard = comparison_summary.get("leaderboard", [])
    if isinstance(leaderboard, list) and leaderboard:
        rows = [row for row in leaderboard if isinstance(row, dict)]
        write_jsonl(leaderboard_path, rows)

    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "leaderboard": str(leaderboard_path),
    }


def _render_markdown(payload: Mapping[str, Any]) -> str:
    baseline = _as_dict(payload.get("baseline"))
    validation = _as_dict(baseline.get("validation"))
    summary = _as_dict(baseline.get("summary"))
    metrics = _as_dict(summary.get("metrics"))

    lines = [
        "# Phase 4 Validation Report",
        "",
        f"- Baseline artifact: {payload.get('baseline_artifact_dir', '')}",
        f"- Baseline passed: {validation.get('passed')}",
        "",
        "## Baseline Metrics",
        "",
        f"- Test macro F1: {metrics.get('test_macro_f1')}",
        f"- Test accuracy: {metrics.get('test_accuracy')}",
        f"- Bayesian primary score: {metrics.get('bayesian_primary_score')}",
        f"- Calibration test ECE: {metrics.get('calibration_test_ece')}",
        "",
        "## Baseline Checks",
        "",
        "| Check | Required | Passed | Details |",
        "| --- | --- | --- | --- |",
    ]

    checks = validation.get("checks", [])
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, dict):
                continue
            name = str(check.get("name", ""))
            required = "yes" if check.get("required", True) else "advisory"
            passed = "yes" if check.get("passed") else "no"
            details = json.dumps(check.get("details", {}), ensure_ascii=False)
            lines.append(f"| {name} | {required} | {passed} | {details} |")

    comparison_summary = _as_dict(payload.get("comparison_summary"))
    leaderboard = comparison_summary.get("leaderboard", [])
    if isinstance(leaderboard, list) and leaderboard:
        lines.extend(
            [
                "",
                "## Leaderboard",
                "",
                "| Rank | Run | Passed | Test Macro F1 | Test Accuracy | Delta F1 vs Baseline |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )

        baseline_name = str(_as_dict(comparison_summary.get("baseline")).get("name", ""))
        delta_lookup = {
            str(item.get("artifact_dir", "")): _as_dict(item.get("deltas"))
            for item in comparison_summary.get("comparisons", [])
            if isinstance(item, dict)
        }

        for row in leaderboard:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", ""))
            artifact_dir = str(row.get("artifact_dir", ""))
            deltas = delta_lookup.get(artifact_dir, {})
            delta_f1 = deltas.get("test_macro_f1") if name != baseline_name else 0.0
            lines.append(
                "| {rank} | {name} | {passed} | {f1} | {acc} | {delta} |".format(
                    rank=row.get("rank", ""),
                    name=name,
                    passed="yes" if row.get("validation_passed") else "no",
                    f1=row.get("test_macro_f1"),
                    acc=row.get("test_accuracy"),
                    delta=delta_f1,
                )
            )

    lines.append("")
    return "\n".join(lines)


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}
