import json
from pathlib import Path

from prert.phase4.pipeline import run_phase4_validation
from prert.phase4.validation import evaluate_phase4_validation


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _prediction_rows(prefix: str, count: int, label: str = "organization") -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        rows.append(
            {
                "example_id": f"{prefix}-{index}",
                "policy_uid": f"policy-{index // 2}",
                "actual_label": label,
                "predicted_label": label,
                "confidence": 0.9,
                "probabilities": {
                    "user": 0.05,
                    "system": 0.05,
                    "organization": 0.9,
                },
                "text": "sample clause",
            }
        )
    return rows


def _write_phase3_artifacts(
    output_dir: Path,
    *,
    validation_rows: int = 3,
    test_rows: int = 3,
    test_macro_f1: float = 0.75,
    test_accuracy: float = 0.8,
    source: str = "polisis::normalized",
    include_bayesian: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_predictions = _prediction_rows("val", validation_rows)
    test_predictions = _prediction_rows("test", test_rows)

    _write_jsonl(output_dir / "validation_predictions.jsonl", validation_predictions)
    _write_jsonl(output_dir / "test_predictions.jsonl", test_predictions)

    _write_json(
        output_dir / "dataset_manifest.json",
        {
            "source": source,
            "input_set": "normalized",
            "total_rows": validation_rows + test_rows + 4,
            "class_distribution": {
                "organization": validation_rows + test_rows + 4,
                "system": 1,
                "user": 1,
            },
            "splits": {
                "train": {"rows": 4},
                "validation": {"rows": validation_rows},
                "test": {"rows": test_rows},
            },
            "policy_overlap": {
                "train_validation": 0,
                "train_test": 0,
                "validation_test": 0,
            },
        },
    )

    _write_json(
        output_dir / "classifier_metrics.json",
        {
            "validation": {
                "rows": validation_rows,
                "accuracy": 0.78,
                "macro_f1": 0.72,
            },
            "test": {
                "rows": test_rows,
                "accuracy": test_accuracy,
                "macro_f1": test_macro_f1,
            },
            "bayesian": {
                "enabled": include_bayesian,
                "primary_score": 0.77 if include_bayesian else None,
            },
            "measurement_targets": {
                "calibration": {
                    "test_ece": 0.11,
                }
            },
        },
    )

    _write_json(
        output_dir / "calibration_test.json",
        {
            "overall": {
                "ece": 0.11,
            }
        },
    )

    _write_json(
        output_dir / "threshold_sweep_test.json",
        {
            "by_label": {
                "organization": [
                    {
                        "threshold": 0.5,
                        "precision": 0.9,
                        "recall": 0.88,
                        "f1": 0.89,
                    }
                ]
            }
        },
    )

    _write_json(
        output_dir / "bootstrap_ci_test.json",
        {
            "metrics": {
                "accuracy": {
                    "mean": test_accuracy,
                    "interval_95": {
                        "lower": max(0.0, test_accuracy - 0.05),
                        "upper": min(1.0, test_accuracy + 0.05),
                    },
                },
                "macro_f1": {
                    "mean": test_macro_f1,
                    "interval_95": {
                        "lower": max(0.0, test_macro_f1 - 0.05),
                        "upper": min(1.0, test_macro_f1 + 0.05),
                    },
                },
            }
        },
    )

    if include_bayesian:
        _write_json(
            output_dir / "bayesian_risk_test.json",
            {
                "levels": {
                    "user": {"evidence_count": 2, "top_contributors": [{"example_id": "u-1"}]},
                    "system": {"evidence_count": 1, "top_contributors": [{"example_id": "s-1"}]},
                    "organization": {"evidence_count": 3, "top_contributors": [{"example_id": "o-1"}]},
                }
            },
        )

    output_files = {
        "dataset_manifest": "dataset_manifest.json",
        "classifier_metrics": "classifier_metrics.json",
        "validation_predictions": "validation_predictions.jsonl",
        "test_predictions": "test_predictions.jsonl",
        "calibration_test": "calibration_test.json",
        "threshold_sweep_test": "threshold_sweep_test.json",
        "bootstrap_ci_test": "bootstrap_ci_test.json",
    }
    if include_bayesian:
        output_files["bayesian_test"] = "bayesian_risk_test.json"

    _write_json(
        output_dir / "phase3_manifest.json",
        {
            "phase": "phase-3",
            "execution_metadata": {
                "executed_at": "2026-01-01T00:00:00Z",
            },
            "inputs": {
                "model_type": "privacybert",
                "input_set": "consolidation-0.75",
            },
            "dataset_manifest": {
                "source": source,
                "input_set": "normalized",
                "splits": {
                    "validation": {"rows": validation_rows},
                    "test": {"rows": test_rows},
                },
                "class_distribution": {
                    "organization": validation_rows + test_rows + 4,
                    "system": 1,
                    "user": 1,
                },
                "policy_overlap": {
                    "train_validation": 0,
                    "train_test": 0,
                    "validation_test": 0,
                },
            },
            "metrics": {
                "validation_macro_f1": 0.72,
                "test_macro_f1": test_macro_f1,
                "validation_accuracy": 0.78,
                "test_accuracy": test_accuracy,
                "bayesian_primary_score": 0.77 if include_bayesian else None,
                "calibration_test_ece": 0.11,
            },
            "output_files": output_files,
        },
    )


def test_phase4_validation_passes_on_valid_artifacts(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "phase-3"
    _write_phase3_artifacts(artifact_dir, include_bayesian=True, source="polisis::normalized")

    report = evaluate_phase4_validation(
        artifact_dir=artifact_dir,
        require_bayesian=True,
        require_polisis=True,
        polisis_advisory=True,
    )

    assert report["validation"]["passed"] is True
    checks = {item["name"]: item for item in report["validation"]["checks"]}
    assert checks["bayesian_evidence_available"]["passed"] is True
    assert checks["polisis_source_required"]["passed"] is True


def test_phase4_validation_fails_when_prediction_count_mismatch(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "phase-3"
    _write_phase3_artifacts(artifact_dir, validation_rows=4, test_rows=3, include_bayesian=True)

    # Force mismatch by truncating validation predictions.
    _write_jsonl(artifact_dir / "validation_predictions.jsonl", _prediction_rows("val", 2))

    report = evaluate_phase4_validation(
        artifact_dir=artifact_dir,
        require_bayesian=True,
    )

    assert report["validation"]["passed"] is False
    failed_required = {
        item["name"]
        for item in report["validation"]["checks"]
        if item.get("required", True) and not item["passed"]
    }
    assert "prediction_counts_match_manifest" in failed_required


def test_phase4_polisis_advisory_is_non_blocking(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "phase-3"
    _write_phase3_artifacts(artifact_dir, include_bayesian=False, source="opp115::consolidation-0.75")

    report = evaluate_phase4_validation(
        artifact_dir=artifact_dir,
        require_bayesian=False,
        polisis_advisory=True,
    )

    assert report["validation"]["passed"] is True
    checks = {item["name"]: item for item in report["validation"]["checks"]}
    assert checks["polisis_source_advisory"]["required"] is False
    assert checks["polisis_source_advisory"]["passed"] is False


def test_phase4_polisis_requirement_is_blocking(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "phase-3"
    _write_phase3_artifacts(artifact_dir, include_bayesian=False, source="opp115::consolidation-0.75")

    report = evaluate_phase4_validation(
        artifact_dir=artifact_dir,
        require_bayesian=False,
        require_polisis=True,
        polisis_advisory=True,
    )

    assert report["validation"]["passed"] is False
    checks = {item["name"]: item for item in report["validation"]["checks"]}
    assert checks["polisis_source_required"]["required"] is True
    assert checks["polisis_source_required"]["passed"] is False
    assert "polisis_source_advisory" not in checks


def test_phase4_pipeline_writes_reports_and_leaderboard(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "phase-3-baseline"
    candidate_dir = tmp_path / "phase-3-candidate"
    output_dir = tmp_path / "phase-4"

    _write_phase3_artifacts(baseline_dir, test_macro_f1=0.70, test_accuracy=0.75, include_bayesian=True)
    _write_phase3_artifacts(candidate_dir, test_macro_f1=0.82, test_accuracy=0.86, include_bayesian=True)

    payload = run_phase4_validation(
        output_dir=output_dir,
        baseline_dir=baseline_dir,
        comparison_dirs=[candidate_dir],
        require_bayesian=True,
    )

    output_files = payload["output_files"]
    assert Path(output_files["json"]).exists()
    assert Path(output_files["markdown"]).exists()
    assert Path(output_files["leaderboard"]).exists()

    leaderboard = payload["comparison_summary"]["leaderboard"]
    assert len(leaderboard) == 2
    assert leaderboard[0]["artifact_dir"] == str(candidate_dir)
    assert leaderboard[0]["rank"] == 1
