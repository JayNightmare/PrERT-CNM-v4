import json
from pathlib import Path

from prert.phase3.acceptance import evaluate_phase3_acceptance


def _write_minimum_artifacts(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "dataset_manifest.json").write_text("{}\n", encoding="utf-8")
    (output_dir / "classifier_metrics.json").write_text("{}\n", encoding="utf-8")
    (output_dir / "validation_predictions.jsonl").write_text("{}\n", encoding="utf-8")
    (output_dir / "test_predictions.jsonl").write_text("{}\n", encoding="utf-8")
    (output_dir / "phase3_manifest.json").write_text("{}\n", encoding="utf-8")

    bayesian_payload = {
        "levels": {
            "user": {"evidence_count": 2, "top_contributors": [{"example_id": "u1"}]},
            "system": {"evidence_count": 1, "top_contributors": [{"example_id": "s1"}]},
            "organization": {"evidence_count": 3, "top_contributors": [{"example_id": "o1"}]},
        }
    }
    (output_dir / "bayesian_risk_test.json").write_text(json.dumps(bayesian_payload) + "\n", encoding="utf-8")
    (output_dir / "bayesian_risk_validation.json").write_text(json.dumps(bayesian_payload) + "\n", encoding="utf-8")

    return {
        "phase": "phase-3",
        "inputs": {
            "model_type": "privacybert",
        },
        "dataset_manifest": {
            "policy_overlap": {
                "train_validation": 0,
                "train_test": 0,
                "validation_test": 0,
            }
        },
        "metrics": {
            "validation_macro_f1": 0.71,
            "test_macro_f1": 0.73,
            "validation_accuracy": 0.82,
            "test_accuracy": 0.81,
            "bayesian_primary_score": 0.62,
        },
        "primary_metric_surface": "bayesian_posterior",
        "output_files": {
            "dataset_manifest": "dataset_manifest.json",
            "classifier_metrics": "classifier_metrics.json",
            "validation_predictions": "validation_predictions.jsonl",
            "test_predictions": "test_predictions.jsonl",
            "bayesian_validation": "bayesian_risk_validation.json",
            "bayesian_test": "bayesian_risk_test.json",
        },
    }


def test_phase3_acceptance_passes_for_privacybert_with_bayesian(tmp_path: Path) -> None:
    output_dir = tmp_path / "phase-3"
    manifest = _write_minimum_artifacts(output_dir)

    report = evaluate_phase3_acceptance(
        output_dir=output_dir,
        manifest=manifest,
        require_privacybert=True,
        require_bayesian=True,
    )

    assert report["acceptance"]["passed"] is True


def test_phase3_acceptance_fails_when_privacybert_required_but_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "phase-3"
    manifest = _write_minimum_artifacts(output_dir)
    manifest["inputs"]["model_type"] = "logreg_tfidf"

    report = evaluate_phase3_acceptance(
        output_dir=output_dir,
        manifest=manifest,
        require_privacybert=True,
        require_bayesian=True,
    )

    assert report["acceptance"]["passed"] is False
    failed_names = {c["name"] for c in report["acceptance"]["checks"] if not c["passed"]}
    assert "privacybert_model_required" in failed_names


def test_phase3_acceptance_polisis_advisory_is_non_blocking(tmp_path: Path) -> None:
    output_dir = tmp_path / "phase-3"
    manifest = _write_minimum_artifacts(output_dir)
    manifest["dataset_manifest"]["source"] = "opp115::consolidation-0.75"

    report = evaluate_phase3_acceptance(
        output_dir=output_dir,
        manifest=manifest,
        require_privacybert=True,
        require_bayesian=True,
        require_polisis=False,
        polisis_advisory=True,
    )

    assert report["acceptance"]["passed"] is True
    checks = {c["name"]: c for c in report["acceptance"]["checks"]}
    assert "polisis_source_advisory" in checks
    assert checks["polisis_source_advisory"]["required"] is False
    assert checks["polisis_source_advisory"]["passed"] is False


def test_phase3_acceptance_can_require_polisis_source(tmp_path: Path) -> None:
    output_dir = tmp_path / "phase-3"
    manifest = _write_minimum_artifacts(output_dir)
    manifest["dataset_manifest"]["source"] = "opp115::consolidation-0.75"

    report = evaluate_phase3_acceptance(
        output_dir=output_dir,
        manifest=manifest,
        require_privacybert=True,
        require_bayesian=True,
        require_polisis=True,
        polisis_advisory=False,
    )

    assert report["acceptance"]["passed"] is False
    failed_names = {c["name"] for c in report["acceptance"]["checks"] if c["required"] and not c["passed"]}
    assert "polisis_source_required" in failed_names
