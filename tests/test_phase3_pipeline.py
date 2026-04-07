import json
from pathlib import Path

import pytest

from prert.phase3.pipeline import run_phase3_pipeline


def _write_labeled_dataset(path: Path) -> None:
    rows = [
        {"example_id": "e1", "text": "Users can opt out of targeted ads.", "label": "user", "policy_uid": "p1", "category": "User Choice/Control"},
        {"example_id": "e2", "text": "Users may delete their profile information.", "label": "user", "policy_uid": "p1", "category": "User Access, Edit and Deletion"},
        {"example_id": "e3", "text": "Encryption is used for data in transit.", "label": "system", "policy_uid": "p2", "category": "Data Security"},
        {"example_id": "e4", "text": "We honor do not track preferences.", "label": "system", "policy_uid": "p2", "category": "Do Not Track"},
        {"example_id": "e5", "text": "We collect account data to provide services.", "label": "organization", "policy_uid": "p3", "category": "First Party Collection/Use"},
        {"example_id": "e6", "text": "We share limited data with payment providers.", "label": "organization", "policy_uid": "p3", "category": "Third Party Sharing/Collection"},
        {"example_id": "e7", "text": "Users can update notification settings.", "label": "user", "policy_uid": "p4", "category": "User Choice/Control"},
        {"example_id": "e8", "text": "Policy updates are posted on this page.", "label": "organization", "policy_uid": "p4", "category": "Policy Change"},
        {"example_id": "e9", "text": "Security controls are reviewed quarterly.", "label": "system", "policy_uid": "p5", "category": "Data Security"},
        {"example_id": "e10", "text": "Retention schedules are enforced across departments.", "label": "organization", "policy_uid": "p5", "category": "Data Retention"},
        {"example_id": "e11", "text": "Users can request account deletion.", "label": "user", "policy_uid": "p6", "category": "User Access, Edit and Deletion"},
        {"example_id": "e12", "text": "Do not track signals are processed by the platform.", "label": "system", "policy_uid": "p6", "category": "Do Not Track"},
    ]

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_phase3_pipeline_writes_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "labeled.jsonl"
    output_dir = tmp_path / "phase-3"
    _write_labeled_dataset(input_path)

    manifest = run_phase3_pipeline(
        output_dir=output_dir,
        labeled_input_path=input_path,
        seed=7,
    )

    assert manifest["phase"] == "phase-3"
    assert manifest["dataset_manifest"]["total_rows"] == 12

    assert (output_dir / "training_dataset.jsonl").exists()
    assert (output_dir / "validation_dataset.jsonl").exists()
    assert (output_dir / "test_dataset.jsonl").exists()
    assert (output_dir / "dataset_manifest.json").exists()
    assert (output_dir / "classifier_checkpoint/model.json").exists()
    assert (output_dir / "classifier_metrics.json").exists()
    assert (output_dir / "classifier_metrics.jsonl").exists()
    assert (output_dir / "validation_predictions.jsonl").exists()
    assert (output_dir / "test_predictions.jsonl").exists()
    assert (output_dir / "calibration_validation.json").exists()
    assert (output_dir / "calibration_test.json").exists()
    assert (output_dir / "threshold_sweep_validation.json").exists()
    assert (output_dir / "threshold_sweep_test.json").exists()
    assert (output_dir / "bootstrap_ci_validation.json").exists()
    assert (output_dir / "bootstrap_ci_test.json").exists()
    assert (output_dir / "bayesian_risk_validation.json").exists()
    assert (output_dir / "bayesian_risk_test.json").exists()
    assert (output_dir / "phase3_manifest.json").exists()
    assert (output_dir.parent / "phase3_run_history.jsonl").exists()


def test_phase3_metrics_in_range(tmp_path: Path) -> None:
    input_path = tmp_path / "labeled.jsonl"
    output_dir = tmp_path / "phase-3"
    _write_labeled_dataset(input_path)

    run_phase3_pipeline(
        output_dir=output_dir,
        labeled_input_path=input_path,
        seed=11,
    )

    metrics_path = output_dir / "classifier_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    for split_name in ("validation", "test"):
        split_metrics = metrics[split_name]
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            value = float(split_metrics[key])
            assert 0.0 <= value <= 1.0

    assert metrics["bayesian"]["enabled"] is True
    primary_score = float(metrics["bayesian"]["primary_score"])
    assert 0.0 <= primary_score <= 1.0

    manifest_path = output_dir / "dataset_manifest.json"
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    overlap = dataset_manifest["policy_overlap"]
    assert overlap["train_validation"] == 0
    assert overlap["train_test"] == 0
    assert overlap["validation_test"] == 0

    prediction_row = json.loads((output_dir / "test_predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    probabilities = prediction_row["probabilities"]
    assert set(probabilities.keys()) == {"user", "system", "organization"}
    assert abs(sum(float(value) for value in probabilities.values()) - 1.0) < 1e-4


def test_phase3_manifest_includes_model_metadata(tmp_path: Path) -> None:
    input_path = tmp_path / "labeled.jsonl"
    output_dir = tmp_path / "phase-3"
    _write_labeled_dataset(input_path)

    manifest = run_phase3_pipeline(
        output_dir=output_dir,
        labeled_input_path=input_path,
        seed=13,
        model_type="naive_bayes",
    )

    assert manifest["inputs"]["model_type"] == "naive_bayes"
    assert manifest["model_summary"]["model_type"] == "multinomial_naive_bayes"
    assert manifest["model_summary"]["training_config"]["max_features"] == 20000
    assert manifest["primary_metric_surface"] == "bayesian_posterior"
    assert manifest["metrics"]["bayesian_primary_score"] is not None
    assert manifest["metrics"]["calibration_test_ece"] >= 0.0
    assert manifest["metrics"]["calibration_test_macro_ece"] >= 0.0
    assert manifest["execution_metadata"]["run_id"]
    assert manifest["execution_metadata"]["executed_at"].endswith("Z")
    assert manifest["output_files"]["bayesian_validation"] == "bayesian_risk_validation.json"
    assert manifest["output_files"]["bayesian_test"] == "bayesian_risk_test.json"
    assert manifest["output_files"]["calibration_test"] == "calibration_test.json"
    assert manifest["output_files"]["threshold_sweep_test"] == "threshold_sweep_test.json"
    assert manifest["output_files"]["bootstrap_ci_test"] == "bootstrap_ci_test.json"

    history_rows = [
        json.loads(line)
        for line in (output_dir.parent / "phase3_run_history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(row["run_id"] == manifest["execution_metadata"]["run_id"] for row in history_rows)


def test_phase3_logreg_tfidf_pipeline(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")

    input_path = tmp_path / "labeled.jsonl"
    output_dir = tmp_path / "phase-3-logreg"
    _write_labeled_dataset(input_path)

    manifest = run_phase3_pipeline(
        output_dir=output_dir,
        labeled_input_path=input_path,
        seed=17,
        model_type="logreg_tfidf",
        min_df=1,
        max_df=1.0,
        max_features=500,
        max_iter=500,
    )

    assert manifest["inputs"]["model_type"] == "logreg_tfidf"
    assert manifest["model_summary"]["model_type"] == "logreg_tfidf"
    assert manifest["model_summary"]["checkpoint_path"].endswith("model.pkl")
    assert (output_dir / "classifier_checkpoint" / "model.pkl").exists()


def test_phase3_can_disable_bayesian_scoring(tmp_path: Path) -> None:
    input_path = tmp_path / "labeled.jsonl"
    output_dir = tmp_path / "phase-3-no-bayes"
    _write_labeled_dataset(input_path)

    manifest = run_phase3_pipeline(
        output_dir=output_dir,
        labeled_input_path=input_path,
        seed=19,
        enable_bayesian_scoring=False,
    )

    metrics = json.loads((output_dir / "classifier_metrics.json").read_text(encoding="utf-8"))
    assert metrics["bayesian"]["enabled"] is False
    assert metrics["bayesian"]["primary_score"] is None
    assert "measurement_targets" in metrics

    assert not (output_dir / "bayesian_risk_validation.json").exists()
    assert not (output_dir / "bayesian_risk_test.json").exists()
    assert manifest["primary_metric_surface"] == "classifier_metrics"
