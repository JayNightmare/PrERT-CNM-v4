import json
from pathlib import Path

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
    assert (output_dir / "phase3_manifest.json").exists()


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

    manifest_path = output_dir / "dataset_manifest.json"
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    overlap = dataset_manifest["policy_overlap"]
    assert overlap["train_validation"] == 0
    assert overlap["train_test"] == 0
    assert overlap["validation_test"] == 0
