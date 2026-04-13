from pathlib import Path

from prert.phase3.io import read_json, read_jsonl
from prert.phase4.synthetic import generate_synthetic_policy_schema_dataset


def test_phase4_synthetic_generator_writes_expected_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic"

    manifest = generate_synthetic_policy_schema_dataset(
        output_dir=output_dir,
        counts_by_band={"low": 2, "medium": 3, "high": 4},
        seed=11,
        include_model_signal=False,
        export_upload_fixtures=True,
    )

    dataset_path = Path(manifest["output_files"]["dataset"])
    manifest_path = Path(manifest["output_files"]["manifest"])
    dictionary_path = Path(manifest["output_files"]["dictionary"])

    assert dataset_path.exists()
    assert manifest_path.exists()
    assert dictionary_path.exists()

    rows = read_jsonl(dataset_path)
    assert len(rows) == 9
    assert {row["compliance_band"] for row in rows} == {"low", "medium", "high"}

    manifest_payload = read_json(manifest_path)
    assert manifest_payload["counts_by_band"] == {"low": 2, "medium": 3, "high": 4}
    assert manifest_payload["upload_fixtures"]["enabled"] is True
    assert manifest_payload["upload_fixtures"]["files_written"] == len(rows) * 2


def test_phase4_synthetic_generator_produces_ordered_band_scores(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic"

    manifest = generate_synthetic_policy_schema_dataset(
        output_dir=output_dir,
        counts_by_band={"low": 3, "medium": 3, "high": 3},
        seed=7,
        include_model_signal=False,
        export_upload_fixtures=False,
    )

    summary = manifest["score_summary"]
    assert summary["low"]["mean"] < summary["medium"]["mean"]
    assert summary["medium"]["mean"] < summary["high"]["mean"]

    assert summary["high"]["in_target_band"] >= 1
    assert summary["low"]["in_target_band"] >= 1
