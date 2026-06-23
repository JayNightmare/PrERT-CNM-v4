import json
import zipfile
from pathlib import Path

from prert.phase3.app350 import run_app350_processing


def test_run_app350_processing_reads_zip_and_writes_auxiliary_rows(tmp_path: Path) -> None:
    input_zip = tmp_path / "APP-350_v1.1.zip"
    output_jsonl = tmp_path / "processed" / "app350_phase3_auxiliary.jsonl"
    output_manifest = tmp_path / "processed" / "app350_phase3_auxiliary_manifest.json"

    policy_real = """
policy_id: 1
policy_name: Example App
policy_type: TRAIN
contains_synthetic: false
segments:
  - segment_id: 0
    segment_text: Example
    sentences:
      - sentence_text: We use cookies for authentication.
        annotations:
          - practice: Identifier_Cookie_or_similar_Tech_1stParty
            modality: PERFORMED
      - sentence_text: You may delete your account.
        annotations:
          - practice: User_Access_Delete
            modality: PERFORMED
      - sentence_text: We collect your email address.
        annotations:
          - practice: Contact_E_Mail_Address_1stParty
            modality: PERFORMED
      - sentence_text: We use cookies and collect your email address.
        annotations:
          - practice: Identifier_Cookie_or_similar_Tech_1stParty
            modality: PERFORMED
          - practice: Contact_E_Mail_Address_1stParty
            modality: PERFORMED
      - sentence_text: We do not collect your GPS location.
        annotations:
          - practice: Location_GPS_1stParty
            modality: NOT_PERFORMED
""".strip()

    policy_synthetic = """
policy_id: 2
policy_name: Synthetic App
policy_type: VALIDATION
contains_synthetic: true
segments:
  - segment_id: 0
    segment_text: Example
    sentences:
      - sentence_text: We share your email address with partners.
        annotations:
          - practice: Contact_E_Mail_Address_3rdParty
            modality: PERFORMED
""".strip()

    with zipfile.ZipFile(input_zip, "w") as archive:
        archive.writestr("APP-350_v1.1/annotations/policy_1.yml", policy_real)
        archive.writestr("APP-350_v1.1/annotations/policy_2.yml", policy_synthetic)

    summary = run_app350_processing(
        input_path=input_zip,
        output_jsonl=output_jsonl,
        output_manifest=output_manifest,
    )

    rows = [
        json.loads(line)
        for line in output_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads(output_manifest.read_text(encoding="utf-8"))

    assert summary["rows_written"] == 3
    assert summary["policies_seen"] == 2
    assert summary["policies_used"] == 1
    assert summary["synthetic_policies_skipped"] == 1
    assert {row["label"] for row in rows} == {"user", "system", "organization"}
    assert all(row["policy_uid"] == "app350::1" for row in rows)
    assert manifest["retained_by_label"] == {"organization": 1, "system": 1, "user": 1}
    assert manifest["dropped_sentences"]["ambiguous_multi_label_sentence"] == 1
    assert manifest["annotation_stats"]["not_performed_annotations"] == 1