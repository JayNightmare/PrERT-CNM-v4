import csv
import json
from pathlib import Path

from prert.phase2.opp115 import build_opp115_public_rows, run_opp115_processing


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow(row)


def test_build_opp115_public_rows_from_annotations(tmp_path: Path) -> None:
    opp115_root = tmp_path / "OPP-115"

    _write_csv(
        opp115_root / "documentation" / "policies_opp115.csv",
        [
            ["Policy UID", "Policy URL", "Policy collection date", "Policy last updated date"],
            ["105", "https://example.com/policy", "2016-02-08", "2016-02-07"],
        ],
    )
    _write_csv(
        opp115_root / "documentation" / "websites_opp115.csv",
        [
            [
                "Site UID",
                "Site URL",
                "Site Human-Readable Name",
                "Policy UID",
                "Site Check Date",
                "In 115 Set?",
                "Comments",
                "Sectoral Data",
            ],
            [
                "1",
                "example.com",
                "Example",
                "105",
                "2016-02-08",
                "Yes",
                "",
                "Business: Retail Trade: Retailers",
            ],
        ],
    )
    _write_csv(
        opp115_root / "annotations" / "105_example.com.csv",
        [
            [
                "A1",
                "batch",
                "121",
                "105",
                "0",
                "First Party Collection/Use",
                "{}",
                "2/8/16",
                "https://example.com/policy",
            ],
            [
                "A2",
                "batch",
                "121",
                "105",
                "1",
                "Data Security",
                "{}",
                "https://example.com/policy",
                "2/8/16",
            ],
        ],
    )

    rows = build_opp115_public_rows(opp115_root=opp115_root, input_set="annotations")
    assert len(rows) == 1

    row = rows[0]
    assert row["event_date"] == "2016-02-08"
    assert row["sector"] == "Business"
    assert row["records_affected"] == 2
    assert row["detection_to_response_hours"] == 24.0
    assert row["country"] == "US"


def test_run_opp115_processing_writes_outputs(tmp_path: Path) -> None:
    opp115_root = tmp_path / "OPP-115"

    _write_csv(
        opp115_root / "documentation" / "policies_opp115.csv",
        [
            ["Policy UID", "Policy URL", "Policy collection date", "Policy last updated date"],
            ["200", "https://example.org/policy", "2016-02-08", ""],
        ],
    )
    _write_csv(
        opp115_root / "documentation" / "websites_opp115.csv",
        [
            [
                "Site UID",
                "Site URL",
                "Site Human-Readable Name",
                "Policy UID",
                "Site Check Date",
                "In 115 Set?",
                "Comments",
                "Sectoral Data",
            ],
            ["2", "example.org", "Org", "200", "2016-02-08", "Yes", "", "News: Newspapers"],
        ],
    )
    _write_csv(
        opp115_root / "annotations" / "200_example.org.csv",
        [
            [
                "B1",
                "batch",
                "121",
                "200",
                "0",
                "Other",
                "{}",
                "2/8/16",
                "https://example.org/policy",
            ]
        ],
    )

    output_csv = tmp_path / "processed" / "opp115_public_mapping.csv"
    output_jsonl = tmp_path / "processed" / "opp115_public_mapping.jsonl"

    summary = run_opp115_processing(
        opp115_root=opp115_root,
        output_csv=output_csv,
        output_jsonl=output_jsonl,
        input_set="annotations",
    )

    assert summary["rows"] == 1
    assert output_csv.exists()
    assert output_jsonl.exists()

    with output_jsonl.open("r", encoding="utf-8") as handle:
        line = handle.readline().strip()
    parsed = json.loads(line)
    assert parsed["policy_uid"] == "200"
    assert parsed["sector"] == "News"