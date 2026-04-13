import csv
import json
from pathlib import Path

from prert.phase3.dataset import build_polisis_clause_examples, map_polisis_category_to_level


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["text", "category", "policy_uid"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_map_polisis_category_to_level_known_and_unknown() -> None:
    assert map_polisis_category_to_level("User Choice/Control") == "user"
    assert map_polisis_category_to_level("Data Security") == "system"
    assert map_polisis_category_to_level("Data Retention") == "organization"
    assert map_polisis_category_to_level("Category Not In Mapping") is None


def test_build_polisis_clause_examples_reads_jsonl_and_csv(tmp_path: Path) -> None:
    polisis_root = tmp_path / "Polisis"
    normalized_dir = polisis_root / "normalized"

    _write_jsonl(
        normalized_dir / "polisis.jsonl",
        [
            {
                "text": "Users can opt out of analytics cookies.",
                "category": "User Choice/Control",
                "policy_uid": "pol-1",
            },
            {
                "text": "Encryption is used for data in transit.",
                "category": "Data Security",
                "policy_uid": "pol-2",
            },
            {
                "text": "Unmapped categories are skipped.",
                "category": "Unmapped Category",
                "policy_uid": "pol-3",
            },
            {
                "text": "Explicit labels are accepted.",
                "label": "organization",
                "policy_uid": "pol-4",
            },
        ],
    )

    _write_csv(
        normalized_dir / "polisis_rows.csv",
        [
            {
                "text": "Retention schedules are documented.",
                "category": "Data Retention",
                "policy_uid": "pol-5",
            },
            {
                "text": "Do not track signals are honored.",
                "category": "Do Not Track",
                "policy_uid": "pol-6",
            },
        ],
    )

    rows = build_polisis_clause_examples(
        polisis_root=polisis_root,
        input_set="normalized",
    )

    assert len(rows) == 5
    assert {row.label for row in rows} == {"user", "system", "organization"}
    assert all(row.source == "polisis_normalized" for row in rows)
    assert all(row.example_id.startswith("polisis::") for row in rows)
    assert "pol-3" not in {row.policy_uid for row in rows}
    assert {row.metadata.get("input_file") for row in rows} == {"polisis.jsonl", "polisis_rows.csv"}
