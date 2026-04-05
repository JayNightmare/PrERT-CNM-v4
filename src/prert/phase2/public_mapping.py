"""Public breach dataset loading and mapping for Phase 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from prert.phase2.io import read_csv, read_jsonl


FIELD_CANDIDATES = {
    "event_date": ["event_date", "date", "incident_date", "disclosure_date"],
    "country": ["country", "jurisdiction", "location"],
    "sector": ["sector", "industry"],
    "records_affected": ["records_affected", "affected_records", "records", "impacted_records"],
    "detection_to_response_hours": [
        "detection_to_response_hours",
        "response_hours",
        "time_to_response_hours",
    ],
    "severity": ["severity", "impact", "criticality"],
}

REQUIRED_FIELDS = ["event_date", "sector", "records_affected"]


def load_public_rows(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None:
        return []
    if not path.exists():
        return []

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".csv":
        return read_csv(path)

    return []


def map_public_rows(rows: List[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
    mapped_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        mapped: Dict[str, Any] = {
            "source": source_name,
            "source_row_index": idx,
        }

        for target_key, candidates in FIELD_CANDIDATES.items():
            value = _pick_value(row, candidates)
            mapped[target_key] = value

        mapped["records_affected"] = _to_int(mapped.get("records_affected"))
        mapped["detection_to_response_hours"] = _to_float(mapped.get("detection_to_response_hours"))

        missing_required = [
            field_name
            for field_name in REQUIRED_FIELDS
            if mapped.get(field_name) in (None, "")
        ]

        mapped["dq_missing_required_fields"] = missing_required
        mapped["dq_valid"] = len(missing_required) == 0

        mapped_rows.append(mapped)

    return mapped_rows


def summarize_public_mapping(mapped_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_rows = sum(1 for row in mapped_rows if row.get("dq_valid"))
    return {
        "input_rows": len(mapped_rows),
        "valid_rows": valid_rows,
        "invalid_rows": len(mapped_rows) - valid_rows,
        "required_fields": REQUIRED_FIELDS,
    }


def _pick_value(row: Dict[str, Any], candidates: List[str]) -> Any:
    row_lc = {str(k).lower(): v for k, v in row.items()}
    for candidate in candidates:
        if candidate.lower() in row_lc:
            return row_lc[candidate.lower()]
    return None


def _to_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(str(value).replace(",", "").strip())
    except ValueError:
        return None


def _to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None
