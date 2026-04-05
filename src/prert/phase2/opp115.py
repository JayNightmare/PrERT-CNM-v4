"""OPP-115 preprocessing helpers for Phase 2 public mapping input."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


INPUT_SET_TO_SUBDIR = {
    "annotations": "annotations",
    "consolidation-0.5": "consolidation/threshold-0.5-overlap-similarity",
    "consolidation-0.75": "consolidation/threshold-0.75-overlap-similarity",
    "consolidation-1.0": "consolidation/threshold-1.0-overlap-similarity",
}

CSV_FIELD_ORDER = [
    "event_date",
    "country",
    "sector",
    "records_affected",
    "detection_to_response_hours",
    "severity",
    "source",
    "policy_uid",
    "policy_url",
    "site_url",
    "site_name",
    "site_check_date",
    "annotation_rows",
    "unique_annotation_ids",
    "unique_segments",
    "unique_categories",
]


@dataclass
class _PolicyAggregate:
    annotation_rows: int = 0
    annotation_ids: set[str] = field(default_factory=set)
    segments: set[int] = field(default_factory=set)
    categories: Counter[str] = field(default_factory=Counter)
    dates: set[str] = field(default_factory=set)
    urls: set[str] = field(default_factory=set)


def run_opp115_processing(
    opp115_root: Path,
    output_csv: Path,
    output_jsonl: Path,
    input_set: str = "consolidation-0.75",
    source_dir: Optional[Path] = None,
    country: str = "US",
) -> Dict[str, Any]:
    rows = build_opp115_public_rows(
        opp115_root=opp115_root,
        input_set=input_set,
        source_dir=source_dir,
        country=country,
    )
    write_opp115_public_mapping(rows, output_csv=output_csv, output_jsonl=output_jsonl)

    return {
        "rows": len(rows),
        "output_csv": str(output_csv),
        "output_jsonl": str(output_jsonl),
        "input_set": input_set,
        "source_dir": str(resolve_input_source_dir(opp115_root, input_set, source_dir)),
    }


def build_opp115_public_rows(
    opp115_root: Path,
    input_set: str = "consolidation-0.75",
    source_dir: Optional[Path] = None,
    country: str = "US",
) -> List[Dict[str, Any]]:
    annotations_dir = resolve_input_source_dir(opp115_root, input_set, source_dir)
    policy_meta = _load_policy_metadata(opp115_root / "documentation" / "policies_opp115.csv")
    site_meta = _load_site_metadata(opp115_root / "documentation" / "websites_opp115.csv")
    aggregates = _aggregate_annotations(annotations_dir)

    policy_uids = set(policy_meta.keys()) | set(aggregates.keys()) | set(site_meta.keys())
    rows: List[Dict[str, Any]] = []

    for policy_uid in sorted(policy_uids, key=_policy_sort_key):
        policy = policy_meta.get(policy_uid, {})
        site = site_meta.get(policy_uid, {})
        aggregate = aggregates.get(policy_uid, _PolicyAggregate())

        collection_date = _normalize_date(policy.get("collection_date"))
        fallback_annotation_date = _first_date(aggregate.dates)
        event_date = collection_date or fallback_annotation_date

        sector = site.get("sector") or "Unknown"
        records_affected = aggregate.annotation_rows
        detection_to_response_hours = _compute_collection_lag_hours(
            policy.get("collection_date"),
            policy.get("last_updated_date"),
        )

        row = {
            "event_date": event_date,
            "country": country,
            "sector": sector,
            "records_affected": records_affected,
            "detection_to_response_hours": detection_to_response_hours,
            "severity": _derive_severity(records_affected, len(aggregate.categories)),
            "source": f"opp115_{input_set}",
            "policy_uid": policy_uid,
            "policy_url": policy.get("policy_url") or _first_sorted(aggregate.urls),
            "site_url": site.get("site_url"),
            "site_name": site.get("site_name"),
            "site_check_date": _normalize_date(site.get("site_check_date")),
            "annotation_rows": aggregate.annotation_rows,
            "unique_annotation_ids": len(aggregate.annotation_ids),
            "unique_segments": len(aggregate.segments),
            "unique_categories": len(aggregate.categories),
        }
        rows.append(row)

    return rows


def write_opp115_public_mapping(
    rows: Iterable[Dict[str, Any]],
    output_csv: Path,
    output_jsonl: Path,
) -> None:
    rows_list = list(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELD_ORDER)
        writer.writeheader()
        for row in rows_list:
            writer.writerow({field: row.get(field) for field in CSV_FIELD_ORDER})

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows_list:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_input_source_dir(opp115_root: Path, input_set: str, source_dir: Optional[Path]) -> Path:
    if source_dir is not None:
        return source_dir

    if input_set not in INPUT_SET_TO_SUBDIR:
        choices = ", ".join(sorted(INPUT_SET_TO_SUBDIR))
        raise ValueError(f"Unsupported input_set '{input_set}'. Choose one of: {choices}")

    return opp115_root / INPUT_SET_TO_SUBDIR[input_set]


def _aggregate_annotations(annotations_dir: Path) -> Dict[str, _PolicyAggregate]:
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotation source directory not found: {annotations_dir}")

    aggregates: Dict[str, _PolicyAggregate] = defaultdict(_PolicyAggregate)

    for csv_path in sorted(annotations_dir.glob("*.csv")):
        file_policy_uid = _extract_policy_uid_from_filename(csv_path)
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row or len(row) < 6:
                    continue

                policy_uid = file_policy_uid or row[3].strip()
                if not policy_uid:
                    continue

                aggregate = aggregates[policy_uid]
                aggregate.annotation_rows += 1
                aggregate.annotation_ids.add(row[0].strip())

                segment_id = _safe_int(row[4])
                if segment_id is not None:
                    aggregate.segments.add(segment_id)

                category = row[5].strip()
                if category:
                    aggregate.categories[category] += 1

                annotation_date, policy_url = _extract_date_and_url(row)
                if annotation_date:
                    aggregate.dates.add(annotation_date)
                if policy_url:
                    aggregate.urls.add(policy_url)

    return aggregates


def _extract_policy_uid_from_filename(csv_path: Path) -> Optional[str]:
    stem = csv_path.stem
    if not stem:
        return None
    if "_" in stem:
        return stem.split("_", 1)[0].strip() or None
    return stem.strip() or None


def _extract_date_and_url(row: List[str]) -> Tuple[Optional[str], Optional[str]]:
    col_7 = row[7].strip() if len(row) > 7 else ""
    col_8 = row[8].strip() if len(row) > 8 else ""

    url = None
    date_candidate = None

    if col_7.startswith("http"):
        url = col_7
        date_candidate = col_8
    elif col_8.startswith("http"):
        url = col_8
        date_candidate = col_7
    else:
        date_candidate = col_7 or col_8

    return _normalize_date(date_candidate), url


def _load_policy_metadata(path: Path) -> Dict[str, Dict[str, Optional[str]]]:
    metadata: Dict[str, Dict[str, Optional[str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            policy_uid = row[0].strip()
            if not policy_uid:
                continue

            metadata[policy_uid] = {
                "policy_url": row[1].strip() or None,
                "collection_date": row[2].strip() or None,
                "last_updated_date": row[3].strip() or None,
            }
    return metadata


def _load_site_metadata(path: Path) -> Dict[str, Dict[str, Optional[str]]]:
    metadata: Dict[str, Dict[str, Optional[str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 5:
                continue

            policy_uid = row[3].strip()
            if not policy_uid:
                continue

            sector_values = [value.strip() for value in row[7:] if value.strip()]
            metadata[policy_uid] = {
                "site_url": row[1].strip() or None,
                "site_name": row[2].strip() or None,
                "site_check_date": row[4].strip() or None,
                "sector": _infer_sector(sector_values),
            }

    return metadata


def _infer_sector(sector_values: List[str]) -> Optional[str]:
    if not sector_values:
        return None

    top_levels: List[str] = []
    for value in sector_values:
        top_level = value.split(":", 1)[0].strip()
        if not top_level:
            continue
        top_levels.append(top_level)

    if not top_levels:
        return None

    top_counts = Counter(top_levels)
    max_count = max(top_counts.values())
    most_common = sorted(name for name, count in top_counts.items() if count == max_count)[0]
    return most_common


def _normalize_date(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    parsed = _parse_date(raw)
    if parsed is None:
        return None
    return parsed.isoformat()


def _parse_date(raw: str) -> Optional[date]:
    value = raw.strip()
    if not value:
        return None

    patterns = ["%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"]
    for pattern in patterns:
        try:
            return datetime.strptime(value, pattern).date()
        except ValueError:
            continue
    return None


def _compute_collection_lag_hours(collection_date: Optional[str], last_updated_date: Optional[str]) -> Optional[float]:
    collected = _parse_date(collection_date or "")
    updated = _parse_date(last_updated_date or "")
    if collected is None or updated is None:
        return None

    delta_days = (collected - updated).days
    if delta_days < 0:
        return None

    return float(delta_days * 24)


def _derive_severity(annotation_rows: int, category_count: int) -> str:
    weighted_score = annotation_rows + (category_count * 10)
    if weighted_score >= 700:
        return "high"
    if weighted_score >= 250:
        return "medium"
    return "low"


def _first_date(values: set[str]) -> Optional[str]:
    if not values:
        return None
    return sorted(values)[0]


def _first_sorted(values: set[str]) -> Optional[str]:
    if not values:
        return None
    return sorted(values)[0]


def _safe_int(raw: str) -> Optional[int]:
    text = str(raw).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _policy_sort_key(policy_uid: str) -> Tuple[int, str]:
    uid_int = _safe_int(policy_uid)
    if uid_int is None:
        return (1, policy_uid)
    return (0, f"{uid_int:08d}")