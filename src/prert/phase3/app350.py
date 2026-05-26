"""APP-350 preprocessing helpers for conservative Phase 3 auxiliary training."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Tuple
import zipfile

from prert.phase3.io import write_json, write_jsonl


SYSTEM_PRACTICE_EXACT = {
    "SSO",
    "Facebook_SSO",
    "Contact_Password_1stParty",
    "Contact_Password_3rdParty",
}

SYSTEM_PRACTICE_PREFIXES = (
    "Identifier_Cookie_or_similar_Tech",
    "Identifier_IP_Address",
    "Identifier_Device_ID",
    "Identifier_Ad_ID",
    "Identifier_MAC",
    "Identifier_IMEI",
    "Identifier_IMSI",
    "Identifier_SIM_Serial",
    "Identifier_SSID_BSSID",
)

ORGANIZATION_PRACTICE_PREFIXES = (
    "Contact_",
    "Demographic_",
    "Location_",
    "Identifier_1stParty",
    "Identifier_3rdParty",
)

USER_PRACTICE_TOKENS = (
    "choice",
    "consent",
    "delete",
    "deletion",
    "access",
    "edit",
    "opt_out",
    "opt-out",
    "optout",
    "rights",
    "request",
)


def run_app350_processing(
    input_path: Path,
    output_jsonl: Path,
    output_manifest: Path,
    include_synthetic: bool = False,
) -> Dict[str, Any]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    retained_rows: List[Dict[str, Any]] = []
    retained_by_label: Counter[str] = Counter()
    retained_by_practice: Counter[str] = Counter()
    dropped_sentences: Counter[str] = Counter()
    annotation_stats: Counter[str] = Counter()
    policy_types: Counter[str] = Counter()

    policies_seen = 0
    policies_used = 0
    synthetic_policies_skipped = 0

    for input_member, document in _iter_app350_documents(input_path):
        policies_seen += 1
        policy_type = str(document.get("policy_type", "unknown")).strip() or "unknown"
        policy_types[policy_type] += 1

        contains_synthetic = bool(document.get("contains_synthetic", False))
        if contains_synthetic and not include_synthetic:
            synthetic_policies_skipped += 1
            continue

        policy_rows, policy_summary = _extract_policy_rows(document, input_member)
        policies_used += 1
        retained_rows.extend(policy_rows)
        retained_by_label.update(policy_summary["retained_by_label"])
        retained_by_practice.update(policy_summary["retained_by_practice"])
        dropped_sentences.update(policy_summary["dropped_sentences"])
        annotation_stats.update(policy_summary["annotation_stats"])

    write_jsonl(output_jsonl, retained_rows)

    summary = {
        "source": "app350",
        "input_path": str(input_path),
        "include_synthetic": include_synthetic,
        "mapping_policy": "conservative-v1-performed-only-sentence-level",
        "policies_seen": policies_seen,
        "policies_used": policies_used,
        "synthetic_policies_skipped": synthetic_policies_skipped,
        "rows_written": len(retained_rows),
        "retained_by_label": dict(sorted(retained_by_label.items())),
        "retained_by_practice": dict(retained_by_practice.most_common()),
        "dropped_sentences": dict(sorted(dropped_sentences.items())),
        "annotation_stats": dict(sorted(annotation_stats.items())),
        "policy_types": dict(sorted(policy_types.items())),
        "output_jsonl": str(output_jsonl),
    }
    write_json(output_manifest, summary)
    return summary


def map_app350_practice_to_level(practice: str) -> str | None:
    normalized = practice.strip()
    if not normalized:
        return None

    lowered = normalized.lower()
    if any(token in lowered for token in USER_PRACTICE_TOKENS):
        return "user"
    if normalized in SYSTEM_PRACTICE_EXACT:
        return "system"
    if normalized.startswith(SYSTEM_PRACTICE_PREFIXES):
        return "system"
    if normalized.startswith(ORGANIZATION_PRACTICE_PREFIXES):
        return "organization"
    return None


def _extract_policy_rows(document: Mapping[str, Any], input_member: str) -> Tuple[List[Dict[str, Any]], Dict[str, Counter[str]]]:
    policy_id = str(document.get("policy_id", "unknown")).strip() or "unknown"
    policy_uid = f"app350::{policy_id}"
    policy_name = str(document.get("policy_name", "")).strip()
    policy_type = str(document.get("policy_type", "unknown")).strip() or "unknown"
    contains_synthetic = bool(document.get("contains_synthetic", False))

    rows: List[Dict[str, Any]] = []
    retained_by_label: Counter[str] = Counter()
    retained_by_practice: Counter[str] = Counter()
    dropped_sentences: Counter[str] = Counter()
    annotation_stats: Counter[str] = Counter()

    for segment in document.get("segments", []) or []:
        segment_id = int(segment.get("segment_id", -1))
        for sentence_index, sentence in enumerate(segment.get("sentences", []) or []):
            sentence_text = _normalise_space(str(sentence.get("sentence_text", "")))
            if not sentence_text:
                dropped_sentences["empty_sentence"] += 1
                continue

            annotations = sentence.get("annotations", []) or []
            if not annotations:
                dropped_sentences["no_sentence_annotations"] += 1
                continue

            mapped_labels = set()
            kept_practices: List[str] = []

            for annotation in annotations:
                annotation_stats["total_annotations"] += 1
                practice = str(annotation.get("practice", "")).strip()
                modality = str(annotation.get("modality", "")).strip().upper()
                if modality != "PERFORMED":
                    annotation_stats["not_performed_annotations"] += 1
                    continue

                annotation_stats["performed_annotations"] += 1
                label = map_app350_practice_to_level(practice)
                if label is None:
                    annotation_stats["unmapped_performed_annotations"] += 1
                    continue

                annotation_stats["mapped_performed_annotations"] += 1
                mapped_labels.add(label)
                kept_practices.append(practice)

            if not kept_practices:
                dropped_sentences["no_mapped_performed_annotations"] += 1
                continue

            if len(mapped_labels) != 1:
                dropped_sentences["ambiguous_multi_label_sentence"] += 1
                continue

            label = next(iter(mapped_labels))
            unique_practices = sorted(set(kept_practices))
            retained_by_label[label] += 1
            retained_by_practice.update(unique_practices)
            rows.append(
                {
                    "example_id": f"app350::{policy_id}::{segment_id}::{sentence_index:04d}",
                    "text": sentence_text,
                    "label": label,
                    "policy_uid": policy_uid,
                    "category": unique_practices[0],
                    "source": "app350",
                    "policy_name": policy_name,
                    "policy_type": policy_type,
                    "input_file": input_member,
                    "contains_synthetic": contains_synthetic,
                    "app350_practices": unique_practices,
                    "segment_id": segment_id,
                    "sentence_index": sentence_index,
                }
            )

    return rows, {
        "retained_by_label": retained_by_label,
        "retained_by_practice": retained_by_practice,
        "dropped_sentences": dropped_sentences,
        "annotation_stats": annotation_stats,
    }


def _iter_app350_documents(input_path: Path) -> Iterator[Tuple[str, Mapping[str, Any]]]:
    yaml = _load_yaml_module()

    if input_path.is_dir():
        annotations_dir = _resolve_annotations_dir(input_path)
        for yaml_path in sorted(annotations_dir.glob("*.yml")):
            with yaml_path.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle)
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid APP-350 annotation payload: {yaml_path}")
            yield yaml_path.name, payload
        return

    if input_path.suffix.lower() == ".zip" and input_path.exists():
        with zipfile.ZipFile(input_path) as archive:
            for member in sorted(archive.namelist()):
                if not _is_annotation_member(member):
                    continue
                payload = yaml.safe_load(archive.read(member).decode("utf-8", "replace"))
                if not isinstance(payload, dict):
                    raise ValueError(f"Invalid APP-350 annotation payload: {member}")
                yield member, payload
        return

    raise FileNotFoundError(f"APP-350 input path not found or unsupported: {input_path}")


def _resolve_annotations_dir(input_path: Path) -> Path:
    candidates = [
        input_path / "annotations",
        input_path / "APP-350_v1.1" / "annotations",
        input_path,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir() and list(candidate.glob("*.yml")):
            return candidate
    raise FileNotFoundError(f"APP-350 annotations directory not found under: {input_path}")


def _is_annotation_member(member: str) -> bool:
    normalized = member.replace("\\", "/")
    return normalized.endswith(".yml") and "/annotations/" in normalized and not normalized.startswith("__MACOSX/")


def _load_yaml_module() -> Any:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "APP-350 preprocessing requires PyYAML. Install project dependencies and re-run."
        ) from exc
    return yaml


def _normalise_space(text: str) -> str:
    return " ".join(text.split())