"""Dataset preparation and deterministic split logic for Phase 3."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from prert.phase2.opp115 import INPUT_SET_TO_SUBDIR
from prert.phase3.io import read_jsonl
from prert.phase3.types import ClauseExample, SplitSummary


LABELS: Tuple[str, str, str] = ("user", "system", "organization")

CATEGORY_TO_LEVEL = {
    "User Choice/Control": "user",
    "User Access, Edit and Deletion": "user",
    "Data Security": "system",
    "Do Not Track": "system",
    "First Party Collection/Use": "organization",
    "Third Party Sharing/Collection": "organization",
    "Data Retention": "organization",
    "Policy Change": "organization",
    "International and Specific Audiences": "organization",
    "Other": "organization",
}


def load_labeled_examples(path: Path, source: str = "labeled_jsonl") -> List[ClauseExample]:
    rows = read_jsonl(path)
    examples: List[ClauseExample] = []

    for index, row in enumerate(rows):
        text = str(row.get("text", "")).strip()
        label = str(row.get("label", "")).strip().lower()
        policy_uid = str(row.get("policy_uid", "manual")).strip() or "manual"
        category = str(row.get("category", "manual")).strip() or "manual"
        example_id = str(row.get("example_id", f"manual::{index:06d}")).strip() or f"manual::{index:06d}"

        if not text or label not in LABELS:
            continue

        metadata = {
            key: value
            for key, value in row.items()
            if key not in {"example_id", "text", "label", "source", "policy_uid", "category"}
        }

        examples.append(
            ClauseExample(
                example_id=example_id,
                text=text,
                label=label,
                source=source,
                policy_uid=policy_uid,
                category=category,
                metadata=metadata,
            )
        )

    return examples


def build_opp115_clause_examples(
    opp115_root: Path,
    input_set: str = "consolidation-0.75",
    source_dir: Optional[Path] = None,
    max_rows: Optional[int] = None,
) -> List[ClauseExample]:
    annotations_dir = resolve_input_source_dir(opp115_root, input_set, source_dir)
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotation source directory not found: {annotations_dir}")

    examples: List[ClauseExample] = []
    source_name = f"opp115_{input_set}"

    for csv_path in sorted(annotations_dir.glob("*.csv")):
        file_policy_uid = _extract_policy_uid_from_filename(csv_path)
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if len(row) < 7:
                    continue

                category = row[5].strip()
                label = map_category_to_level(category)
                if label is None:
                    continue

                annotation_id = row[0].strip()
                policy_uid = (row[3].strip() if len(row) > 3 else "") or file_policy_uid or "unknown"
                segment_raw = row[4].strip() if len(row) > 4 else ""
                segment_id = _safe_int(segment_raw)

                selected_texts = _extract_selected_texts(row[6])
                if selected_texts:
                    clause_text = " ".join(selected_texts)
                else:
                    clause_text = category

                clause_text = _normalize_space(clause_text)
                if not clause_text:
                    continue

                suffix = annotation_id if annotation_id else f"{len(examples):06d}"
                example_id = f"opp115::{policy_uid}::{suffix}"
                examples.append(
                    ClauseExample(
                        example_id=example_id,
                        text=clause_text,
                        label=label,
                        source=source_name,
                        policy_uid=policy_uid,
                        category=category,
                        metadata={
                            "annotation_id": annotation_id,
                            "segment_id": segment_id,
                            "input_file": csv_path.name,
                        },
                    )
                )

                if max_rows is not None and len(examples) >= max_rows:
                    return examples

    return examples


def split_examples_by_policy(
    examples: Sequence[ClauseExample],
    seed: int = 42,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> Dict[str, List[ClauseExample]]:
    if not examples:
        return {"train": [], "validation": [], "test": []}

    by_policy: Dict[str, List[ClauseExample]] = defaultdict(list)
    for example in examples:
        by_policy[example.policy_uid].append(example)

    policy_uids = list(by_policy.keys())
    rnd = random.Random(seed)
    rnd.shuffle(policy_uids)

    total_policies = len(policy_uids)
    train_count = max(1, int(total_policies * train_ratio))
    validation_count = max(1, int(total_policies * validation_ratio)) if total_policies >= 3 else 0

    if train_count + validation_count >= total_policies:
        validation_count = max(0, total_policies - train_count - 1)

    train_ids = set(policy_uids[:train_count])
    validation_ids = set(policy_uids[train_count : train_count + validation_count])
    test_ids = set(policy_uids[train_count + validation_count :])

    if not test_ids and validation_ids:
        moved = next(iter(validation_ids))
        validation_ids.remove(moved)
        test_ids.add(moved)

    splits = {
        "train": _collect_split_examples(by_policy, train_ids),
        "validation": _collect_split_examples(by_policy, validation_ids),
        "test": _collect_split_examples(by_policy, test_ids),
    }

    _ensure_label_coverage(splits, by_policy, minimum_train_policies=2)
    return splits


def build_dataset_manifest(
    splits: Mapping[str, Sequence[ClauseExample]],
    seed: int,
    source: str,
    input_set: str,
) -> Dict[str, Any]:
    split_summaries = {
        split: _split_summary(split, rows).as_dict()
        for split, rows in splits.items()
    }

    overlap_report = _policy_overlap_report(splits)

    total_rows = sum(summary["rows"] for summary in split_summaries.values())
    global_distribution = Counter()
    for rows in splits.values():
        global_distribution.update(example.label for example in rows)

    return {
        "phase": "phase-3",
        "seed": seed,
        "source": source,
        "input_set": input_set,
        "total_rows": total_rows,
        "class_distribution": dict(sorted(global_distribution.items())),
        "splits": split_summaries,
        "policy_overlap": overlap_report,
        "labels": list(LABELS),
    }


def resolve_input_source_dir(opp115_root: Path, input_set: str, source_dir: Optional[Path]) -> Path:
    if source_dir is not None:
        return source_dir

    if input_set not in INPUT_SET_TO_SUBDIR:
        choices = ", ".join(sorted(INPUT_SET_TO_SUBDIR))
        raise ValueError(f"Unsupported input_set '{input_set}'. Choose one of: {choices}")

    return opp115_root / INPUT_SET_TO_SUBDIR[input_set]


def map_category_to_level(category: str) -> Optional[str]:
    normalized = category.strip()
    if not normalized:
        return None

    if normalized in CATEGORY_TO_LEVEL:
        return CATEGORY_TO_LEVEL[normalized]

    lowered = normalized.lower()
    if "user" in lowered:
        return "user"
    if "security" in lowered or "track" in lowered:
        return "system"
    return "organization"


def _extract_selected_texts(payload: str) -> List[str]:
    payload = payload.strip()
    if not payload:
        return []

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return []

    texts: List[str] = []
    _walk_selected_text(parsed, texts)

    deduped: List[str] = []
    seen: Set[str] = set()
    for value in texts:
        normalized = _normalize_space(value)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _walk_selected_text(node: Any, out: List[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "selectedText" and isinstance(value, str):
                out.append(value)
            _walk_selected_text(value, out)
        return

    if isinstance(node, list):
        for value in node:
            _walk_selected_text(value, out)


def _extract_policy_uid_from_filename(csv_path: Path) -> Optional[str]:
    stem = csv_path.stem
    if not stem:
        return None
    if "_" in stem:
        return stem.split("_", 1)[0].strip() or None
    return stem.strip() or None


def _collect_split_examples(by_policy: Dict[str, List[ClauseExample]], policy_ids: Set[str]) -> List[ClauseExample]:
    rows: List[ClauseExample] = []
    for policy_uid in sorted(policy_ids):
        rows.extend(by_policy.get(policy_uid, []))
    return rows


def _policy_overlap_report(splits: Mapping[str, Sequence[ClauseExample]]) -> Dict[str, int]:
    split_ids: Dict[str, Set[str]] = {
        split: {row.policy_uid for row in rows}
        for split, rows in splits.items()
    }
    return {
        "train_validation": len(split_ids["train"].intersection(split_ids["validation"])),
        "train_test": len(split_ids["train"].intersection(split_ids["test"])),
        "validation_test": len(split_ids["validation"].intersection(split_ids["test"])),
    }


def _split_summary(split: str, rows: Sequence[ClauseExample]) -> SplitSummary:
    class_distribution = Counter(row.label for row in rows)
    policy_uids = {row.policy_uid for row in rows}
    return SplitSummary(
        split=split,
        rows=len(rows),
        unique_policies=len(policy_uids),
        class_distribution=dict(sorted(class_distribution.items())),
    )


def _ensure_label_coverage(
    splits: Dict[str, List[ClauseExample]],
    by_policy: Dict[str, List[ClauseExample]],
    minimum_train_policies: int,
) -> None:
    train_policies = {row.policy_uid for row in splits["train"]}

    for split_name in ("validation", "test"):
        split_labels = {row.label for row in splits[split_name]}
        missing_labels = [label for label in LABELS if label not in split_labels]
        for missing_label in missing_labels:
            candidate_policy = _find_policy_with_label(
                candidate_policies=train_policies,
                by_policy=by_policy,
                target_label=missing_label,
            )
            if candidate_policy is None:
                continue
            if len(train_policies) <= minimum_train_policies:
                continue

            moved_rows = by_policy[candidate_policy]
            splits["train"] = [row for row in splits["train"] if row.policy_uid != candidate_policy]
            splits[split_name].extend(moved_rows)
            train_policies.remove(candidate_policy)


def _find_policy_with_label(
    candidate_policies: Set[str],
    by_policy: Dict[str, List[ClauseExample]],
    target_label: str,
) -> Optional[str]:
    for policy_uid in sorted(candidate_policies):
        rows = by_policy.get(policy_uid, [])
        if any(row.label == target_label for row in rows):
            return policy_uid
    return None


def _safe_int(raw: str) -> Optional[int]:
    text = str(raw).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _normalize_space(text: str) -> str:
    return " ".join(text.split())
