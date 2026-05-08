"""ISO 27001 parser tailored to clause hierarchy and bullet requirement formatting."""

from __future__ import annotations

from pathlib import Path
import re
from typing import List, Tuple

from .docx_reader import read_docx_text
from .schema import ControlRecord, make_normalized_id, normalize_whitespace, stable_hash


CLAUSE_RE = re.compile(r"^((?:[1-9]\d*)(?:\.[0-9]+){0,4}|A\.[0-9]+(?:\.[0-9]+)?)\s+(.+)$")
# A2: accept lower- and upper-case letters, digits, and roman numerals so
# atomic ISO controls aren't merged into their parent clause.
BULLET_RE = re.compile(r"^\s*([A-Za-z]|[ivxIVX]+|\d{1,2})\)\s*;?\s+(.*)$")
ALT_BULLET_RE = re.compile(r"^\s*([A-Za-z]|[ivxIVX]+|\d{1,2})[.:]\s+(.*)$")


def parse_iso_controls(
    path: Path,
    *,
    regulation: str = "ISO27001",
    source_document_id: str = "iso-27001",
) -> List[ControlRecord]:
    text = read_docx_text(path)
    return parse_iso_controls_from_text(
        text,
        source_path=str(path),
        regulation=regulation,
        source_document_id=source_document_id,
    )


def parse_iso_controls_from_text(
    text: str,
    source_path: str = "ISO_27001_Standard-1",
    *,
    regulation: str = "ISO27001",
    source_document_id: str = "iso-27001",
) -> List[ControlRecord]:
    lines = [line.rstrip("\n") for line in text.splitlines()]
    start = _find_body_start(lines)

    records: List[ControlRecord] = []
    active_id: str | None = None
    active_title = ""
    active_body: List[str] = []

    def flush_clause() -> None:
        nonlocal active_id, active_title, active_body
        if active_id is None:
            return

        body_text = "\n".join(active_body)
        clause_records = _clause_to_records(
            source_path=source_path,
            clause_id=active_id,
            clause_title=active_title,
            body_text=body_text,
            regulation=regulation,
            source_document_id=source_document_id,
        )
        records.extend(clause_records)

        active_id = None
        active_title = ""
        active_body = []

    for raw_line in lines[start:]:
        line = _clean_for_match(raw_line)
        if _is_noise_line(line):
            continue

        clause_match = CLAUSE_RE.match(line)
        if clause_match and not _looks_like_toc_entry(line):
            flush_clause()
            active_id = clause_match.group(1)
            active_title = normalize_whitespace(clause_match.group(2))
            continue

        if active_id is not None:
            active_body.append(raw_line)

    flush_clause()
    return _dedupe_records(records)


_BODY_START_RE = re.compile(r"^1[\.\)\s]+Scope\s*$", re.IGNORECASE)


def _find_body_start(lines: List[str]) -> int:
    # A6: tolerate "1 Scope", "1. Scope", "1) Scope" formatting variants.
    start_index = 0
    for idx, raw in enumerate(lines):
        line = _clean_for_match(raw)
        if _looks_like_toc_entry(line):
            continue
        if _BODY_START_RE.match(line):
            start_index = idx
            break
    return start_index


def _clause_to_records(
    *,
    source_path: str,
    clause_id: str,
    clause_title: str,
    body_text: str,
    regulation: str,
    source_document_id: str,
) -> List[ControlRecord]:
    records: List[ControlRecord] = []

    bullet_blocks = _split_bullets(body_text)
    preface_text = bullet_blocks[0][1] if bullet_blocks and bullet_blocks[0][0] == "_preface" else ""

    if preface_text:
        records.append(
            _make_record(
                source_path=source_path,
                native_id=clause_id,
                title=clause_title,
                text=preface_text,
                clause_id=clause_id,
                regulation=regulation,
                source_document_id=source_document_id,
            )
        )

    for bullet_id, bullet_text in bullet_blocks:
        if bullet_id == "_preface":
            continue
        native_id = f"{clause_id}.{bullet_id}"
        records.append(
            _make_record(
                source_path=source_path,
                native_id=native_id,
                title=f"{clause_title} ({bullet_id})",
                text=bullet_text,
                clause_id=clause_id,
                regulation=regulation,
                source_document_id=source_document_id,
            )
        )

    if not records:
        clean_text = normalize_whitespace(body_text)
        if clean_text:
            # A7: surface fallback-record creation so silent extraction
            # failures (no preface and no bullets parsed) become auditable.
            import logging

            logging.getLogger(__name__).debug(
                "iso_parser fallback record for clause %s (regulation=%s) — "
                "bullets did not split; storing whole body as single record",
                clause_id,
                regulation,
            )
            records.append(
                _make_record(
                    source_path=source_path,
                    native_id=clause_id,
                    title=clause_title,
                    text=clean_text,
                    clause_id=clause_id,
                    regulation=regulation,
                    source_document_id=source_document_id,
                )
            )

    return records


def _split_bullets(body_text: str) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    preface_lines: List[str] = []

    active_id: str | None = None
    active_lines: List[str] = []

    for raw in body_text.splitlines():
        line = raw.strip()
        if not line or _is_noise_line(line):
            continue

        bullet_match = BULLET_RE.match(line)
        if not bullet_match:
            bullet_match = ALT_BULLET_RE.match(line)
        if bullet_match:
            if active_id is None:
                preface_text = normalize_whitespace(" ".join(preface_lines))
                if preface_text:
                    entries.append(("_preface", preface_text))
            else:
                entries.append((active_id, normalize_whitespace(" ".join(active_lines))))

            active_id = bullet_match.group(1)
            active_lines = [bullet_match.group(2)]
            continue

        if active_id is None:
            preface_lines.append(line)
        else:
            active_lines.append(line)

    if active_id is not None:
        entries.append((active_id, normalize_whitespace(" ".join(active_lines))))
    elif preface_lines:
        entries.append(("_preface", normalize_whitespace(" ".join(preface_lines))))

    return [(eid, etext) for eid, etext in entries if etext]


def _looks_like_toc_entry(line: str) -> bool:
    return bool(re.search(r"\.{4,}", line))


def _is_noise_line(line: str) -> bool:
    if not line:
        return True

    if line.startswith("© ISO") or line.startswith("ISO/IEC"):
        return True

    if re.fullmatch(r"[ivxlcdm]+", line.lower()):
        return True

    return False


def _clean_for_match(raw_line: str) -> str:
    # Remove invisible control-like chars that often appear in extracted PDF text.
    cleaned = raw_line.replace("\u200b", "").replace("\ufeff", "").replace("\u2060", "")
    cleaned = cleaned.replace("\xa0", " ")
    return cleaned.strip()


def _dedupe_records(records: List[ControlRecord]) -> List[ControlRecord]:
    best_by_id: dict[str, ControlRecord] = {}
    order: List[str] = []

    for record in records:
        key = record.normalized_id
        if key not in best_by_id:
            best_by_id[key] = record
            order.append(key)
            continue

        prev = best_by_id[key]
        # Keep the richer record text when duplicates are encountered.
        if len(record.text) > len(prev.text):
            best_by_id[key] = record

    return [best_by_id[key] for key in order]


def _make_record(
    *,
    source_path: str,
    native_id: str,
    title: str,
    text: str,
    clause_id: str,
    regulation: str,
    source_document_id: str,
) -> ControlRecord:
    first_segment = native_id.split(".", 1)[0]
    normalized_id = make_normalized_id(regulation, native_id)
    # A3: full SHA1 hex digest; truncation caused birthday collisions in Chroma.
    record_id = stable_hash(f"{regulation}:{native_id}:{text}")

    return ControlRecord(
        record_id=record_id,
        regulation=regulation,
        source_document_id=source_document_id,
        source_path=source_path,
        native_id=native_id,
        normalized_id=normalized_id,
        title=title,
        text=text,
        hierarchy_path=[first_segment, clause_id, native_id],
        chapter=first_segment,
        section=clause_id,
        clause=native_id,
        parser_confidence=0.92,
        metadata={
            "format_profile": "iso_clause_bullet",
            "ground_truth_source": True,
        },
    )
