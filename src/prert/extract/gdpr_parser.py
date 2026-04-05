"""GDPR parser tailored to article/sub-clause formatting in the repository TXT file."""

from __future__ import annotations

from pathlib import Path
import re
from typing import List, Tuple

from .schema import ControlRecord, make_normalized_id, normalize_whitespace, stable_hash


ARTICLE_HEADER_RE = re.compile(r"^Article\s+(\d+)\s*$")
CHAPTER_HEADER_RE = re.compile(r"^CHAPTER\s+([IVXLC]+)\s*$")
SUBCLAUSE_RE = re.compile(r"^\s*(\d+)\.\s+(.*)$")


def parse_gdpr_controls(path: Path) -> List[ControlRecord]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return parse_gdpr_controls_from_text(text, source_path=str(path))


def parse_gdpr_controls_from_text(text: str, source_path: str = "GDPR-2016_679") -> List[ControlRecord]:
    lines = [line.rstrip("\n") for line in text.splitlines()]

    records: List[ControlRecord] = []
    chapter_id = ""
    chapter_title = ""

    current_article: str | None = None
    current_title = ""
    body_lines: List[str] = []

    def flush_article() -> None:
        nonlocal current_article, current_title, body_lines
        if current_article is None:
            return

        article_key = f"Article {current_article}"
        body_text = "\n".join(body_lines)
        subclauses = _split_subclauses(body_text)

        if not subclauses:
            clause_id = article_key
            record = _make_record(
                regulation="GDPR",
                source_path=source_path,
                source_document_id="gdpr-2016_679",
                native_id=clause_id,
                title=current_title or article_key,
                text=normalize_whitespace(body_text),
                chapter=chapter_id,
                section=article_key,
                clause=clause_id,
                hierarchy=[chapter_id, chapter_title, article_key],
            )
            records.append(record)
        else:
            for sub_id, sub_text in subclauses:
                clause_id = f"{article_key}.{sub_id}"
                record = _make_record(
                    regulation="GDPR",
                    source_path=source_path,
                    source_document_id="gdpr-2016_679",
                    native_id=clause_id,
                    title=current_title or article_key,
                    text=normalize_whitespace(sub_text),
                    chapter=chapter_id,
                    section=article_key,
                    clause=clause_id,
                    hierarchy=[chapter_id, chapter_title, article_key, f"{sub_id}"],
                )
                records.append(record)

        current_article = None
        current_title = ""
        body_lines = []

    i = 0
    while i < len(lines):
        raw_line = lines[i]
        line = raw_line.strip()

        chapter_match = CHAPTER_HEADER_RE.match(line)
        if chapter_match:
            flush_article()
            chapter_id = f"CHAPTER {chapter_match.group(1)}"
            chapter_title = ""
            j = i + 1
            while j < len(lines):
                probe = lines[j].strip()
                if not probe:
                    j += 1
                    continue
                if CHAPTER_HEADER_RE.match(probe) or ARTICLE_HEADER_RE.match(probe):
                    break
                chapter_title = probe
                break
            i += 1
            continue

        article_match = ARTICLE_HEADER_RE.match(line)
        if article_match:
            flush_article()
            current_article = article_match.group(1)
            current_title = ""

            j = i + 1
            while j < len(lines):
                probe = lines[j].strip()
                if not probe:
                    j += 1
                    continue
                if CHAPTER_HEADER_RE.match(probe) or ARTICLE_HEADER_RE.match(probe):
                    break
                if _is_noise_line(probe):
                    j += 1
                    continue
                current_title = probe
                i = j
                break
            i += 1
            continue

        if current_article is not None and not _is_noise_line(line):
            body_lines.append(raw_line)

        i += 1

    flush_article()
    return records


def _split_subclauses(article_text: str) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    active_id: str | None = None
    active_lines: List[str] = []

    for raw_line in article_text.splitlines():
        line = raw_line.strip()
        if not line or _is_noise_line(line):
            continue

        match = SUBCLAUSE_RE.match(line)
        if match:
            if active_id is not None:
                entries.append((active_id, " ".join(active_lines).strip()))
            active_id = match.group(1)
            active_lines = [match.group(2)]
            continue

        if active_id is not None:
            active_lines.append(line)

    if active_id is not None:
        entries.append((active_id, " ".join(active_lines).strip()))

    return [(cid, ctext) for cid, ctext in entries if ctext]


def _is_noise_line(line: str) -> bool:
    if not line:
        return True

    noise_patterns = (
        "Official Journal of the European Union",
        "EN",
        "L 119/",
        "4.5.2016",
    )

    if any(token in line for token in noise_patterns):
        return True

    if re.fullmatch(r"\(\d+\)", line):
        return True

    return False


def _make_record(
    *,
    regulation: str,
    source_path: str,
    source_document_id: str,
    native_id: str,
    title: str,
    text: str,
    chapter: str,
    section: str,
    clause: str,
    hierarchy: List[str],
) -> ControlRecord:
    normalized_id = make_normalized_id(regulation, native_id)
    record_id = stable_hash(f"{regulation}:{native_id}:{text}")[:24]
    clean_hierarchy = [item for item in hierarchy if item]

    return ControlRecord(
        record_id=record_id,
        regulation=regulation,
        source_document_id=source_document_id,
        source_path=source_path,
        native_id=native_id,
        normalized_id=normalized_id,
        title=title,
        text=text,
        hierarchy_path=clean_hierarchy,
        chapter=chapter,
        section=section,
        clause=clause,
        parser_confidence=0.95,
        metadata={
            "format_profile": "gdpr_article_subclause",
            "ground_truth_source": True,
        },
    )
