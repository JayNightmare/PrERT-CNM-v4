"""NIST Privacy Framework parser tailored to PF subcategory identifier formatting."""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
from typing import List

from .schema import ControlRecord, make_normalized_id, normalize_whitespace, stable_hash


SUBCATEGORY_RE = re.compile(r"^\s*([A-Z]{2}\.[A-Z]{2}-P\d+)\s*:\s*(.+)$")


def parse_nist_controls(pdf_path: Path, txt_cache_path: Path | None = None) -> List[ControlRecord]:
    text = extract_text_from_pdf(pdf_path, txt_cache_path=txt_cache_path)
    return parse_nist_controls_from_text(text, source_path=str(pdf_path))


def parse_nist_controls_from_text(text: str, source_path: str = "NIST-1.1") -> List[ControlRecord]:
    lines = [line.rstrip("\n") for line in text.splitlines()]
    records: List[ControlRecord] = []

    active_id: str | None = None
    active_lines: List[str] = []

    def flush_entry() -> None:
        nonlocal active_id, active_lines
        if active_id is None:
            return

        body = normalize_whitespace(" ".join(active_lines))
        if not body:
            active_id = None
            active_lines = []
            return

        family = active_id.split(".", 1)[0]
        category = active_id.split("-", 1)[0]
        normalized_id = make_normalized_id("NISTPF", active_id)
        record_id = stable_hash(f"NISTPF:{active_id}:{body}")[:24]

        metadata = {
            "format_profile": "nist_pf_subcategory",
            "ground_truth_source": True,
            "status": "moved" if body.lower().startswith("moved to ") else "active",
        }

        records.append(
            ControlRecord(
                record_id=record_id,
                regulation="NISTPF",
                source_document_id="nist-pf-1.1",
                source_path=source_path,
                native_id=active_id,
                normalized_id=normalized_id,
                title=active_id,
                text=body,
                hierarchy_path=[family, category, active_id],
                chapter=family,
                section=category,
                clause=active_id,
                parser_confidence=0.9,
                metadata=metadata,
            )
        )

        active_id = None
        active_lines = []

    for raw_line in lines:
        line = raw_line.strip()
        if _is_noise_line(line):
            continue

        match = SUBCATEGORY_RE.match(line)
        if match:
            flush_entry()
            active_id = match.group(1)
            active_lines = [match.group(2)]
            continue

        if active_id is not None:
            active_lines.append(line)

    flush_entry()
    return _dedupe_records(records)


def extract_text_from_pdf(pdf_path: Path, txt_cache_path: Path | None = None) -> str:
    if txt_cache_path and txt_cache_path.exists():
        return txt_cache_path.read_text(encoding="utf-8", errors="ignore")

    if shutil.which("pdftotext"):
        cmd = ["pdftotext", "-layout", str(pdf_path), "-"]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        text = proc.stdout
    else:
        text = _extract_with_pypdf(pdf_path)

    if txt_cache_path is not None:
        txt_cache_path.parent.mkdir(parents=True, exist_ok=True)
        txt_cache_path.write_text(text, encoding="utf-8")

    return text


def _extract_with_pypdf(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "PDF extraction requires pdftotext or pypdf. Install pypdf as fallback."
        ) from exc

    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _is_noise_line(line: str) -> bool:
    if not line:
        return True

    noise_prefixes = (
        "CSWP",
        "NIST Privacy Framework",
        "April",
        "National Institute of Standards",
        "Table of Contents",
    )

    if line.startswith(noise_prefixes):
        return True

    if re.fullmatch(r"[0-9]+", line):
        return True

    return False


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
        prev_status = str(prev.metadata.get("status", "active")).lower()
        curr_status = str(record.metadata.get("status", "active")).lower()

        # Prefer active entries over moved references, then longer informative text.
        if prev_status == "moved" and curr_status != "moved":
            best_by_id[key] = record
            continue
        if curr_status == prev_status and len(record.text) > len(prev.text):
            best_by_id[key] = record

    return [best_by_id[key] for key in order]
