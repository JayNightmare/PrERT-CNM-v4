"""Line-based chunking with 16 KiB document safety for Chroma records."""

from __future__ import annotations

from typing import Iterable, List

from prert.extract.schema import ControlChunk, ControlRecord, stable_hash


MAX_DOCUMENT_BYTES = 16 * 1024


def chunk_records(
    records: Iterable[ControlRecord],
    *,
    max_document_bytes: int = MAX_DOCUMENT_BYTES,
    max_lines_per_chunk: int = 40,
) -> List[ControlChunk]:
    chunks: List[ControlChunk] = []
    for record in records:
        chunks.extend(
            chunk_record(
                record,
                max_document_bytes=max_document_bytes,
                max_lines_per_chunk=max_lines_per_chunk,
            )
        )
    return chunks


def chunk_record(
    record: ControlRecord,
    *,
    max_document_bytes: int = MAX_DOCUMENT_BYTES,
    max_lines_per_chunk: int = 40,
) -> List[ControlChunk]:
    lines = [line.strip() for line in record.text.splitlines() if line.strip()]
    if not lines:
        lines = [record.text.strip() or ""]

    chunks: List[ControlChunk] = []
    active: List[str] = []
    chunk_index = 0

    for line in lines:
        if not line:
            continue

        line_segments = _split_line_to_fit(line, max_document_bytes)
        for seg in line_segments:
            candidate = _join_lines(active + [seg])
            if active and (
                len(candidate.encode("utf-8")) > max_document_bytes
                or len(active) >= max_lines_per_chunk
            ):
                chunks.append(_build_chunk(record, chunk_index, _join_lines(active)))
                chunk_index += 1
                active = [seg]
            else:
                active.append(seg)

    if active:
        chunks.append(_build_chunk(record, chunk_index, _join_lines(active)))

    return chunks


def _split_line_to_fit(line: str, max_document_bytes: int) -> List[str]:
    if len(line.encode("utf-8")) <= max_document_bytes:
        return [line]

    pieces: List[str] = []
    current = ""

    for token in line.split(" "):
        probe = token if not current else f"{current} {token}"
        if len(probe.encode("utf-8")) <= max_document_bytes:
            current = probe
            continue

        if current:
            pieces.append(current)

        if len(token.encode("utf-8")) <= max_document_bytes:
            current = token
            continue

        char_buf = ""
        for ch in token:
            test = f"{char_buf}{ch}"
            if len(test.encode("utf-8")) <= max_document_bytes:
                char_buf = test
            else:
                pieces.append(char_buf)
                char_buf = ch
        current = char_buf

    if current:
        pieces.append(current)

    return pieces


def _join_lines(lines: List[str]) -> str:
    return "\n".join(lines).strip()


def _build_chunk(record: ControlRecord, chunk_index: int, text: str) -> ControlChunk:
    chunk_key = f"{record.normalized_id}:{chunk_index}:{text}"
    chunk_id = stable_hash(chunk_key)[:24]

    metadata = {
        "regulation": record.regulation,
        "control_id": record.normalized_id,
        "native_control_id": record.native_id,
        "source_document_id": record.source_document_id,
        "source_path": record.source_path,
        "chunk_index": chunk_index,
        "chapter": record.chapter,
        "section": record.section,
        "clause": record.clause,
    }
    metadata.update(record.metadata)

    return ControlChunk(
        chunk_id=chunk_id,
        regulation=record.regulation,
        source_document_id=record.source_document_id,
        control_id=record.normalized_id,
        chunk_index=chunk_index,
        text=text,
        metadata=metadata,
    )
