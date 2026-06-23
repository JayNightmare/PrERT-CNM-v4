"""Phase 1 extraction CLI: regulation-specific parsing and chunk generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from prert.chunking import chunk_records
from prert.extract import parse_gdpr_controls, parse_iso_controls, parse_nist_controls
from prert.extract.iso_sources import IsoDocxSource, discover_iso_docx_sources
from prert.extract.schema import ControlChunk, ControlRecord


def main() -> None:
    args = _parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    iso_sources = _resolve_iso_sources(args)
    if not iso_sources:
        raise RuntimeError("No ISO DOCX files found. Provide --iso-path values or set --iso-dir correctly.")

    _cleanup_stale_iso_outputs(output_dir, iso_sources)

    gdpr_records = parse_gdpr_controls(args.gdpr_path)
    nist_records = parse_nist_controls(args.nist_path)

    iso_records_by_stem: dict[str, list[ControlRecord]] = {}
    for source in iso_sources:
        records = parse_iso_controls(
            source.path,
            regulation=source.regulation,
            source_document_id=source.source_document_id,
        )
        iso_records_by_stem[source.output_stem] = records

    all_records: List[ControlRecord] = []
    all_records.extend(gdpr_records)
    for source in iso_sources:
        all_records.extend(iso_records_by_stem[source.output_stem])
    all_records.extend(nist_records)

    _write_records(output_dir / "controls_gdpr.jsonl", gdpr_records)
    for source in iso_sources:
        _write_records(output_dir / f"controls_{source.output_stem}.jsonl", iso_records_by_stem[source.output_stem])
    _write_records(output_dir / "controls_nistpf.jsonl", nist_records)
    _write_records(output_dir / "controls_all.jsonl", all_records)

    print(f"Wrote {len(gdpr_records)} GDPR control rows")
    for source in iso_sources:
        count = len(iso_records_by_stem[source.output_stem])
        print(f"Wrote {count} {source.display_name} control rows ({source.output_stem})")
    print(f"Wrote {len(nist_records)} NIST PF control rows")

    if args.chunk:
        gdpr_chunks = chunk_records(gdpr_records)
        nist_chunks = chunk_records(nist_records)
        iso_chunks_by_stem: dict[str, list[ControlChunk]] = {}
        for source in iso_sources:
            iso_chunks_by_stem[source.output_stem] = chunk_records(iso_records_by_stem[source.output_stem])

        all_chunks: List[ControlChunk] = [*gdpr_chunks, *nist_chunks]
        for source in iso_sources:
            all_chunks.extend(iso_chunks_by_stem[source.output_stem])

        _write_chunks(output_dir / "chunks_gdpr.jsonl", gdpr_chunks)
        for source in iso_sources:
            _write_chunks(output_dir / f"chunks_{source.output_stem}.jsonl", iso_chunks_by_stem[source.output_stem])
        _write_chunks(output_dir / "chunks_nistpf.jsonl", nist_chunks)
        _write_chunks(output_dir / "chunks_all.jsonl", all_chunks)

        print(f"Wrote {len(gdpr_chunks)} GDPR chunks")
        for source in iso_sources:
            count = len(iso_chunks_by_stem[source.output_stem])
            print(f"Wrote {count} {source.display_name} chunks ({source.output_stem})")
        print(f"Wrote {len(nist_chunks)} NIST PF chunks")


def _parse_args() -> argparse.Namespace:
    default_root = Path.cwd()
    default_regulations_dir = default_root / "docs/Standards/Regulations"

    parser = argparse.ArgumentParser(description="Extract Phase 1 controls from GDPR, ISO standards, and NIST")
    parser.add_argument(
        "--gdpr-path",
        type=Path,
        default=default_regulations_dir / "GDPR-2016_679.docx",
    )
    parser.add_argument(
        "--iso-dir",
        type=Path,
        default=default_regulations_dir,
        help="Directory containing ISO DOCX standards.",
    )
    parser.add_argument(
        "--iso-path",
        type=Path,
        action="append",
        default=None,
        help="Optional explicit ISO DOCX path. Provide multiple times to process a subset.",
    )
    parser.add_argument(
        "--nist-path",
        type=Path,
        default=default_regulations_dir / "NIST-1.1.docx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_root / "artifacts/phase-1",
    )
    parser.add_argument(
        "--chunk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate Chroma-safe chunks in addition to control rows.",
    )

    return parser.parse_args()


def _resolve_iso_sources(args: argparse.Namespace) -> list[IsoDocxSource]:
    explicit_paths: list[Path] | None = None
    if args.iso_path:
        explicit_paths = [path.resolve() for path in args.iso_path]

    sources = discover_iso_docx_sources(args.iso_dir.resolve(), explicit_paths=explicit_paths)
    return sources


def _cleanup_stale_iso_outputs(output_dir: Path, iso_sources: list[IsoDocxSource]) -> None:
    expected_controls = {f"controls_{source.output_stem}.jsonl" for source in iso_sources}
    expected_chunks = {f"chunks_{source.output_stem}.jsonl" for source in iso_sources}

    for path in output_dir.glob("controls_iso*.jsonl"):
        if path.name not in expected_controls:
            path.unlink(missing_ok=True)

    for path in output_dir.glob("chunks_iso*.jsonl"):
        if path.name not in expected_chunks:
            path.unlink(missing_ok=True)


def _write_records(path: Path, records: Iterable[ControlRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = record.as_dict()
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_chunks(path: Path, chunks: Iterable[ControlChunk]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            payload = chunk.as_dict()
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
