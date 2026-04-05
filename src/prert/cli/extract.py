"""Phase 1 extraction CLI: regulation-specific parsing and chunk generation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable, List

from prert.chunking import chunk_records
from prert.extract import parse_gdpr_controls, parse_iso_controls, parse_nist_controls
from prert.extract.schema import ControlChunk, ControlRecord


def main() -> None:
    args = _parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gdpr_records = parse_gdpr_controls(args.gdpr_path)
    iso_records = parse_iso_controls(args.iso_path)
    nist_records = parse_nist_controls(args.nist_pdf_path, txt_cache_path=args.nist_txt_cache)

    all_records: List[ControlRecord] = []
    all_records.extend(gdpr_records)
    all_records.extend(iso_records)
    all_records.extend(nist_records)

    _write_records(output_dir / "controls_gdpr.jsonl", gdpr_records)
    _write_records(output_dir / "controls_iso27001.jsonl", iso_records)
    _write_records(output_dir / "controls_nistpf.jsonl", nist_records)
    _write_records(output_dir / "controls_all.jsonl", all_records)

    print(f"Wrote {len(gdpr_records)} GDPR control rows")
    print(f"Wrote {len(iso_records)} ISO 27001 control rows")
    print(f"Wrote {len(nist_records)} NIST PF control rows")

    if args.chunk:
        gdpr_chunks = chunk_records(gdpr_records)
        iso_chunks = chunk_records(iso_records)
        nist_chunks = chunk_records(nist_records)
        all_chunks: List[ControlChunk] = [*gdpr_chunks, *iso_chunks, *nist_chunks]

        _write_chunks(output_dir / "chunks_gdpr.jsonl", gdpr_chunks)
        _write_chunks(output_dir / "chunks_iso27001.jsonl", iso_chunks)
        _write_chunks(output_dir / "chunks_nistpf.jsonl", nist_chunks)
        _write_chunks(output_dir / "chunks_all.jsonl", all_chunks)

        print(f"Wrote {len(gdpr_chunks)} GDPR chunks")
        print(f"Wrote {len(iso_chunks)} ISO 27001 chunks")
        print(f"Wrote {len(nist_chunks)} NIST PF chunks")


def _parse_args() -> argparse.Namespace:
    default_root = Path.cwd()

    parser = argparse.ArgumentParser(description="Extract Phase 1 controls from GDPR, ISO 27001, and NIST")
    parser.add_argument(
        "--gdpr-path",
        type=Path,
        default=default_root / "docs/Standards/Regulations/TX/GDPR-2016_679.txt",
    )
    parser.add_argument(
        "--iso-path",
        type=Path,
        default=default_root / "docs/Standards/Regulations/TX/ISO_27001_Standard-1.txt",
    )
    parser.add_argument(
        "--nist-pdf-path",
        type=Path,
        default=default_root / "docs/Standards/Regulations/NIST-1.1.pdf",
    )
    parser.add_argument(
        "--nist-txt-cache",
        type=Path,
        default=default_root / "artifacts/phase-1/NIST-1.1.txt",
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
