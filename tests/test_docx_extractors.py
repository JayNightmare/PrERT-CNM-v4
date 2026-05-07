from pathlib import Path

from docx import Document

from prert.cli.migrate import _collection_name, _discover_chunk_files
from prert.extract.gdpr_parser import parse_gdpr_controls
from prert.extract.iso_parser import parse_iso_controls
from prert.extract.iso_sources import discover_iso_docx_sources, make_iso_docx_source
from prert.extract.nist_parser import parse_nist_controls


def _write_docx(path: Path, lines: list[str]) -> None:
    doc = Document()
    for line in lines:
        doc.add_paragraph(line)
    doc.save(path)


def test_iso_docx_entrypoint_preserves_clause_and_bullets(tmp_path: Path) -> None:
    path = tmp_path / "ISO_27001_Standard-1.docx"
    _write_docx(
        path,
        [
            "1 Scope",
            "4.1 Understanding the organization and its context",
            "The organization shall define context.",
            "a) Include internal issues.",
            "b) Include external issues.",
            "4.2 Understanding the needs and expectations of interested parties",
            "The organization shall determine parties.",
        ],
    )

    records = parse_iso_controls(path, regulation="ISO27002", source_document_id="iso-27002-2022")
    ids = {record.native_id for record in records}

    assert "4.1" in ids
    assert "4.1.a" in ids
    assert "4.1.b" in ids
    assert "4.2" in ids
    assert {record.regulation for record in records} == {"ISO27002"}
    assert {record.source_document_id for record in records} == {"iso-27002-2022"}


def test_gdpr_docx_entrypoint_extracts_articles(tmp_path: Path) -> None:
    path = tmp_path / "GDPR-2016_679.docx"
    _write_docx(
        path,
        [
            "CHAPTER I",
            "General provisions",
            "Article 1",
            "Subject-matter and objectives",
            "1. This Regulation lays down rules.",
            "2. This Regulation protects rights.",
        ],
    )

    records = parse_gdpr_controls(path)
    ids = {record.native_id for record in records}
    assert "Article 1.1" in ids
    assert "Article 1.2" in ids


def test_nist_docx_entrypoint_extracts_subcategories(tmp_path: Path) -> None:
    path = tmp_path / "NIST-1.1.docx"
    _write_docx(
        path,
        [
            "ID.IM-P4: Data actions of systems and products are inventoried.",
            "Continuation line for ID.IM-P4.",
            "GV.RM-P2: The organization's risk tolerance is defined.",
        ],
    )

    records = parse_nist_controls(path)
    ids = {record.native_id for record in records}
    assert "ID.IM-P4" in ids
    assert "GV.RM-P2" in ids
    assert len(records) == 2


def test_iso_source_discovery_uses_numeric_family(tmp_path: Path) -> None:
    (tmp_path / "GDPR-2016_679.docx").touch()
    (tmp_path / "BS EN ISO-IEC 27002-2022.docx").touch()
    (tmp_path / "BS ISO-IEC 15944-12-2025.docx").touch()

    sources = discover_iso_docx_sources(tmp_path)
    stems = [source.output_stem for source in sources]
    assert stems == ["iso15944_12", "iso27002"]

    source = make_iso_docx_source(tmp_path / "BS ISO-IEC 15944-12-2025.docx")
    assert source.regulation == "ISO15944_12"
    assert source.source_document_id == "iso-15944-12-2025"


def test_migration_discovers_dynamic_iso_chunk_files(tmp_path: Path) -> None:
    (tmp_path / "chunks_gdpr.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "chunks_nistpf.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "chunks_iso27001.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "chunks_iso27002.jsonl").write_text("{}\n", encoding="utf-8")

    discovered = _discover_chunk_files(tmp_path)
    keys = [key for key, _ in discovered]

    assert "gdpr" in keys
    assert "nistpf" in keys
    assert "iso27001" in keys
    assert "iso27002" in keys

    assert _collection_name("gdpr", "") == "gdpr_controls"
    assert _collection_name("nistpf", "") == "nist_controls"
    assert _collection_name("iso27002", "") == "iso27002_controls"
