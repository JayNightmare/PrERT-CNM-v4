from prert.extract.gdpr_parser import parse_gdpr_controls_from_text
from prert.extract.iso_parser import parse_iso_controls_from_text
from prert.extract.nist_parser import parse_nist_controls_from_text


def test_gdpr_parser_ignores_preamble_article_mentions() -> None:
    sample = """
Whereas:
Article 8(1) should not be parsed as an article header.
CHAPTER I
General provisions
Article 1
Subject-matter and objectives
1. This Regulation lays down rules.
2. This Regulation protects rights.
Article 2
Material scope
1. This Regulation applies.
"""
    records = parse_gdpr_controls_from_text(sample, source_path="sample")
    ids = {r.native_id for r in records}

    assert "Article 1.1" in ids
    assert "Article 1.2" in ids
    assert "Article 2.1" in ids
    assert all("8(1)" not in native_id for native_id in ids)


def test_iso_parser_handles_clause_and_bullets() -> None:
    sample = """
1 Scope
4.1 Understanding the organization and its context
The organization shall define context.
a) Include internal issues.
b) Include external issues.
4.2 Understanding the needs and expectations of interested parties
The organization shall determine parties.
"""
    records = parse_iso_controls_from_text(sample, source_path="sample")
    ids = {r.native_id for r in records}

    assert "4.1" in ids
    assert "4.1.a" in ids
    assert "4.1.b" in ids
    assert "4.2" in ids


def test_nist_parser_extracts_subcategory_ids() -> None:
    sample = """
ID.IM-P4: Data actions of systems and products are inventoried.
Continuation line for ID.IM-P4.
GV.RM-P2: The organization's risk tolerance is defined.
"""
    records = parse_nist_controls_from_text(sample, source_path="sample")
    ids = {r.native_id for r in records}

    assert "ID.IM-P4" in ids
    assert "GV.RM-P2" in ids
    assert len(records) == 2
