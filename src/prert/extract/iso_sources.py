"""Discovery and metadata helpers for ISO standard DOCX sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence


_ISO_IEC_FAMILY_RE = re.compile(r"ISO[-_\s]?IEC[-_\s]?(\d+(?:-\d+)*)", re.IGNORECASE)
_ISO_FAMILY_RE = re.compile(r"ISO[-_\s]?(\d+(?:-\d+)*)", re.IGNORECASE)
_YEAR_RE = re.compile(r"((?:19|20)\d{2})")


@dataclass(frozen=True)
class IsoDocxSource:
    path: Path
    output_stem: str
    regulation: str
    source_document_id: str
    display_name: str


def discover_iso_docx_sources(regulations_dir: Path, explicit_paths: Sequence[Path] | None = None) -> list[IsoDocxSource]:
    if explicit_paths:
        candidates = [Path(path) for path in explicit_paths]
    else:
        candidates = sorted(regulations_dir.glob("*.docx"))

    sources: list[IsoDocxSource] = []
    for candidate in candidates:
        if not is_iso_docx_file(candidate):
            continue
        sources.append(make_iso_docx_source(candidate))

    by_stem: dict[str, IsoDocxSource] = {}
    for source in sources:
        if source.output_stem in by_stem:
            existing = by_stem[source.output_stem]
            raise ValueError(
                f"Duplicate ISO output stem '{source.output_stem}' for '{existing.path.name}' and '{source.path.name}'"
            )
        by_stem[source.output_stem] = source

    return [by_stem[key] for key in sorted(by_stem.keys())]


def is_iso_docx_file(path: Path) -> bool:
    if path.suffix.lower() != ".docx":
        return False

    upper_name = path.name.upper()
    if "GDPR" in upper_name or "NIST" in upper_name:
        return False

    return "ISO" in upper_name


def make_iso_docx_source(path: Path) -> IsoDocxSource:
    family_raw = _extract_family(path.stem)
    family_key = family_raw.replace("-", "_")
    output_stem = f"iso{family_key}"
    regulation = f"ISO{family_key}"

    year = _extract_year(path.stem)
    if year:
        source_document_id = f"iso-{family_raw.lower()}-{year}"
    else:
        source_document_id = f"iso-{family_raw.lower()}"

    display_name = f"ISO {family_raw}"
    return IsoDocxSource(
        path=path,
        output_stem=output_stem,
        regulation=regulation,
        source_document_id=source_document_id,
        display_name=display_name,
    )


def _extract_family(name: str) -> str:
    match = _ISO_IEC_FAMILY_RE.search(name)
    if not match:
        match = _ISO_FAMILY_RE.search(name)
    if not match:
        raise ValueError(f"Could not derive ISO family from filename '{name}'")

    token = match.group(1)
    parts = token.split("-")
    if len(parts) >= 2 and re.fullmatch(r"(?:19|20)\d{2}", parts[-1]):
        parts = parts[:-1]

    return "-".join(parts)


def _extract_year(name: str) -> str | None:
    matches = _YEAR_RE.findall(name)
    if not matches:
        return None
    return matches[-1]
