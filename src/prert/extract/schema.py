"""Common extraction schema shared across regulation parsers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha1
from typing import Any, Dict, List
import re


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def stable_hash(value: str) -> str:
    return sha1(value.encode("utf-8")).hexdigest()


def make_normalized_id(regulation: str, native_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", native_id.strip())
    return f"{regulation.lower()}::{cleaned}"


@dataclass
class ControlRecord:
    record_id: str
    regulation: str
    source_document_id: str
    source_path: str
    native_id: str
    normalized_id: str
    title: str
    text: str
    hierarchy_path: List[str]
    chapter: str = ""
    section: str = ""
    clause: str = ""
    parser_confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["title"] = normalize_whitespace(payload["title"])
        payload["text"] = normalize_whitespace(payload["text"])
        return payload


@dataclass
class ControlChunk:
    chunk_id: str
    regulation: str
    source_document_id: str
    control_id: str
    chunk_index: int
    text: str
    metadata: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["text"] = normalize_whitespace(payload["text"])
        return payload
