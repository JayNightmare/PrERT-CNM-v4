"""Deterministic unit tests for the CNM memory index.

Uses a small synthetic corpus and a stub encoder so tests run offline and
do not require sentence-transformers / a network fetch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from prert.phase3.cnm.memory import (
    MemoryEntry,
    MemoryIndex,
    build_memory_index,
    load_memory_index,
)


class _StubEncoder:
    """Deterministic hash-based encoder for tests."""

    def __init__(self) -> None:
        self.dim = 8
        self.backend = "stub"
        self.model_name = "stub-encoder"

    def encode(
        self, texts: Sequence[str], batch_size: int = 64, normalise: bool = True
    ) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            # Deterministic pseudo-embedding from character sums.
            for j in range(self.dim):
                vectors[i, j] = sum((ord(c) * (j + 1)) % 7 for c in text[:32])
        if normalise:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms
        return vectors.astype(np.float32)


def _entries():
    return [
        MemoryEntry(
            control_id="gdpr-5",
            text="Personal data shall be processed lawfully",
            source="GDPR",
        ),
        MemoryEntry(
            control_id="iso-27001-a5",
            text="Information security policies shall be defined",
            source="ISO/IEC 27001",
        ),
        MemoryEntry(
            control_id="nist-53-ac1",
            text="Access control policy and procedures",
            source="NIST SP 800-53",
        ),
    ]


def test_top_k_returns_expected_ordering():
    encoder = _StubEncoder()
    entries = _entries()
    embeddings = encoder.encode([e.text for e in entries])
    index = MemoryIndex(
        entries=entries, embeddings=embeddings, encoder_name="stub-encoder"
    )

    query = encoder.encode(["access control"])[0]
    top = index.top_k(query, k=2)
    assert len(top) == 2
    assert all(0 <= i < len(entries) for i in top)


def test_top_k_zero_or_empty():
    entries = _entries()
    embeddings = np.eye(len(entries), 8, dtype=np.float32)
    index = MemoryIndex(
        entries=entries, embeddings=embeddings, encoder_name="stub-encoder"
    )
    assert index.top_k(np.zeros(8, dtype=np.float32), k=0) == []


def test_save_and_load_roundtrip(tmp_path: Path):
    encoder = _StubEncoder()
    entries = _entries()
    embeddings = encoder.encode([e.text for e in entries])
    original = MemoryIndex(
        entries=entries, embeddings=embeddings, encoder_name="stub-encoder"
    )
    original.save(tmp_path)

    loaded = load_memory_index(tmp_path)
    assert len(loaded) == len(original)
    assert loaded.dim == original.dim
    assert loaded.encoder_name == "stub-encoder"
    np.testing.assert_allclose(loaded.embeddings, original.embeddings, atol=1e-6)
    for a, b in zip(loaded.entries, original.entries):
        assert a.control_id == b.control_id
        assert a.text == b.text
        assert a.source == b.source


def test_build_from_jsonl(tmp_path: Path, monkeypatch):
    controls_path = tmp_path / "controls.jsonl"
    with controls_path.open("w", encoding="utf-8") as h:
        for e in _entries():
            h.write(json.dumps(e.as_dict()) + "\n")

    # Patch encoder so the test doesn't hit the network.
    import prert.phase3.cnm.memory as memory_module

    monkeypatch.setattr(memory_module, "TextEncoder", lambda **kwargs: _StubEncoder())

    out = tmp_path / "index"
    index = build_memory_index(
        controls_path=controls_path, output_dir=out, encoder_name="stub-encoder"
    )
    assert len(index) == 3
    assert (out / "embeddings.npz").exists()
    assert (out / "entries.jsonl").exists()
    assert (out / "index_metadata.json").exists()


def test_build_from_wrapped_json(tmp_path: Path, monkeypatch):
    controls_path = tmp_path / "controls.json"
    payload = {"controls": [e.as_dict() for e in _entries()]}
    controls_path.write_text(json.dumps(payload), encoding="utf-8")

    import prert.phase3.cnm.memory as memory_module

    monkeypatch.setattr(memory_module, "TextEncoder", lambda **kwargs: _StubEncoder())

    out = tmp_path / "index"
    index = build_memory_index(
        controls_path=controls_path, output_dir=out, encoder_name="stub-encoder"
    )
    assert len(index) == 3


def test_from_dict_handles_alternate_keys():
    entry = MemoryEntry.from_dict(
        {
            "id": "x-1",
            "control_text": "Data minimisation",
            "framework": "GDPR",
            "section_id": "Art. 5",
            "extra": "kept in metadata",
        }
    )
    assert entry.control_id == "x-1"
    assert entry.text == "Data minimisation"
    assert entry.source == "GDPR"
    assert entry.section == "Art. 5"
    assert entry.metadata.get("extra") == "kept in metadata"


def test_from_dict_flattens_nested_metadata():
    entry = MemoryEntry.from_dict(
        {
            "id": "x-2",
            "control_text": "Storage limitation",
            "framework": "GDPR",
            "metadata": {
                "format_profile": "gdpr_article_subclause",
                "ground_truth_source": True,
            },
        }
    )
    assert entry.metadata == {
        "format_profile": "gdpr_article_subclause",
        "ground_truth_source": True,
    }


def test_empty_controls_raises(tmp_path: Path, monkeypatch):
    controls_path = tmp_path / "empty.jsonl"
    controls_path.write_text("", encoding="utf-8")

    import prert.phase3.cnm.memory as memory_module

    monkeypatch.setattr(memory_module, "TextEncoder", lambda **kwargs: _StubEncoder())

    with pytest.raises(ValueError):
        build_memory_index(controls_path=controls_path, output_dir=tmp_path / "idx")
