"""Retriever unit tests using the stub encoder from test_memory_index."""

from __future__ import annotations

import numpy as np
import pytest

from prert.phase3.cnm.memory import MemoryEntry, MemoryIndex
from prert.phase3.cnm.retriever import Retriever

from tests.test_memory_index import _StubEncoder, _entries


def _build_index() -> MemoryIndex:
    encoder = _StubEncoder()
    entries = _entries()
    embeddings = encoder.encode([e.text for e in entries])
    return MemoryIndex(
        entries=entries, embeddings=embeddings, encoder_name="stub-encoder"
    )


def test_retrieve_returns_top_k():
    index = _build_index()
    retriever = Retriever(index=index, query_encoder=_StubEncoder())
    results = retriever.retrieve("access control policy", k=2)
    assert len(results) == 2
    assert results[0].rank == 0
    assert results[1].rank == 1
    assert results[0].score >= results[1].score


def test_retrieve_zero_k():
    index = _build_index()
    retriever = Retriever(index=index, query_encoder=_StubEncoder())
    assert retriever.retrieve("anything", k=0) == []


def test_query_cache():
    index = _build_index()
    encoder = _StubEncoder()
    retriever = Retriever(index=index, query_encoder=encoder, cache_size=8)
    r1 = retriever.retrieve("access", k=1)
    r2 = retriever.retrieve("access", k=1)
    assert r1[0].entry.control_id == r2[0].entry.control_id


def test_empty_index():
    index = MemoryIndex(
        entries=[], embeddings=np.zeros((0, 8), dtype=np.float32), encoder_name="stub"
    )
    retriever = Retriever(index=index, query_encoder=_StubEncoder())
    assert retriever.retrieve("test", k=5) == []
