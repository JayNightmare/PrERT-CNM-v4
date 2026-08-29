"""Contextual Neural Memory (CNM) for PrERT-CNM Phase 3.

Retrieval-augmented memory over the harmonised standards catalogue. Given a
clause, the CNM retrieves the top-k semantically-similar controls from
Phase 1's harmonised catalogue and concatenates their text into the PrivBERT
input, giving the classifier standards-aware context at inference time.

Public API:
    from prert.phase3.cnm import build_memory_index, MemoryIndex, Retriever
    from prert.phase3.cnm import CNMPrivacyBertClassifier, CNMConfig

The design is deliberately dependency-light: numpy-based cosine index,
sentence-transformers (or a transformers fallback) for the encoder, and a
thin subclass of PrivacyBertClassifier for training/inference. No chromadb
is required for the paper's headline index.
"""

from __future__ import annotations

from prert.phase3.cnm.memory import (
    MemoryEntry,
    MemoryIndex,
    build_memory_index,
    load_memory_index,
)
from prert.phase3.cnm.retriever import Retriever, RetrievedControl
from prert.phase3.cnm.classifier import CNMPrivacyBertClassifier, CNMConfig

__all__ = [
    "MemoryEntry",
    "MemoryIndex",
    "build_memory_index",
    "load_memory_index",
    "Retriever",
    "RetrievedControl",
    "CNMPrivacyBertClassifier",
    "CNMConfig",
]
