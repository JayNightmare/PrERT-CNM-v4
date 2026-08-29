"""Retriever: cosine top-k over a MemoryIndex, with query encoding cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from prert.phase3.cnm.embed import TextEncoder
from prert.phase3.cnm.memory import MemoryEntry, MemoryIndex


@dataclass
class RetrievedControl:
    entry: MemoryEntry
    score: float
    rank: int


class Retriever:
    """Encodes query text once per unique clause, retrieves top-k controls.

    The query encoder is required to share tokenisation geometry with the
    encoder used to build the index — i.e. same model name. If a different
    model is passed we log and continue (cosine still returns something, but
    the geometry differs).
    """

    def __init__(
        self,
        index: MemoryIndex,
        query_encoder: TextEncoder | None = None,
        cache_size: int = 4096,
    ) -> None:
        self.index = index
        self.query_encoder = query_encoder or TextEncoder(model_name=index.encoder_name)
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_order: List[str] = []
        self._cache_size = int(cache_size)

    def _encode_query(self, text: str) -> np.ndarray:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        vector = self.query_encoder.encode([text], normalise=True)[0]
        # Simple FIFO cache.
        if len(self._cache_order) >= self._cache_size:
            oldest = self._cache_order.pop(0)
            self._cache.pop(oldest, None)
        self._cache[text] = vector
        self._cache_order.append(text)
        return vector

    def retrieve(self, text: str, k: int) -> List[RetrievedControl]:
        if k <= 0 or len(self.index) == 0 or not text.strip():
            return []
        query = self._encode_query(text)
        top_idx = self.index.top_k(query, k)
        scores = self.index.similarities(query, top_idx)
        results: List[RetrievedControl] = []
        for rank, (idx, score) in enumerate(zip(top_idx, scores)):
            results.append(
                RetrievedControl(
                    entry=self.index.entries[idx], score=float(score), rank=rank
                )
            )
        return results

    def retrieve_batch(
        self, texts: Sequence[str], k: int
    ) -> List[List[RetrievedControl]]:
        return [self.retrieve(text, k) for text in texts]
