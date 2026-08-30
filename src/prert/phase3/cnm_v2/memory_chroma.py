"""ChromaDB-backed memory index for CNMv2.

Replaces the numpy-based MemoryIndex used in CNMv1. Chroma is already a
project dependency (pyproject) and gives us:

  * Persistent on-disk collection with metadata search
  * Query API that returns (ids, distances, documents, metadata) in one call
  * A stable identifier per control for use in traceability tables

The memory index still exposes a `numpy_matrix()` view of the embeddings
for the training-time precompute path — during training we don't want to
call Chroma once per example, so we materialise the embedding matrix once
at fit-start and do batched cosine top-k in numpy. Chroma is used at
inference time and for the traceability trace file.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

_LOGGER = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "prert_cnm_controls"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class ControlRecord:
    control_id: str
    text: str
    source: str = ""
    section: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "control_id": self.control_id,
            "text": self.text,
            "source": self.source,
            "section": self.section,
            **self.metadata,
        }


class ChromaMemoryIndex:
    """Chroma-backed control-catalogue memory.

    Persists the collection under `<persist_dir>/chroma/`. The collection
    name is fixed so the same index is reused across builds. A JSON
    metadata sidecar (`index_metadata.json`) records the encoder name,
    dimension, corpus hash, and control-id ordering.
    """

    def __init__(
        self,
        persist_dir: Path,
        encoder_name: str = DEFAULT_EMBEDDING_MODEL,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.encoder_name = encoder_name
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._records: List[ControlRecord] = []
        self._embeddings_matrix: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Chroma plumbing
    # ------------------------------------------------------------------
    def _ensure_client(self):
        if self._client is None:
            import chromadb  # local import — Chroma is heavy

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            (self.persist_dir / "chroma").mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir / "chroma")
            )
        return self._client

    def _ensure_collection(self):
        if self._collection is None:
            client = self._ensure_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"encoder_name": self.encoder_name},
            )
        return self._collection

    # ------------------------------------------------------------------
    # Build / load
    # ------------------------------------------------------------------
    def build(
        self,
        records: Sequence[ControlRecord],
        embeddings: np.ndarray,
        overwrite: bool = True,
    ) -> None:
        """Populate the Chroma collection and persist metadata."""
        if len(records) != embeddings.shape[0]:
            raise ValueError(
                f"records ({len(records)}) and embeddings ({embeddings.shape[0]}) size mismatch"
            )
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        client = self._ensure_client()
        if overwrite:
            try:
                client.delete_collection(self.collection_name)
            except Exception:  # noqa: BLE001
                pass
            self._collection = None

        collection = self._ensure_collection()
        # Chroma stores embeddings as python lists — this is unavoidable.
        collection.add(
            ids=[r.control_id for r in records],
            embeddings=embeddings.tolist(),
            documents=[r.text for r in records],
            metadatas=[
                {"source": r.source, "section": r.section, **r.metadata}
                for r in records
            ],
        )
        self._records = list(records)
        self._embeddings_matrix = embeddings

        metadata = {
            "encoder_name": self.encoder_name,
            "dim": int(embeddings.shape[1]),
            "count": len(records),
            "collection_name": self.collection_name,
            "corpus_hash": self._corpus_hash(records),
            "control_id_order": [r.control_id for r in records],
        }
        (self.persist_dir / "index_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _LOGGER.info(
            "Built Chroma memory index: n=%d dim=%d at %s",
            len(records),
            embeddings.shape[1],
            self.persist_dir,
        )

    def load(self) -> None:
        """Load an existing Chroma collection + its embedding matrix."""
        client = self._ensure_client()
        try:
            self._collection = client.get_collection(name=self.collection_name)
        except Exception as exc:  # noqa: BLE001
            raise FileNotFoundError(
                f"Chroma collection '{self.collection_name}' not found at {self.persist_dir}"
            ) from exc

        metadata_path = self.persist_dir / "index_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.encoder_name = metadata["encoder_name"]

        # Pull the full collection back in a stable order — collection order
        # matches the training-time embedding matrix row order.
        control_id_order = metadata["control_id_order"]
        result = self._collection.get(
            ids=control_id_order,
            include=["embeddings", "documents", "metadatas"],
        )
        # Chroma may re-order results; reorder to match control_id_order.
        by_id = {
            _id: (emb, doc, meta)
            for _id, emb, doc, meta in zip(
                result["ids"],
                result["embeddings"],
                result["documents"],
                result["metadatas"],
            )
        }
        emb_rows: List[List[float]] = []
        self._records = []
        for cid in control_id_order:
            emb, doc, meta = by_id[cid]
            emb_rows.append(emb)
            self._records.append(
                ControlRecord(
                    control_id=cid,
                    text=doc,
                    source=str(meta.get("source", "")),
                    section=str(meta.get("section", "")),
                    metadata={
                        k: v for k, v in meta.items() if k not in {"source", "section"}
                    },
                )
            )
        self._embeddings_matrix = np.array(emb_rows, dtype=np.float32)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._records)

    def numpy_matrix(self) -> np.ndarray:
        if self._embeddings_matrix is None:
            raise RuntimeError("Index not built or loaded")
        return self._embeddings_matrix

    def records(self) -> List[ControlRecord]:
        return self._records

    def batch_top_k(
        self, query_matrix: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batched top-k cosine over `query_matrix` (Q, D) → (indices (Q,k), scores (Q,k)).

        Both query_matrix and stored embeddings are L2-normalised, so cosine
        similarity = inner product. When `k == 0` (CNM disabled / ablation
        baseline) the returned arrays have shape (Q, 0) — downstream code must
        handle empty retrievals explicitly.
        """
        if self._embeddings_matrix is None:
            raise RuntimeError("Index not built or loaded")
        if query_matrix.dtype != np.float32:
            query_matrix = query_matrix.astype(np.float32)
        Q = query_matrix.shape[0]
        if k <= 0:
            # Cleanly-typed empty result; do not call argpartition with k-1=-1.
            return (
                np.zeros((Q, 0), dtype=np.int64),
                np.zeros((Q, 0), dtype=np.float32),
            )
        sims = query_matrix @ self._embeddings_matrix.T  # (Q, M)
        k = min(k, sims.shape[1])
        top_idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        row_scores = np.take_along_axis(sims, top_idx, axis=1)
        order = np.argsort(-row_scores, axis=1, kind="stable")
        indices = np.take_along_axis(top_idx, order, axis=1)
        scores = np.take_along_axis(row_scores, order, axis=1)
        return indices.astype(np.int64), scores.astype(np.float32)

    def query_chroma(
        self, embeddings: np.ndarray, k: int
    ) -> List[List[Dict[str, Any]]]:
        """Chroma-native query path (used for the inference traceability trace).

        Returns per-query lists of dicts with control_id/text/source/score.
        """
        collection = self._ensure_collection()
        result = collection.query(
            query_embeddings=embeddings.tolist(),
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        out: List[List[Dict[str, Any]]] = []
        for ids, docs, metas, dists in zip(
            result["ids"], result["documents"], result["metadatas"], result["distances"]
        ):
            row: List[Dict[str, Any]] = []
            for i, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists)):
                # Chroma returns cosine DISTANCE by default (1 - cos_sim).
                row.append(
                    {
                        "rank": i,
                        "control_id": cid,
                        "text": doc,
                        "source": str(meta.get("source", "")) if meta else "",
                        "section": str(meta.get("section", "")) if meta else "",
                        "score": float(1.0 - dist),
                    }
                )
            out.append(row)
        return out

    def _corpus_hash(self, records: Sequence[ControlRecord]) -> str:
        h = hashlib.sha256()
        for r in records:
            h.update(r.control_id.encode("utf-8"))
            h.update(b"\x1f")
            h.update(r.text.encode("utf-8"))
            h.update(b"\x1e")
        return h.hexdigest()
