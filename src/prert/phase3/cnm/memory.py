"""Persistent, numpy-backed memory index for the CNM.

The index stores:
  - `embeddings`: float32 matrix (N, D), L2-normalised, saved as .npz
  - `entries`: JSONL list of MemoryEntry records (control_id, text, source, ...)
  - `metadata`: JSON with encoder name, dim, count, build config

Cosine similarity == inner product on the normalised matrix.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from prert.phase3.cnm.embed import DEFAULT_EMBEDDING_MODEL, TextEncoder

_LOGGER = logging.getLogger(__name__)


def _coerce_chroma_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_coerce_chroma_value(item) for item in value]
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _flatten_metadata_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in data.items():
        flat_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_metadata_dict(value, prefix=flat_key))
        else:
            out[flat_key] = _coerce_chroma_value(value)
    return out


@dataclass
class MemoryEntry:
    control_id: str
    text: str
    source: str = ""  # e.g. "GDPR", "ISO/IEC 27001", "NIST SP 800-53"
    section: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MemoryEntry":
        metadata: Dict[str, Any] = {}
        for key, value in payload.items():
            if key in {
                "control_id",
                "id",
                "text",
                "control_text",
                "body",
                "source",
                "framework",
                "section",
                "section_id",
            }:
                continue
            if key == "metadata" and isinstance(value, dict):
                metadata.update(_flatten_metadata_dict(value))
            else:
                metadata[key] = _coerce_chroma_value(value)
        return cls(
            control_id=str(payload.get("control_id") or payload.get("id") or ""),
            text=str(
                payload.get("text")
                or payload.get("control_text")
                or payload.get("body")
                or ""
            ),
            source=str(payload.get("source") or payload.get("framework") or ""),
            section=str(payload.get("section") or payload.get("section_id") or ""),
            metadata=metadata,
        )


class MemoryIndex:
    """A retrieval-augmented memory over control text.

    Cosine (== inner product on normalised vectors) top-k search.
    """

    def __init__(
        self,
        entries: Sequence[MemoryEntry],
        embeddings: np.ndarray,
        encoder_name: str,
        build_config: Dict[str, Any] | None = None,
    ) -> None:
        if len(entries) != embeddings.shape[0]:
            raise ValueError(
                f"entries ({len(entries)}) must have same length as embeddings ({embeddings.shape[0]})"
            )
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        self.entries: List[MemoryEntry] = list(entries)
        self.embeddings: np.ndarray = embeddings
        self.encoder_name = encoder_name
        self.dim = int(embeddings.shape[1]) if embeddings.size else 0
        self.build_config: Dict[str, Any] = dict(build_config or {})

    def __len__(self) -> int:
        return len(self.entries)

    def top_k(self, query_vector: np.ndarray, k: int) -> List[int]:
        """Return the indices of the top-k most similar entries."""
        if k <= 0 or self.embeddings.shape[0] == 0:
            return []
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        # Assumes rows already L2-normalised.
        sims = self.embeddings @ query_vector.T  # (N, 1)
        sims = sims.reshape(-1)
        k = min(k, sims.shape[0])
        # argpartition then sort the top slice for deterministic ordering.
        part = np.argpartition(-sims, k - 1)[:k]
        ordered = part[np.argsort(-sims[part], kind="stable")]
        return [int(i) for i in ordered]

    def similarities(
        self, query_vector: np.ndarray, indices: Sequence[int]
    ) -> List[float]:
        if not indices:
            return []
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        rows = self.embeddings[list(indices)]
        return [float(v) for v in (rows @ query_vector.T).reshape(-1)]

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_dir / "embeddings.npz", embeddings=self.embeddings)
        with (output_dir / "entries.jsonl").open("w", encoding="utf-8") as handle:
            for entry in self.entries:
                handle.write(json.dumps(entry.as_dict(), ensure_ascii=False) + "\n")
        metadata = {
            "encoder_name": self.encoder_name,
            "dim": self.dim,
            "count": len(self.entries),
            "build_config": self.build_config,
            "corpus_hash": self._corpus_hash(),
        }
        with (output_dir / "index_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        _LOGGER.info(
            "Saved CNM memory index to %s (n=%d, dim=%d)",
            output_dir,
            len(self),
            self.dim,
        )

    def _corpus_hash(self) -> str:
        hasher = hashlib.sha256()
        for entry in self.entries:
            hasher.update(entry.control_id.encode("utf-8"))
            hasher.update(b"\x1f")
            hasher.update(entry.text.encode("utf-8"))
            hasher.update(b"\x1e")
        return hasher.hexdigest()


def load_memory_index(index_dir: Path) -> MemoryIndex:
    """Load a MemoryIndex previously saved by MemoryIndex.save()."""
    with (index_dir / "index_metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    embeddings = np.load(index_dir / "embeddings.npz")["embeddings"].astype(np.float32)
    entries: List[MemoryEntry] = []
    with (index_dir / "entries.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(MemoryEntry.from_dict(json.loads(line)))
    return MemoryIndex(
        entries=entries,
        embeddings=embeddings,
        encoder_name=metadata["encoder_name"],
        build_config=metadata.get("build_config", {}),
    )


def _load_controls_file(controls_path: Path) -> List[MemoryEntry]:
    """Load a controls file (JSON list or JSONL) into MemoryEntry records.

    Dispatch order:
      - .jsonl suffix   -> parse line-by-line JSONL
      - .json suffix    -> parse as a single JSON document (list or wrapper dict)
      - anything else   -> try JSONL first, then fall back to a single JSON parse
    """
    text = controls_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    suffix = controls_path.suffix.lower()
    raw: Iterable[Dict[str, Any]] = []

    def _from_json_document(parsed: Any) -> Iterable[Dict[str, Any]]:
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("controls", "entries", "items", "data"):
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
            raise ValueError(
                f"Cannot find controls list in {controls_path}; "
                "expected a top-level list or a dict with key 'controls'/'entries'/'items'/'data'."
            )
        raise ValueError(f"Unsupported controls format in {controls_path}")

    if suffix == ".jsonl":
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif suffix == ".json":
        raw = _from_json_document(json.loads(text))
    else:
        # Unknown suffix — try JSONL first (line-by-line), then single JSON.
        try:
            raw = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError:
            raw = _from_json_document(json.loads(text))

    entries = [MemoryEntry.from_dict(item) for item in raw]
    # Drop entries with no text — they can't be embedded meaningfully.
    entries = [e for e in entries if e.text.strip()]
    return entries


def build_memory_index(
    controls_path: Path,
    output_dir: Path,
    encoder_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    device: str | None = None,
) -> MemoryIndex:
    """Build and persist a MemoryIndex from a controls JSON/JSONL file."""
    entries = _load_controls_file(controls_path)
    if not entries:
        raise ValueError(f"No usable controls found in {controls_path}")
    _LOGGER.info("Encoding %d controls with %s", len(entries), encoder_name)
    encoder = TextEncoder(model_name=encoder_name, device=device)
    texts = [entry.text for entry in entries]
    embeddings = encoder.encode(texts, batch_size=batch_size, normalise=True)
    build_config = {
        "controls_path": str(controls_path),
        "encoder_backend": encoder.backend,
        "batch_size": batch_size,
    }
    index = MemoryIndex(
        entries=entries,
        embeddings=embeddings,
        encoder_name=encoder_name,
        build_config=build_config,
    )
    index.save(output_dir)
    return index
