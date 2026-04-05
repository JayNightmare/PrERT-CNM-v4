"""Schema and embedding configuration for Chroma Cloud dense+sparse retrieval."""

# pyright: reportMissingImports=false

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ChromaSchemaBundle:
    schema: Any = None
    dense_embedding_function: Any = None
    sparse_key: str = "sparse_embedding"


def build_ground_truth_schema(api_key_env_var: str = "CHROMA_API_KEY") -> ChromaSchemaBundle:
    """Builds a schema bundle with Qwen dense and Splade sparse embedding support.

    Returns an empty bundle if the SDK classes are unavailable at runtime.
    """
    try:
        from chromadb import K, Schema  # type: ignore[import-not-found]
        from chromadb import SparseVectorIndexConfig  # type: ignore[import-not-found]
        from chromadb.utils.embedding_functions import (
            ChromaCloudQwenEmbeddingFunction,
            ChromaCloudSpladeEmbeddingFunction,
        )  # type: ignore[import-not-found]
    except Exception:
        return ChromaSchemaBundle()

    dense_ef = _build_qwen_embedding_function(ChromaCloudQwenEmbeddingFunction)
    sparse_ef = _build_splade_embedding_function(ChromaCloudSpladeEmbeddingFunction)

    schema = Schema()
    sparse_key = "sparse_embedding"

    if sparse_ef is not None:
        try:
            source_key = str(getattr(K, "DOCUMENT", "#document"))
            cfg = SparseVectorIndexConfig(
                source_key=source_key,
                embedding_function=sparse_ef,
            )
            schema.create_index(config=cfg, key=sparse_key)
        except Exception:
            pass

    return ChromaSchemaBundle(
        schema=schema,
        dense_embedding_function=dense_ef,
        sparse_key=sparse_key,
    )


def _build_qwen_embedding_function(ef_cls: Any) -> Optional[Any]:
    model = "QWEN3_EMBEDDING_0p6B"

    # Chroma Cloud docs require both model and task for Qwen.
    for task_name in ("retrieval_document", "semantic_search", "nl_to_code"):
        try:
            return ef_cls(model=model, task=task_name)
        except Exception:
            continue
    return None


def _build_splade_embedding_function(ef_cls: Any) -> Optional[Any]:
    model = "SPLADE_PP_EN_V1"
    try:
        return ef_cls(model=model)
    except Exception:
        return None
