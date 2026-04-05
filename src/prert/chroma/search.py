"""Search payload builders for dense, sparse, and hybrid retrieval."""

from __future__ import annotations

from typing import Any, Dict


def build_dense_search(query: str, *, limit: int = 10) -> Any:
    return _build_search_payload(
        query=query,
        limit=limit,
        mode="dense",
    )


def build_sparse_search(query: str, *, limit: int = 10, sparse_key: str = "sparse_embedding") -> Any:
    return _build_search_payload(
        query=query,
        limit=limit,
        mode="sparse",
        sparse_key=sparse_key,
    )


def build_hybrid_search(
    query: str,
    *,
    limit: int = 10,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    rrf_k: int = 60,
    sparse_key: str = "sparse_embedding",
    group_by_key: str = "source_document_id",
    dedup_per_group: int = 1,
) -> Any:
    return _build_search_payload(
        query=query,
        limit=limit,
        mode="hybrid",
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        rrf_k=rrf_k,
        sparse_key=sparse_key,
        group_by_key=group_by_key,
        dedup_per_group=dedup_per_group,
    )


def _build_search_payload(query: str, *, limit: int, mode: str, **kwargs: Any) -> Any:
    """Build SDK-native Search payload when available, else return a fallback descriptor.

    The fallback descriptor can be converted to OpenAPI search payload by callers.
    """
    try:
        from chromadb import GroupBy, K, Knn, MinK, Rrf, Search  # type: ignore[import-not-found]

        if mode == "dense":
            ranking = Knn(query=query, return_rank=False, limit=max(limit * 20, 100))
        elif mode == "sparse":
            ranking = Knn(
                query=query,
                key=kwargs.get("sparse_key", "sparse_embedding"),
                return_rank=False,
                limit=max(limit * 20, 100),
            )
        else:
            dense_rank = Knn(
                query=query,
                return_rank=True,
                limit=max(limit * 20, 100),
            )
            sparse_rank = Knn(
                query=query,
                key=kwargs.get("sparse_key", "sparse_embedding"),
                return_rank=True,
                limit=max(limit * 20, 100),
            )
            ranking = Rrf(
                ranks=[dense_rank, sparse_rank],
                weights=[kwargs.get("dense_weight", 0.7), kwargs.get("sparse_weight", 0.3)],
                k=kwargs.get("rrf_k", 60),
            )

        search = Search().rank(ranking)

        group_key = kwargs.get("group_by_key")
        if group_key:
            search = search.group_by(
                GroupBy(
                    keys=K(group_key),
                    aggregate=MinK(keys=K.SCORE, k=kwargs.get("dedup_per_group", 1)),
                )
            )

        return (
            search.limit(limit)
            .select(
                K.DOCUMENT,
                K.SCORE,
                K("regulation"),
                K("control_id"),
                K("source_document_id"),
                K("chunk_index"),
            )
        )
    except Exception:
        return _fallback_payload(query=query, limit=limit, mode=mode, **kwargs)


def _fallback_payload(query: str, *, limit: int, mode: str, **kwargs: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "query": query,
        "mode": mode,
        "limit": limit,
        "sparse_key": kwargs.get("sparse_key", "sparse_embedding"),
        "rrf_k": kwargs.get("rrf_k", 60),
        "dense_weight": kwargs.get("dense_weight", 0.7),
        "sparse_weight": kwargs.get("sparse_weight", 0.3),
        "group_by_key": kwargs.get("group_by_key", "source_document_id"),
        "dedup_per_group": kwargs.get("dedup_per_group", 1),
    }
    return payload
