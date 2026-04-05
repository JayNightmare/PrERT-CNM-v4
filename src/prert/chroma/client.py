"""SDK-first Chroma Cloud client with OpenAPI fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from prert.config import ChromaSettings


@dataclass
class OpenApiCollectionHandle:
    api: "OpenApiChromaClient"
    name: str
    collection_id: str

    def upsert(
        self,
        *,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        return self.api.upsert(
            collection_id=self.collection_id,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def search(self, searches: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.api.search(collection_id=self.collection_id, searches=searches)


class OpenApiChromaClient:
    def __init__(self, settings: ChromaSettings, timeout: float = 30.0) -> None:
        base_url = settings.host
        if not base_url.startswith("http://") and not base_url.startswith("https://"):
            base_url = f"https://{base_url}"

        self._settings = settings
        self._http = httpx.Client(
            base_url=base_url,
            headers={
                "x-chroma-token": settings.api_key,
                "content-type": "application/json",
            },
            timeout=timeout,
        )
        self._embed_base_url = "https://embed.trychroma.com"
        self._dense_embedding_model = "Qwen/Qwen3-Embedding-0.6B"
        self._resolved_database = settings.database
        self._database_resolved = False

    def close(self) -> None:
        self._http.close()

    def get_or_create_collection(
        self,
        *,
        name: str,
        schema: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OpenApiCollectionHandle:
        collection = self._find_collection_by_name(name)
        if collection is None:
            collection = self._create_collection(name=name, schema=schema, metadata=metadata)

        collection_id = collection.get("id") or collection.get("collection_id")
        if not collection_id:
            raise RuntimeError("OpenAPI collection payload did not include an id")

        return OpenApiCollectionHandle(api=self, name=name, collection_id=collection_id)

    def upsert(
        self,
        *,
        collection_id: str,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        if embeddings is None:
            embeddings = self._generate_dense_embeddings(documents)

        payload: Dict[str, Any] = {
            "ids": ids,
            "documents": documents,
        }
        if metadatas is not None:
            payload["metadatas"] = metadatas
        if embeddings is not None:
            payload["embeddings"] = embeddings

        path = self._collection_path(collection_id, "upsert")
        resp = self._http.post(path, json=payload)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def _generate_dense_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        payload: Dict[str, Any] = {
            "texts": texts,
            "instructions": "retrieval_document",
        }
        headers = {
            "x-chroma-token": self._settings.api_key,
            "x-chroma-embedding-model": self._dense_embedding_model,
            "content-type": "application/json",
        }

        resp = httpx.post(
            f"{self._embed_base_url}/embed",
            headers=headers,
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise RuntimeError("Dense embeddings API did not return an embeddings list")

        return embeddings

    def search(self, *, collection_id: str, searches: List[Dict[str, Any]]) -> Dict[str, Any]:
        path = self._collection_path(collection_id, "search")
        payload = {"searches": searches}
        resp = self._http.post(path, json=payload)
        resp.raise_for_status()
        return resp.json()

    def _find_collection_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        resp = self._http.get(self._collections_base_path())
        resp.raise_for_status()
        collections = resp.json()
        for collection in collections:
            if collection.get("name") == name:
                return collection
        return None

    def _create_collection(
        self,
        *,
        name: str,
        schema: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": name,
            "get_or_create": True,
        }
        if metadata:
            payload["metadata"] = metadata
        if schema is not None:
            payload["schema"] = self._safe_json_schema(schema)

        resp = self._http.post(self._collections_base_path(), json=payload)
        resp.raise_for_status()
        return resp.json()

    def _safe_json_schema(self, schema: Any) -> Any:
        if isinstance(schema, dict):
            return schema
        if hasattr(schema, "to_dict"):
            return schema.to_dict()
        if hasattr(schema, "model_dump"):
            return schema.model_dump()
        return None

    def _collections_base_path(self) -> str:
        s = self._settings
        database = self._effective_database()
        return f"/api/v2/tenants/{s.tenant}/databases/{database}/collections"

    def _collection_path(self, collection_id: str, suffix: str) -> str:
        base = self._collections_base_path()
        return f"{base}/{collection_id}/{suffix}"

    def _effective_database(self) -> str:
        if not self._database_resolved:
            self._resolved_database = self._resolve_database_name()
            self._database_resolved = True
        return self._resolved_database

    def _resolve_database_name(self) -> str:
        requested = self._settings.database

        if self._database_exists(requested):
            return requested

        databases = self._identity_databases()
        match = next((db for db in databases if db.lower() == requested.lower()), None)
        if match:
            return match

        if databases:
            db_csv = ", ".join(databases)
            raise RuntimeError(
                f"Chroma database '{requested}' is not accessible for tenant '{self._settings.tenant}'. "
                f"Available databases: {db_csv}. Check CHROMA_DATABASE in .env (case-sensitive)."
            )

        raise RuntimeError(
            f"Chroma database '{requested}' is not accessible for tenant '{self._settings.tenant}'. "
            "Check CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE values."
        )

    def _database_exists(self, database_name: str) -> bool:
        path = f"/api/v2/tenants/{self._settings.tenant}/databases/{database_name}"
        resp = self._http.get(path)
        if resp.status_code == 200:
            return True
        if resp.status_code in (403, 404):
            return False
        resp.raise_for_status()
        return False

    def _identity_databases(self) -> List[str]:
        resp = self._http.get("/api/v2/auth/identity")
        resp.raise_for_status()
        payload = resp.json()
        databases = payload.get("databases")
        if isinstance(databases, list):
            return [str(db) for db in databases]
        return []


class ChromaCloudClient:
    """Unified wrapper around SDK and OpenAPI fallback clients."""

    def __init__(self, settings: ChromaSettings, timeout: float = 30.0) -> None:
        self.settings = settings
        self._fallback = OpenApiChromaClient(settings, timeout=timeout)
        self._sdk_client = self._build_sdk_client(settings)
        self._collection_cache: Dict[str, Any] = {}

    @property
    def using_sdk(self) -> bool:
        return self._sdk_client is not None

    def close(self) -> None:
        self._fallback.close()

    def get_or_create_collection(
        self,
        *,
        name: str,
        schema: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding_function: Any = None,
    ) -> Any:
        if self._sdk_client is not None:
            try:
                collection = self._get_or_create_sdk_collection(
                    name=name,
                    schema=schema,
                    metadata=metadata,
                    embedding_function=embedding_function,
                )
                self._collection_cache[name] = collection
                return collection
            except Exception:
                pass

        collection = self._fallback.get_or_create_collection(name=name, schema=schema, metadata=metadata)
        self._collection_cache[name] = collection
        return collection

    def upsert(
        self,
        *,
        collection_name: str,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> Any:
        collection = self._collection_cache.get(collection_name)
        if collection is None:
            collection = self.get_or_create_collection(name=collection_name)

        if self._sdk_client is not None and not isinstance(collection, OpenApiCollectionHandle):
            kwargs: Dict[str, Any] = {
                "ids": ids,
                "documents": documents,
            }
            if metadatas is not None:
                kwargs["metadatas"] = metadatas
            if embeddings is not None:
                kwargs["embeddings"] = embeddings
            return collection.upsert(**kwargs)

        return collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def search(
        self,
        *,
        collection_name: str,
        search_payload: Any = None,
        query_text: Optional[str] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Any:
        collection = self._collection_cache.get(collection_name)
        if collection is None:
            collection = self.get_or_create_collection(name=collection_name)

        if self._sdk_client is not None and not isinstance(collection, OpenApiCollectionHandle):
            if search_payload is not None and hasattr(collection, "search"):
                return collection.search(search_payload)
            if query_text is not None and hasattr(collection, "query"):
                kwargs: Dict[str, Any] = {
                    "query_texts": [query_text],
                    "n_results": n_results,
                }
                if where is not None:
                    kwargs["where"] = where
                return collection.query(**kwargs)
            raise ValueError("SDK search requires either a Search payload or query_text")

        if search_payload is None:
            raise ValueError("OpenAPI fallback currently requires explicit search payload")

        searches = search_payload if isinstance(search_payload, list) else [search_payload]
        return collection.search(searches=searches)

    def _get_or_create_sdk_collection(
        self,
        *,
        name: str,
        schema: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding_function: Any = None,
    ) -> Any:
        client = self._sdk_client
        kwargs: Dict[str, Any] = {"name": name}

        if metadata is not None:
            kwargs["metadata"] = metadata
        if schema is not None:
            kwargs["schema"] = schema
        if embedding_function is not None:
            kwargs["embedding_function"] = embedding_function

        if hasattr(client, "get_or_create_collection"):
            return client.get_or_create_collection(**kwargs)

        if hasattr(client, "create_collection"):
            kwargs["get_or_create"] = True
            return client.create_collection(**kwargs)

        raise RuntimeError("Cloud SDK client does not expose collection creation methods")

    def _build_sdk_client(self, settings: ChromaSettings) -> Any:
        try:
            import chromadb  # type: ignore[import-not-found]
        except Exception:
            return None

        # First, try the documented CloudClient constructor.
        try:
            return chromadb.CloudClient(
                tenant=settings.tenant,
                database=settings.database,
                api_key=settings.api_key,
            )
        except Exception:
            pass

        # Fallback: derive explicit cloud host/port/ssl values when host is customized.
        cloud_host, cloud_port, enable_ssl = self._derive_cloud_endpoint(settings.host)
        try:
            return chromadb.CloudClient(
                tenant=settings.tenant,
                database=settings.database,
                api_key=settings.api_key,
                cloud_host=cloud_host,
                cloud_port=cloud_port,
                enable_ssl=enable_ssl,
            )
        except Exception:
            pass

        return None

    @staticmethod
    def _derive_cloud_endpoint(host_value: str) -> tuple[str, int, bool]:
        host_value = host_value.strip()
        if host_value.startswith("http://") or host_value.startswith("https://"):
            parsed = urlparse(host_value)
            cloud_host = parsed.hostname or "api.trychroma.com"
            enable_ssl = parsed.scheme != "http"
            if parsed.port is not None:
                cloud_port = parsed.port
            else:
                cloud_port = 443 if enable_ssl else 80
            return cloud_host, cloud_port, enable_ssl

        return host_value or "api.trychroma.com", 443, True
