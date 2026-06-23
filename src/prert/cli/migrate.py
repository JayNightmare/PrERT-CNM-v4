"""Migration CLI for loading extracted Phase 1 chunks into Chroma Cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import httpx


BASE_REGULATION_FILES = {
    "gdpr": "chunks_gdpr.jsonl",
    "nistpf": "chunks_nistpf.jsonl",
}


def main() -> None:
    from prert.chroma import ChromaCloudClient, build_ground_truth_schema
    from prert.config import ChromaSettings

    args = _parse_args()

    settings = ChromaSettings.from_env(args.env_file)
    schema_bundle = build_ground_truth_schema()
    client = ChromaCloudClient(settings)
    resetter = _CollectionResetter(settings) if args.replace_existing and not args.dry_run else None

    total_rows = 0

    try:
        for regulation, path in _discover_chunk_files(args.input_dir):
            if not path.exists():
                print(f"Skipping {regulation}: {path} does not exist")
                continue

            rows = list(_load_jsonl(path))
            if not rows:
                print(f"Skipping {regulation}: no rows")
                continue

            collection_name = _collection_name(regulation, args.collection_prefix)
            total_rows += len(rows)

            print(f"Preparing collection '{collection_name}' with {len(rows)} rows")
            if args.dry_run:
                continue

            if resetter and resetter.delete_if_exists(collection_name):
                print(f"Reset collection '{collection_name}'")

            client.get_or_create_collection(
                name=collection_name,
                schema=schema_bundle.schema,
                embedding_function=schema_bundle.dense_embedding_function,
                metadata={
                    "regulation": regulation,
                    "ground_truth": True,
                    "shard_strategy": "regulation",
                    "sparse_key": schema_bundle.sparse_key,
                },
            )

            for batch in _batched(rows, args.batch_size):
                ids = [row["chunk_id"] for row in batch]
                docs = [row["text"] for row in batch]
                metadatas = [row.get("metadata", {}) for row in batch]

                client.upsert(
                    collection_name=collection_name,
                    ids=ids,
                    documents=docs,
                    metadatas=metadatas,
                )

            print(f"Uploaded {len(rows)} rows to {collection_name}")

    finally:
        client.close()
        if resetter:
            resetter.close()

    print(f"Migration complete. Total rows processed: {total_rows}")


def _parse_args() -> argparse.Namespace:
    root = Path.cwd()

    parser = argparse.ArgumentParser(description="Migrate Phase 1 chunks to Chroma Cloud")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root / "artifacts/phase-1",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=root / ".env",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned operations without writing to Chroma Cloud.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--collection-prefix",
        type=str,
        default="",
        help="Optional prefix for collection names.",
    )
    parser.add_argument(
        "--replace-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete matching target collections before upload for exact-count synchronization.",
    )

    return parser.parse_args()


def _collection_name(regulation: str, prefix: str) -> str:
    suffix_map = {
        "gdpr": "gdpr_controls",
        "nistpf": "nist_controls",
    }
    if regulation.startswith("iso"):
        suffix = f"{regulation}_controls"
    else:
        suffix = suffix_map[regulation]
    return f"{prefix}{suffix}" if prefix else suffix


def _discover_chunk_files(input_dir: Path) -> list[tuple[str, Path]]:
    discovered: list[tuple[str, Path]] = []

    for regulation, filename in BASE_REGULATION_FILES.items():
        discovered.append((regulation, input_dir / filename))

    iso_chunk_paths = sorted(input_dir.glob("chunks_iso*.jsonl"))
    for path in iso_chunk_paths:
        key = path.stem.replace("chunks_", "", 1)
        discovered.append((key, path))

    return discovered


def _load_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _batched(rows: List[Dict], size: int) -> Iterable[List[Dict]]:
    for idx in range(0, len(rows), size):
        yield rows[idx : idx + size]


class _CollectionResetter:
    def __init__(self, settings) -> None:
        base_url = settings.host
        if not base_url.startswith("http://") and not base_url.startswith("https://"):
            base_url = f"https://{base_url}"

        self._settings = settings
        self._http = httpx.Client(
            base_url=base_url,
            headers={"x-chroma-token": settings.api_key, "content-type": "application/json"},
            timeout=30.0,
        )
        self._resolved_db = self._resolve_database_name()

    def close(self) -> None:
        self._http.close()

    def delete_if_exists(self, collection_name: str) -> bool:
        delete_resp = self._http.delete(f"{self._collections_path()}/{collection_name}")
        if delete_resp.status_code == 404:
            return False
        delete_resp.raise_for_status()
        return True

    def _collections_path(self) -> str:
        tenant = self._settings.tenant
        return f"/api/v2/tenants/{tenant}/databases/{self._resolved_db}/collections"

    def _resolve_database_name(self) -> str:
        requested = self._settings.database
        identity_resp = self._http.get("/api/v2/auth/identity")
        identity_resp.raise_for_status()
        identity = identity_resp.json()
        databases = [str(item) for item in identity.get("databases", [])]
        return next((db for db in databases if db.lower() == requested.lower()), requested)


if __name__ == "__main__":
    main()
