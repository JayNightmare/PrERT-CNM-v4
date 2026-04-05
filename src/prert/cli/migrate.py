"""Migration CLI for loading extracted Phase 1 chunks into Chroma Cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from prert.chroma import ChromaCloudClient, build_ground_truth_schema
from prert.config import ChromaSettings


REGULATION_FILES = {
    "gdpr": "chunks_gdpr.jsonl",
    "iso27001": "chunks_iso27001.jsonl",
    "nistpf": "chunks_nistpf.jsonl",
}


def main() -> None:
    args = _parse_args()

    settings = ChromaSettings.from_env(args.env_file)
    schema_bundle = build_ground_truth_schema()
    client = ChromaCloudClient(settings)

    total_rows = 0

    try:
        for regulation, filename in REGULATION_FILES.items():
            path = args.input_dir / filename
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

    return parser.parse_args()


def _collection_name(regulation: str, prefix: str) -> str:
    suffix_map = {
        "gdpr": "gdpr_controls",
        "iso27001": "iso27001_controls",
        "nistpf": "nist_controls",
    }
    suffix = suffix_map[regulation]
    return f"{prefix}{suffix}" if prefix else suffix


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


if __name__ == "__main__":
    main()
