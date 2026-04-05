# Phase 1 Implementation Runbook

This runbook covers the implementation that was added for:

- Regulation-specific control extraction for GDPR, ISO 27001, and NIST PF 1.1.
- Agent-friendly control schema + chunk metadata.
- Chroma Cloud ingestion with regulation-sharded collections.
- Hybrid search payload builders (dense + sparse + RRF) and GroupBy dedup support.

## Added Components

- Python package under `src/prert/`.
- Extractors:
     - `src/prert/extract/gdpr_parser.py`
     - `src/prert/extract/iso_parser.py`
     - `src/prert/extract/nist_parser.py`
- Chunking:
     - `src/prert/chunking/line_chunker.py`
- Chroma integration:
     - `src/prert/chroma/client.py` (SDK first, OpenAPI fallback)
     - `src/prert/chroma/schema.py` (Qwen dense + Splade sparse)
     - `src/prert/chroma/search.py` (dense/sparse/hybrid builders)
- CLIs:
     - `src/prert/cli/extract.py`
     - `src/prert/cli/migrate.py`
- Script wrappers:
     - `scripts/extract_phase1_controls.py`
     - `scripts/migrate_to_chroma.py`
- Tests:
     - `tests/test_extractors.py`
     - `tests/test_chunking.py`

## Install

From repo root:

```bash
python3 -m pip install -e .
```

Optional dev tools:

```bash
python3 -m pip install -e .[dev]
```

## Run Extraction (Phase 1)

```bash
PYTHONPATH=src python3 scripts/extract_phase1_controls.py \
  --chunk \
  --output-dir artifacts/phase-1
```

Expected outputs:

- `artifacts/phase-1/controls_gdpr.jsonl`
- `artifacts/phase-1/controls_iso27001.jsonl`
- `artifacts/phase-1/controls_nistpf.jsonl`
- `artifacts/phase-1/chunks_gdpr.jsonl`
- `artifacts/phase-1/chunks_iso27001.jsonl`
- `artifacts/phase-1/chunks_nistpf.jsonl`
- plus combined `controls_all.jsonl` and `chunks_all.jsonl`

## Dry-Run Chroma Migration

```bash
PYTHONPATH=src python3 scripts/migrate_to_chroma.py \
  --input-dir artifacts/phase-1 \
  --dry-run
```

This verifies collection sharding and row counts without writing to cloud.

## Live Chroma Migration

```bash
PYTHONPATH=src python3 scripts/migrate_to_chroma.py \
  --input-dir artifacts/phase-1
```

Default collection shards:

- `gdpr_controls`
- `iso27001_controls`
- `nist_controls`

Optional prefix:

```bash
PYTHONPATH=src python3 scripts/migrate_to_chroma.py \
  --input-dir artifacts/phase-1 \
  --collection-prefix prert_
```

## Validation

Run tests:

```bash
PYTHONPATH=src python3 -m pytest tests -q
```

## Chroma MCP (Optional)

For improved doc search while developing, see:

- https://docs.trychroma.com/mcp

Use it as a companion for docs lookup; runtime ingestion/search in this repo remains implemented via Python SDK + OpenAPI fallback.

## Notes

- Chunking is line-based and enforces Chroma's 16 KiB per-document limit.
- Each chunk includes metadata fields required for dedup and traceability:
     - `source_document_id`
     - `control_id`
     - `chunk_index`
     - `regulation`
- GDPR extraction excludes recitals and focuses on operative article/sub-clause structure.

---

## Navigation

[⬅ Back](05-phase-4-validation-and-reporting.md) | [Next ⮕](07-phase2-implementation-runbook.md)
