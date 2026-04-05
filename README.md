# PrERT-CNM-v4

Phase 1 implementation for regulation-specific control extraction and Chroma Cloud ingestion.

This repository extracts ground-truth controls from:

- GDPR (articles and sub-clauses)
- ISO/IEC 27001
- NIST Privacy Framework 1.1

It then normalizes and chunks the controls for search and ingestion into Chroma Cloud.

## What Is Implemented

- Regulation-specific parsers (no generic one-size-fits-all parser)
- Agent-friendly normalized control schema
- 16 KiB-safe line-based chunking
- Chroma Cloud migration with regulation-sharded collections
- SDK-first Chroma client with OpenAPI fallback
- Search payload builders for dense, sparse, and hybrid (RRF) retrieval
- Unit tests for parser and chunking behavior

## Project Structure

- `src/prert/extract/` - GDPR, ISO, and NIST parsers + control schema
- `src/prert/chunking/` - line chunking with 16 KiB limit handling
- `src/prert/chroma/` - Chroma client, schema config, and search payload builders
- `src/prert/cli/` - extraction and migration CLIs
- `scripts/` - script wrappers for phase commands
- `artifacts/phase-1/` - generated control and chunk outputs
- `tests/` - parser/chunking tests

## Prerequisites

- Python 3.11+
- Optional: `pdftotext` installed locally (used for NIST PDF extraction)
- Chroma Cloud credentials in `.env`

Required environment variables:

- `CHROMA_HOST`
- `CHROMA_API_KEY`
- `CHROMA_TENANT`
- `CHROMA_DATABASE`
- `CHROMA_COLLECTION_NAME` (optional default collection label)

## Install

From repository root:

```bash
python3 -m pip install -e .
```

Optional development dependencies:

```bash
python3 -m pip install -e .[dev]
```

## Commands To Run

## 1) Extract Controls and Chunks

```bash
PYTHONPATH=src python3 scripts/extract_phase1_controls.py \
  --chunk \
  --output-dir artifacts/phase-1
```

Equivalent package script:

```bash
prert-extract --chunk --output-dir artifacts/phase-1
```

## 2) Run Unit Tests

```bash
PYTHONPATH=src python3 -m pytest tests -q
```

## 3) Dry-Run Migration to Chroma (No Writes)

```bash
PYTHONPATH=src python3 scripts/migrate_to_chroma.py \
  --input-dir artifacts/phase-1 \
  --dry-run
```

Equivalent package script:

```bash
prert-migrate --input-dir artifacts/phase-1 --dry-run
```

## 4) Live Migration to Chroma

```bash
PYTHONPATH=src python3 scripts/migrate_to_chroma.py \
  --input-dir artifacts/phase-1
```

Optional collection prefix:

```bash
PYTHONPATH=src python3 scripts/migrate_to_chroma.py \
  --input-dir artifacts/phase-1 \
  --collection-prefix prert_
```

## 5) Run Phase 2 Baseline Pipeline

Default run (uses Phase 1 controls and writes to `artifacts/phase-2`):

```bash
PYTHONPATH=src python3 scripts/run_phase2_metrics.py
```

Equivalent package script:

```bash
prert-phase2
```

Optional public dataset mapping:

```bash
PYTHONPATH=src python3 scripts/run_phase2_metrics.py \
	--public-input path/to/public_breach_data.csv
```

## Expected Console Output

## Extraction (example)

```text
Wrote 103 GDPR control rows
Wrote 69 ISO 27001 control rows
Wrote 139 NIST PF control rows
Wrote 103 GDPR chunks
Wrote 70 ISO 27001 chunks
Wrote 140 NIST PF chunks
```

## Dry-run migration (example)

```text
Preparing collection 'gdpr_controls' with 103 rows
Preparing collection 'iso27001_controls' with 70 rows
Preparing collection 'nist_controls' with 140 rows
Migration complete. Total rows processed: 313
```

## Tests (example)

```text
....                                                                     [100%]
4 passed in 0.15s
```

## Phase 2 (example)

```text
Phase 2 pipeline complete
Mapped controls: 237 / 237
Metric specs: 237
Synthetic events: 711
Public mapped rows: 0
Baseline score rows: 723
```

## Output Files

Generated under `artifacts/phase-1/`:

- `controls_gdpr.jsonl`
- `controls_iso27001.jsonl`
- `controls_nistpf.jsonl`
- `controls_all.jsonl`
- `chunks_gdpr.jsonl`
- `chunks_iso27001.jsonl`
- `chunks_nistpf.jsonl`
- `chunks_all.jsonl`
- `NIST-1.1.txt` (cached PDF extraction)

Current baseline row counts:

- `controls_gdpr.jsonl`: 103
- `controls_iso27001.jsonl`: 69
- `controls_nistpf.jsonl`: 139
- `chunks_gdpr.jsonl`: 103
- `chunks_iso27001.jsonl`: 70
- `chunks_nistpf.jsonl`: 140

## Phase 2 Outputs

Generated under `artifacts/phase-2/`:

- `metric_specs.jsonl`
- `synthetic_events.jsonl`
- `public_data_mapped.jsonl`
- `baseline_scores.jsonl`
- `phase2_manifest.json`
- `synthetic_data_dictionary.md`

## Data Format

All outputs are JSONL (one JSON object per line).

## Control record format (`controls_*.jsonl`)

Example fields:

```json
{
	"record_id": "3672973e6fc2f8823ea66176",
	"regulation": "GDPR",
	"source_document_id": "gdpr-2016_679",
	"source_path": "docs/Standards/Regulations/TX/GDPR-2016_679.txt",
	"native_id": "Article 1",
	"normalized_id": "gdpr::Article_1",
	"title": "Subject-matter and objectives",
	"text": "...",
	"hierarchy_path": ["CHAPTER I", "General provisions", "Article 1"],
	"chapter": "CHAPTER I",
	"section": "Article 1",
	"clause": "Article 1",
	"parser_confidence": 0.95,
	"metadata": {
		"format_profile": "gdpr_article_subclause",
		"ground_truth_source": true
	}
}
```

## Chunk record format (`chunks_*.jsonl`)

Example fields:

```json
{
	"chunk_id": "ee2cfd5603c8be0b857fc8b3",
	"regulation": "GDPR",
	"source_document_id": "gdpr-2016_679",
	"control_id": "gdpr::Article_1",
	"chunk_index": 0,
	"text": "...",
	"metadata": {
		"regulation": "GDPR",
		"control_id": "gdpr::Article_1",
		"native_control_id": "Article 1",
		"source_document_id": "gdpr-2016_679",
		"source_path": "docs/Standards/Regulations/TX/GDPR-2016_679.txt",
		"chunk_index": 0,
		"chapter": "CHAPTER I",
		"section": "Article 1",
		"clause": "Article 1",
		"format_profile": "gdpr_article_subclause",
		"ground_truth_source": true
	}
}
```

## Database Format (Chroma Cloud)

## Collection sharding strategy

Mutually exclusive phase-1 shards by regulation:

- `gdpr_controls`
- `iso27001_controls`
- `nist_controls`

## What is stored

- `id`: `chunk_id`
- `document`: chunk `text`
- `metadata`: chunk metadata object

## Schema/index strategy

- Dense embeddings: Chroma Cloud Qwen
- Sparse embeddings: Chroma Cloud Splade (`sparse_embedding` key)
- Hybrid retrieval: RRF (Reciprocal Rank Fusion)
- Dedup target: Group by `source_document_id` (with optional refinement to include `control_id`)

## Search Modes

Implemented payload builders in `src/prert/chroma/search.py`:

- Dense-only
- Sparse-only
- Hybrid (dense+sparse via RRF)

Hybrid defaults:

- dense weight: 0.7
- sparse weight: 0.3
- `k`: 60

## Operational Run Order

1. Install dependencies.
2. Run extraction to produce controls and chunks.
3. Run tests.
4. Run migration dry-run.
5. Run live migration.
6. Execute retrieval checks and quality evaluation.

## Troubleshooting

- If `pdftotext` is unavailable, parser falls back to `pypdf`.
- If Chroma SDK features are unavailable at runtime, OpenAPI fallback is used.
- If a migration reports missing files, run extraction first.
- If outputs differ from baseline counts, inspect source document updates and parser changes.

## Additional Reference

Detailed phase runbook:

- `docs/Project/Execution-Playbook/06-phase1-implementation-runbook.md`
