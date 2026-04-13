# PrERT-CNM-v4

> This repository contains the implementation for Phase 1 of the Privacy Evaluation and Risk Quantification Tool (PrERT) project, focused on regulation-specific control extraction, chunking, and Chroma Cloud ingestion. It also includes initial work towards Phase 2 metrics definition and synthetic data generation.

## Total Time Spent on Project PrERT

|                                                                                                                           v1                                                                                                                            |                                                                                                                        v2                                                                                                                         |                                                                                                                        v3                                                                                                                         |                                                                                                                        v4                                                                                                                         |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [![wakatime](https://wakatime.com/badge/user/2d4d1d3d-9942-415a-87fc-0530a909486d/project/8e8bea5e-4532-4823-9e8a-e64b5aef2c5e.svg)](https://wakatime.com/badge/user/2d4d1d3d-9942-415a-87fc-0530a909486d/project/8e8bea5e-4532-4823-9e8a-e64b5aef2c5e) | [![v2](https://wakatime.com/badge/user/2d4d1d3d-9942-415a-87fc-0530a909486d/project/4cfafaba-bafb-4ed8-8e0d-9d074a369e55.svg)](https://wakatime.com/badge/user/2d4d1d3d-9942-415a-87fc-0530a909486d/project/4cfafaba-bafb-4ed8-8e0d-9d074a369e55) | [![v3](https://wakatime.com/badge/user/2d4d1d3d-9942-415a-87fc-0530a909486d/project/21793439-1f64-4645-9090-cf7e1ecc0411.svg)](https://wakatime.com/badge/user/2d4d1d3d-9942-415a-87fc-0530a909486d/project/21793439-1f64-4645-9090-cf7e1ecc0411) | [![v4](https://wakatime.com/badge/user/2d4d1d3d-9942-415a-87fc-0530a909486d/project/bbab7b0b-e4bf-4f9e-ba2a-507491705ea4.svg)](https://wakatime.com/badge/user/2d4d1d3d-9942-415a-87fc-0530a909486d/project/bbab7b0b-e4bf-4f9e-ba2a-507491705ea4) |

|     Total     |
| :-----------: |
| **2d 2h 56m** |

## Phase Breakdowns

### Phase 1: Control Extraction and Chroma Ingestion

Phase 1 implementation for regulation-specific control extraction and Chroma Cloud ingestion.

This extracts ground-truth controls from:

- GDPR (articles and sub-clauses)
- ISO/IEC 27001
- NIST Privacy Framework 1.1

It then normalizes and chunks the controls for search and ingestion into Chroma Cloud.

### Phase 2: Metrics Definition, Synthetic Data Generation, and Baseline Scoring

Phase 2 implementation for multi-level privacy metrics definition, synthetic data generation, and baseline scoring.

This defines user, system, and organization-level metrics based on Phase 1 controls, generates synthetic observations for testing, and produces baseline scoring outputs. It also supports optional enrichment with public breach datasets.

### Phase 3: AI Prototype for Privacy Clause Classification and Risk Scoring

Phase 3 implementation for an AI prototype that combines PrivacyBERT-based clause classification with Bayesian/probabilistic risk modeling.

This fine-tunes PrivacyBERT on OPP-115 data and supports first-class Polisis ingestion through normalized JSONL/CSV inputs, then integrates outputs with a risk scoring model to produce user privacy quantification outputs.

### Phase 4: Prototype Validation, Benchmarking, and Final Reporting

Phase 4 implementation for validating the Phase 3 prototype, benchmarking against defined metrics, and delivering final project outputs.

This tests the prototype on real and synthetic data, benchmarks outputs against defined metrics, refines the model based on validation findings, and delivers the final report and validated tool.

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
python -m pip install -e .
```

Optional development dependencies:

```bash
python -m pip install -e .[dev]
```

## Commands To Run

## 1) Extract Controls and Chunks

```bash
PYTHONPATH=src python scripts/extract_phase1_controls.py \
  --chunk \
  --output-dir artifacts/phase-1
```

Equivalent package script:

```bash
prert-extract --chunk --output-dir artifacts/phase-1
```

## 2) Run Unit Tests

```bash
PYTHONPATH=src python -m pytest tests -q
```

## 3) Dry-Run Migration to Chroma (No Writes)

```bash
PYTHONPATH=src python scripts/migrate_to_chroma.py \
  --input-dir artifacts/phase-1 \
  --dry-run
```

Equivalent package script:

```bash
prert-migrate --input-dir artifacts/phase-1 --dry-run
```

## 4) Live Migration to Chroma

```bash
PYTHONPATH=src python scripts/migrate_to_chroma.py \
  --input-dir artifacts/phase-1
```

Optional collection prefix:

```bash
PYTHONPATH=src python scripts/migrate_to_chroma.py \
  --input-dir artifacts/phase-1 \
  --collection-prefix prert_
```

## 5) Run Phase 2 Baseline Pipeline

Default run (uses Phase 1 controls and writes to `artifacts/phase-2`):

```bash
PYTHONPATH=src python scripts/run_phase2_metrics.py
```

Equivalent package script:

```bash
prert-phase2
```

Optional public dataset mapping:

```bash
PYTHONPATH=src python scripts/run_phase2_metrics.py \
	--public-input path/to/public_breach_data.csv
```

## 6) Preprocess OPP-115 For Phase 2 Public Mapping

Generate flat OPP-115 exports compatible with `--public-input`:

```bash
PYTHONPATH=src python scripts/process_opp115_for_phase2.py
```

Equivalent package script:

```bash
prert-opp115
```

Then run Phase 2 with processed OPP-115:

```bash
PYTHONPATH=src python scripts/run_phase2_metrics.py \
	--public-input data/processed/opp115_public_mapping.csv
```

## 7) Run Phase 3 Baseline / Freeze

Run Phase 3 baseline:

```bash
PYTHONPATH=src python scripts/run_phase3_baseline.py
```

Equivalent package script:

```bash
prert-phase3
```

Run Phase 3 acceptance freeze:

```bash
PYTHONPATH=src python scripts/run_phase3_acceptance_freeze.py \
  --strict
```

Equivalent package script:

```bash
prert-phase3-freeze --strict
```

## 8) Run Phase 4 Validation Tool (MVP)

Validate one or more Phase 3 artifact folders without retraining:

```bash
PYTHONPATH=src python scripts/run_phase4_validation.py \
  --baseline-dir artifacts/phase-3-freeze \
  --comparison-dirs artifacts/phase-3-nb artifacts/phase-3-logreg artifacts/phase-3-privacybert \
  --output-dir artifacts/phase-4
```

Equivalent package script:

```bash
prert-phase4 \
  --baseline-dir artifacts/phase-3-freeze \
  --comparison-dirs artifacts/phase-3-nb artifacts/phase-3-logreg artifacts/phase-3-privacybert \
  --output-dir artifacts/phase-4
```

## 9) Run Compliance Web GUI (Supervisor Workflow)

Launch a browser-based interface that accepts:

- A company privacy policy file (`.txt`, `.md`, `.pdf`)
- A database schema file (`.sql`, `.txt`, `.json`, `.yaml`, `.yml`)

Script launch:

```bash
PYTHONPATH=src python scripts/run_phase4_web.py --port 8501
```

Equivalent package command:

```bash
prert-phase4-web --port 8501
```

Then open:

```text
http://localhost:8501
```

## 10) Generate Synthetic Policy/Schema Compliance Data

Generate synthetic privacy-policy and database-schema pairs with low/medium/high compliance bands:

```bash
PYTHONPATH=src python scripts/run_phase4_synthetic_data.py \
  --output-dir artifacts/phase-4/synthetic-compliance \
  --low-count 8 \
  --medium-count 8 \
  --high-count 8 \
  --seed 42 \
  --export-upload-fixtures
```

Equivalent package command:

```bash
prert-phase4-synth \
  --output-dir artifacts/phase-4/synthetic-compliance \
  --low-count 8 \
  --medium-count 8 \
  --high-count 8 \
  --seed 42 \
  --export-upload-fixtures
```

Generated outputs include:

- `synthetic_policy_schema_pairs.jsonl`
- `synthetic_policy_schema_manifest.json`
- `synthetic_policy_schema_dictionary.md`
- `upload-fixtures/` (policy/schema files for direct GUI upload testing when enabled)

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
