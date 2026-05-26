# PrERT-CNM-v4

> This repository contains the working implementation for Phases 1 to 4 of the Privacy Evaluation and Risk Quantification Tool (PrERT) project. It covers regulation extraction, metrics and synthetic data generation, privacy-clause classification and Bayesian risk scoring, and Phase 4 validation artefacts. The current emphasis is on strengthening privacy-focused model quality, dataset suitability, and reproducibility for a defensible future publication attempt.

## Total Time Spent on Project PrERT

|    Version     |                                                                                                                                                                                                                                       Coding Time Spent | Research Time Spent |
| :------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | ------------------: |
|       v1       | [![wakatime](https://wakatime.com/badge/user/2d4d1d3d-9942-415a-87fc-0530a909486d/project/8e8bea5e-4532-4823-9e8a-e64b5aef2c5e.svg)](https://wakatime.com/badge/user/2d4d1d3d-9942-415a-87fc-0530a909486d/project/8e8bea5e-4532-4823-9e8a-e64b5aef2c5e) |            20 hours |
|       v2       | [![wakatime](https://wakatime.com/badge/user/2d4d1d3d-9942-415a-87fc-0530a909486d/project/4cfafaba-bafb-4ed8-8e0d-9d074a369e55.svg)](https://wakatime.com/badge/user/2d4d1d3d-9942-415a-87fc-0530a909486d/project/4cfafaba-bafb-4ed8-8e0d-9d074a369e55) |            15 hours |
|       v3       | [![wakatime](https://wakatime.com/badge/user/2d4d1d3d-9942-415a-87fc-0530a909486d/project/21793439-1f64-4645-9090-cf7e1ecc0411.svg)](https://wakatime.com/badge/user/2d4d1d3d-9942-415a-87fc-0530a909486d/project/21793439-1f64-4645-9090-cf7e1ecc0411) |            25 hours |
|       v4       |                                                                                                             [![wakatime](https://wakatime.com/badge/github/JayNightmare/PrERT-CNM-v4.svg)](https://wakatime.com/badge/github/JayNightmare/PrERT-CNM-v4) |            10 hours |
|                |
| **Total Sums** |                                                                                                                                                                                                                                      **56 hrs 56 mins** |          **70 hrs** |
|                |
|   **Total**    |                                                                                                                                                                                                                                     **126 hrs 56 mins** |                     |

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

The default Phase 3 transformer backbone now points to the published PrivBERT checkpoint, while remaining overrideable through the CLI for controlled benchmark runs.

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
- Phase 2 metric definition, synthetic observation generation, and baseline scoring
- Phase 3 clause classification, calibration, bootstrap, and Bayesian risk outputs
- Phase 4 validation, leaderboard reporting, and demo-bundle preparation
- Unit and integration tests across extraction, dataset processing, modelling, and validation workflows

## Project Structure

- `src/prert/extract/` - GDPR, ISO, and NIST parsers + control schema
- `src/prert/chunking/` - line chunking with 16 KiB limit handling
- `src/prert/chroma/` - Chroma client, schema config, and search payload builders
- `src/prert/phase2/` - metric definitions, synthetic data generation, and scoring
- `src/prert/phase3/` - dataset preparation, classifier training, analytics, and acceptance outputs
- `src/prert/phase4/` - validation, compliance assessment, and synthetic evaluation assets
- `src/prert/cli/` - end-to-end CLI entry points for all implemented phases
- `scripts/` - script wrappers for phase commands
- `artifacts/` - committed benchmark artefacts, reports, and demo-ready outputs
- `data/` - dataset notes and local raw/processed data locations
- `docs/` - execution playbooks, proposal material, and review notes
- `deployment/` - demo bundle manifests and deployment-facing assets
- `tests/` - parser, pipeline, acceptance, and validation tests

## Prerequisites

- Python 3.11+
- Chroma Cloud credentials in `.env` for Chroma-backed extraction and migration workflows
- Regulation source DOCX files under `docs/Standards/Regulations/`
- OPP-115 extracted under `data/raw/OPP-115` for the default Phase 3 training path

Required environment variables:

- `CHROMA_HOST`
- `CHROMA_API_KEY`
- `CHROMA_TENANT`
- `CHROMA_DATABASE`
- `CHROMA_COLLECTION_NAME` (optional default collection label)

Start by copying `.env.example` to `.env` and filling in the Chroma values that apply to your workspace.

## Install

From repository root:

```bash
python -m pip install -e .
```

Optional development dependencies:

```bash
python -m pip install -e .[dev]
```

For a reproducible local setup, use `.env.example` as the starting point for environment configuration and the canonical command reference for the supported run order.

## Quick Start Command Hub

Canonical command reference: [docs/Project/Execution-Playbook/12-command-reference.md](docs/Project/Execution-Playbook/12-command-reference.md)

Primary command style:

```bash
prert <command>
```

Preflight setup check:

```bash
prert doctor
```

Guided command order:

```bash
prert guide --goal full
```

Interactive command picker:

```bash
prert interactive
```

Golden path (recommended):

1. `prert extract --chunk --output-dir artifacts/phase-1`
2. `prert migrate --input-dir artifacts/phase-1`
3. `prert phase2`
4. `prert phase3`
5. `prert phase4 --baseline-dir artifacts/phase-3-freeze`

If you need script wrappers, they are still supported and mapped in the canonical command reference.

## Detailed Command Examples

The examples below retain script-wrapper forms for compatibility.
For the canonical command style and the full matrix, use [docs/Project/Execution-Playbook/12-command-reference.md](docs/Project/Execution-Playbook/12-command-reference.md).

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

Require Polisis source evidence as a blocking benchmark criterion:

```bash
PYTHONPATH=src python scripts/run_phase4_validation.py \
  --baseline-dir artifacts/phase-3-freeze \
  --comparison-dirs artifacts/phase-3-nb artifacts/phase-3-logreg artifacts/phase-3-privacybert \
  --output-dir artifacts/phase-4 \
  --require-polisis \
  --strict
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

## 11) Prepare Live Demo Bundle (Supervisor Presentation)

Create a deployment-safe bundle that includes:

- Baseline benchmark files for Phase 4 validation
- Naive Bayes `model.json` checkpoint for model-signal scoring in the web app

Run:

```bash
PYTHONPATH=src python scripts/prepare_phase4_demo_bundle.py \
  --include-nb-artifacts \
  --overwrite
```

This writes files under:

```text
deployment/demo-assets/
```

Generated structure includes:

- `deployment/demo-assets/phase-3-freeze/` (baseline benchmark files)
- `deployment/demo-assets/phase-3-nb/` (optional comparison benchmark files)
- `deployment/demo-assets/phase-3-nb/classifier_checkpoint/model.json` (web app model signal)

## 12) Launch Web App With Deployment Paths

The app now supports environment-variable overrides for hosted demos:

- `PRERT_PHASE4_MODEL_PATH`
- `PRERT_PHASE4_BASELINE_DIR`
- `PRERT_PHASE4_COMPARISON_DIRS` (comma or newline separated)
- `PRERT_PHASE4_OUTPUT_DIR`

Example:

```bash
export PRERT_PHASE4_MODEL_PATH="deployment/demo-assets/phase-3-nb/classifier_checkpoint/model.json"
export PRERT_PHASE4_BASELINE_DIR="deployment/demo-assets/phase-3-freeze"
export PRERT_PHASE4_COMPARISON_DIRS="deployment/demo-assets/phase-3-nb"
export PRERT_PHASE4_OUTPUT_DIR="artifacts/phase-4"

prert-phase4-web --port 8501
```

If variables are not set, the app auto-falls back to bundled deployment paths first, then local `artifacts/` paths.

## 13) Streamlit Community Cloud (Public Demo URL)

1. Push your repository (including `deployment/demo-assets/`) to GitHub.
2. In Streamlit Community Cloud, create a new app from your repo.
3. Set the main file path to:
      - `src/prert/phase4/web_app.py`
4. Add optional app environment variables from section 12 (use `.streamlit/secrets.toml.example` as a template).
5. Runtime defaults are preconfigured in `.streamlit/config.toml`.
6. Deploy and share the generated URL with your supervisor.

## 14) Auto-Update WakaTime Total (No Manual Summing)

Update the `Total` row in this README by summing all `v*` WakaTime badges:

```bash
PYTHONPATH=src python scripts/update_wakatime_totals.py --write
```

This repo also includes a scheduled workflow at `.github/workflows/update-wakatime-total.yml`.

- Runs daily and on manual dispatch.
- Commits README changes only when the total has changed.

## Expected Console Output

## Extraction (example)

```text
Wrote 98 GDPR control rows
Wrote 32 ISO 27001 control rows (iso27001)
Wrote 102 ISO 27002 control rows (iso27002)
Wrote 60 ISO 27005 control rows (iso27005)
Wrote 102 ISO 27018 control rows (iso27018)
Wrote 125 ISO 27701 control rows (iso27701)
Wrote 38 ISO 29100 control rows (iso29100)
Wrote 182 ISO 29151 control rows (iso29151)
Wrote 59 ISO 15944-12 control rows (iso15944_12)
Wrote 138 NIST PF control rows
```

## Dry-run migration (example)

```text
Preparing collection 'gdpr_controls' with 98 rows
Preparing collection 'iso27001_controls' with 33 rows
Preparing collection 'iso27002_controls' with 105 rows
Preparing collection 'iso15944_12_controls' with 67 rows
Preparing collection 'nist_controls' with 139 rows
Migration complete. Total rows processed: 952
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
- `controls_iso*.jsonl` (one file per ISO standard, for example `controls_iso27002.jsonl`)
- `controls_nistpf.jsonl`
- `controls_all.jsonl`
- `chunks_gdpr.jsonl`
- `chunks_iso*.jsonl` (one file per ISO standard, for example `chunks_iso27002.jsonl`)
- `chunks_nistpf.jsonl`
- `chunks_all.jsonl`

Current baseline row counts:

- `controls_gdpr.jsonl`: 98
- `controls_nistpf.jsonl`: 138
- `chunks_gdpr.jsonl`: 98
- `chunks_nistpf.jsonl`: 139
- Per-standard ISO control/chunk counts are generated dynamically and validated against `docs/Standards/Regulations/iso_clause_id_baseline.json`.

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
	"source_path": "docs/Standards/Regulations/GDPR-2016_679.docx",
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
		"source_path": "docs/Standards/Regulations/GDPR-2016_679.docx",
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
- `iso*_controls` (one collection per ISO standard, for example `iso27002_controls`)
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

- Phase 1 extraction is DOCX-only for GDPR, ISO standards, and NIST PF inputs.
- If Chroma SDK features are unavailable at runtime, OpenAPI fallback is used.
- If a migration reports missing files, run extraction first.
- If outputs differ from baseline counts, inspect source document updates and parser changes.

## Additional Reference

Detailed phase runbook:

- `docs/Project/Execution-Playbook/06-phase1-implementation-runbook.md`
