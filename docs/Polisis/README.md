# Polisis Source Material

This folder stores Polisis and OPP-115 reference material used to support taxonomy alignment and dataset-ingestion work elsewhere in the repository.

## Current Baseline

The folder currently contains:

- `sec18-harkous.pdf`

## Relationship To The Repository

This material supports review and alignment work for:

- Step 1 taxonomy alignment decisions
- OPP-115 ingestion and normalization work under `src/prert/phase2/opp115.py`
- Phase 3 Polisis harmonization and ingestion under `src/prert/phase3/dataset.py`
- provenance and contributor context for label interpretation

## Scope Boundary

This folder is for source references only.

- It is not a generated-artifact location.
- It is not a direct runtime input folder for raw runtime ingestion.
- Runtime behavior is implemented in code and configuration elsewhere in the repository.

## Runtime Handoff For Phase 3

For runtime Phase 3 ingestion, place normalized Polisis files under:

- `data/raw/Polisis/normalized`

Supported normalized file types:

- `.jsonl`
- `.csv`

Normalized row contract for ingestion:

- Required: `text`
- Preferred for harmonization: `category`
- Optional: `label`, `policy_uid`, `example_id`

Rows with categories that cannot be harmonized to `user|system|organization` are skipped.

## Contributor Expectations

- Keep source materials identifiable and repository-specific.
- Preserve provenance when adding new Polisis or OPP-115 reference documents.
- Update nearby documentation when the role of a new source file is not obvious from its name alone.
