# Artifacts

This folder stores generated outputs from the PrERT-CNM pipeline.
The artifacts are organized by phase so you can trace how controls, datasets,
models, metrics, and validation outputs were produced.

## Folder layout

- `phase-1/`
- `phase-2/`
- `phase-3/`
- `phase-3-freeze/`
- `phase-3-logreg/`
- `phase-3-nb/`
- `phase-3-no-bayes/`
- `phase-3-privacybert/`
- `phase-4/`
- `phase3_run_history.jsonl`

## Phase 1: Controls and chunks

Phase 1 artifacts are the canonical control corpus and chunked retrieval index.

- `controls_all.jsonl`
  - Full set of normalized controls across all supported frameworks.
  - Use this as the source of truth for regulation coverage.
- `controls_<framework>.jsonl`
  - Framework-specific subset files.
  - Examples: `controls_gdpr.jsonl`, `controls_iso27001.jsonl`, `controls_nistpf.jsonl`.
- `chunks_all.jsonl`
  - Chunked version of all controls for retrieval workflows.
- `chunks_<framework>.jsonl`
  - Framework-specific chunk files used for targeted retrieval.

Each line in these files is a JSON object.

## Phase 2: Baselines and manifests

Phase 2 captures baseline scoring runs and processing metadata.

- `baseline_scores.jsonl`
  - Baseline model score entries.
  - Useful for comparing future experiments.
- `phase2_manifest.json`
  - Run metadata such as parameters, inputs, and summary metrics.

## Phase 3: Classifier experiments and acceptance

Phase 3 folders track multiple model and pipeline variants.

- `phase-3/`
  - General Phase 3 outputs.
- `phase-3-freeze/`
  - Acceptance or release-candidate snapshots.
- `phase-3-logreg/`
  - Logistic regression variant outputs.
- `phase-3-nb/`
  - Naive Bayes variant outputs.
- `phase-3-no-bayes/`
  - Ablation outputs with Bayesian components disabled.
- `phase-3-privacybert/`
  - PrivacyBERT-specific training and evaluation outputs.
- `phase3_run_history.jsonl`
  - Historical log of Phase 3 runs and checkpoints.

## Phase 4: Policy compliance and synthetic validation

Phase 4 artifacts support policy-level compliance assessment, synthetic
data generation, and validation workflows.

- `phase-4/`
  - Compliance run outputs, validation reports, and related manifests.

## File formats

- `*.jsonl`
  - Newline-delimited JSON. One JSON object per line.
  - Preferred for large append-only logs and datasets.
- `*.json`
  - Single JSON document for manifests and summaries.

## Reproducibility notes

- Keep generated artifacts under phase-specific folders.
- Do not manually edit generated files unless clearly marked as curated.
- Record run parameters in manifests to preserve comparability.
- Prefer `controls_all.jsonl` when complete framework coverage is required.

## Related scripts

Common scripts that produce or consume these artifacts are in the scripts
folder, including:

- `scripts/extract_phase1_controls.py`
- `scripts/run_phase2_metrics.py`
- `scripts/run_phase3_baseline.py`
- `scripts/run_phase3_acceptance_freeze.py`
- `scripts/run_phase4_validation.py`
- `scripts/run_phase4_synthetic_data.py`

## Quick inspection examples

Count rows in a JSONL artifact:

```bash
wc -l artifacts/phase-1/controls_all.jsonl
```

Preview first entries:

```bash
head -n 3 artifacts/phase-1/controls_all.jsonl
```

Filter by framework key (example):

```bash
rg "ISO_27001|ISO27001" artifacts/phase-1/controls_all.jsonl
```
