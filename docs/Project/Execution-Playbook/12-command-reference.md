# PrERT Command Reference

This is the canonical command map for running PrERT.

## Command Style

Preferred command style:

```bash
prert <command>
```

Compatibility command style (still supported):

```bash
PYTHONPATH=src python scripts/<wrapper>.py
```

## Golden Path (Recommended End-to-End)

1. Validate setup

```bash
prert doctor
```

1. Phase 1 extraction and chunking

```bash
prert extract --chunk --output-dir artifacts/phase-1
```

1. Phase 1 Chroma ingestion

```bash
prert migrate --input-dir artifacts/phase-1
```

1. Phase 2 metrics baseline

```bash
prert phase2
```

1. Phase 3 baseline model

```bash
prert phase3
```

1. Phase 4 validation

```bash
prert phase4 --baseline-dir artifacts/phase-3-freeze
```

## Command Matrix

| Area    | Command                   | Purpose                                                  | Typical Output                             |
| ------- | ------------------------- | -------------------------------------------------------- | ------------------------------------------ |
| Setup   | `prert doctor`            | Validate prerequisites and critical inputs               | PASS/FAIL checks                           |
| Help    | `prert guide --goal full` | Show guided run order                                    | Next-step command list                     |
| Help    | `prert interactive`       | Pick commands from a guided menu and optionally execute  | Interactive workflow selection             |
| Phase 1 | `prert extract`           | Extract controls/chunks from DOCX regulation sources     | `artifacts/phase-1/*.jsonl`                |
| Phase 1 | `prert migrate`           | Load Phase 1 chunks into Chroma Cloud                    | Chroma collections                         |
| Phase 2 | `prert phase2`            | Generate metric specs, synthetic events, baseline scores | `artifacts/phase-2/*`                      |
| Phase 2 | `prert opp115`            | Build flattened OPP-115 mapping files                    | `data/processed/opp115_public_mapping.*`   |
| Phase 3 | `prert phase3`            | Train and evaluate baseline model                        | `artifacts/phase-3*/*`                     |
| Phase 3 | `prert phase3-freeze`     | Run acceptance freeze workflow                           | `artifacts/phase-3-freeze/*`               |
| Phase 4 | `prert phase4`            | Validate and compare artifact sets                       | `artifacts/phase-4/*`                      |
| Phase 4 | `prert phase4-web`        | Launch Gradio compliance studio                          | Local or public web app                    |
| Phase 4 | `prert phase4-synth`      | Generate synthetic policy/schema fixtures                | `artifacts/phase-4/synthetic-compliance/*` |

## Workflow Presets

Full workflow:

```bash
prert guide --goal full
```

Interactive full workflow:

```bash
prert interactive --goal full
```

Non-interactive selection with direct execution:

```bash
prert interactive --goal phase1 --select 1 --execute
```

Phase-specific examples:

```bash
prert guide --goal phase1
prert guide --goal phase2
prert guide --goal phase3
prert guide --goal phase4
prert guide --goal validation
```

## Legacy Wrapper Mapping

| Preferred             | Wrapper Equivalent                                              |
| --------------------- | --------------------------------------------------------------- |
| `prert extract`       | `PYTHONPATH=src python scripts/extract_phase1_controls.py`      |
| `prert migrate`       | `PYTHONPATH=src python scripts/migrate_to_chroma.py`            |
| `prert phase2`        | `PYTHONPATH=src python scripts/run_phase2_metrics.py`           |
| `prert opp115`        | `PYTHONPATH=src python scripts/process_opp115_for_phase2.py`    |
| `prert phase3`        | `PYTHONPATH=src python scripts/run_phase3_baseline.py`          |
| `prert phase3-freeze` | `PYTHONPATH=src python scripts/run_phase3_acceptance_freeze.py` |
| `prert phase4`        | `PYTHONPATH=src python scripts/run_phase4_validation.py`        |
| `prert phase4-web`    | `PYTHONPATH=src python scripts/run_phase4_web.py`               |
| `prert phase4-synth`  | `PYTHONPATH=src python scripts/run_phase4_synthetic_data.py`    |
