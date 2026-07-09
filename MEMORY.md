# PrERT-CNM-v4 Project Memory

## Project Overview

PrERT-CNM-v4 is an open-source framework for **Strengthening User Privacy through AI-Driven Risk Quantification and International Standards Alignment**. It operates through a multi-phase pipeline:

- **Phase 1**: Regulation extraction — parses GDPR, NIST, ISO standards into normalised JSONL and ingests into Chroma Cloud vector DB.
- **Phase 2**: Metric generation — builds privacy metrics and generates synthetic observation data.
- **Phase 3**: Classification & risk scoring — PrivacyBERT fine-tuned on OPP-115 + Polisis; Bayesian risk scoring via Naive Bayes.
- **Phase 4**: Validation & GUI — Gradio web app for compliance assessment, benchmark validation, synthetic data generation, and artifact exploration.

## Current Stage

The project is currently in a **Phase 4 maturity and dissemination stage**:

- Core compliance and benchmarking workflows are implemented and being hardened for reproducible validation runs.
- Synthetic policy/schema generation assets are expanded for broader demonstration coverage.
- Documentation has been elevated to include thesis-ready materials, references, and presentation artifacts.

## Architecture Decisions

- **Lazy package exports**: Phase 4 `__init__.py` uses `__getattr__` for deferred imports to minimise startup cost.
- **Dual assessment paths**: `assess_policy_schema_compliance` (policy + schema) and `assess_policy_compliance` (policy-only) coexist in `compliance_assessor.py`. The web app routes based on whether a schema file is uploaded.
- **Regulation reference data**: `REGULATION_CONTROLS` dict embeds GDPR articles, NIST subcategories, and ISO 27701 controls mapped to each `POLICY_CHECK_SPECS` check_id. This is deterministic keyword matching — no LLM or vector DB at runtime.
- **Per-regulation verdicts**: Each policy clause produces independent `RegulationVerdict` objects per regulation control, with `compliant` bool, `reason` string, and `cited_clauses` list of exact policy text.
- **Benchmark-centric validation**: Live PrivacyBERT benchmarking now supports clearer primary-model interpretation and release-oriented reporting guidance.
- **Risk in compliance flow**: Bayesian risk assessment is integrated into compliance pipeline execution, reducing drift between scoring and compliance outputs.

## Active Tasks

### Completed

- Policy-only compliance assessment (`assess_policy_compliance`) with source citations
- Per-regulation independent scoring (GDPR, NIST, ISO/IEC)
- Gradio GUI wired for policy-only assessment, benchmark results, synthetic data, and artifact exploration
- Full test suite: 9 new tests in `test_phase4_policy_compliance.py`, all 24 Phase 4 tests passing
- Live PrivacyBERT benchmarking workflow plus primary-model identification refinements
- Synthetic policy/document class expansion for Phase 4 synthetic schema generation
- Bayesian risk integration into compliance pipeline with legacy artifact cleanup
- Thesis documentation package and presentation assets added for project dissemination

### Next Steps

- Lock a reproducible benchmark baseline for report snapshots and release notes.
- Consider adding CCPA/LGPD regulation frameworks if scope expands.
- Optional GUI polish: add regulation-level visual summaries (bar/radar).
- Package final end-to-end demo evidence (screenshots and walk-through) for thesis defense use.

## Key Files

| File                                       | Purpose                                                           |
| ------------------------------------------ | ----------------------------------------------------------------- |
| `src/prert/phase4/compliance_assessor.py`  | Core assessment engine — both policy+schema and policy-only paths |
| `src/prert/phase4/web_app.py`              | Gradio GUI entry point for all Phase 4 workflows                   |
| `src/prert/phase4/pipeline.py`             | Phase 4 artifact validation orchestrator                          |
| `src/prert/phase4/__init__.py`             | Package exports (lazy)                                            |
| `tests/test_phase4_policy_compliance.py`   | Tests for policy-only assessment                                  |
| `tests/test_phase4_compliance_assessor.py` | Tests for policy+schema assessment                                |
