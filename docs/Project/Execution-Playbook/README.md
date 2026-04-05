# Privacy Risk Quantification Execution Playbook

This playbook expands the proposal into an implementation-ready execution plan.

## Purpose

Provide practical guidance to complete each of the 4 phases with:

- Clear goals and boundaries.
- Recommended technology stack with rationale.
- Actionable step-by-step tasks.
- Quality gates and completion criteria.
- Risks and mitigation strategies.

## Visualisations

- [Phase 1-2 Progress and Accuracy Dashboard](09-phase1-phase2-progress-dashboard.md) - Charts and figure tables for extraction progress, mapping coverage, risk distribution, and scenario stability.
- [Phase 3 Visual Dashboard](11-phase3-visual-dashboard.md) - Dataset mix, split profile, held-out classifier indicators, baseline-vs-upgraded model comparisons, and confusion flow.

## Implementation Status Snapshot

- Phase 1: implemented and operational via `scripts/run_phase1_pipeline.py`.
- Phase 2: implemented and operational via `scripts/run_phase2_metrics.py`.
- Phase 3 model backends: implemented and operational via `scripts/run_phase3_baseline.py` (`--model-type naive_bayes|logreg_tfidf|privacybert`).
- Phase 3 Bayesian risk output layer: implemented as default artifact output (`bayesian_risk_validation.json`, `bayesian_risk_test.json`) and recorded in `phase3_manifest.json` as the primary metric surface when enabled.
- Latest comparable benchmark artifacts: `artifacts/phase-3-nb/` and `artifacts/phase-3-logreg/`.
- Latest recorded combined regression command completed successfully: `PYTHONPATH=src pytest -q tests/test_phase2_pipeline.py tests/test_phase3_pipeline.py`
- Phase 3 next increment: strengthen PrivacyBERT training workflow, calibrate Bayesian priors/posteriors with benchmarked risk metrics, and finalize API integration.

## Document Index

1. `01-tech-stack.md`
      - Recommended architecture, tools, and package choices.
      - Rationale and alternatives.
2. `02-phase-1-standards-mapping.md`
      - Detailed execution of standards-to-indicators mapping.
3. `03-phase-2-metrics-and-data.md`
      - Detailed execution of metric design and synthetic/public data preparation.
4. `04-phase-3-ai-prototype.md`
      - Detailed execution of PrivacyBERT + Bayesian scoring prototype.
5. `05-phase-4-validation-and-reporting.md`
      - Detailed execution of validation, benchmarking, and reporting.
6. `06-phase1-implementation-runbook.md`
      - Commands and operational steps for running the implemented Phase 1 pipeline.
7. `07-phase2-implementation-runbook.md`
      - Commands and operational steps for running the implemented Phase 2 pipeline.
8. `08-phase2-technical-documentation.md`
      - Architecture, rationale, data contracts, scoring logic, and quality gates for Phase 2.
9. [09-phase1-phase2-progress-dashboard.md](09-phase1-phase2-progress-dashboard.md)
      - Visual progress dashboard with charts and figure tables for Phase 1 and Phase 2 quality indicators.
10. [10-phase3-implementation-runbook.md](10-phase3-implementation-runbook.md)
       - Commands and operational steps for running the implemented Phase 3 classifier backends and Bayesian risk output pipeline.
11. [11-phase3-visual-dashboard.md](11-phase3-visual-dashboard.md)
       - Visual dashboard for Phase 3 dataset profile, held-out classifier quality metrics, confusion flow, and Bayesian scoring context.

## Suggested Working Rhythm

- Weekly cadence: plan on Monday, execute Tue-Thu, review Friday.
- End of each phase: produce a signed-off artifact package before moving to next phase.
- Keep a living assumptions log and decision log to reduce rework.

## Phase Dependency Map

- Phase 1 output (indicator catalog) is required by Phase 2.
- Phase 2 output (metric definitions + data assets) is required by Phase 3.
- Phase 3 output (integrated prototype) is required by Phase 4.

If one phase slips, prioritize preserving quality gates over artificial schedule adherence.

## Standard Artifact Folder Convention (Recommended)

```text
project-root/
  artifacts/
    phase-1/
      standards-mapping.csv
      indicator-catalog.md
      traceability-matrix.md
    phase-2/
      metric-specs.md
      synthetic-data-dictionary.md
      baseline-results.csv
    phase-3/
      model-card.md
      scoring-spec.md
      prototype-demo.md
    phase-4/
      benchmark-report.md
      validation-summary.md
      final-report.md
```

## Exit Criteria for Entire Project

- A standards-aligned indicator framework exists and is traceable.
- User/system/organization-level metrics are defined and tested.
- AI prototype classifies clauses and produces explainable risk scores.
- Validation demonstrates benchmarked performance and documented limitations.
- Final report includes roadmap for ecosystem-level extensions.

---

## Navigation

[⬅ Back](09-phase1-phase2-progress-dashboard.md) | [Next ⮕](01-tech-stack.md)
