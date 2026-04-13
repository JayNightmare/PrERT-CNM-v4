# Phase 4 Detailed Plan: Validation, Benchmarking, and Final Reporting

Timeline target: Month 4

## Goal

Validate the prototype against defined metrics, benchmark performance, document findings, and deliver final report artifacts.

## Scope

In scope:

- Test execution on real and synthetic datasets.
- Benchmarking against phase-defined metrics.
- Error analysis and refinement loop.
- Final documentation and roadmap.

Out of scope:

- Enterprise production rollout.
- Full ecosystem-level implementation.

## Inputs

- Integrated Phase 3 prototype.
- Synthetic and public/real evaluation datasets.
- Benchmark definitions and acceptance thresholds.

## MVP Validation Tool (Artifact-Based)

The current Phase 4 MVP validates existing Phase 3 artifacts and compares one-or-more runs without retraining.

Primary command:

```bash
PYTHONPATH=src python scripts/run_phase4_validation.py \
     --baseline-dir artifacts/phase-3-freeze \
     --comparison-dirs artifacts/phase-3-nb artifacts/phase-3-logreg artifacts/phase-3-privacybert \
     --output-dir artifacts/phase-4
```

Equivalent package command:

```bash
prert-phase4 \
     --baseline-dir artifacts/phase-3-freeze \
     --comparison-dirs artifacts/phase-3-nb artifacts/phase-3-logreg artifacts/phase-3-privacybert \
     --output-dir artifacts/phase-4
```

MVP outputs:

- `artifacts/phase-4/phase4_validation_report.json`
- `artifacts/phase-4/phase4_validation_report.md`
- `artifacts/phase-4/phase4_leaderboard.jsonl`

Supervisor-facing GUI workflow:

```bash
PYTHONPATH=src python scripts/run_phase4_web.py --port 8501
```

The GUI allows uploads for:

- company privacy policy documents
- database schema files

and returns an evidence-backed compliance score with detected risks.

Notes:

- `--require-bayesian` upgrades Bayesian evidence checks from advisory to required.
- `--strict` exits with status code `2` when baseline required checks fail.

## Steps to Complete

1. Build validation test plan

- Define scenario matrix:
     - normal operations,
     - high-risk policy clauses,
     - missing/incomplete data,
     - conflicting indicator evidence.
- Define success thresholds for each scenario.

2. Execute benchmark suite

- Run batch benchmarks on all scenarios.
- Capture classification and risk-scoring metrics with confidence intervals.

3. Perform robustness and sensitivity analysis

- Perturb key inputs and observe score stability.
- Test model behavior under class imbalance and sparse evidence.

4. Conduct error analysis

- Identify top failure modes in clause classification and score estimation.
- Trace errors back to data, mapping, or model assumptions.

5. Refine prototype where justified

- Apply targeted fixes only when they improve benchmark outcomes and interpretability.
- Re-run impacted benchmarks and log deltas.

6. Finalize reporting package

- Prepare validation summary.
- Prepare benchmark report.
- Prepare final report with limitations and future roadmap.

7. Completion and handoff

- Ensure all artifacts are versioned and reproducible.
- Conduct final review and sign-off.

## Deliverables

- Benchmark report with scenario-level results.
- Validation summary with risk/limitation analysis.
- Final report and roadmap.
- Packaged validated prototype.

## Recommended Acceptance Checks

- Benchmarks executed for all predefined scenarios.
- Results include both performance and uncertainty metrics.
- Known limitations and assumptions are explicitly documented.
- Final report is reproducible from versioned artifacts.

## Risks and Mitigations

- Risk: Benchmark definitions are too vague.
     - Mitigation: Lock thresholds before large test execution.
- Risk: Last-minute model changes invalidate comparisons.
     - Mitigation: Freeze baseline and track all post-freeze deltas.
- Risk: Report quality suffers due to time compression.
     - Mitigation: Start report skeleton in Week 1 and update continuously.

## Recommended Week-by-Week Breakdown

Week 1:

- Finalize test plan and benchmark scripts.

Week 2:

- Execute full benchmark suite and collect outputs.

Week 3:

- Perform error analysis and targeted refinements.

Week 4:

- Finalize reports, artifacts, and sign-off package.

---

## Navigation

[⬅ Back](04-phase-3-ai-prototype.md) | [Next ⮕](06-phase1-implementation-runbook.md)
