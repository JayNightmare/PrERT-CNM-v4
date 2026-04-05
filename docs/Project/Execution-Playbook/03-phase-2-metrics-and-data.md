# Phase 2 Detailed Plan: Metrics Definition and Data Preparation

Timeline target: Month 2

## Goal

Define measurable privacy-risk metrics across user, system, and organization levels, then validate feasibility using synthetic and public data.

## Scope

In scope:

- Metric formula/spec design.
- Synthetic data generation for controlled testing.
- Public breach data alignment (ENISA, PRC references).
- Conceptual scoping for digital ecosystem indicators.

Out of scope:

- Final production calibration across all industries.
- Full ecosystem-level implementation.

## Inputs

- Phase 1 indicator catalog and traceability matrix.
- ENISA and PRC breach datasets.
- Proposed metric dimensions from proposal.

## Steps to Complete

1. Define metric schema

- For each indicator, define:
     - metric id,
     - formula,
     - required fields,
     - normalization rule,
     - confidence weighting,
     - missing data handling.

2. Build synthetic data generator

- Create representative entities: users, systems, vendors, incidents.
- Inject controlled edge cases (missing consent records, delayed breach response, weak safeguards).
- Create dataset variants for normal, stressed, and adversarial conditions.

3. Integrate public breach context

- Map ENISA/PRC attributes to internal schema.
- Add transformation pipelines with data quality checks.

4. Prototype scoring at three levels

- User-level metrics (control/consent exposure).
- System-level metrics (encryption posture, sharing exposure).
- Organization-level metrics (response time, safeguard maturity).

5. Define draft composite scoring strategy

- Choose aggregation method (weighted sum, Bayesian prior-informed, or hybrid).
- Document score interpretation bands (low/medium/high risk).

6. Document ecosystem-level future scope

- Capture cross-border transfer and interoperability indicators as future implementation backlog.

7. Freeze Phase 2 baseline

- Publish metric spec, synthetic data dictionary, and baseline results.

## Deliverables

- Metric specification document.
- Synthetic data generation scripts and data dictionary.
- Public-data mapping report.
- Baseline metric result snapshots.
- Ecosystem scope note.

## Recommended Acceptance Checks

- Every Phase 1 indicator maps to a metric or has explicit deferral rationale.
- Synthetic data covers normal and edge-case scenarios.
- Metric outputs are numerically stable and interpretable.
- At least one end-to-end run from raw input to scored outputs is reproducible.

## Risks and Mitigations

- Risk: Synthetic data does not reflect real patterns.
     - Mitigation: Seed distributions from public breach statistics where feasible.
- Risk: Overly complex metric formulas.
     - Mitigation: Start simple, add complexity only if it improves explainability.
- Risk: Missing fields in public datasets.
     - Mitigation: Add imputation policy and confidence penalties.

## Recommended Week-by-Week Breakdown

Week 1:

- Finalize metric schema and formula drafts.

Week 2:

- Build synthetic generator and produce first datasets.

Week 3:

- Add public-data mappings and run baseline scoring.

Week 4:

- Review outputs, refine formulas, freeze Phase 2 artifacts.

---

## Navigation

[⬅ Back](02-phase-1-standards-mapping.md) | [Next ⮕](04-phase-3-ai-prototype.md)
