# Phase 1 Detailed Plan: Standards-to-Indicators Mapping

Timeline target: Month 1

## Goal

Convert standards and regulatory clauses into measurable privacy indicators that can be used for downstream scoring.

## Scope

In scope:

- ISO 27001, NIST, GDPR, and related regulatory clauses.
- Indicators tied to consent, data minimization, encryption, third-party sharing, and transparency/accountability dimensions.

Out of scope:

- Full legal interpretation beyond technical measurable mapping.
- Final risk score modeling (Phase 3).

## Inputs

- `Proposal.pdf`
- Source standards/regulatory documents.
- Prior related references (Polisis, PrivacyBERT literature context).

## Steps to Complete

1. Build a standards corpus

- Collect and normalize all relevant clauses into a single structured table.
- Include metadata fields: source, section, clause text, jurisdiction, control category.

2. Define an indicator taxonomy

- Create top-level dimensions: notice/transparency, consent/control, data minimization, retention, security, sharing, redress.
- Define naming rules for indicators (for example `CONSENT_EXPLICIT_OPT_IN`).

3. Translate clauses into measurable indicators

- For each clause, define:
     - indicator definition,
     - data needed to measure it,
     - acceptable value range,
     - scoring direction (higher is better/worse).

4. Build traceability matrix

- Map `clause -> indicator -> metric candidate`.
- Ensure many-to-many relationships are documented explicitly.

5. Run internal quality review

- Check for overlap, ambiguity, and unmeasurable indicators.
- Remove indicators that lack feasible data signals.

6. Freeze Phase 1 baseline

- Version the indicator catalog and traceability matrix.
- Publish assumptions and unresolved questions.

## Deliverables

- Standards mapping table (CSV/Parquet).
- Indicator catalog (Markdown).
- Traceability matrix (Markdown/CSV).
- Phase 1 review memo.

## Recommended Acceptance Checks

- 100 percent of in-scope clauses map to at least one indicator or have documented rationale for exclusion.
- Indicator definitions are testable with identifiable data inputs.
- Duplicate indicators reduced via normalization pass.
- At least one reviewer sign-off on traceability consistency.

## Risks and Mitigations

- Risk: Too many low-value indicators.
     - Mitigation: Prioritize indicators that are observable in policy text/logs.
- Risk: Ambiguous legal language.
     - Mitigation: Add confidence levels and interpretation notes per mapping.
- Risk: Scope creep from ecosystem-level topics.
     - Mitigation: Keep ecosystem items tagged as future scope only.

## Recommended Week-by-Week Breakdown

Week 1:

- Collect and normalize clause corpus.

Week 2:

- Draft taxonomy and first-pass indicator mapping.

Week 3:

- Build traceability matrix and resolve ambiguities.

Week 4:

- Review, baseline freeze, and handoff package for Phase 2.

---

## Navigation

[⬅ Back](01-tech-stack.md) | [Next ⮕](03-phase-2-metrics-and-data.md)
