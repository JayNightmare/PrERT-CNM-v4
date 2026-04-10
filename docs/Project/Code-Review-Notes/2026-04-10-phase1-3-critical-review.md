# Phase 1-3 Critical Review (Proposal Alignment)

- Date: 2026-04-10
- Reviewer: GitHub Copilot (GPT-5.3-Codex)
- Scope: Proposal alignment check for Phases 1, 2, and 3 using repository evidence only. No code changes performed.

## Executive Verdict

- Phase 1: Partially on track, but at-risk on proposal-deliverable completeness.
- Phase 2: On track for baseline implementation, with modeling maturity gaps.
- Phase 3: Functionally strong baseline, but not fully proposal-aligned due to dataset and reproducibility gaps.

## Findings (Ordered by Severity)

### 1. High: Phase 3 proposal requirement (OPP-115 + Polisis) is not fully implemented in code

What I found:

- The proposal requires PrivacyBERT fine-tuned on both OPP-115 and Polisis.
- Repository docs also claim OPP-115 + Polisis.
- Implementation currently ingests OPP-115 or generic labeled JSONL only; no Polisis-specific ingestion/harmonization pipeline exists in source code.

Evidence:

- Proposal requirement: [docs/Project/Proposal-extracted.md](/docs/Project/Proposal-extracted.md#L25)
- Repo claim: [README.md](/README.md#L39)
- Implemented dataset path (OPP-115): [src/prert/phase3/dataset.py](/src/prert/phase3/dataset.py#L68), [src/prert/phase3/dataset.py](/src/prert/phase3/dataset.py#L79)
- No Polisis references in source tree: grep result `NO_POLISIS_MATCH_IN_SRC`

Impact:

- Phase 3 is close, but currently falls short of explicit proposal wording unless Polisis is supplied manually as labeled JSONL outside the repo process.

Recommendation:

- Add a first-class Polisis loader + label harmonization path (with tests), or explicitly revise docs/acceptance criteria to state Polisis is optional/manual.

### 2. High: Reproducibility/auditability gap in committed artifacts

What I found:

- Runbooks and dashboard docs reference machine-readable artifacts (manifests, metrics JSON, Bayesian outputs), but committed artifact folders mostly contain markdown summaries.
- `artifacts/phase-1` is absent from committed artifacts while Phase 2 default CLI input depends on it.

Evidence:

- Current committed artifact files: markdown-only summary set in [artifacts](/artifacts)
- Phase 2 default dependency on Phase 1 controls: [src/prert/cli/phase2.py](/src/prert/cli/phase2.py#L39)
- Phase 3 runbook expected outputs include JSON/JSONL: [docs/Project/Execution-Playbook/10-phase3-implementation-runbook.md](/docs/Project/Execution-Playbook/10-phase3-implementation-runbook.md#L144), [docs/Project/Execution-Playbook/10-phase3-implementation-runbook.md](/docs/Project/Execution-Playbook/10-phase3-implementation-runbook.md#L151), [docs/Project/Execution-Playbook/10-phase3-implementation-runbook.md](/docs/Project/Execution-Playbook/10-phase3-implementation-runbook.md#L155)
- Dashboard expects those artifacts and explicitly skips when missing: [scripts/generate_phase3_dashboard_figures.py](/scripts/generate_phase3_dashboard_figures.py#L619), [scripts/generate_phase3_dashboard_figures.py](/scripts/generate_phase3_dashboard_figures.py#L622)
- Dashboard data-source assumptions: [docs/Project/Execution-Playbook/11-phase3-visual-dashboard.md](/docs/Project/Execution-Playbook/11-phase3-visual-dashboard.md#L16), [docs/Project/Execution-Playbook/11-phase3-visual-dashboard.md](/docs/Project/Execution-Playbook/11-phase3-visual-dashboard.md#L18)

Impact:

- Third-party verification of reported metrics from a fresh checkout is blocked.

Recommendation:

- Commit a minimal reproducibility bundle (small sample manifests/metrics) or publish a documented artifact registry with checksums and retrieval steps.

### 3. High: Phase 1 deliverable wording vs implemented deliverable type is misaligned

What I found:

- Proposal and project docs indicate Phase 1 output should be standards-to-indicator/metrics mapping (indicator catalog and traceability).
- Implementation is strong on control extraction/chunking for GDPR/ISO/NIST but does not expose a concrete Phase 1 indicator catalog/traceability artifact in repository outputs.

Evidence:

- Proposal Phase 1 deliverable: [docs/Project/Proposal-extracted.md](/docs/Project/Proposal-extracted.md#L39)
- Dependency map says indicator catalog is required for Phase 2: [docs/README.md](/docs/README.md#L68)
- Implemented Phase 1 extraction scope (GDPR/ISO/NIST): [README.md](/README.md#L21), [README.md](/README.md#L23)
- Extract CLI wired only for GDPR/ISO/NIST parsers: [src/prert/cli/extract.py](/src/prert/cli/extract.py#L12), [src/prert/cli/extract.py](/src/prert/cli/extract.py#L22)

Impact:

- Stakeholder interpretation risk: completed engineering work may still appear non-compliant with planned deliverable format.

Recommendation:

- Generate and store explicit `clause -> indicator -> metric` traceability outputs (even if auto-derived from current controls/specs).

### 4. Medium: Bayesian risk layer is lightweight and not tightly coupled to Phase 2 metric evidence

What I found:

- Bayesian updates are based on predicted label and confidence accumulation, not explicit evidence from Phase 2 metric contracts or standards-linked likelihood terms.

Evidence:

- Bayesian update logic: [src/prert/phase3/risk.py](/src/prert/phase3/risk.py#L77), [src/prert/phase3/risk.py](/src/prert/phase3/risk.py#L83)
- No direct Phase 2 metric input in Phase 3 data-prep path (OPP-115/labeled only): [src/prert/phase3/dataset.py](/src/prert/phase3/dataset.py#L68)

Impact:

- Good prototype behavior, but weaker claim to standards-grounded Bayesian semantics than proposal language implies.

Recommendation:

- Add an adapter that maps clause predictions into Phase 2 metric evidence nodes before Bayesian updates.

### 5. Medium: Phase 2 level assignment remains keyword-heuristic with organization fallback

What I found:

- Level classification (`user/system/organization`) is keyword-based and defaults to `organization` when no keyword hits.

Evidence:

- Heuristic keyword sets and fallback: [src/prert/phase2/metrics.py](/src/prert/phase2/metrics.py#L11), [src/prert/phase2/metrics.py](/src/prert/phase2/metrics.py#L145), [src/prert/phase2/metrics.py](/src/prert/phase2/metrics.py#L157)

Impact:

- Baseline is valid, but risk of systematic level skew and weaker interpretability at scale.

Recommendation:

- Introduce semantic level classifier or at least calibration/audit report for fallback-assigned controls.

### 6. Medium: PrivacyBERT path is not fully exercised in tests

What I found:

- Test suite is strong overall, but PrivacyBERT training path is not directly CI-tested in `test_phase3_pipeline.py`; optional model test shown is for sklearn/logreg.

Evidence:

- Optional sklearn path in tests: [tests/test_phase3_pipeline.py](/tests/test_phase3_pipeline.py#L138)

Impact:

- Regressions in transformer path may be detected late.

Recommendation:

- Add a lightweight mocked/smoke PrivacyBERT test (or nightly integration test) to protect the accepted Phase 3 path.

## Positive Signals

- Architecture is phase-structured and cleanly modular for extraction, phase2 metrics/scoring, and phase3 classification/risk.
- Acceptance-gate mechanics for Phase 3 are present and proposal-aware.
- Relevant tests for Phases 1-3 pass in this review session.

Validation run in this review:

- `pytest -q tests/test_extractors.py tests/test_chunking.py tests/test_phase2_opp115_processor.py tests/test_phase2_pipeline.py tests/test_phase3_pipeline.py tests/test_phase3_analytics.py tests/test_phase3_acceptance.py`
- Result: `18 passed in 26.94s`

## Phase-by-Phase Status

### Phase 1

Status: Partially on track

Rationale:

- Engineering implementation for standards extraction is strong.
- Deliverable framing (indicator catalog + traceability) is not yet concretely represented as a committed artifact.

### Phase 2

Status: On track (baseline)

Rationale:

- End-to-end baseline pipeline exists with synthetic generation, mapping, and scoring.
- Public mapping is generic and practical, but still not clearly ENISA/PRC-specific in implementation semantics.

### Phase 3

Status: Mostly on track, but proposal-alignment risk remains

Rationale:

- Classifier + Bayesian + acceptance flow are implemented and tested at baseline level.
- Explicit OPP-115 + Polisis implementation parity is missing in source.
- Reproducibility artifacts are under-committed for external audit.

## Suggested Next Gate Before Declaring Full Track

1. Add explicit Polisis ingestion/harmonization path (or formally revise scope wording).
2. Publish traceability artifact for Phase 1 (`clause -> indicator -> metric`).
3. Commit or externally publish machine-readable Phase 2/3 artifacts referenced by dashboards/runbooks.
4. Add PrivacyBERT-specific CI smoke coverage.
