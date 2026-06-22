# Changelog

## [0.6.0] - 2026-06-22

### Added
- **Policy-only compliance assessment** (`assess_policy_compliance`): Evaluates a privacy policy against GDPR, NIST Privacy Framework, and ISO/IEC 27701 without requiring a database schema.
- **Source citations**: Every compliance verdict now cites the exact policy clause text that supports the pass/fail determination.
- **Per-regulation independent scoring**: Each extracted policy claim produces independent `RegulationVerdict` objects for each regulation framework (GDPR articles, NIST subcategories, ISO 27701 controls) with a boolean `compliant` field and human-readable `reason`.
- **`REGULATION_CONTROLS` reference data**: Maps all 8 compliance check areas to 30 specific regulation controls across three frameworks.
- **`RegulationVerdict` and `PolicyClaimResult` dataclasses**: Structured output types for per-regulation scoring.
- **Streamlit GUI**: Schema upload is now optional — uploading only a policy triggers the new policy-only assessment path with per-regulation verdict tables and regulation summary columns.
- **9 new tests** in `test_phase4_policy_compliance.py` covering structured output, citation integrity, regulation coverage, and grading consistency.
