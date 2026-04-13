# Synthetic Policy/Schema Dataset Dictionary

Synthetic records for Phase 4 supervisor workflow testing.
Each row represents one privacy-policy and database-schema pair with an assessor output.

## JSONL Fields

- sample_id: Stable synthetic sample identifier.
- compliance_band: Intended compliance band (`low`, `medium`, `high`).
- sample_index: Zero-based index within each compliance band.
- target_score_min: Lower bound for intended compliance score range.
- target_score_max: Upper bound for intended compliance score range.
- within_target_band: Whether generated assessment score fell in the intended range.
- policy_text: Synthetic privacy policy text block.
- schema_text: Synthetic database schema text block.
- assessment: Structured output from `assess_policy_schema_compliance`.
- metadata: Generator metadata (seed, version, model-signal setting).

## Intended Use

- Regression tests for compliance scoring behavior.
- Manual supervisor upload demos in the Streamlit GUI.
- Stressing low/medium/high compliance scenarios with reproducible seed control.

## Non-Use

- Not representative of legal advice or production compliance attestation.
- Not a substitute for domain expert policy review.
