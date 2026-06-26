# Synthetic Data Dictionary (Phase 2)

## File: synthetic_events.jsonl

- observation_id: Stable synthetic row identifier.
- scenario: One of normal, stressed, adversarial.
- entity_type: user, system, or organization.
- entity_id: Deterministic entity key for reproducible runs.
- metric_id: Link to metric_specs.metric_id.
- level: Metric level (user/system/organization).
- total_checks: Number of checks observed for this metric instance.
- failure_count: Number of failed checks.
- missing_fields: Count of missing required fields for penalty application.
- observed_confidence: Observed confidence in [0, 1].
- metadata.generator: Synthetic generator version identifier.

## File: metric_specs.jsonl

- metric_id: Stable metric identifier.
- control_id: Phase 1 normalized control id mapped to this metric.
- regulation: GDPR, ISO27001, or NIST PF source family.
- level: user/system/organization.
- formula: Score formula string.
- required_fields: Required data fields before scoring.
- normalization_rule: Rule used to constrain score range.
- confidence_weight: Prior confidence weight from parser confidence.
- missing_data_handling: Declared policy used in score penalty.

## File: public_data_mapped.jsonl

- source: Input file name.
- source_row_index: Row index from public dataset input.
- event_date/country/sector: Canonical mapped categorical fields.
- records_affected: Canonical integer impact volume.
- detection_to_response_hours: Canonical float timing feature.
- severity: Canonical impact tag if available.
- dq_missing_required_fields: Required fields missing after mapping.
- dq_valid: True if row passes required field checks.

## File: baseline_scores.jsonl

- row_type: metric, level_summary, or scenario_summary.
- compliance_score: Compliance-oriented score in [0, 1].
- risk_score: Converted risk score in [0, 1].
- risk_band: low, medium, or high risk.
