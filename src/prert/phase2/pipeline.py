"""End-to-end Phase 2 pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from prert.phase2.io import read_jsonl, write_json, write_jsonl
from prert.phase2.metrics import build_metric_coverage_summary, build_metric_specs
from prert.phase2.public_mapping import load_public_rows, map_public_rows, summarize_public_mapping
from prert.phase2.scoring import score_observations
from prert.phase2.synthetic import generate_synthetic_observations


def run_phase2_pipeline(
    controls_path: Path,
    output_dir: Path,
    public_input_path: Optional[Path] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    controls = read_jsonl(controls_path)
    metric_specs = build_metric_specs(controls)
    coverage_summary = build_metric_coverage_summary(controls, metric_specs)

    observations = generate_synthetic_observations(metric_specs, seed=seed)
    metric_rows, level_rows, scenario_rows = score_observations(metric_specs, observations)

    public_rows = load_public_rows(public_input_path)
    public_source_name = public_input_path.name if public_input_path else "none"
    mapped_public_rows = map_public_rows(public_rows, source_name=public_source_name)
    public_summary = summarize_public_mapping(mapped_public_rows)

    write_jsonl(output_dir / "metric_specs.jsonl", (spec.as_dict() for spec in metric_specs))
    write_jsonl(output_dir / "synthetic_events.jsonl", (row.as_dict() for row in observations))
    write_jsonl(output_dir / "public_data_mapped.jsonl", mapped_public_rows)
    write_jsonl(output_dir / "baseline_scores.jsonl", [*metric_rows, *level_rows, *scenario_rows])

    _write_data_dictionary(output_dir / "synthetic_data_dictionary.md")

    manifest = {
        "phase": "phase-2",
        "seed": seed,
        "inputs": {
            "controls_path": str(controls_path),
            "public_input_path": str(public_input_path) if public_input_path else "",
        },
        "coverage_summary": coverage_summary,
        "public_mapping_summary": public_summary,
        "output_counts": {
            "metric_specs": len(metric_specs),
            "synthetic_events": len(observations),
            "public_data_mapped": len(mapped_public_rows),
            "metric_score_rows": len(metric_rows),
            "level_summary_rows": len(level_rows),
            "scenario_summary_rows": len(scenario_rows),
            "baseline_score_rows_total": len(metric_rows) + len(level_rows) + len(scenario_rows),
        },
    }

    write_json(output_dir / "phase2_manifest.json", manifest)
    return manifest


def _write_data_dictionary(path: Path) -> None:
    text = """# Synthetic Data Dictionary (Phase 2)

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
"""
    path.write_text(text, encoding="utf-8")
