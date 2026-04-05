"""Scoring logic for Phase 2 metric baseline runs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from prert.phase2.types import MetricSpec, SyntheticObservation


LEVEL_WEIGHTS = {
    "user": 0.4,
    "system": 0.35,
    "organization": 0.25,
}


def score_observations(
    metric_specs: Iterable[MetricSpec],
    observations: Iterable[SyntheticObservation],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    spec_by_id = {spec.metric_id: spec for spec in metric_specs}

    metric_rows: List[Dict[str, Any]] = []
    level_accumulator: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    for obs in observations:
        spec = spec_by_id.get(obs.metric_id)
        if spec is None:
            continue

        raw_score = 1.0 - (obs.failure_count / max(obs.total_checks, 1))
        normalized_score = _clamp(raw_score)
        missing_penalty = min(0.4, 0.05 * obs.missing_fields)
        confidence_adjusted_score = _clamp(
            normalized_score * (1.0 - missing_penalty) * obs.observed_confidence * spec.confidence_weight
        )
        risk_score = _clamp(1.0 - confidence_adjusted_score)

        metric_row = {
            "row_type": "metric",
            "scenario": obs.scenario,
            "level": obs.level,
            "entity_id": obs.entity_id,
            "metric_id": obs.metric_id,
            "control_id": spec.control_id,
            "total_checks": obs.total_checks,
            "failure_count": obs.failure_count,
            "missing_fields": obs.missing_fields,
            "raw_score": round(raw_score, 6),
            "normalized_score": round(normalized_score, 6),
            "confidence_adjusted_score": round(confidence_adjusted_score, 6),
            "risk_score": round(risk_score, 6),
            "risk_band": _risk_band(risk_score),
        }
        metric_rows.append(metric_row)

        level_accumulator[(obs.scenario, obs.level)].append(confidence_adjusted_score)

    level_rows = _build_level_rows(level_accumulator)
    scenario_rows = _build_scenario_rows(level_rows)

    return metric_rows, level_rows, scenario_rows


def _build_level_rows(level_accumulator: Dict[Tuple[str, str], List[float]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for (scenario, level), values in sorted(level_accumulator.items()):
        avg_score = sum(values) / max(len(values), 1)
        risk_score = _clamp(1.0 - avg_score)
        rows.append(
            {
                "row_type": "level_summary",
                "scenario": scenario,
                "level": level,
                "sample_size": len(values),
                "compliance_score": round(avg_score, 6),
                "risk_score": round(risk_score, 6),
                "risk_band": _risk_band(risk_score),
            }
        )

    return rows


def _build_scenario_rows(level_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_scenario: Dict[str, Dict[str, float]] = defaultdict(dict)

    for row in level_rows:
        scenario = str(row["scenario"])
        level = str(row["level"])
        by_scenario[scenario][level] = float(row["compliance_score"])

    rows: List[Dict[str, Any]] = []
    for scenario, score_map in sorted(by_scenario.items()):
        weighted_score = 0.0
        total_weight = 0.0
        for level, weight in LEVEL_WEIGHTS.items():
            if level in score_map:
                weighted_score += score_map[level] * weight
                total_weight += weight

        if total_weight == 0:
            continue

        weighted_score /= total_weight
        risk_score = _clamp(1.0 - weighted_score)

        rows.append(
            {
                "row_type": "scenario_summary",
                "scenario": scenario,
                "composite_method": "weighted_sum_v1",
                "level_weights": LEVEL_WEIGHTS,
                "compliance_score": round(weighted_score, 6),
                "risk_score": round(risk_score, 6),
                "risk_band": _risk_band(risk_score),
            }
        )

    return rows


def _risk_band(risk_score: float) -> str:
    if risk_score >= 0.67:
        return "high"
    if risk_score >= 0.34:
        return "medium"
    return "low"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
