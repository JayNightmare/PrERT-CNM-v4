"""Synthetic observation generation for Phase 2 scoring."""

from __future__ import annotations

from random import Random
from typing import Dict, Iterable, List

from prert.extract.schema import stable_hash
from prert.phase2.types import MetricSpec, SyntheticObservation


SCENARIO_PROFILES: Dict[str, Dict[str, float]] = {
    "normal": {
        "failure_rate": 0.08,
        "missing_rate": 0.04,
        "confidence_low": 0.85,
        "confidence_high": 0.98,
    },
    "stressed": {
        "failure_rate": 0.22,
        "missing_rate": 0.12,
        "confidence_low": 0.72,
        "confidence_high": 0.92,
    },
    "adversarial": {
        "failure_rate": 0.4,
        "missing_rate": 0.2,
        "confidence_low": 0.55,
        "confidence_high": 0.85,
    },
}


def generate_synthetic_observations(
    metric_specs: Iterable[MetricSpec],
    seed: int = 42,
    scenarios: Iterable[str] = ("normal", "stressed", "adversarial"),
) -> List[SyntheticObservation]:
    rnd = Random(seed)
    observations: List[SyntheticObservation] = []

    for metric_index, spec in enumerate(metric_specs):
        for scenario in scenarios:
            profile = SCENARIO_PROFILES[scenario]
            total_checks = _sample_total_checks(spec.level, rnd)
            failure_count = _sample_failure_count(total_checks, profile["failure_rate"], rnd)
            missing_fields = _sample_missing_fields(profile["missing_rate"], rnd)
            confidence = round(rnd.uniform(profile["confidence_low"], profile["confidence_high"]), 4)

            entity_type = spec.level
            entity_id = f"{entity_type}-{metric_index:04d}"
            observation_id = "obs::" + stable_hash(f"{scenario}|{spec.metric_id}|{entity_id}")[:18]

            observations.append(
                SyntheticObservation(
                    observation_id=observation_id,
                    scenario=scenario,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    metric_id=spec.metric_id,
                    level=spec.level,
                    total_checks=total_checks,
                    failure_count=failure_count,
                    missing_fields=missing_fields,
                    observed_confidence=confidence,
                    metadata={
                        "generator": "phase2_synthetic_v1",
                        "scenario_profile": scenario,
                    },
                )
            )

    return observations


def _sample_total_checks(level: str, rnd: Random) -> int:
    if level == "user":
        return rnd.randint(20, 80)
    if level == "system":
        return rnd.randint(30, 100)
    return rnd.randint(25, 90)


def _sample_failure_count(total_checks: int, failure_rate: float, rnd: Random) -> int:
    failures = 0
    for _ in range(total_checks):
        if rnd.random() < failure_rate:
            failures += 1
    return failures


def _sample_missing_fields(missing_rate: float, rnd: Random) -> int:
    count = 0
    for _ in range(6):
        if rnd.random() < missing_rate:
            count += 1
    return count
