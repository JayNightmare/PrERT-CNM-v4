"""Typed structures used by the Phase 2 pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class MetricSpec:
    metric_id: str
    control_id: str
    regulation: str
    level: str
    metric_name: str
    formula: str
    required_fields: List[str]
    normalization_rule: str
    confidence_weight: float
    missing_data_handling: str
    status: str = "active"
    deferral_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SyntheticObservation:
    observation_id: str
    scenario: str
    entity_type: str
    entity_id: str
    metric_id: str
    level: str
    total_checks: int
    failure_count: int
    missing_fields: int
    observed_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
