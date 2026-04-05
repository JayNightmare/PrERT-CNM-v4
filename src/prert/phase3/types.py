"""Typed structures used by the Phase 3 baseline pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class ClauseExample:
    example_id: str
    text: str
    label: str
    source: str
    policy_uid: str
    category: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SplitSummary:
    split: str
    rows: int
    unique_policies: int
    class_distribution: Dict[str, int]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
