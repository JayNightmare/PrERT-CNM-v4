"""Metric specification generation for Phase 2."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from prert.extract.schema import stable_hash
from prert.phase2.types import MetricSpec


USER_KEYWORDS = {
    "consent",
    "data subject",
    "right",
    "erasure",
    "rectification",
    "portability",
    "transparent",
    "notice",
    "access",
    "withdraw",
}

SYSTEM_KEYWORDS = {
    "encryption",
    "integrity",
    "availability",
    "resilience",
    "security",
    "pseudonym",
    "incident",
    "breach",
    "access control",
    "vulnerability",
}

ORG_KEYWORDS = {
    "policy",
    "governance",
    "risk",
    "controller",
    "processor",
    "training",
    "assessment",
    "audit",
    "management",
    "compliance",
    "vendor",
}


FORMULAS = {
    "user": "1 - (consent_or_rights_failures / max(total_user_events, 1))",
    "system": "1 - (security_or_safeguard_failures / max(total_system_checks, 1))",
    "organization": "1 - (governance_or_response_gaps / max(total_org_checks, 1))",
}


REQUIRED_FIELDS = {
    "user": [
        "total_user_events",
        "consent_or_rights_failures",
        "missing_user_fields",
    ],
    "system": [
        "total_system_checks",
        "security_or_safeguard_failures",
        "missing_system_fields",
    ],
    "organization": [
        "total_org_checks",
        "governance_or_response_gaps",
        "missing_org_fields",
    ],
}


def build_metric_specs(controls: Iterable[Dict[str, Any]]) -> List[MetricSpec]:
    specs: List[MetricSpec] = []

    for control in controls:
        control_id = str(control.get("normalized_id", "")).strip()
        if not control_id:
            continue

        regulation = str(control.get("regulation", "unknown")).strip() or "unknown"
        level = classify_level(control)
        metric_name = make_metric_name(control)
        metric_id = "metric::" + stable_hash(f"{control_id}|{level}")[:16]

        parser_confidence = float(control.get("parser_confidence", 0.8))
        confidence_weight = min(1.0, max(0.1, parser_confidence))

        spec = MetricSpec(
            metric_id=metric_id,
            control_id=control_id,
            regulation=regulation,
            level=level,
            metric_name=metric_name,
            formula=FORMULAS[level],
            required_fields=REQUIRED_FIELDS[level],
            normalization_rule="clamp(score, 0.0, 1.0)",
            confidence_weight=confidence_weight,
            missing_data_handling="impute_with_penalty: penalty=min(0.4, 0.05*missing_fields)",
            status="active",
            metadata={
                "native_id": control.get("native_id", ""),
                "chapter": control.get("chapter", ""),
                "section": control.get("section", ""),
            },
        )
        specs.append(spec)

    return specs


def build_metric_coverage_summary(
    controls: Iterable[Dict[str, Any]],
    specs: Iterable[MetricSpec],
) -> Dict[str, Any]:
    control_ids = {
        str(control.get("normalized_id", "")).strip()
        for control in controls
        if str(control.get("normalized_id", "")).strip()
    }
    mapped_ids = {spec.control_id for spec in specs}

    missing_controls = sorted(control_ids - mapped_ids)
    deferred_count = sum(1 for spec in specs if spec.status != "active")

    level_counts: Dict[str, int] = {"user": 0, "system": 0, "organization": 0}
    for spec in specs:
        level_counts[spec.level] = level_counts.get(spec.level, 0) + 1

    return {
        "total_controls": len(control_ids),
        "mapped_controls": len(mapped_ids),
        "missing_controls": missing_controls,
        "active_metric_count": sum(1 for spec in specs if spec.status == "active"),
        "deferred_metric_count": deferred_count,
        "level_counts": level_counts,
    }


def classify_level(control: Dict[str, Any]) -> str:
    title = str(control.get("title", "")).lower()
    text = str(control.get("text", "")).lower()
    bag = f"{title} {text}"

    if _contains_any(bag, USER_KEYWORDS):
        return "user"
    if _contains_any(bag, SYSTEM_KEYWORDS):
        return "system"
    if _contains_any(bag, ORG_KEYWORDS):
        return "organization"

    # Default fallback keeps metrics complete instead of deferred.
    return "organization"


def make_metric_name(control: Dict[str, Any]) -> str:
    native_id = str(control.get("native_id", "")).strip()
    title = str(control.get("title", "")).strip()
    regulation = str(control.get("regulation", "")).strip()

    pieces = [part for part in [regulation, native_id, title] if part]
    if not pieces:
        return "Unnamed Metric"

    return " | ".join(pieces)


def _contains_any(text: str, phrases: Iterable[str]) -> bool:
    return any(phrase in text for phrase in phrases)
