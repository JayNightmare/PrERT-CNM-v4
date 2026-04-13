"""Policy-and-schema compliance assessment helpers for Phase 4 GUI workflows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from prert.phase3.classifier import NaiveBayesTextClassifier


LABELS: tuple[str, str, str] = ("user", "system", "organization")


PII_FIELD_PATTERNS: tuple[str, ...] = (
    "name",
    "first_name",
    "last_name",
    "full_name",
    "email",
    "phone",
    "mobile",
    "address",
    "city",
    "state",
    "zip",
    "postal",
    "country",
    "dob",
    "birth",
    "ssn",
    "tax",
    "passport",
    "national_id",
    "driver",
    "ip",
    "cookie",
    "device",
    "location",
)

SENSITIVE_FIELD_PATTERNS: tuple[str, ...] = (
    "health",
    "medical",
    "biometric",
    "genetic",
    "race",
    "ethnicity",
    "religion",
    "political",
    "sexual",
    "criminal",
    "bank",
    "credit_card",
    "card_number",
    "account_number",
    "salary",
    "income",
)


@dataclass(frozen=True)
class PolicyCheckSpec:
    check_id: str
    title: str
    weight: float
    keywords: tuple[str, ...]


POLICY_CHECK_SPECS: tuple[PolicyCheckSpec, ...] = (
    PolicyCheckSpec(
        check_id="consent_transparency",
        title="Consent And Transparency",
        weight=12.0,
        keywords=(
            "consent",
            "opt in",
            "opt-out",
            "opt out",
            "permission",
            "notice",
            "transparent",
        ),
    ),
    PolicyCheckSpec(
        check_id="user_rights",
        title="User Access And Deletion Rights",
        weight=12.0,
        keywords=(
            "access",
            "delete",
            "erasure",
            "right to be forgotten",
            "correct",
            "rectify",
            "portability",
        ),
    ),
    PolicyCheckSpec(
        check_id="security_safeguards",
        title="Security Safeguards",
        weight=12.0,
        keywords=(
            "encrypt",
            "encryption",
            "security",
            "secure",
            "safeguard",
            "access control",
            "multi-factor",
        ),
    ),
    PolicyCheckSpec(
        check_id="data_retention",
        title="Data Retention And Deletion Windows",
        weight=10.0,
        keywords=(
            "retention",
            "retain",
            "storage period",
            "delete after",
            "deletion schedule",
            "archive",
        ),
    ),
    PolicyCheckSpec(
        check_id="third_party_sharing",
        title="Third-Party Sharing Disclosures",
        weight=10.0,
        keywords=(
            "third party",
            "third-party",
            "share",
            "vendor",
            "processor",
            "affiliate",
        ),
    ),
    PolicyCheckSpec(
        check_id="purpose_limitation",
        title="Purpose Limitation",
        weight=8.0,
        keywords=(
            "purpose",
            "why we collect",
            "for the purpose of",
            "necessary",
            "legitimate interest",
        ),
    ),
    PolicyCheckSpec(
        check_id="incident_response",
        title="Breach And Incident Response",
        weight=8.0,
        keywords=(
            "breach",
            "incident",
            "notify",
            "notification",
            "unauthorized",
            "compromise",
        ),
    ),
    PolicyCheckSpec(
        check_id="contact_and_dpo",
        title="Contact Mechanism / DPO",
        weight=8.0,
        keywords=(
            "contact",
            "data protection officer",
            "dpo",
            "privacy team",
            "complaint",
        ),
    ),
)


def assess_policy_schema_compliance(
    policy_text: str,
    schema_text: str,
    model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    normalized_policy = _normalize_space(policy_text)
    normalized_schema = schema_text.strip()

    clauses = split_policy_clauses(policy_text)
    if not clauses and normalized_policy:
        clauses = [normalized_policy]
    fields = extract_schema_fields(normalized_schema)
    pii_fields, sensitive_fields = classify_schema_fields(fields)

    policy_checks: List[Dict[str, Any]] = []
    policy_presence: Dict[str, bool] = {}

    for spec in POLICY_CHECK_SPECS:
        score, matched, evidence = _score_policy_check(spec, clauses, normalized_policy)
        policy_checks.append(
            {
                "check_id": spec.check_id,
                "title": spec.title,
                "score": round(score, 2),
                "max_score": round(spec.weight, 2),
                "passed": score >= (spec.weight * 0.4),
                "matched_keywords": matched,
                "evidence": evidence,
            }
        )
        policy_presence[spec.check_id] = score > 0.0

    schema_alignment = _score_schema_alignment(
        pii_fields=pii_fields,
        sensitive_fields=sensitive_fields,
        policy_presence=policy_presence,
        policy_text=normalized_policy,
    )

    model_signal = _score_model_signal(clauses=clauses, model_path=model_path)

    total_score = sum(item["score"] for item in policy_checks)
    total_score += float(schema_alignment["score"])
    total_score += float(model_signal["score"])

    overall_score = max(0.0, min(100.0, round(total_score, 2)))
    grade = _grade_from_score(overall_score)
    status = _status_from_score(overall_score)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "overall_score": overall_score,
        "grade": grade,
        "status": status,
        "summary": {
            "clauses_analyzed": len(clauses),
            "schema_fields_detected": len(fields),
            "pii_fields_detected": len(pii_fields),
            "sensitive_fields_detected": len(sensitive_fields),
        },
        "policy_checks": policy_checks,
        "schema_analysis": schema_alignment,
        "model_signal": model_signal,
        "detected_fields": {
            "all_fields": fields,
            "pii_fields": pii_fields,
            "sensitive_fields": sensitive_fields,
        },
    }


def split_policy_clauses(policy_text: str) -> List[str]:
    text = policy_text.strip()
    if not text:
        return []

    chunks = re.split(r"\r?\n\s*\r?\n|\r?\n|(?<=[.!?])\s+", text)
    clauses: List[str] = []
    seen: set[str] = set()

    for chunk in chunks:
        normalized = _normalize_space(chunk)
        if len(normalized) < 24:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        clauses.append(normalized)

    return clauses


def extract_schema_fields(schema_text: str) -> List[str]:
    if not schema_text.strip():
        return []

    fields = set()
    text = schema_text.strip()

    # Try JSON-based schema extraction first.
    try:
        payload = json.loads(text)
        _walk_json_fields(payload, fields)
    except json.JSONDecodeError:
        pass

    # SQL-style extraction.
    for line in text.splitlines():
        raw = line.strip().strip(",")
        if not raw:
            continue
        if raw.lower().startswith(("create table", "primary key", "foreign key", "constraint", "index", ")")):
            continue

        match = re.match(r'^[`"\[]?([A-Za-z_][A-Za-z0-9_]*)[`"\]]?\s+[A-Za-z]', raw)
        if match:
            fields.add(match.group(1).lower())
            continue

        colon_match = re.match(r'^[`"\[]?([A-Za-z_][A-Za-z0-9_]*)[`"\]]?\s*:\s*', raw)
        if colon_match:
            fields.add(colon_match.group(1).lower())

    return sorted(fields)


def classify_schema_fields(fields: Sequence[str]) -> Tuple[List[str], List[str]]:
    pii: List[str] = []
    sensitive: List[str] = []

    for field in fields:
        lowered = field.lower()
        if any(token in lowered for token in PII_FIELD_PATTERNS):
            pii.append(field)
        if any(token in lowered for token in SENSITIVE_FIELD_PATTERNS):
            sensitive.append(field)

    return sorted(set(pii)), sorted(set(sensitive))


def resolve_default_model_path(project_root: Optional[Path] = None) -> Optional[Path]:
    root = project_root or Path.cwd()
    candidates = [
        root / "artifacts/phase-3-nb/classifier_checkpoint/model.json",
        root / "artifacts/phase-3/classifier_checkpoint/model.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _score_policy_check(
    spec: PolicyCheckSpec,
    clauses: Sequence[str],
    normalized_policy_text: str,
) -> Tuple[float, List[str], List[str]]:
    lowered_policy = normalized_policy_text.lower()
    matched = [keyword for keyword in spec.keywords if keyword in lowered_policy]

    if not matched:
        return 0.0, [], []

    coverage = min(1.0, len(matched) / max(2.0, len(spec.keywords) * 0.35))
    score = spec.weight * coverage

    evidence = _find_clause_evidence(clauses=clauses, keywords=matched, max_items=3)
    return score, matched, evidence


def _score_schema_alignment(
    pii_fields: Sequence[str],
    sensitive_fields: Sequence[str],
    policy_presence: Mapping[str, bool],
    policy_text: str,
) -> Dict[str, Any]:
    max_score = 15.0
    penalties = 0.0
    details: List[str] = []

    has_security = bool(policy_presence.get("security_safeguards", False))
    has_rights = bool(policy_presence.get("user_rights", False))
    has_retention = bool(policy_presence.get("data_retention", False))
    has_sharing = bool(policy_presence.get("third_party_sharing", False))

    if pii_fields and not has_rights:
        penalties += 4.0
        details.append("PII fields detected but user rights disclosures are weak.")
    if pii_fields and not has_security:
        penalties += 4.0
        details.append("PII fields detected but security safeguards are weak.")
    if pii_fields and not has_sharing:
        penalties += 2.0
        details.append("PII fields detected but third-party sharing disclosures are weak.")
    if pii_fields and not has_retention:
        penalties += 2.0
        details.append("PII fields detected but retention details are weak.")

    lowered_policy = policy_text.lower()
    if sensitive_fields and not has_security:
        penalties += 3.0
        details.append("Sensitive fields detected but security language is insufficient.")
    if sensitive_fields and "sensitive" not in lowered_policy and "special category" not in lowered_policy:
        penalties += 2.0
        details.append("Sensitive fields detected but no explicit sensitive-data policy language found.")

    raw_score = max_score - penalties
    score = max(0.0, min(max_score, raw_score))

    return {
        "title": "Schema Alignment",
        "score": round(score, 2),
        "max_score": max_score,
        "passed": score >= (max_score * 0.6),
        "details": details,
        "pii_fields_count": len(pii_fields),
        "sensitive_fields_count": len(sensitive_fields),
    }


def _score_model_signal(clauses: Sequence[str], model_path: Optional[Path]) -> Dict[str, Any]:
    max_score = 5.0

    resolved_model_path = model_path
    if resolved_model_path is None:
        resolved_model_path = resolve_default_model_path()

    if resolved_model_path is None or not resolved_model_path.exists():
        return {
            "title": "Model Signal",
            "score": 0.0,
            "max_score": max_score,
            "passed": False,
            "details": ["No baseline Naive Bayes checkpoint found; model signal skipped."],
            "model_used": "none",
            "label_distribution": {},
            "avg_confidence": None,
        }

    try:
        classifier = NaiveBayesTextClassifier.load(resolved_model_path)
    except Exception as exc:  # pragma: no cover - defensive path
        return {
            "title": "Model Signal",
            "score": 0.0,
            "max_score": max_score,
            "passed": False,
            "details": [f"Failed to load model checkpoint: {exc}"],
            "model_used": str(resolved_model_path),
            "label_distribution": {},
            "avg_confidence": None,
        }

    if not clauses:
        return {
            "title": "Model Signal",
            "score": 0.0,
            "max_score": max_score,
            "passed": False,
            "details": ["Policy text did not produce analyzable clauses."],
            "model_used": str(resolved_model_path),
            "label_distribution": {},
            "avg_confidence": None,
        }

    label_counts: Dict[str, int] = {label: 0 for label in LABELS}
    confidences: List[float] = []

    for clause in clauses:
        probabilities = classifier.predict_proba(clause)
        predicted = max(probabilities.items(), key=lambda item: item[1])[0]
        confidence = float(probabilities.get(predicted, 0.0))
        if predicted in label_counts:
            label_counts[predicted] += 1
        confidences.append(confidence)

    represented = sum(1 for count in label_counts.values() if count > 0)
    avg_confidence = sum(confidences) / len(confidences)

    score = 0.0
    score += min(3.0, represented) / 3.0 * 3.0
    score += max(0.0, min(2.0, avg_confidence * 2.0))

    return {
        "title": "Model Signal",
        "score": round(score, 2),
        "max_score": max_score,
        "passed": score >= 2.5,
        "details": [
            f"Model checkpoint used: {resolved_model_path}",
            f"Label coverage: {represented}/3",
            f"Average confidence: {avg_confidence:.3f}",
        ],
        "model_used": str(resolved_model_path),
        "label_distribution": label_counts,
        "avg_confidence": round(avg_confidence, 6),
    }


def _find_clause_evidence(clauses: Sequence[str], keywords: Sequence[str], max_items: int) -> List[str]:
    evidence: List[str] = []
    seen: set[str] = set()
    lowered_keywords = [keyword.lower() for keyword in keywords]

    for clause in clauses:
        lowered = clause.lower()
        if not any(keyword in lowered for keyword in lowered_keywords):
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        evidence.append(clause)
        if len(evidence) >= max_items:
            break

    return evidence


def _walk_json_fields(node: Any, out: set[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            cleaned = str(key).strip().lower()
            if cleaned:
                out.add(cleaned)
            _walk_json_fields(value, out)
        return

    if isinstance(node, list):
        for value in node:
            _walk_json_fields(value, out)


def _normalize_space(text: str) -> str:
    return " ".join(str(text).split())


def _grade_from_score(score: float) -> str:
    if score >= 85.0:
        return "A"
    if score >= 70.0:
        return "B"
    if score >= 55.0:
        return "C"
    if score >= 40.0:
        return "D"
    return "F"


def _status_from_score(score: float) -> str:
    if score >= 85.0:
        return "Compliant"
    if score >= 70.0:
        return "Mostly Compliant"
    if score >= 55.0:
        return "Needs Attention"
    return "High Compliance Risk"
