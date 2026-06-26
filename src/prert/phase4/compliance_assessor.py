"""Policy-and-schema compliance assessment helpers for Phase 4 GUI workflows.

Includes a policy-only assessment path (assess_policy_compliance) that evaluates
privacy policy clauses independently against GDPR, NIST, and ISO 27701 regulation
controls, producing per-regulation pass/fail verdicts with source citations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from prert.phase3.classifier import NaiveBayesTextClassifier
from prert.phase3.risk import compute_bayesian_risk


LABELS: tuple[str, str, str] = ("user", "system", "organization")
MODEL_PATH_ENV: str = "PRERT_PHASE4_MODEL_PATH"


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
class RegulationControl:
    """A single regulation requirement mapped to a compliance check."""
    regulation: str
    control_id: str
    title: str
    requirement: str


@dataclass(frozen=True)
class RegulationVerdict:
    """Pass/fail result for a single policy claim against a single regulation control."""
    regulation: str
    control_id: str
    control_title: str
    compliant: bool
    reason: str
    cited_clauses: list[str]
    remediation_advice: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "regulation": self.regulation,
            "control_id": self.control_id,
            "control_title": self.control_title,
            "compliant": self.compliant,
            "reason": self.reason,
            "cited_clauses": list(self.cited_clauses),
            "remediation_advice": self.remediation_advice,
        }


@dataclass
class PolicyClaimResult:
    """Assessment of a single policy clause against all regulation frameworks."""
    claim_index: int
    claim_text: str
    check_id: str
    check_title: str
    regulation_verdicts: list[RegulationVerdict]
    predicted_label: str
    confidence: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "claim_index": self.claim_index,
            "claim_text": self.claim_text,
            "check_id": self.check_id,
            "check_title": self.check_title,
            "regulation_verdicts": [v.as_dict() for v in self.regulation_verdicts],
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
        }


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


REGULATION_CONTROLS: Dict[str, List[RegulationControl]] = {
    "consent_transparency": [
        RegulationControl(
            regulation="GDPR",
            control_id="Article 6",
            title="Lawfulness of processing",
            requirement="Processing shall be lawful only if the data subject has given consent to the processing of their personal data for one or more specific purposes.",
        ),
        RegulationControl(
            regulation="GDPR",
            control_id="Article 7",
            title="Conditions for consent",
            requirement="The controller shall be able to demonstrate that the data subject has consented to processing of their personal data.",
        ),
        RegulationControl(
            regulation="NIST",
            control_id="CT.PO-P1",
            title="Transparency policies",
            requirement="Policies, processes, and procedures for transparency are established and in place to inform individuals about data processing purposes and practices.",
        ),
        RegulationControl(
            regulation="ISO_27701",
            control_id="A.7.2.3",
            title="Determining information for consent",
            requirement="The organisation shall determine and document the information needed for PII principals to give their consent for processing.",
        ),
        RegulationControl(
            regulation="ISO_27701",
            control_id="A.7.2.4",
            title="Obtaining consent",
            requirement="The organisation shall obtain and record consent from PII principals as required by applicable legislation.",
        ),
    ],
    "user_rights": [
        RegulationControl(
            regulation="GDPR",
            control_id="Article 15",
            title="Right of access by the data subject",
            requirement="The data subject shall have the right to obtain confirmation as to whether personal data concerning them is being processed and access to that data.",
        ),
        RegulationControl(
            regulation="GDPR",
            control_id="Article 17",
            title="Right to erasure ('right to be forgotten')",
            requirement="The data subject shall have the right to obtain the erasure of personal data concerning them without undue delay.",
        ),
        RegulationControl(
            regulation="GDPR",
            control_id="Article 20",
            title="Right to data portability",
            requirement="The data subject shall have the right to receive personal data in a structured, commonly used and machine-readable format.",
        ),
        RegulationControl(
            regulation="NIST",
            control_id="CT.DM-P3",
            title="Data access and correction",
            requirement="Mechanisms for individuals to access, correct, and delete their personal data are in place.",
        ),
        RegulationControl(
            regulation="ISO_27701",
            control_id="A.7.3.2",
            title="Determining information for PII principals",
            requirement="The organisation shall provide PII principals with clear, accessible information about data processing and their rights.",
        ),
        RegulationControl(
            regulation="ISO_27701",
            control_id="A.7.3.6",
            title="Access, correction and erasure",
            requirement="The organisation shall implement policies and procedures to meet obligations related to PII principals' requests for access, correction, or erasure.",
        ),
    ],
    "security_safeguards": [
        RegulationControl(
            regulation="GDPR",
            control_id="Article 32",
            title="Security of processing",
            requirement="The controller and processor shall implement appropriate technical and organisational measures to ensure a level of security appropriate to the risk, including encryption and access controls.",
        ),
        RegulationControl(
            regulation="NIST",
            control_id="PR.DS-P1",
            title="Data security",
            requirement="Data-at-rest and data-in-transit are protected using safeguards commensurate with the risk.",
        ),
        RegulationControl(
            regulation="NIST",
            control_id="PR.AC-P1",
            title="Access control",
            requirement="Access to data and devices is limited to authorised individuals, processes, or devices and managed consistently.",
        ),
        RegulationControl(
            regulation="ISO_27701",
            control_id="A.7.4.5",
            title="PII de-identification and deletion at end of processing",
            requirement="The organisation shall de-identify or delete PII at the end of processing unless there is a requirement to retain it.",
        ),
    ],
    "data_retention": [
        RegulationControl(
            regulation="GDPR",
            control_id="Article 5.1(e)",
            title="Storage limitation",
            requirement="Personal data shall be kept in a form which permits identification of data subjects for no longer than is necessary for the purposes for which the data are processed.",
        ),
        RegulationControl(
            regulation="NIST",
            control_id="CT.DM-P7",
            title="Data retention and disposal",
            requirement="Retention schedules and disposal methods are established and data is disposed of according to policy.",
        ),
        RegulationControl(
            regulation="ISO_27701",
            control_id="A.7.4.4",
            title="PII minimisation objectives",
            requirement="The organisation shall define and document data minimisation objectives and retention periods.",
        ),
    ],
    "third_party_sharing": [
        RegulationControl(
            regulation="GDPR",
            control_id="Article 28",
            title="Processor",
            requirement="Processing by a processor shall be governed by a contract that sets out the subject-matter and duration of the processing, the nature and purpose, and the obligations of the processor.",
        ),
        RegulationControl(
            regulation="NIST",
            control_id="CT.PO-P4",
            title="Third-party data sharing",
            requirement="Policies for disclosing data processing activities to third parties including sharing, trading, and selling are established.",
        ),
        RegulationControl(
            regulation="ISO_27701",
            control_id="A.7.5.1",
            title="Identifying basis for PII transfer",
            requirement="The organisation shall identify and document the relevant basis for transfers of PII to third parties.",
        ),
    ],
    "purpose_limitation": [
        RegulationControl(
            regulation="GDPR",
            control_id="Article 5.1(b)",
            title="Purpose limitation",
            requirement="Personal data shall be collected for specified, explicit and legitimate purposes and not further processed in a manner incompatible with those purposes.",
        ),
        RegulationControl(
            regulation="NIST",
            control_id="CT.PO-P2",
            title="Purpose specification",
            requirement="Purposes for data processing are identified and communicated to individuals.",
        ),
        RegulationControl(
            regulation="ISO_27701",
            control_id="A.7.2.1",
            title="Identify and document purpose",
            requirement="The organisation shall identify and document the specific purposes for which the PII will be processed.",
        ),
    ],
    "incident_response": [
        RegulationControl(
            regulation="GDPR",
            control_id="Article 33",
            title="Notification of a personal data breach to the supervisory authority",
            requirement="In the case of a personal data breach, the controller shall notify the supervisory authority within 72 hours after having become aware of it.",
        ),
        RegulationControl(
            regulation="GDPR",
            control_id="Article 34",
            title="Communication of a personal data breach to the data subject",
            requirement="When the breach is likely to result in a high risk to the rights and freedoms of natural persons, the controller shall communicate the breach to the data subject without undue delay.",
        ),
        RegulationControl(
            regulation="NIST",
            control_id="RS.CO-P1",
            title="Response communication",
            requirement="Response activities are coordinated with internal and external stakeholders, including breach notifications.",
        ),
        RegulationControl(
            regulation="ISO_27701",
            control_id="A.7.3.9",
            title="PII breach notification",
            requirement="The organisation shall notify PII principals of breaches involving their PII when required by legislation or contract.",
        ),
    ],
    "contact_and_dpo": [
        RegulationControl(
            regulation="GDPR",
            control_id="Article 37",
            title="Designation of the data protection officer",
            requirement="The controller and the processor shall designate a data protection officer where core activities require regular and systematic monitoring of data subjects on a large scale.",
        ),
        RegulationControl(
            regulation="NIST",
            control_id="GV.PO-P3",
            title="Roles and responsibilities",
            requirement="Roles and responsibilities for the workforce are established with respect to privacy.",
        ),
        RegulationControl(
            regulation="ISO_27701",
            control_id="A.7.2.8",
            title="Records related to processing PII",
            requirement="The organisation shall determine and maintain records necessary to demonstrate compliance with its obligations for processing PII, including contact mechanisms.",
        ),
    ],
}


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
    env_model_path = os.getenv(MODEL_PATH_ENV, "").strip()

    if env_model_path:
        candidate = Path(env_model_path).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.exists():
            return candidate

    candidates = [
        root / "deployment/demo-assets/phase-3-nb/classifier_checkpoint/model.json",
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


def assess_policy_compliance(
    policy_text: str,
    model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Assess a privacy policy against GDPR, NIST, and ISO 27701 independently.

    Unlike assess_policy_schema_compliance, this function requires **only** the
    privacy policy text. Each extracted clause is evaluated against every
    regulation framework, producing per-regulation pass/fail verdicts with
    source citations (the exact policy text supporting each verdict).
    """
    normalized_policy = _normalize_space(policy_text)
    clauses = split_policy_clauses(policy_text)
    if not clauses and normalized_policy:
        clauses = [normalized_policy]

    claims: List[PolicyClaimResult] = []
    regulation_tallies: Dict[str, Dict[str, int]] = {}
    regulation_control_totals: Dict[str, int] = {}
    predictions_for_risk = []

    resolved_model_path = model_path
    if resolved_model_path is None:
        resolved_model_path = resolve_default_model_path()
        
    classifier = None
    if resolved_model_path is not None and resolved_model_path.exists():
        try:
            classifier = NaiveBayesTextClassifier.load(resolved_model_path)
        except Exception:
            pass

    for claim_index, clause in enumerate(clauses):
        predicted_label = "unknown"
        confidence = 0.0
        if classifier is not None:
            probabilities = classifier.predict_proba(clause)
            predicted_label = max(probabilities.items(), key=lambda item: item[1])[0]
            confidence = float(probabilities.get(predicted_label, 0.0))
            
        predictions_for_risk.append({
            "predicted_label": predicted_label,
            "confidence": confidence,
            "text": clause,
        })
        
        lowered_clause = clause.lower()

        for spec in POLICY_CHECK_SPECS:
            matched_keywords = [
                kw for kw in spec.keywords if kw in lowered_clause
            ]
            if not matched_keywords:
                continue

            controls = REGULATION_CONTROLS.get(spec.check_id, [])
            verdicts: List[RegulationVerdict] = []

            for control in controls:
                compliant = _clause_satisfies_control(
                    clause=clause,
                    matched_keywords=matched_keywords,
                    control=control,
                )
                reason = _build_verdict_reason(
                    compliant=compliant,
                    clause=clause,
                    control=control,
                    matched_keywords=matched_keywords,
                )
                remediation_advice = ""
                if compliant:
                    remediation_advice = f"Clause meets {control.control_id} requirements. To improve further, ensure the language is highly specific."
                else:
                    remediation_advice = f"Update the policy to address {control.control_id}: {control.requirement}"

                verdicts.append(
                    RegulationVerdict(
                        regulation=control.regulation,
                        control_id=control.control_id,
                        control_title=control.title,
                        compliant=compliant,
                        reason=reason,
                        cited_clauses=[clause],
                        remediation_advice=remediation_advice,
                    )
                )

                reg = control.regulation
                if reg not in regulation_tallies:
                    regulation_tallies[reg] = {"pass": 0, "fail": 0}
                if compliant:
                    regulation_tallies[reg]["pass"] += 1
                else:
                    regulation_tallies[reg]["fail"] += 1

            claims.append(
                PolicyClaimResult(
                    claim_index=claim_index,
                    claim_text=clause,
                    check_id=spec.check_id,
                    check_title=spec.title,
                    regulation_verdicts=verdicts,
                    predicted_label=predicted_label,
                    confidence=confidence,
                )
            )

    for controls in REGULATION_CONTROLS.values():
        for control in controls:
            reg = control.regulation
            regulation_control_totals[reg] = regulation_control_totals.get(reg, 0) + 1

    regulation_summary: Dict[str, Dict[str, Any]] = {}
    for reg, totals in sorted(regulation_tallies.items()):
        total_evaluated = totals["pass"] + totals["fail"]
        coverage_pct = round(
            (totals["pass"] / total_evaluated * 100) if total_evaluated > 0 else 0.0, 2
        )
        regulation_summary[reg] = {
            "pass_count": totals["pass"],
            "fail_count": totals["fail"],
            "total_evaluated": total_evaluated,
            "total_controls": regulation_control_totals.get(reg, 0),
            "compliance_pct": coverage_pct,
        }

    for reg, total in sorted(regulation_control_totals.items()):
        if reg not in regulation_summary:
            regulation_summary[reg] = {
                "pass_count": 0,
                "fail_count": 0,
                "total_evaluated": 0,
                "total_controls": total,
                "compliance_pct": 0.0,
            }

    model_signal = _score_model_signal(clauses=clauses, model_path=model_path)

    total_pass = sum(t.get("pass_count", 0) for t in regulation_summary.values())
    total_controls = sum(t.get("total_controls", 0) for t in regulation_summary.values())
    raw_score = (total_pass / total_controls * 100) if total_controls > 0 else 0.0
    overall_score = max(0.0, min(100.0, round(raw_score, 2)))
    grade = _grade_from_score(overall_score)
    status = _status_from_score(overall_score)

    bayesian_risk = None
    if predictions_for_risk and classifier is not None:
        bayesian_risk = compute_bayesian_risk(predictions_for_risk)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": "policy_only",
        "overall_score": overall_score,
        "grade": grade,
        "status": status,
        "summary": {
            "clauses_analyzed": len(clauses),
            "claims_generated": len(claims),
            "regulations_evaluated": list(regulation_summary.keys()),
        },
        "claims": [c.as_dict() for c in claims],
        "regulation_summary": regulation_summary,
        "model_signal": model_signal,
        "bayesian_risk": bayesian_risk,
    }


def _clause_satisfies_control(
    clause: str,
    matched_keywords: List[str],
    control: RegulationControl,
) -> bool:
    """Determine whether a policy clause satisfies a regulation control.

    Uses keyword overlap between the clause and the control's requirement text
    as a deterministic relevance signal. A clause passes if at least two of its
    matched keywords also appear in the control requirement, or if the clause
    contains substantive language that maps to the control's domain.
    """
    lowered_requirement = control.requirement.lower()
    lowered_clause = clause.lower()

    requirement_overlap = sum(
        1 for kw in matched_keywords if kw in lowered_requirement
    )
    if requirement_overlap >= 2:
        return True

    control_signals = _extract_control_signals(control)
    signal_hits = sum(1 for signal in control_signals if signal in lowered_clause)
    return signal_hits >= 1 and requirement_overlap >= 1


def _extract_control_signals(control: RegulationControl) -> List[str]:
    """Extract key terms from a control's requirement for matching."""
    signal_map: Dict[str, List[str]] = {
        "Article 6": ["lawful", "legal basis", "legitimate"],
        "Article 7": ["demonstrate consent", "withdraw"],
        "Article 15": ["access", "obtain confirmation"],
        "Article 17": ["erasure", "right to be forgotten", "delete"],
        "Article 20": ["portability", "machine-readable"],
        "Article 28": ["processor", "contract", "sub-processor"],
        "Article 32": ["encrypt", "security measure", "access control"],
        "Article 33": ["notify", "72 hours", "supervisory authority"],
        "Article 34": ["communicate", "high risk", "breach"],
        "Article 37": ["data protection officer", "dpo"],
        "Article 5.1(b)": ["specified purpose", "legitimate purpose", "compatible"],
        "Article 5.1(e)": ["no longer than necessary", "storage limitation", "retention period"],
        "CT.PO-P1": ["transparency", "inform"],
        "CT.PO-P2": ["purpose", "communicated"],
        "CT.PO-P4": ["third party", "sharing", "disclosure"],
        "CT.DM-P3": ["access", "correct", "delete"],
        "CT.DM-P7": ["retention schedule", "disposal"],
        "PR.DS-P1": ["protect", "safeguard"],
        "PR.AC-P1": ["access control", "authorised"],
        "RS.CO-P1": ["response", "notification"],
        "GV.PO-P3": ["roles", "responsibilities", "privacy team"],
        "A.7.2.1": ["purpose", "documented"],
        "A.7.2.3": ["consent", "information"],
        "A.7.2.4": ["obtain consent", "record"],
        "A.7.2.8": ["records", "demonstrate compliance", "contact"],
        "A.7.3.2": ["clear", "accessible information"],
        "A.7.3.6": ["access", "correction", "erasure"],
        "A.7.3.9": ["breach notification", "notify"],
        "A.7.4.4": ["minimisation", "retention"],
        "A.7.4.5": ["de-identify", "deletion"],
        "A.7.5.1": ["transfer", "basis"],
    }
    return signal_map.get(control.control_id, [])


def _build_verdict_reason(
    compliant: bool,
    clause: str,
    control: RegulationControl,
    matched_keywords: List[str],
) -> str:
    """Build a human-readable reason for a regulation verdict."""
    keywords_csv = ", ".join(f"'{kw}'" for kw in matched_keywords[:4])
    if compliant:
        return (
            f"Policy clause addresses {control.control_id} ({control.title}). "
            f"Matched keywords [{keywords_csv}] align with the requirement: "
            f"\"{control.requirement[:120]}...\""
        )
    return (
        f"Policy clause mentions [{keywords_csv}] but does not sufficiently "
        f"address {control.control_id} ({control.title}): "
        f"\"{control.requirement[:120]}...\""
    )


def _status_from_score(score: float) -> str:
    if score >= 85.0:
        return "Compliant"
    if score >= 70.0:
        return "Mostly Compliant"
    if score >= 55.0:
        return "Needs Attention"
    return "High Compliance Risk"
