"""Synthetic policy+schema dataset generation for Phase 4 compliance workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional

from prert.phase3.io import write_json, write_jsonl
from prert.phase4.compliance_assessor import assess_policy_schema_compliance


BAND_ORDER = ("low", "medium", "high")

DEFAULT_COUNTS_BY_BAND: Dict[str, int] = {
    "low": 6,
    "medium": 6,
    "high": 6,
}

TARGET_SCORE_RANGES: Dict[str, tuple[float, float]] = {
    "low": (0.0, 40.0),
    "medium": (45.0, 74.0),
    "high": (75.0, 100.0),
}


LOW_POLICY_FRAGMENTS = [
    "Using this service implies consent to our baseline data handling terms.",
    "We may share account information with partners to operate and improve services.",
    "Policy terms may change over time without detailed implementation guarantees.",
]

MEDIUM_POLICY_FRAGMENT_GROUPS = [
    [
        "We provide a privacy notice and request consent before enabling optional data uses.",
        "Users can access account records and request deletion through support channels.",
        "Data is protected with encryption and access control safeguards in production systems.",
        "Limited third-party vendors process billing data under contractual obligations.",
    ],
    [
        "We publish transparent notice details and collect consent for non-essential processing.",
        "Users may access and delete personal data associated with their account profile.",
        "Security controls include encryption, secure storage, and role-based access control.",
        "We share required data with third-party processors and vetted service vendors.",
    ],
]

HIGH_POLICY_FRAGMENT_GROUPS = [
    [
        "We provide transparent privacy notice details, request explicit consent, and offer opt out controls for optional processing.",
        "Users can access, correct, and delete personal data, and data portability requests are supported through the privacy portal.",
        "Security safeguards include encryption, multi-factor authentication, and strict access control protections for sensitive systems.",
        "We disclose each third-party processor, vendor, and affiliate category involved in data sharing operations.",
        "We define retention schedules, storage period limits, and delete after timelines for archived records.",
        "Data is collected only for documented purpose limitations and necessary legitimate interest operations.",
        "If an incident or breach occurs, we send notification updates and investigate unauthorized compromise rapidly.",
        "Users can contact the privacy team or Data Protection Officer (DPO) to submit a complaint or request.",
        "Special category and sensitive data handling controls are documented in dedicated governance procedures.",
    ],
    [
        "Our policy is transparent, includes clear notice language, and requires consent with opt-out controls for optional uses.",
        "Individuals may access, rectify, erase, and port their personal data through automated rights workflows.",
        "We enforce encryption at rest and in transit, multi-factor access, and layered safeguards for regulated services.",
        "Third-party sharing disclosures identify each processor, vendor role, and affiliate transfer category.",
        "Retention and archive controls define storage periods and automated delete after windows for stale records.",
        "Purpose limitation statements describe why data is collected and what processing is necessary.",
        "Incident response procedures cover breach notification obligations and unauthorized access triage.",
        "Data subjects can contact our DPO or privacy team for complaints, access requests, and escalation support.",
        "Sensitive data categories receive additional handling safeguards and policy-level restrictions.",
    ],
]


SCHEMA_TEMPLATES = {
    "low": [
        """
CREATE TABLE customer_profiles_{suffix} (
  profile_id BIGINT PRIMARY KEY,
  full_name VARCHAR(255),
  email VARCHAR(255),
  phone VARCHAR(30),
  home_address TEXT,
  city VARCHAR(100),
  country VARCHAR(80),
  dob DATE,
  ssn VARCHAR(20),
  passport_number VARCHAR(30),
  credit_card_number VARCHAR(40),
  account_number VARCHAR(40),
  salary_amount DECIMAL(12,2),
  health_diagnosis TEXT,
  biometric_hash TEXT,
  ip_address VARCHAR(64),
  device_id VARCHAR(80)
);
""".strip(),
        """
CREATE TABLE user_records_{suffix} (
  user_id BIGINT PRIMARY KEY,
  first_name VARCHAR(120),
  last_name VARCHAR(120),
  email VARCHAR(255),
  phone VARCHAR(32),
  postal_address TEXT,
  birth_date DATE,
  national_id VARCHAR(40),
  bank_account_number VARCHAR(40),
  income_band VARCHAR(40),
  medical_notes TEXT,
  biometric_template TEXT,
  ip_address VARCHAR(64),
  device_identifier VARCHAR(64)
);
""".strip(),
    ],
    "medium": [
        """
CREATE TABLE accounts_{suffix} (
  account_id BIGINT PRIMARY KEY,
  email VARCHAR(255),
  phone VARCHAR(30),
  address_line1 TEXT,
  country VARCHAR(80),
  loyalty_tier VARCHAR(40),
  last_login_ip VARCHAR(64),
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);
""".strip(),
        """
CREATE TABLE billing_profiles_{suffix} (
  profile_id BIGINT PRIMARY KEY,
  email VARCHAR(255),
  customer_name VARCHAR(200),
  billing_city VARCHAR(100),
  card_last4 VARCHAR(4),
  account_reference VARCHAR(40),
  last_login_ip VARCHAR(64),
  created_at TIMESTAMP
);
""".strip(),
    ],
    "high": [
        """
CREATE TABLE privacy_controlled_accounts_{suffix} (
  account_id UUID PRIMARY KEY,
  pseudonymous_user_id VARCHAR(80),
  email VARCHAR(255),
  consent_state VARCHAR(32),
  retention_expiry_at TIMESTAMP,
  processor_region VARCHAR(32),
  encryption_key_ref VARCHAR(64),
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);
""".strip(),
        """
CREATE TABLE governed_customer_registry_{suffix} (
  customer_uuid UUID PRIMARY KEY,
  privacy_subject_id VARCHAR(80),
  email VARCHAR(255),
  data_portability_token VARCHAR(128),
  retention_window_days INTEGER,
  processor_vendor_code VARCHAR(32),
  security_tier VARCHAR(32),
  created_at TIMESTAMP
);
""".strip(),
    ],
}


def generate_synthetic_policy_schema_dataset(
    output_dir: Path,
    counts_by_band: Optional[Mapping[str, int]] = None,
    seed: int = 42,
    include_model_signal: bool = False,
    model_path: Optional[Path] = None,
    export_upload_fixtures: bool = False,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_counts = _resolve_counts_by_band(counts_by_band)
    rnd = Random(seed)

    rows: List[Dict[str, Any]] = []
    model_path_for_assessment = model_path if include_model_signal else Path("__phase4_synthetic_model_signal_disabled__.json")

    running_index = 0
    for band in BAND_ORDER:
        count = resolved_counts[band]
        low, high = TARGET_SCORE_RANGES[band]
        for band_index in range(count):
            running_index += 1
            sample_id = f"synth-{running_index:05d}"
            policy_text = _render_policy_text(band=band, rnd=rnd)
            schema_text = _render_schema_text(band=band, rnd=rnd, suffix=f"{running_index:05d}")

            assessment = assess_policy_schema_compliance(
                policy_text=policy_text,
                schema_text=schema_text,
                model_path=model_path_for_assessment,
            )

            overall_score = float(assessment["overall_score"])
            rows.append(
                {
                    "sample_id": sample_id,
                    "compliance_band": band,
                    "sample_index": band_index,
                    "target_score_min": low,
                    "target_score_max": high,
                    "within_target_band": low <= overall_score <= high,
                    "policy_text": policy_text,
                    "schema_text": schema_text,
                    "assessment": {
                        "overall_score": overall_score,
                        "grade": assessment["grade"],
                        "status": assessment["status"],
                        "summary": assessment["summary"],
                        "schema_analysis": assessment["schema_analysis"],
                        "policy_checks": assessment["policy_checks"],
                        "model_signal": assessment["model_signal"],
                    },
                    "metadata": {
                        "generator": "phase4_synthetic_policy_schema_v1",
                        "seed": seed,
                        "include_model_signal": bool(include_model_signal),
                    },
                }
            )

    dataset_path = output_dir / "synthetic_policy_schema_pairs.jsonl"
    manifest_path = output_dir / "synthetic_policy_schema_manifest.json"
    dictionary_path = output_dir / "synthetic_policy_schema_dictionary.md"

    write_jsonl(dataset_path, rows)

    fixture_summary: Dict[str, Any] = {
        "enabled": bool(export_upload_fixtures),
        "output_dir": "",
        "files_written": 0,
    }
    if export_upload_fixtures:
        fixture_summary = _write_upload_fixtures(output_dir=output_dir, rows=rows)

    score_summary = _compute_score_summary(rows)
    manifest = {
        "phase": "phase-4",
        "artifact_type": "synthetic_policy_schema_dataset",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "seed": seed,
        "counts_by_band": resolved_counts,
        "target_score_ranges": {
            band: {
                "min": TARGET_SCORE_RANGES[band][0],
                "max": TARGET_SCORE_RANGES[band][1],
            }
            for band in BAND_ORDER
        },
        "score_summary": score_summary,
        "upload_fixtures": fixture_summary,
        "output_files": {
            "dataset": str(dataset_path),
            "manifest": str(manifest_path),
            "dictionary": str(dictionary_path),
        },
    }

    write_json(manifest_path, manifest)
    dictionary_path.write_text(_render_dictionary_markdown(), encoding="utf-8")
    return manifest


def _resolve_counts_by_band(counts_by_band: Optional[Mapping[str, int]]) -> Dict[str, int]:
    resolved = dict(DEFAULT_COUNTS_BY_BAND)
    if counts_by_band is not None:
        for band, value in counts_by_band.items():
            if band not in resolved:
                raise ValueError(f"Unsupported compliance band '{band}'. Choose from: {', '.join(BAND_ORDER)}")
            numeric = int(value)
            if numeric < 0:
                raise ValueError(f"Count for band '{band}' must be non-negative")
            resolved[band] = numeric
    return resolved


def _render_policy_text(band: str, rnd: Random) -> str:
    if band == "low":
        fragments = list(LOW_POLICY_FRAGMENTS)
        rnd.shuffle(fragments)
        selected = fragments[: rnd.randint(2, 3)]
    elif band == "medium":
        selected = list(rnd.choice(MEDIUM_POLICY_FRAGMENT_GROUPS))
    elif band == "high":
        selected = list(rnd.choice(HIGH_POLICY_FRAGMENT_GROUPS))
    else:
        raise ValueError(f"Unsupported compliance band: {band}")

    return "\n\n".join(selected)


def _render_schema_text(band: str, rnd: Random, suffix: str) -> str:
    templates = SCHEMA_TEMPLATES[band]
    template = rnd.choice(templates)
    return template.format(suffix=suffix)


def _compute_score_summary(rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    per_band_scores: Dict[str, List[float]] = {band: [] for band in BAND_ORDER}
    per_band_hits: Dict[str, int] = {band: 0 for band in BAND_ORDER}

    for row in rows:
        band = str(row.get("compliance_band", ""))
        if band not in per_band_scores:
            continue
        assessment = row.get("assessment")
        if not isinstance(assessment, dict):
            continue
        score = float(assessment.get("overall_score", 0.0))
        per_band_scores[band].append(score)
        if bool(row.get("within_target_band", False)):
            per_band_hits[band] += 1

    result: Dict[str, Any] = {}
    for band in BAND_ORDER:
        scores = per_band_scores[band]
        if not scores:
            result[band] = {
                "count": 0,
                "in_target_band": 0,
                "minimum": None,
                "mean": None,
                "maximum": None,
            }
            continue

        result[band] = {
            "count": len(scores),
            "in_target_band": per_band_hits[band],
            "minimum": round(min(scores), 4),
            "mean": round(mean(scores), 4),
            "maximum": round(max(scores), 4),
        }

    return result


def _write_upload_fixtures(output_dir: Path, rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    fixture_dir = output_dir / "upload-fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    files_written = 0
    for row in rows:
        sample_id = str(row.get("sample_id", "sample"))
        policy_path = fixture_dir / f"{sample_id}-policy.md"
        schema_path = fixture_dir / f"{sample_id}-schema.sql"
        policy_text = str(row.get("policy_text", "")).strip()
        schema_text = str(row.get("schema_text", "")).strip()

        policy_path.write_text(policy_text + "\n", encoding="utf-8")
        schema_path.write_text(schema_text + "\n", encoding="utf-8")
        files_written += 2

    return {
        "enabled": True,
        "output_dir": str(fixture_dir),
        "files_written": files_written,
    }


def _render_dictionary_markdown() -> str:
    return "\n".join(
        [
            "# Synthetic Policy/Schema Dataset Dictionary",
            "",
            "Synthetic records for Phase 4 supervisor workflow testing.",
            "Each row represents one privacy-policy and database-schema pair with an assessor output.",
            "",
            "## JSONL Fields",
            "",
            "- sample_id: Stable synthetic sample identifier.",
            "- compliance_band: Intended compliance band (`low`, `medium`, `high`).",
            "- sample_index: Zero-based index within each compliance band.",
            "- target_score_min: Lower bound for intended compliance score range.",
            "- target_score_max: Upper bound for intended compliance score range.",
            "- within_target_band: Whether generated assessment score fell in the intended range.",
            "- policy_text: Synthetic privacy policy text block.",
            "- schema_text: Synthetic database schema text block.",
            "- assessment: Structured output from `assess_policy_schema_compliance`.",
            "- metadata: Generator metadata (seed, version, model-signal setting).",
            "",
            "## Intended Use",
            "",
            "- Regression tests for compliance scoring behavior.",
            "- Manual supervisor upload demos in the Streamlit GUI.",
            "- Stressing low/medium/high compliance scenarios with reproducible seed control.",
            "",
            "## Non-Use",
            "",
            "- Not representative of legal advice or production compliance attestation.",
            "- Not a substitute for domain expert policy review.",
            "",
        ]
    )
