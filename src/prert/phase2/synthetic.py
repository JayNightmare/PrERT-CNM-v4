"""Synthetic policy and observation generation for Phase 2 scoring."""

from __future__ import annotations

from collections import defaultdict
from random import Random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from prert.extract.schema import stable_hash
from prert.phase2.types import MetricSpec, SyntheticObservation, SyntheticPolicyClaim, SyntheticPolicyDocument


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

SCENARIO_COMPLIANCE_BANDS = {
    "normal": "high",
    "stressed": "medium",
    "adversarial": "low",
}

CLAIM_STATUS_PATTERNS = {
    "normal": ("compliant", "compliant", "compliant", "partial", "compliant", "compliant", "noncompliant"),
    "stressed": ("compliant", "partial", "partial", "noncompliant", "compliant", "partial"),
    "adversarial": ("noncompliant", "partial", "noncompliant", "compliant", "partial", "noncompliant"),
}

CLAIM_STRENGTH_RANGES = {
    "compliant": (0.80, 1),
    "partial": (0.45, 0.79),
    "noncompliant": (0.05, 0.44),
}

CLAIM_TYPES = {
    "user": (
        "consent transparency",
        "access and deletion rights",
        "data portability",
        "cookie preference control",
    ),
    "system": (
        "encryption safeguards",
        "access control",
        "incident monitoring",
        "data integrity controls",
    ),
    "organization": (
        "retention governance",
        "processor oversight",
        "privacy accountability",
        "audit readiness",
    ),
}

ORGANIZATION_PROFILES = (
    ("NexusCart", "ecommerce", "US"),
    ("HealthBridge Labs", "digital health", "EU"),
    ("OrbitLearn", "education technology", "UK"),
    ("CivicCloud", "public services", "CA"),
    ("FinTrail", "financial services", "US"),
    ("AtlasHome", "consumer IoT", "AU"),
)

PolicyClaimLink = Tuple[SyntheticPolicyDocument, SyntheticPolicyClaim]


def generate_synthetic_observations(
    metric_specs: Iterable[MetricSpec],
    seed: int = 42,
    scenarios: Iterable[str] = ("normal", "stressed", "adversarial"),
    policy_documents: Optional[Sequence[SyntheticPolicyDocument]] = None,
) -> List[SyntheticObservation]:
    rnd = Random(seed)
    observations: List[SyntheticObservation] = []
    policy_claim_index = _index_policy_claims(policy_documents or [])

    for spec in metric_specs:
        for scenario in scenarios:
            profile = SCENARIO_PROFILES[scenario]
            policy_claim_link = _select_policy_claim_link(policy_claim_index, scenario, spec, rnd)
            total_checks = _sample_total_checks(spec.level, rnd)
            failure_rate = _claim_adjusted_failure_rate(profile["failure_rate"], policy_claim_link)
            failure_count = _sample_failure_count(total_checks, failure_rate, rnd)
            # B5: cap missing_fields at len(required_fields) to honour the
            # MetricSpec contract instead of using a magic constant of 6.
            missing_field_cap = max(1, len(spec.required_fields))
            missing_fields = _sample_missing_fields(
                profile["missing_rate"], rnd, missing_field_cap
            )
            confidence = round(rnd.uniform(profile["confidence_low"], profile["confidence_high"]), 4)

            entity_type = spec.level
            # B1: derive entity_id deterministically from the control id so
            # the same control always maps to the same synthetic entity
            # across runs, regardless of metric_specs ordering.
            entity_id = f"{entity_type}-{stable_hash(spec.control_id)[:12]}"
            # B2: observation_id no longer needs entity_id since
            # (scenario, metric_id) is already unique — keeps ids stable
            # if entity_id derivation changes again later.
            observation_id = "obs::" + stable_hash(f"{scenario}|{spec.metric_id}")[:18]

            metadata = {
                "generator": "phase2_synthetic_v2",
                "scenario_profile": scenario,
                "compliance_band": SCENARIO_COMPLIANCE_BANDS.get(scenario, "unknown"),
            }
            if policy_claim_link is not None:
                policy, claim = policy_claim_link
                metadata.update(
                    {
                        "policy_id": policy.policy_id,
                        "policy_claim_id": claim.claim_id,
                        "claim_compliance_status": claim.compliance_status,
                        "claim_compliance_strength": claim.compliance_strength,
                        "claim_text_excerpt": claim.text[:240],
                    },
                )

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
                    metadata=metadata,
                )
            )

    return observations


def generate_synthetic_policy_documents(
    metric_specs: Iterable[MetricSpec],
    seed: int = 42,
    scenarios: Iterable[str] = ("normal", "stressed", "adversarial"),
    policies_per_scenario: int = 6,
    min_claims: int = 5,
    max_claims: int = 9,
) -> List[SyntheticPolicyDocument]:
    specs = list(metric_specs)
    if not specs:
        return []

    rnd = Random(seed + 7919)
    policies: List[SyntheticPolicyDocument] = []
    claims_low = max(2, int(min_claims))
    claims_high = max(claims_low, int(max_claims))

    for scenario in scenarios:
        compliance_band = SCENARIO_COMPLIANCE_BANDS.get(scenario, "unknown")
        for policy_index in range(max(1, int(policies_per_scenario))):
            organization, sector, region = ORGANIZATION_PROFILES[(policy_index + len(scenario)) % len(ORGANIZATION_PROFILES)]
            policy_id = "policy::" + stable_hash(f"{scenario}|{policy_index}|{seed}|phase2")[:16]
            claim_count = rnd.randint(claims_low, claims_high)
            selected_specs = _select_policy_specs(specs, claim_count, rnd)
            claims: List[SyntheticPolicyClaim] = []

            for claim_index, spec in enumerate(selected_specs):
                compliance_status = _claim_status_for_index(scenario, claim_index)
                strength_low, strength_high = CLAIM_STRENGTH_RANGES[compliance_status]
                compliance_strength = round(rnd.uniform(strength_low, strength_high), 4)
                claim_type = rnd.choice(CLAIM_TYPES.get(spec.level, CLAIM_TYPES["organization"]))
                claim_text = _render_claim_text(
                    spec=spec,
                    organization=organization,
                    claim_type=claim_type,
                    compliance_status=compliance_status,
                    rnd=rnd,
                )
                claim_id = "claim::" + stable_hash(f"{policy_id}|{claim_index}|{spec.metric_id}")[:18]
                claims.append(
                    SyntheticPolicyClaim(
                        claim_id=claim_id,
                        control_id=spec.control_id,
                        metric_id=spec.metric_id,
                        level=spec.level,
                        regulation=spec.regulation,
                        claim_type=claim_type,
                        compliance_status=compliance_status,
                        compliance_strength=compliance_strength,
                        text=claim_text,
                        expected_failure=compliance_status != "compliant",
                        metadata={
                            "metric_name": spec.metric_name,
                            "normalization_rule": spec.normalization_rule,
                        },
                    )
                )

            policy_text = _render_policy_text(
                organization=organization,
                sector=sector,
                region=region,
                compliance_band=compliance_band,
                claims=claims,
                policy_index=policy_index,
            )
            policies.append(
                SyntheticPolicyDocument(
                    policy_id=policy_id,
                    scenario=scenario,
                    compliance_band=compliance_band,
                    organization=organization,
                    sector=sector,
                    region=region,
                    effective_date=f"2025-{(policy_index % 12) + 1:02d}-01",
                    policy_text=policy_text,
                    claims=claims,
                    metadata={
                        "generator": "phase2_synthetic_policy_v1",
                        "claim_count": len(claims),
                        "target_compliance_band": compliance_band,
                    },
                )
            )

    return policies


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


def _sample_missing_fields(missing_rate: float, rnd: Random, max_fields: int = 6) -> int:
    count = 0
    for _ in range(max_fields):
        if rnd.random() < missing_rate:
            count += 1
    return count


def _select_policy_specs(specs: Sequence[MetricSpec], claim_count: int, rnd: Random) -> List[MetricSpec]:
    by_level: Dict[str, List[MetricSpec]] = defaultdict(list)
    for spec in specs:
        by_level[spec.level].append(spec)

    selected: List[MetricSpec] = []
    for level in ("user", "system", "organization"):
        if by_level[level] and len(selected) < claim_count:
            selected.append(rnd.choice(by_level[level]))

    while len(selected) < claim_count:
        selected.append(rnd.choice(specs))

    rnd.shuffle(selected)
    return selected


def _claim_status_for_index(scenario: str, claim_index: int) -> str:
    pattern = CLAIM_STATUS_PATTERNS.get(scenario, CLAIM_STATUS_PATTERNS["stressed"])
    return pattern[claim_index % len(pattern)]


def _render_claim_text(
    spec: MetricSpec,
    organization: str,
    claim_type: str,
    compliance_status: str,
    rnd: Random,
) -> str:
    native_id = str(spec.metadata.get("native_id", "")).strip() or spec.control_id
    control_ref = f"{spec.regulation} {native_id}"
    level = spec.level

    templates = {
        ("user", "compliant"): (
            "{org} gives individuals clear notice before {claim_type}, records consent choices, and lets users access, correct, delete, or export personal information through self-service and support channels.",
            "Individuals can withdraw optional consent for {claim_type}, review the categories involved, and receive a dated confirmation when a request is completed.",
        ),
        ("user", "partial"): (
            "{org} accepts privacy requests related to {claim_type}, but response timelines, identity verification steps, and appeal options are described only at a high level.",
            "Users may contact support about {claim_type}; some rights are available, although deletion limits and portability formats are not fully specified.",
        ),
        ("user", "noncompliant"): (
            "Use of the service is treated as broad consent for {claim_type}, and {org} may deny access or deletion requests when operationally inconvenient.",
            "The policy references user choices for {claim_type}, but it does not provide a reliable method to withdraw consent or exercise privacy rights.",
        ),
        ("system", "compliant"): (
            "{org} protects data tied to {claim_type} with encryption, least-privilege access, monitoring, incident response review, and recurring control testing.",
            "Production systems use documented safeguards for {claim_type}, including access logging, key management, vulnerability remediation, and incident escalation.",
        ),
        ("system", "partial"): (
            "{org} uses general security controls for {claim_type}, but the policy does not define monitoring frequency, encryption scope, or incident review ownership.",
            "Security safeguards are applied to {claim_type}; however, control testing and breach notification timing are not consistently described.",
        ),
        ("system", "noncompliant"): (
            "{org} may store data for {claim_type} in operational tools without documented encryption, access review, or incident response commitments.",
            "The policy says security matters for {claim_type}, but it does not commit to specific safeguards, monitoring, or breach handling practices.",
        ),
        ("organization", "compliant"): (
            "{org} maintains governance for {claim_type} through documented retention schedules, processor contracts, audit evidence, and accountable privacy ownership.",
            "Vendor and governance controls for {claim_type} include documented purpose limits, retention review, training, and periodic compliance assessment.",
        ),
        ("organization", "partial"): (
            "{org} reviews {claim_type} during periodic governance meetings, but processor evidence, retention exceptions, and owner sign-off are not consistently documented.",
            "The policy describes accountability for {claim_type}, although vendor review cadence and audit artifacts are incomplete.",
        ),
        ("organization", "noncompliant"): (
            "{org} may share or retain data for {claim_type} as business needs change, without a defined owner, processor review, or deletion schedule.",
            "Governance for {claim_type} is handled case by case, and the policy does not require audits, documented retention limits, or processor oversight.",
        ),
    }

    template = rnd.choice(templates.get((level, compliance_status), templates[("organization", compliance_status)]))
    return f"{template.format(org=organization, claim_type=claim_type)} This claim is mapped to {control_ref}."


def _render_policy_text(
    organization: str,
    sector: str,
    region: str,
    compliance_band: str,
    claims: Sequence[SyntheticPolicyClaim],
    policy_index: int,
) -> str:
    sections: Dict[str, List[SyntheticPolicyClaim]] = defaultdict(list)
    for claim in claims:
        sections[claim.level].append(claim)

    lines = [
        f"# {organization} Privacy Policy",
        "",
        f"Effective date: 2025-{(policy_index % 12) + 1:02d}-01",
        f"Scope: This synthetic policy represents a {sector} service operating in {region} with a {compliance_band} compliance posture.",
        "",
        "## Information We Collect And Why",
        "We collect account, device, transaction, support, and usage information to provide the service, prevent abuse, and respond to requests.",
        "",
    ]

    section_titles = {
        "user": "## Individual Rights And Choices",
        "system": "## Security And Operational Safeguards",
        "organization": "## Governance, Retention, And Sharing",
    }
    for level in ("user", "system", "organization"):
        if not sections[level]:
            continue
        lines.extend([section_titles[level], ""])
        for claim in sections[level]:
            lines.append(f"- {claim.text}")
        lines.append("")

    lines.extend(
        [
            "## Contact",
            f"Privacy questions can be sent to privacy@{organization.lower().replace(' ', '')}.example.",
        ]
    )
    return "\n".join(lines)


def _index_policy_claims(
    policy_documents: Sequence[SyntheticPolicyDocument],
) -> Dict[str, Dict[Tuple[str, str], List[PolicyClaimLink]]]:
    by_metric: Dict[Tuple[str, str], List[PolicyClaimLink]] = defaultdict(list)
    by_level: Dict[Tuple[str, str], List[PolicyClaimLink]] = defaultdict(list)

    for policy in policy_documents:
        for claim in policy.claims:
            link = (policy, claim)
            by_metric[(policy.scenario, claim.metric_id)].append(link)
            by_level[(policy.scenario, claim.level)].append(link)

    return {"by_metric": by_metric, "by_level": by_level}


def _select_policy_claim_link(
    policy_claim_index: Dict[str, Dict[Tuple[str, str], List[PolicyClaimLink]]],
    scenario: str,
    spec: MetricSpec,
    rnd: Random,
) -> Optional[PolicyClaimLink]:
    metric_matches = policy_claim_index.get("by_metric", {}).get((scenario, spec.metric_id), [])
    if metric_matches:
        return rnd.choice(metric_matches)

    level_matches = policy_claim_index.get("by_level", {}).get((scenario, spec.level), [])
    if level_matches:
        return rnd.choice(level_matches)
    return None


def _claim_adjusted_failure_rate(base_failure_rate: float, link: Optional[PolicyClaimLink]) -> float:
    if link is None:
        return base_failure_rate

    _, claim = link
    if claim.compliance_status == "compliant":
        return min(base_failure_rate, 0.12) * 0.65
    if claim.compliance_status == "partial":
        return max(base_failure_rate, 0.18)
    return max(base_failure_rate, 0.45)
