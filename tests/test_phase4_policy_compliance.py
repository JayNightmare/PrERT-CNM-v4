"""Tests for the policy-only compliance assessment path (assess_policy_compliance)."""

from pathlib import Path

from prert.phase4.compliance_assessor import (
    assess_policy_compliance,
    split_policy_clauses,
    REGULATION_CONTROLS,
)


COMPREHENSIVE_POLICY = """
We provide transparent notice and request explicit consent before collecting any personal data.
Users can opt out of optional data uses at any time through the privacy settings.

Users may access, correct, and delete their data at any time by contacting our support team.
We support data portability and will provide records in a machine-readable format on request.

We use encryption and access controls to secure personal information in transit and at rest.

We retain personal data only as long as necessary for the stated purpose and delete it after
the retention period expires according to our published deletion schedule.

We disclose third-party processors and share data only under contractual obligations.
Our vendor agreements govern the purpose and duration of any third-party processing.

Data is collected for specified, legitimate purposes and is not processed beyond those purposes.

In case of a data breach, we notify the supervisory authority within 72 hours and communicate
the breach to affected users without undue delay.

Contact our data protection officer at dpo@example.com for any privacy requests or complaints.
"""


def test_assess_policy_compliance_returns_structured_output() -> None:
    result = assess_policy_compliance(
        policy_text=COMPREHENSIVE_POLICY,
        model_path=Path("does-not-exist-model.json"),
    )

    assert result["mode"] == "policy_only"
    assert 0.0 <= result["overall_score"] <= 100.0
    assert result["grade"] in {"A", "B", "C", "D", "F"}
    assert result["status"] in {
        "Compliant",
        "Mostly Compliant",
        "Needs Attention",
        "High Compliance Risk",
    }
    assert result["summary"]["clauses_analyzed"] > 0
    assert result["summary"]["claims_generated"] > 0
    assert isinstance(result["claims"], list)
    assert isinstance(result["regulation_summary"], dict)


def test_claims_have_regulation_verdicts_with_citations() -> None:
    result = assess_policy_compliance(
        policy_text=COMPREHENSIVE_POLICY,
        model_path=Path("does-not-exist-model.json"),
    )

    assert len(result["claims"]) > 0

    for claim in result["claims"]:
        assert "claim_text" in claim
        assert "check_id" in claim
        assert "check_title" in claim
        assert isinstance(claim["regulation_verdicts"], list)
        assert len(claim["regulation_verdicts"]) > 0

        for verdict in claim["regulation_verdicts"]:
            assert verdict["regulation"] in {"GDPR", "NIST", "ISO_27701"}
            assert isinstance(verdict["compliant"], bool)
            assert isinstance(verdict["reason"], str)
            assert len(verdict["reason"]) > 0
            assert isinstance(verdict["cited_clauses"], list)
            assert len(verdict["cited_clauses"]) > 0


def test_all_three_regulation_frameworks_appear() -> None:
    result = assess_policy_compliance(
        policy_text=COMPREHENSIVE_POLICY,
        model_path=Path("does-not-exist-model.json"),
    )

    regulation_summary = result["regulation_summary"]
    assert "GDPR" in regulation_summary
    assert "NIST" in regulation_summary
    assert "ISO_27701" in regulation_summary

    for reg, summary in regulation_summary.items():
        assert "pass_count" in summary
        assert "fail_count" in summary
        assert "total_controls" in summary
        assert "compliance_pct" in summary
        assert summary["total_controls"] > 0


def test_per_regulation_verdicts_are_independent() -> None:
    """Each claim's verdicts should evaluate each regulation independently."""
    result = assess_policy_compliance(
        policy_text=COMPREHENSIVE_POLICY,
        model_path=Path("does-not-exist-model.json"),
    )

    for claim in result["claims"]:
        regulations_seen = set()
        for verdict in claim["regulation_verdicts"]:
            regulations_seen.add(verdict["regulation"])

        assert len(regulations_seen) >= 2, (
            f"Claim '{claim['check_id']}' should have verdicts from multiple "
            f"regulations but only has: {regulations_seen}"
        )


def test_empty_policy_returns_zero_claims() -> None:
    result = assess_policy_compliance(
        policy_text="",
        model_path=Path("does-not-exist-model.json"),
    )

    assert result["summary"]["clauses_analyzed"] == 0
    assert result["summary"]["claims_generated"] == 0
    assert result["claims"] == []


def test_single_clause_maps_to_correct_check() -> None:
    result = assess_policy_compliance(
        policy_text="We request consent before collecting personal data and provide transparent notice to users.",
        model_path=Path("does-not-exist-model.json"),
    )

    check_ids = {claim["check_id"] for claim in result["claims"]}
    assert "consent_transparency" in check_ids


def test_cited_clauses_contain_original_text() -> None:
    """Source citations must contain the actual policy text, not synthesised text."""
    policy = "We apply encryption and multi-factor access controls to secure all personal data."

    result = assess_policy_compliance(
        policy_text=policy,
        model_path=Path("does-not-exist-model.json"),
    )

    for claim in result["claims"]:
        for verdict in claim["regulation_verdicts"]:
            for cited in verdict["cited_clauses"]:
                assert "encryption" in cited.lower() or "access control" in cited.lower()


def test_regulation_controls_cover_all_check_specs() -> None:
    """Every POLICY_CHECK_SPECS check_id must have a REGULATION_CONTROLS entry."""
    from prert.phase4.compliance_assessor import POLICY_CHECK_SPECS

    for spec in POLICY_CHECK_SPECS:
        assert spec.check_id in REGULATION_CONTROLS, (
            f"Missing REGULATION_CONTROLS entry for check_id: {spec.check_id}"
        )
        controls = REGULATION_CONTROLS[spec.check_id]
        regulations_covered = {c.regulation for c in controls}
        assert "GDPR" in regulations_covered, f"GDPR missing for {spec.check_id}"
        assert "NIST" in regulations_covered, f"NIST missing for {spec.check_id}"
        assert "ISO_27701" in regulations_covered, f"ISO_27701 missing for {spec.check_id}"


def test_grading_boundaries_consistent() -> None:
    """A comprehensive policy should score well; an empty one should score poorly."""
    good_result = assess_policy_compliance(
        policy_text=COMPREHENSIVE_POLICY,
        model_path=Path("does-not-exist-model.json"),
    )

    empty_result = assess_policy_compliance(
        policy_text="This document contains no privacy-relevant content whatsoever.",
        model_path=Path("does-not-exist-model.json"),
    )

    assert good_result["overall_score"] > empty_result["overall_score"]
