import json
from pathlib import Path

from prert.phase2.pipeline import run_phase2_pipeline


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_controls(path: Path) -> None:
    controls = [
        {
            "normalized_id": "gdpr::Article_7",
            "regulation": "GDPR",
            "native_id": "Article 7",
            "title": "Conditions for consent",
            "text": "The data subject shall have the right to withdraw consent.",
            "parser_confidence": 0.95,
        },
        {
            "normalized_id": "iso::8.2",
            "regulation": "ISO27001",
            "native_id": "8.2",
            "title": "Information security risk assessment",
            "text": "The organization shall define and apply risk assessment process.",
            "parser_confidence": 0.9,
        },
        {
            "normalized_id": "nist::PR.AC-P1",
            "regulation": "NISTPF",
            "native_id": "PR.AC-P1",
            "title": "Identities and credentials",
            "text": "Access control and encryption are implemented.",
            "parser_confidence": 0.85,
        },
    ]

    with path.open("w", encoding="utf-8") as handle:
        for row in controls:
            handle.write(json.dumps(row) + "\n")


def test_phase2_pipeline_writes_isolated_outputs(tmp_path: Path) -> None:
    controls_path = tmp_path / "controls_all.jsonl"
    output_dir = tmp_path / "phase-2"
    _write_controls(controls_path)

    manifest = run_phase2_pipeline(
        controls_path=controls_path,
        output_dir=output_dir,
        public_input_path=None,
        seed=7,
    )

    assert manifest["coverage_summary"]["mapped_controls"] == 3
    assert manifest["coverage_summary"]["missing_controls"] == []
    assert manifest["output_counts"]["synthetic_policies"] == 18

    assert (output_dir / "metric_specs.jsonl").exists()
    assert (output_dir / "synthetic_policies.jsonl").exists()
    assert (output_dir / "synthetic_events.jsonl").exists()
    assert (output_dir / "public_data_mapped.jsonl").exists()
    assert (output_dir / "baseline_scores.jsonl").exists()
    assert (output_dir / "phase2_manifest.json").exists()


def test_phase2_scores_stay_in_range(tmp_path: Path) -> None:
    controls_path = tmp_path / "controls_all.jsonl"
    output_dir = tmp_path / "phase-2"
    _write_controls(controls_path)

    run_phase2_pipeline(
        controls_path=controls_path,
        output_dir=output_dir,
        public_input_path=None,
        seed=11,
    )

    with (output_dir / "baseline_scores.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if "risk_score" in row:
                assert 0.0 <= float(row["risk_score"]) <= 1.0
            if "compliance_score" in row:
                assert 0.0 <= float(row["compliance_score"]) <= 1.0


def test_phase2_synthetic_policies_have_multiple_varied_claims(tmp_path: Path) -> None:
    controls_path = tmp_path / "controls_all.jsonl"
    output_dir = tmp_path / "phase-2"
    _write_controls(controls_path)

    run_phase2_pipeline(
        controls_path=controls_path,
        output_dir=output_dir,
        public_input_path=None,
        seed=13,
    )

    policies = _read_jsonl(output_dir / "synthetic_policies.jsonl")
    events = _read_jsonl(output_dir / "synthetic_events.jsonl")

    assert policies
    assert {policy["compliance_band"] for policy in policies} == {"high", "medium", "low"}

    claim_statuses = set()
    for policy in policies:
        claims = policy["claims"]
        assert len(claims) >= 5
        assert len(policy["policy_text"].splitlines()) > len(claims)
        claim_statuses.update(claim["compliance_status"] for claim in claims)

    assert {"compliant", "partial", "noncompliant"}.issubset(claim_statuses)
    assert all(row["metadata"].get("policy_id") for row in events)
    assert all(row["metadata"].get("policy_claim_id") for row in events)
