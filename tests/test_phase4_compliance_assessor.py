from pathlib import Path

from prert.phase4.compliance_assessor import (
    assess_policy_schema_compliance,
    classify_schema_fields,
    extract_schema_fields,
    split_policy_clauses,
)


def test_split_policy_clauses_filters_short_and_duplicates() -> None:
    text = """
    Privacy Policy Notice.

    We obtain consent before collecting personal information and users can request access or deletion.

    We obtain consent before collecting personal information and users can request access or deletion.

    We apply encryption and access controls to protect data.
    """
    clauses = split_policy_clauses(text)
    assert len(clauses) == 2
    assert "consent" in clauses[0].lower()


def test_split_policy_clauses_handles_single_space_sentences() -> None:
    text = (
        "We request consent before enabling optional processing. "
        "Users can request access and deletion of account data. "
        "We protect personal data with encryption and access controls."
    )
    clauses = split_policy_clauses(text)

    assert len(clauses) == 3


def test_extract_schema_fields_from_sql_and_json() -> None:
    sql_schema = """
    CREATE TABLE customers (
      id BIGINT,
      email VARCHAR(255),
      phone VARCHAR(25),
      created_at TIMESTAMP
    );
    """
    json_schema = '{"users": {"first_name": "string", "last_name": "string", "ssn": "string"}}'

    sql_fields = extract_schema_fields(sql_schema)
    json_fields = extract_schema_fields(json_schema)

    assert "email" in sql_fields
    assert "phone" in sql_fields
    assert "first_name" in json_fields
    assert "ssn" in json_fields


def test_classify_schema_fields_detects_pii_and_sensitive() -> None:
    fields = ["email", "phone", "health_status", "salary_amount", "created_at"]
    pii, sensitive = classify_schema_fields(fields)

    assert "email" in pii
    assert "phone" in pii
    assert "health_status" in sensitive
    assert "salary_amount" in sensitive


def test_assess_policy_schema_compliance_returns_structured_score() -> None:
    policy = """
    We provide transparent notice and request consent before data collection.
    Users may access, correct, and delete their data at any time.
    We use encryption and access controls to secure personal information.
    We disclose third-party processors and retention periods.
    In case of a data breach, we notify affected users.
    Contact our data protection officer for privacy requests.
    """

    schema = """
    CREATE TABLE customer_profiles (
      customer_id BIGINT,
      full_name VARCHAR(255),
      email VARCHAR(255),
      phone VARCHAR(25),
      health_notes TEXT,
      created_at TIMESTAMP
    );
    """

    result = assess_policy_schema_compliance(
        policy_text=policy,
        schema_text=schema,
        model_path=Path("does-not-exist-model.json"),
    )

    assert 0.0 <= result["overall_score"] <= 100.0
    assert result["grade"] in {"A", "B", "C", "D", "F"}
    assert "policy_checks" in result
    assert "schema_analysis" in result
    assert result["summary"]["clauses_analyzed"] > 0
    assert result["summary"]["schema_fields_detected"] > 0


def test_assessor_preserves_clause_boundaries_for_evidence() -> None:
    policy = """
    We provide a privacy notice and request consent before enabling optional data uses.

    Users can access account records and request deletion through support channels.

    Data is protected with encryption and access control safeguards in production systems.

    Limited third-party vendors process billing data under contractual obligations.
    """

    schema = """
    CREATE TABLE billing_profiles (
      profile_id BIGINT PRIMARY KEY,
      email VARCHAR(255),
      customer_name VARCHAR(200),
      billing_city VARCHAR(100),
      card_last4 VARCHAR(4),
      account_reference VARCHAR(40),
      last_login_ip VARCHAR(64),
      created_at TIMESTAMP
    );
    """

    result = assess_policy_schema_compliance(
        policy_text=policy,
        schema_text=schema,
        model_path=Path("does-not-exist-model.json"),
    )

    assert result["summary"]["clauses_analyzed"] == 4

    consent_check = next(
        item
        for item in result["policy_checks"]
        if item["check_id"] == "consent_transparency"
    )
    assert len(consent_check["evidence"]) == 1
    assert "optional data uses" in consent_check["evidence"][0].lower()
    assert "third-party vendors process billing data" not in consent_check["evidence"][0].lower()
