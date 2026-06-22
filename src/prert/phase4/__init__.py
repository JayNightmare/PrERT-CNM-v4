"""Phase 4 artifact-based validation package.

Keep package exports lazy so artifact validation can import the minimal surface
it needs without triggering GUI or modelling helpers unnecessarily.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prert.phase4.compliance_assessor import assess_policy_compliance
    from prert.phase4.compliance_assessor import assess_policy_schema_compliance
    from prert.phase4.pipeline import run_phase4_validation
    from prert.phase4.synthetic import generate_synthetic_policy_schema_dataset
    from prert.phase4.validation import evaluate_phase4_validation

__all__ = [
    "assess_policy_compliance",
    "assess_policy_schema_compliance",
    "generate_synthetic_policy_schema_dataset",
    "run_phase4_validation",
    "evaluate_phase4_validation",
]


def __getattr__(name: str) -> Any:
    if name == "assess_policy_compliance":
        from prert.phase4.compliance_assessor import assess_policy_compliance

        return assess_policy_compliance
    if name == "assess_policy_schema_compliance":
        from prert.phase4.compliance_assessor import assess_policy_schema_compliance

        return assess_policy_schema_compliance
    if name == "generate_synthetic_policy_schema_dataset":
        from prert.phase4.synthetic import generate_synthetic_policy_schema_dataset

        return generate_synthetic_policy_schema_dataset
    if name == "run_phase4_validation":
        from prert.phase4.pipeline import run_phase4_validation

        return run_phase4_validation
    if name == "evaluate_phase4_validation":
        from prert.phase4.validation import evaluate_phase4_validation

        return evaluate_phase4_validation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
