"""Phase 3 baseline classifier package.

Keep package exports lazy so lightweight submodules such as
`prert.phase3.classifier` do not pull in the full pipeline surface on import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
	from prert.phase3.acceptance import evaluate_phase3_acceptance, write_phase3_acceptance_report
	from prert.phase3.pipeline import run_phase3_pipeline

__all__ = [
	"run_phase3_pipeline",
	"evaluate_phase3_acceptance",
	"write_phase3_acceptance_report",
]


def __getattr__(name: str) -> Any:
	if name == "run_phase3_pipeline":
		from prert.phase3.pipeline import run_phase3_pipeline

		return run_phase3_pipeline
	if name == "evaluate_phase3_acceptance":
		from prert.phase3.acceptance import evaluate_phase3_acceptance

		return evaluate_phase3_acceptance
	if name == "write_phase3_acceptance_report":
		from prert.phase3.acceptance import write_phase3_acceptance_report

		return write_phase3_acceptance_report
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
