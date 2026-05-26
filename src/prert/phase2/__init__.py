"""Phase 2 metric and data preparation package.

Keep package exports lazy so metadata helpers such as `prert.phase2.opp115`
can be imported without pulling in extraction dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
	from prert.phase2.pipeline import run_phase2_pipeline

__all__ = ["run_phase2_pipeline"]


def __getattr__(name: str) -> Any:
	if name == "run_phase2_pipeline":
		from prert.phase2.pipeline import run_phase2_pipeline

		return run_phase2_pipeline
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
