"""End-to-end orchestration for Phase 4 artifact-based validation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from prert.phase4.comparison import compare_validation_reports
from prert.phase4.reporting import write_phase4_validation_outputs
from prert.phase4.validation import evaluate_phase4_validation


def run_phase4_validation(
    output_dir: Path,
    baseline_dir: Path,
    comparison_dirs: Optional[Sequence[Path]] = None,
    require_bayesian: bool = False,
    polisis_advisory: bool = True,
    ece_threshold: float = 0.20,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_report = evaluate_phase4_validation(
        artifact_dir=baseline_dir,
        require_bayesian=require_bayesian,
        polisis_advisory=polisis_advisory,
        ece_threshold=ece_threshold,
    )

    candidate_dirs = [path for path in (comparison_dirs or []) if path != baseline_dir]
    comparison_reports = [
        evaluate_phase4_validation(
            artifact_dir=artifact_dir,
            require_bayesian=require_bayesian,
            polisis_advisory=polisis_advisory,
            ece_threshold=ece_threshold,
        )
        for artifact_dir in candidate_dirs
    ]

    comparison_summary = compare_validation_reports(baseline_report, comparison_reports)

    payload = {
        "phase": "phase-4",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "baseline_artifact_dir": str(baseline_dir),
        "comparison_artifact_dirs": [str(path) for path in candidate_dirs],
        "baseline": baseline_report,
        "comparisons": comparison_reports,
        "comparison_summary": comparison_summary,
    }

    output_files = write_phase4_validation_outputs(output_dir=output_dir, payload=payload)
    payload["output_files"] = output_files
    return payload
