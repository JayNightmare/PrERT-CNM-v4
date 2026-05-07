"""End-to-end orchestration for Phase 4 artifact-based validation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from prert.phase4.comparison import compare_validation_reports
from prert.phase4.reporting import write_phase4_validation_outputs
from prert.phase4.validation import evaluate_phase4_validation


def run_phase4_validation(
    output_dir: Path,
    baseline_dir: Path,
    comparison_dirs: Optional[Sequence[Path]] = None,
    require_bayesian: bool = False,
    require_polisis: bool = False,
    polisis_advisory: bool = True,
    ece_threshold: float = 0.20,
    status_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    _emit_status(
        status_callback,
        {
            "event": "start",
            "output_dir": str(output_dir),
        },
    )

    _emit_status(
        status_callback,
        {
            "event": "baseline_start",
            "artifact_dir": str(baseline_dir),
        },
    )

    baseline_report = evaluate_phase4_validation(
        artifact_dir=baseline_dir,
        require_bayesian=require_bayesian,
        require_polisis=require_polisis,
        polisis_advisory=polisis_advisory,
        ece_threshold=ece_threshold,
    )

    _emit_status(
        status_callback,
        {
            "event": "baseline_complete",
            "artifact_dir": str(baseline_dir),
        },
    )

    candidate_dirs = [path for path in (comparison_dirs or []) if path != baseline_dir]
    comparison_reports = []
    for index, artifact_dir in enumerate(candidate_dirs, start=1):
        _emit_status(
            status_callback,
            {
                "event": "comparison_start",
                "artifact_dir": str(artifact_dir),
                "index": index,
                "total": len(candidate_dirs),
            },
        )
        comparison_reports.append(
            evaluate_phase4_validation(
                artifact_dir=artifact_dir,
                require_bayesian=require_bayesian,
                require_polisis=require_polisis,
                polisis_advisory=polisis_advisory,
                ece_threshold=ece_threshold,
            )
        )
        _emit_status(
            status_callback,
            {
                "event": "comparison_complete",
                "artifact_dir": str(artifact_dir),
                "index": index,
                "total": len(candidate_dirs),
            },
        )

    _emit_status(status_callback, {"event": "summary_start"})
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

    _emit_status(
        status_callback,
        {
            "event": "complete",
            "output_files": output_files,
        },
    )

    return payload


def _emit_status(
    status_callback: Optional[Callable[[Dict[str, Any]], None]],
    event: Dict[str, Any],
) -> None:
    if status_callback is None:
        return

    try:
        status_callback(event)
    except Exception:
        # Status updates are informational and should never fail pipeline execution.
        return
