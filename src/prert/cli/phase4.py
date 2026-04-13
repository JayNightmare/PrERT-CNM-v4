"""Phase 4 CLI: artifact-based validation and benchmark comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from prert.phase4 import run_phase4_validation


def main() -> None:
    args = _parse_args()

    report = run_phase4_validation(
        output_dir=args.output_dir,
        baseline_dir=args.baseline_dir,
        comparison_dirs=args.comparison_dirs,
        require_bayesian=args.require_bayesian,
        polisis_advisory=not args.disable_polisis_advisory,
        ece_threshold=args.ece_threshold,
    )

    baseline_validation = report["baseline"]["validation"]
    passed = bool(baseline_validation["passed"])

    print("Phase 4 validation complete")
    print(f"Baseline artifact: {report['baseline_artifact_dir']}")
    print(f"Baseline passed: {passed}")
    print(f"Comparisons: {len(report.get('comparisons', []))}")
    print(f"Validation report (json): {report['output_files']['json']}")
    print(f"Validation report (markdown): {report['output_files']['markdown']}")

    if args.strict and not passed:
        raise SystemExit(2)


def _parse_args() -> argparse.Namespace:
    root = Path.cwd()
    parser = argparse.ArgumentParser(description="Run Phase 4 artifact-based validation")

    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=root / "artifacts/phase-3-freeze",
        help="Artifact directory containing a Phase 3 manifest and outputs.",
    )
    parser.add_argument(
        "--comparison-dirs",
        type=Path,
        nargs="*",
        default=[],
        help="Optional additional artifact directories for side-by-side comparison.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "artifacts/phase-4",
        help="Output directory for Phase 4 validation reports.",
    )
    parser.add_argument(
        "--require-bayesian",
        action="store_true",
        help="Require Bayesian evidence artifacts for required checks.",
    )
    parser.add_argument(
        "--disable-polisis-advisory",
        action="store_true",
        help="Disable non-blocking Polisis advisory check.",
    )
    parser.add_argument(
        "--ece-threshold",
        type=float,
        default=0.20,
        help="Calibration ECE advisory threshold.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 2 when baseline required checks fail.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
