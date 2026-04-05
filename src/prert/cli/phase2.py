"""Phase 2 CLI: metric specs, synthetic data, public mapping, and baseline scoring."""

from __future__ import annotations

import argparse
from pathlib import Path

from prert.phase2 import run_phase2_pipeline


def main() -> None:
    args = _parse_args()

    manifest = run_phase2_pipeline(
        controls_path=args.controls_path,
        output_dir=args.output_dir,
        public_input_path=args.public_input,
        seed=args.seed,
    )

    coverage = manifest["coverage_summary"]
    outputs = manifest["output_counts"]

    print("Phase 2 pipeline complete")
    print(f"Mapped controls: {coverage['mapped_controls']} / {coverage['total_controls']}")
    print(f"Metric specs: {outputs['metric_specs']}")
    print(f"Synthetic events: {outputs['synthetic_events']}")
    print(f"Public mapped rows: {outputs['public_data_mapped']}")
    print(f"Baseline score rows: {outputs['baseline_score_rows_total']}")


def _parse_args() -> argparse.Namespace:
    root = Path.cwd()

    parser = argparse.ArgumentParser(description="Run Phase 2 metrics and data-preparation pipeline")
    parser.add_argument(
        "--controls-path",
        type=Path,
        default=root / "artifacts/phase-1/controls_all.jsonl",
        help="Path to Phase 1 control catalog JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "artifacts/phase-2",
        help="Phase 2 artifact output directory.",
    )
    parser.add_argument(
        "--public-input",
        type=Path,
        default=None,
        help="Optional public breach dataset (.csv or .jsonl).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible synthetic generation.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
