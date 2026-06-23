"""CLI for normalising APP-350 into a conservative Phase 3 auxiliary dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from prert.phase3.app350 import run_app350_processing


def main() -> None:
    args = _parse_args()

    summary = run_app350_processing(
        input_path=args.input_path,
        output_jsonl=args.output_jsonl,
        output_manifest=args.output_manifest,
        include_synthetic=args.include_synthetic,
    )

    print("APP-350 preprocessing complete")
    print(f"Policies seen: {summary['policies_seen']}")
    print(f"Policies used: {summary['policies_used']}")
    print(f"Synthetic policies skipped: {summary['synthetic_policies_skipped']}")
    print(f"Rows written: {summary['rows_written']}")
    print(f"JSONL output: {summary['output_jsonl']}")
    print(f"Manifest output: {args.output_manifest}")


def _parse_args() -> argparse.Namespace:
    root = Path.cwd()
    parser = argparse.ArgumentParser(description="Normalise APP-350 into a conservative Phase 3 auxiliary JSONL dataset")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=root / "data/raw/APP-350_v1.1.zip",
        help="APP-350 zip archive or extracted directory.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=root / "data/processed/app350_phase3_auxiliary.jsonl",
        help="Output JSONL path for auxiliary labelled rows.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=root / "data/processed/app350_phase3_auxiliary_manifest.json",
        help="Output JSON manifest path with provenance and drop statistics.",
    )
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Include APP-350 policies marked as containing synthetic sentences.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()