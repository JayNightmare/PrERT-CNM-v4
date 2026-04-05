"""CLI for preprocessing OPP-115 into a Phase 2 public input dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from prert.phase2.opp115 import INPUT_SET_TO_SUBDIR, run_opp115_processing


def main() -> None:
    args = _parse_args()

    summary = run_opp115_processing(
        opp115_root=args.opp115_root,
        output_csv=args.output_csv,
        output_jsonl=args.output_jsonl,
        input_set=args.input_set,
        source_dir=args.source_dir,
        country=args.country,
    )

    print("OPP-115 preprocessing complete")
    print(f"Rows written: {summary['rows']}")
    print(f"Input set: {summary['input_set']}")
    print(f"Source directory: {summary['source_dir']}")
    print(f"CSV output: {summary['output_csv']}")
    print(f"JSONL output: {summary['output_jsonl']}")


def _parse_args() -> argparse.Namespace:
    root = Path.cwd()
    parser = argparse.ArgumentParser(description="Preprocess OPP-115 into Phase 2 public mapping input")
    parser.add_argument(
        "--opp115-root",
        type=Path,
        default=root / "data/raw/OPP-115",
        help="Root directory of the OPP-115 corpus.",
    )
    parser.add_argument(
        "--input-set",
        type=str,
        default="consolidation-0.75",
        choices=sorted(INPUT_SET_TO_SUBDIR.keys()),
        help="Annotation set to aggregate.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Optional override for annotation source directory.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=root / "data/processed/opp115_public_mapping.csv",
        help="Output CSV path for flat public mapping rows.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=root / "data/processed/opp115_public_mapping.jsonl",
        help="Output JSONL path for flat public mapping rows.",
    )
    parser.add_argument(
        "--country",
        type=str,
        default="US",
        help="Country value used in mapped output rows.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()