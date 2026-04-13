"""CLI for generating synthetic policy+schema compliance datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from prert.phase4.synthetic import generate_synthetic_policy_schema_dataset


def main() -> None:
    args = _parse_args()

    manifest = generate_synthetic_policy_schema_dataset(
        output_dir=args.output_dir,
        counts_by_band={
            "low": args.low_count,
            "medium": args.medium_count,
            "high": args.high_count,
        },
        seed=args.seed,
        include_model_signal=args.include_model_signal,
        model_path=args.model_path,
        export_upload_fixtures=args.export_upload_fixtures,
    )

    print("Phase 4 synthetic data generation complete")
    print(f"Output dataset: {manifest['output_files']['dataset']}")
    print(f"Output manifest: {manifest['output_files']['manifest']}")
    print(f"Output dictionary: {manifest['output_files']['dictionary']}")
    print("Band score summary:")
    for band in ("low", "medium", "high"):
        band_summary = manifest["score_summary"][band]
        print(
            " - {band}: count={count}, in-target={in_target}, min={minimum}, mean={mean}, max={maximum}".format(
                band=band,
                count=band_summary["count"],
                in_target=band_summary["in_target_band"],
                minimum=band_summary["minimum"],
                mean=band_summary["mean"],
                maximum=band_summary["maximum"],
            )
        )


def _parse_args() -> argparse.Namespace:
    root = Path.cwd()
    parser = argparse.ArgumentParser(
        description="Generate synthetic privacy-policy/schema records with varied compliance bands"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "artifacts/phase-4/synthetic-compliance",
        help="Output directory for synthetic dataset artifacts.",
    )
    parser.add_argument("--low-count", type=int, default=6, help="Number of low-compliance samples to generate.")
    parser.add_argument(
        "--medium-count",
        type=int,
        default=6,
        help="Number of medium-compliance samples to generate.",
    )
    parser.add_argument("--high-count", type=int, default=6, help="Number of high-compliance samples to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic synthetic generation.")
    parser.add_argument(
        "--include-model-signal",
        action="store_true",
        help="Enable model-signal scoring during synthetic assessment.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional model checkpoint path used only when --include-model-signal is set.",
    )
    parser.add_argument(
        "--export-upload-fixtures",
        action="store_true",
        help="Write per-sample policy/schema files for direct GUI upload tests.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
