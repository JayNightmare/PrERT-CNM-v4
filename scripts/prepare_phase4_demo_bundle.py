"""Prepare a deployment-safe Phase 4 demo artifact bundle.

This script copies the minimum files needed to run:
- Compliance Assessment model-signal scoring (Naive Bayes model.json)
- Benchmark Validation baseline checks and optional comparison checks
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Dict, List


BASELINE_REQUIRED_FILES: tuple[str, ...] = (
    "phase3_manifest.json",
    "dataset_manifest.json",
    "classifier_metrics.json",
    "validation_predictions.jsonl",
    "test_predictions.jsonl",
)

BASELINE_OPTIONAL_FILES: tuple[str, ...] = (
    "bayesian_risk_test.json",
    "classifier_metrics.jsonl",
)


def main() -> None:
    args = _parse_args()

    output_root = args.output_root.resolve()
    baseline_target = output_root / "phase-3-freeze"
    nb_target = output_root / "phase-3-nb"

    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Output directory is not empty: {output_root}. Use --overwrite to replace files."
        )

    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)

    copied: Dict[str, List[str]] = {
        "baseline_required": [],
        "baseline_optional": [],
        "nb_required": [],
        "nb_optional": [],
        "model_checkpoint": [],
    }

    _copy_phase3_subset(
        source_dir=args.source_freeze_dir.resolve(),
        target_dir=baseline_target,
        copied=copied,
        required_bucket="baseline_required",
        optional_bucket="baseline_optional",
    )

    if args.include_nb_artifacts:
        _copy_phase3_subset(
            source_dir=args.source_nb_dir.resolve(),
            target_dir=nb_target,
            copied=copied,
            required_bucket="nb_required",
            optional_bucket="nb_optional",
        )

    _copy_nb_model_checkpoint(
        source_nb_dir=args.source_nb_dir.resolve(),
        target_root=output_root,
        copied=copied,
    )

    bundle_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_freeze_dir": str(args.source_freeze_dir.resolve()),
        "source_nb_dir": str(args.source_nb_dir.resolve()),
        "output_root": str(output_root),
        "include_nb_artifacts": bool(args.include_nb_artifacts),
        "files": copied,
    }

    manifest_path = output_root / "phase4_demo_bundle_manifest.json"
    manifest_path.write_text(json.dumps(bundle_manifest, indent=2) + "\n", encoding="utf-8")

    print("Phase 4 demo bundle ready")
    print(f"Output root: {output_root}")
    print(f"Manifest: {manifest_path}")


def _parse_args() -> argparse.Namespace:
    root = Path.cwd()

    parser = argparse.ArgumentParser(description="Prepare deployable Phase 4 demo assets")
    parser.add_argument(
        "--source-freeze-dir",
        type=Path,
        default=root / "artifacts/phase-3-freeze",
        help="Source baseline artifact directory used by benchmark validation.",
    )
    parser.add_argument(
        "--source-nb-dir",
        type=Path,
        default=root / "artifacts/phase-3-nb",
        help="Source Naive Bayes artifact directory containing model.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "deployment/demo-assets",
        help="Output directory for bundled deployment assets.",
    )
    parser.add_argument(
        "--include-nb-artifacts",
        action="store_true",
        help="Also copy phase-3-nb benchmark artifacts for leaderboard comparisons.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output root before writing.",
    )

    return parser.parse_args()


def _copy_phase3_subset(
    source_dir: Path,
    target_dir: Path,
    copied: Dict[str, List[str]],
    required_bucket: str,
    optional_bucket: str,
) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)

    for name in BASELINE_REQUIRED_FILES:
        src = source_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Required file is missing: {src}")
        dst = target_dir / name
        shutil.copy2(src, dst)
        copied[required_bucket].append(str(dst))

    for name in BASELINE_OPTIONAL_FILES:
        src = source_dir / name
        if not src.exists():
            continue
        dst = target_dir / name
        shutil.copy2(src, dst)
        copied[optional_bucket].append(str(dst))


def _copy_nb_model_checkpoint(source_nb_dir: Path, target_root: Path, copied: Dict[str, List[str]]) -> None:
    source_model = source_nb_dir / "classifier_checkpoint/model.json"
    if not source_model.exists():
        raise FileNotFoundError(f"Naive Bayes model checkpoint not found: {source_model}")

    target_model = target_root / "phase-3-nb/classifier_checkpoint/model.json"
    target_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_model, target_model)
    copied["model_checkpoint"].append(str(target_model))


if __name__ == "__main__":
    main()
