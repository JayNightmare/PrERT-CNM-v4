"""Phase 3 CLI: baseline clause classifier training and held-out evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from prert.phase2.opp115 import INPUT_SET_TO_SUBDIR
from prert.phase3 import run_phase3_pipeline


def main() -> None:
    args = _parse_args()

    manifest = run_phase3_pipeline(
        output_dir=args.output_dir,
        opp115_root=args.opp115_root,
        input_set=args.input_set,
        source_dir=args.source_dir,
        labeled_input_path=args.labeled_input_path,
        model_type=args.model_type,
        random_state=args.random_state,
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        max_df=args.max_df,
        c=args.c,
        max_iter=args.max_iter,
        seed=args.seed,
        max_rows=args.max_rows,
    )

    metrics = manifest["metrics"]
    dataset = manifest["dataset_manifest"]

    print("Phase 3 baseline pipeline complete")
    print(f"Total rows: {dataset['total_rows']}")
    print(f"Validation macro F1: {metrics['validation_macro_f1']}")
    print(f"Test macro F1: {metrics['test_macro_f1']}")
    print(f"Validation accuracy: {metrics['validation_accuracy']}")
    print(f"Test accuracy: {metrics['test_accuracy']}")


def _parse_args() -> argparse.Namespace:
    root = Path.cwd()
    parser = argparse.ArgumentParser(description="Run Phase 3 baseline classifier pipeline")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "artifacts/phase-3",
        help="Phase 3 artifact output directory.",
    )
    parser.add_argument(
        "--opp115-root",
        type=Path,
        default=root / "data/raw/OPP-115",
        help="Root directory of OPP-115 corpus used when no labeled input is provided.",
    )
    parser.add_argument(
        "--input-set",
        type=str,
        default="consolidation-0.75",
        choices=sorted(INPUT_SET_TO_SUBDIR.keys()),
        help="Annotation set to aggregate from OPP-115.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Optional override for OPP-115 annotation source directory.",
    )
    parser.add_argument(
        "--labeled-input-path",
        type=Path,
        default=None,
        help="Optional pre-labeled JSONL dataset (text,label,policy_uid) to bypass OPP-115 parsing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split behavior.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="naive_bayes",
        choices=("naive_bayes", "logreg_tfidf"),
        help="Classifier backend for Phase 3.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for classifier initialization.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=20000,
        help="Maximum number of TF-IDF features for logreg_tfidf.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        help="Upper bound for n-gram range in TF-IDF vectorization.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Minimum document frequency for TF-IDF terms.",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=0.95,
        help="Maximum document frequency for TF-IDF terms.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for logistic regression.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for logistic regression solver.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum number of examples to ingest.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
