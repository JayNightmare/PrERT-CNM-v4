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
        privacybert_model_name=args.privacybert_model_name,
        privacybert_epochs=args.privacybert_epochs,
        privacybert_batch_size=args.privacybert_batch_size,
        privacybert_learning_rate=args.privacybert_learning_rate,
        privacybert_max_length=args.privacybert_max_length,
        enable_bayesian_scoring=not args.disable_bayesian_scoring,
        bayesian_priors_path=args.bayesian_priors_path,
        bayesian_top_k=args.bayesian_top_k,
        seed=args.seed,
        max_rows=args.max_rows,
        run_id=args.run_id,
        calibration_bins=args.calibration_bins,
        bootstrap_resamples=args.bootstrap_resamples,
    )

    metrics = manifest["metrics"]
    dataset = manifest["dataset_manifest"]

    print("Phase 3 baseline pipeline complete")
    print(f"Total rows: {dataset['total_rows']}")
    print(f"Validation macro F1: {metrics['validation_macro_f1']}")
    print(f"Test macro F1: {metrics['test_macro_f1']}")
    print(f"Validation accuracy: {metrics['validation_accuracy']}")
    print(f"Test accuracy: {metrics['test_accuracy']}")
    if metrics.get("bayesian_primary_score") is not None:
        print(f"Bayesian primary score (test): {metrics['bayesian_primary_score']}")


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
        choices=("naive_bayes", "logreg_tfidf", "privacybert"),
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
        "--privacybert-model-name",
        type=str,
        default="bert-base-uncased",
        help="Transformers model name or path used for privacybert backend.",
    )
    parser.add_argument(
        "--privacybert-epochs",
        type=float,
        default=2.0,
        help="Training epochs for privacybert backend.",
    )
    parser.add_argument(
        "--privacybert-batch-size",
        type=int,
        default=8,
        help="Per-device batch size for privacybert backend.",
    )
    parser.add_argument(
        "--privacybert-learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate for privacybert backend.",
    )
    parser.add_argument(
        "--privacybert-max-length",
        type=int,
        default=256,
        help="Maximum token length for privacybert backend.",
    )
    parser.add_argument(
        "--disable-bayesian-scoring",
        action="store_true",
        help="Disable Bayesian posterior risk scoring outputs.",
    )
    parser.add_argument(
        "--bayesian-priors-path",
        type=Path,
        default=None,
        help="Optional JSON file with Bayesian alpha/beta priors by level.",
    )
    parser.add_argument(
        "--bayesian-top-k",
        type=int,
        default=5,
        help="Top contributing clauses retained per level in Bayesian outputs.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum number of examples to ingest.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional explicit run identifier recorded in run history artifacts.",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of bins for reliability and ECE calibration analytics.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for confidence interval estimation.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
