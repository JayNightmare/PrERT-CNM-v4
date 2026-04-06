"""Phase 3 acceptance-freeze CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from prert.phase2.opp115 import INPUT_SET_TO_SUBDIR
from prert.phase3 import run_phase3_pipeline
from prert.phase3.acceptance import evaluate_phase3_acceptance, write_phase3_acceptance_report


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
    )

    report = evaluate_phase3_acceptance(
        output_dir=args.output_dir,
        manifest=manifest,
        require_privacybert=not args.allow_non_privacybert,
        require_bayesian=not args.disable_bayesian_scoring,
    )
    report_paths = write_phase3_acceptance_report(args.output_dir, report)

    passed = bool(report["acceptance"]["passed"])
    print("Phase 3 acceptance freeze complete")
    print(f"Output directory: {args.output_dir}")
    print(f"Acceptance passed: {passed}")
    print(f"Acceptance report (json): {report_paths['json']}")
    print(f"Acceptance report (markdown): {report_paths['markdown']}")

    if args.strict and not passed:
        raise SystemExit(2)


def _parse_args() -> argparse.Namespace:
    root = Path.cwd()
    parser = argparse.ArgumentParser(description="Run Phase 3 acceptance freeze workflow")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "artifacts/phase-3-freeze",
        help="Artifact output directory for freeze run and acceptance report.",
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split behavior.")
    parser.add_argument(
        "--model-type",
        type=str,
        default="privacybert",
        choices=("naive_bayes", "logreg_tfidf", "privacybert"),
        help="Classifier backend for freeze run. Default is privacybert for proposal compliance.",
    )
    parser.add_argument(
        "--allow-non-privacybert",
        action="store_true",
        help="Allow acceptance pass even when model-type is not privacybert.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random state for classifier initialization.")
    parser.add_argument("--max-features", type=int, default=20000, help="Maximum number of TF-IDF features.")
    parser.add_argument("--ngram-max", type=int, default=2, help="Upper bound for n-gram range.")
    parser.add_argument("--min-df", type=int, default=2, help="Minimum document frequency for TF-IDF terms.")
    parser.add_argument("--max-df", type=float, default=0.95, help="Maximum document frequency for TF-IDF terms.")
    parser.add_argument("--c", type=float, default=1.0, help="Inverse regularization strength for logistic regression.")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum iterations for logistic regression solver.")

    parser.add_argument(
        "--privacybert-model-name",
        type=str,
        default="bert-base-uncased",
        help="Transformers model name or path used for privacybert backend.",
    )
    parser.add_argument("--privacybert-epochs", type=float, default=2.0, help="Training epochs for privacybert backend.")
    parser.add_argument("--privacybert-batch-size", type=int, default=8, help="Per-device batch size for privacybert backend.")
    parser.add_argument("--privacybert-learning-rate", type=float, default=5e-5, help="Learning rate for privacybert backend.")
    parser.add_argument("--privacybert-max-length", type=int, default=256, help="Maximum token length for privacybert backend.")

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
    parser.add_argument("--max-rows", type=int, default=None, help="Optional maximum number of examples to ingest.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 2 if acceptance checks do not pass.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
