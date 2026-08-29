"""CNM CLIs.

- `prert-cnm-build`: build a memory index from a controls JSON/JSONL file.
- `prert-cnm-phase3`: run Phase 3 pipeline with CNM enabled (single k).
- `prert-cnm-ablation`: run Phase 3 pipeline across a sweep of top-k values.

Usage examples:

    # 1. Build the index (one-time)
    python -m prert.phase3.cnm.cli build \\
        --controls-path artifacts/phase-1/controls.jsonl \\
        --output-dir artifacts/cnm-index

    # 2. Run Phase 3 with CNM (k=5)
    python -m prert.phase3.cnm.cli phase3 \\
        --cnm-index artifacts/cnm-index \\
        --top-k 5 \\
        --output-dir artifacts/phase-3-cnm-k5

    # 3. Full ablation sweep
    python -m prert.phase3.cnm.cli ablation \\
        --cnm-index artifacts/cnm-index \\
        --k-values 0,1,3,5,10 \\
        --output-dir artifacts/phase-3-cnm-ablation
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from prert.config import load_dotenv_if_available
from prert.phase2.opp115 import INPUT_SET_TO_SUBDIR
from prert.phase3.classifier import DEFAULT_PRIVACYBERT_MODEL_NAME
from prert.phase3.cnm.classifier import CNMConfig
from prert.phase3.cnm.embed import DEFAULT_EMBEDDING_MODEL
from prert.phase3.cnm.memory import build_memory_index
from prert.phase3.cnm.pipeline import run_cnm_ablation, run_phase3_cnm_pipeline
from prert.phase3.dataset import POLISIS_INPUT_SET_TO_SUBDIR

_LOGGER = logging.getLogger("prert.phase3.cnm.cli")


def _add_phase3_args(parser: argparse.ArgumentParser) -> None:
    root = Path.cwd()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--opp115-root", type=Path, default=root / "data/raw/OPP-115")
    parser.add_argument(
        "--input-set",
        type=str,
        default="consolidation-0.75",
        choices=sorted(INPUT_SET_TO_SUBDIR.keys()),
    )
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--polisis-root", type=Path, default=None)
    parser.add_argument(
        "--polisis-input-set",
        type=str,
        default="normalized",
        choices=sorted(POLISIS_INPUT_SET_TO_SUBDIR.keys()),
    )
    parser.add_argument("--polisis-source-dir", type=Path, default=None)
    parser.add_argument("--labeled-input-path", type=Path, default=None)
    parser.add_argument("--auxiliary-labeled-input-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--privacybert-model-name", type=str, default=DEFAULT_PRIVACYBERT_MODEL_NAME
    )
    parser.add_argument("--privacybert-epochs", type=float, default=2.0)
    parser.add_argument("--privacybert-batch-size", type=int, default=8)
    parser.add_argument("--privacybert-learning-rate", type=float, default=5e-5)
    parser.add_argument("--privacybert-max-length", type=int, default=256)
    parser.add_argument(
        "--privacybert-loss-type",
        type=str,
        default="focal",
        choices=("ce", "weighted_ce", "focal"),
    )
    parser.add_argument("--privacybert-focal-gamma", type=float, default=2.0)
    parser.add_argument("--privacybert-label-smoothing", type=float, default=0.05)
    parser.add_argument("--privacybert-weight-decay", type=float, default=0.01)
    parser.add_argument("--privacybert-warmup-steps", type=int, default=0)
    parser.add_argument("--privacybert-early-stopping-patience", type=int, default=1)
    parser.add_argument("--disable-bayesian-scoring", action="store_true")
    parser.add_argument("--bayesian-priors-path", type=Path, default=None)
    parser.add_argument("--bayesian-top-k", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)


def _phase3_kwargs(args: argparse.Namespace) -> dict:
    return {
        "opp115_root": args.opp115_root,
        "input_set": args.input_set,
        "source_dir": args.source_dir,
        "polisis_root": args.polisis_root,
        "polisis_input_set": args.polisis_input_set,
        "polisis_source_dir": args.polisis_source_dir,
        "labeled_input_path": args.labeled_input_path,
        "auxiliary_labeled_input_path": args.auxiliary_labeled_input_path,
        "random_state": args.random_state,
        "privacybert_model_name": args.privacybert_model_name,
        "privacybert_epochs": args.privacybert_epochs,
        "privacybert_batch_size": args.privacybert_batch_size,
        "privacybert_learning_rate": args.privacybert_learning_rate,
        "privacybert_max_length": args.privacybert_max_length,
        "privacybert_loss_type": args.privacybert_loss_type,
        "privacybert_focal_gamma": args.privacybert_focal_gamma,
        "privacybert_label_smoothing": args.privacybert_label_smoothing,
        "privacybert_weight_decay": args.privacybert_weight_decay,
        "privacybert_warmup_steps": args.privacybert_warmup_steps,
        "privacybert_early_stopping_patience": args.privacybert_early_stopping_patience,
        "enable_bayesian_scoring": not args.disable_bayesian_scoring,
        "bayesian_priors_path": args.bayesian_priors_path,
        "bayesian_top_k": args.bayesian_top_k,
        "seed": args.seed,
        "max_rows": args.max_rows,
        "run_id": args.run_id,
        "calibration_bins": args.calibration_bins,
        "bootstrap_resamples": args.bootstrap_resamples,
    }


def _cmd_build(args: argparse.Namespace) -> int:
    index = build_memory_index(
        controls_path=args.controls_path,
        output_dir=args.output_dir,
        encoder_name=args.encoder,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(
        f"Built CNM memory index: n={len(index)}, dim={index.dim}, encoder={index.encoder_name}"
    )
    print(f"Saved to: {args.output_dir}")
    return 0


def _cmd_phase3(args: argparse.Namespace) -> int:
    cnm_config = CNMConfig(
        index_dir=args.cnm_index,
        top_k=args.top_k,
        max_control_chars=args.max_control_chars,
        include_source_tag=not args.no_source_tag,
        augment_training=not args.no_train_augment,
        trace_output_path=args.trace_output,
    )
    print(
        f"[CNM] Running with top_k={cnm_config.top_k}, augment_training={cnm_config.augment_training}, "
        f"index={cnm_config.index_dir}. If you do NOT see a log line 'Monkey-patched train_classifier' "
        f"followed by 'Training CNMPrivacyBertClassifier' below, the CNM shim did not fire and the run "
        f"will silently be an ordinary PrivBERT baseline.",
        flush=True,
    )
    manifest = run_phase3_cnm_pipeline(
        output_dir=args.output_dir,
        cnm_config=cnm_config,
        **_phase3_kwargs(args),
    )
    metrics = manifest["metrics"]
    model_type = metrics.get("model_type", "unknown")
    print("Phase 3 CNM pipeline complete")
    print(f"top_k={args.top_k}  index={args.cnm_index}  model_type={model_type}")
    if model_type != "cnm_privacybert":
        print(
            "[CNM][WARNING] metrics.model_type is not 'cnm_privacybert'. The CNM wrapper "
            "did NOT train. Rerun after checking the shim; results below are a plain-PrivBERT baseline."
        )
    print(f"Validation macro F1: {metrics['validation_macro_f1']}")
    print(f"Test macro F1: {metrics['test_macro_f1']}")
    print(f"Test accuracy: {metrics['test_accuracy']}")
    if metrics.get("bayesian_primary_score") is not None:
        print(f"Bayesian primary score (test): {metrics['bayesian_primary_score']}")
    return 0


def _cmd_ablation(args: argparse.Namespace) -> int:
    k_values = [int(x) for x in args.k_values.split(",")]
    print(
        f"[CNM] Ablation over k={k_values}. Watch for 'Monkey-patched train_classifier' + "
        f"'Training CNMPrivacyBertClassifier' at the start of each k>0 run. Any k>0 run whose "
        f"model_type comes back as 'privacybert' (not 'cnm_privacybert') is a silent shim failure.",
        flush=True,
    )
    results = run_cnm_ablation(
        output_root=args.output_dir,
        cnm_index_dir=args.cnm_index,
        k_values=k_values,
        **_phase3_kwargs(args),
    )
    # Sanity check: every k>0 run must have model_type=cnm_privacybert
    bad = [
        k
        for k, m in results.items()
        if k > 0 and m["metrics"].get("model_type") != "cnm_privacybert"
    ]
    if bad:
        print(
            f"[CNM][ERROR] Runs with k={bad} produced model_type='privacybert' "
            f"instead of 'cnm_privacybert'. The CNM shim did NOT fire for those runs. "
            f"Do NOT report these numbers as CNM results."
        )
    print("=== CNM ablation summary ===")
    print(
        f"{'k':>4}  {'val_macroF1':>12}  {'test_macroF1':>13}  {'test_acc':>9}  {'primary':>9}"
    )
    for k, manifest in results.items():
        m = manifest["metrics"]
        prim = m.get("bayesian_primary_score", "n/a")
        print(
            f"{k:>4}  {m['validation_macro_f1']:>12.4f}  "
            f"{m['test_macro_f1']:>13.4f}  {m['test_accuracy']:>9.4f}  {str(prim):>9}"
        )
    # Write a summary JSON for downstream paper-table generation.
    import json

    summary_path = args.output_dir / "ablation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "k_values": k_values,
                "results": {
                    str(k): {
                        "validation_macro_f1": manifest["metrics"][
                            "validation_macro_f1"
                        ],
                        "test_macro_f1": manifest["metrics"]["test_macro_f1"],
                        "test_accuracy": manifest["metrics"]["test_accuracy"],
                        "bayesian_primary_score": manifest["metrics"].get(
                            "bayesian_primary_score"
                        ),
                    }
                    for k, manifest in results.items()
                },
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    print(f"Wrote ablation summary to {summary_path}")
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    load_dotenv_if_available(None)

    parser = argparse.ArgumentParser(
        description="CNM (Contextual Neural Memory) commands."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser(
        "build", help="Build a CNM memory index from a controls file."
    )
    p_build.add_argument("--controls-path", type=Path, required=True)
    p_build.add_argument("--output-dir", type=Path, required=True)
    p_build.add_argument("--encoder", type=str, default=DEFAULT_EMBEDDING_MODEL)
    p_build.add_argument("--batch-size", type=int, default=64)
    p_build.add_argument("--device", type=str, default=None)
    p_build.set_defaults(func=_cmd_build)

    p_phase3 = sub.add_parser("phase3", help="Run Phase 3 pipeline with CNM enabled.")
    p_phase3.add_argument("--cnm-index", type=Path, required=True)
    p_phase3.add_argument("--top-k", type=int, default=5)
    p_phase3.add_argument("--max-control-chars", type=int, default=300)
    p_phase3.add_argument("--no-source-tag", action="store_true")
    p_phase3.add_argument(
        "--no-train-augment",
        action="store_true",
        help="Retrieval only at inference (retrieval-at-test-time only).",
    )
    p_phase3.add_argument(
        "--trace-output",
        type=Path,
        default=None,
        help="JSONL path to record retrieval traces.",
    )
    _add_phase3_args(p_phase3)
    p_phase3.set_defaults(func=_cmd_phase3)

    p_ab = sub.add_parser("ablation", help="Run a top-k ablation sweep.")
    p_ab.add_argument("--cnm-index", type=Path, required=True)
    p_ab.add_argument("--k-values", type=str, default="0,1,3,5,10")
    p_ab.add_argument("--max-control-chars", type=int, default=300)
    _add_phase3_args(p_ab)
    p_ab.set_defaults(func=_cmd_ablation)

    args = parser.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
