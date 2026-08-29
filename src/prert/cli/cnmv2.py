"""CNMv2 CLI for ChromaDB indexing, Phase 3 pipeline execution, and ablation sweeps."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List

from prert.config import load_dotenv_if_available
from prert.phase2.opp115 import INPUT_SET_TO_SUBDIR
from prert.phase3.classifier import DEFAULT_PRIVACYBERT_MODEL_NAME
from prert.phase3.cnm_v2.classifier_v2 import CNMv2Classifier, CNMv2TrainingConfig
from prert.phase3.cnm.embed import DEFAULT_EMBEDDING_MODEL, TextEncoder
from prert.phase3.cnm.memory import _load_controls_file
from prert.phase3.cnm_v2.memory_chroma import ChromaMemoryIndex, ControlRecord
from prert.phase3.cnm_v2.pipeline_v2 import run_phase3_cnmv2_pipeline
from prert.phase3.dataset import POLISIS_INPUT_SET_TO_SUBDIR

_LOGGER = logging.getLogger("prert.cli.cnmv2")


def _memory_entries_to_control_records(entries: List[Any]) -> List[ControlRecord]:
    return [
        ControlRecord(
            control_id=e.control_id or f"ctrl-{i:04d}",
            text=e.text,
            source=e.source,
            section=e.section,
            metadata=e.metadata,
        )
        for i, e in enumerate(entries)
    ]


def _cmd_build_chroma(args: argparse.Namespace) -> int:
    entries = _load_controls_file(args.controls_path)
    if not entries:
        raise SystemExit(f"No controls found in {args.controls_path}")
    _LOGGER.info("Loaded %d controls from %s", len(entries), args.controls_path)

    encoder = TextEncoder(model_name=args.encoder, device=args.device)
    texts = [e.text for e in entries]
    embeddings = encoder.encode(texts, batch_size=args.batch_size, normalise=True)
    records = _memory_entries_to_control_records(entries)

    memory = ChromaMemoryIndex(persist_dir=args.output_dir, encoder_name=args.encoder)
    memory.build(records=records, embeddings=embeddings, overwrite=True)

    print(f"Built ChromaDB memory index: n={len(records)} dim={embeddings.shape[1]}")
    print(f"Persisted at {args.output_dir}")
    return 0


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
    parser.add_argument("--privacybert-epochs", type=float, default=3.0)
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
    # CNMv2-specific
    parser.add_argument("--softmax-temperature", type=float, default=1.0)
    parser.add_argument("--gate-hidden", type=int, default=128)
    parser.add_argument(
        "--freeze-retrieval-below-epoch",
        type=int,
        default=0,
        help="Keep retrieval head + gate frozen until this epoch",
    )
    parser.add_argument("--retrieval-consistency-weight", type=float, default=0.1)
    parser.add_argument("--trace-output", type=Path, default=None)


def _phase3_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
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


def _cnm_config_from_args(args: argparse.Namespace) -> CNMv2TrainingConfig:
    return CNMv2TrainingConfig(
        index_dir=args.cnm_index,
        top_k=args.top_k,
        softmax_temperature=args.softmax_temperature,
        gate_hidden=args.gate_hidden,
        freeze_retrieval_below_epoch=args.freeze_retrieval_below_epoch,
        retrieval_consistency_weight=args.retrieval_consistency_weight,
        trace_output_path=args.trace_output,
    )


def _cmd_phase3(args: argparse.Namespace) -> int:
    cnm_config = _cnm_config_from_args(args)
    print(
        f"[CNMv2] top_k={cnm_config.top_k} "
        f"freeze_below_epoch={cnm_config.freeze_retrieval_below_epoch} "
        f"consistency_w={cnm_config.retrieval_consistency_weight} "
        f"index={cnm_config.index_dir}",
        flush=True,
    )
    manifest = run_phase3_cnmv2_pipeline(
        output_dir=args.output_dir,
        cnm_config=cnm_config,
        **_phase3_kwargs(args),
    )
    m = manifest["metrics"]
    print(f"[CNMv2] complete. model_type={m.get('model_type')}")
    print(f"  val_macro_f1  = {m['validation_macro_f1']}")
    print(f"  test_macro_f1 = {m['test_macro_f1']}")
    print(f"  test_accuracy = {m['test_accuracy']}")
    if m.get("bayesian_primary_score") is not None:
        print(f"  primary       = {m['bayesian_primary_score']}")
    if m.get("model_type") != "cnmv2_privacybert":
        print(f"[CNMv2][ERROR] model_type mismatch — expected 'cnmv2_privacybert'")
    return 0


def _gpu_status() -> str:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "(no output)"
    except Exception as exc:  # pragma: no cover - nvidia-smi may be absent
        return f"(nvidia-smi unavailable: {exc})"


def _confirm_continue(prompt: str) -> bool:
    reply = input(f"{prompt} [y/N] ").strip().lower()
    return reply in ("y", "yes")


def _cmd_ablation(args: argparse.Namespace) -> int:
    k_values = [int(x) for x in args.k_values.split(",")]
    print(f"[CNMv2] Ablation over k={k_values}", flush=True)

    results: Dict[int, Dict[str, Any]] = {}
    for fold_index, k in enumerate(k_values):
        run_dir = args.output_dir / f"k={k}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"{args.run_id}-k{k}" if args.run_id else f"ablation-k{k}"

        cmd = [
            sys.executable,
            "-m",
            "prert.cli.cnmv2",
            "phase3",
            "--cnm-index",
            str(args.cnm_index),
            "--top-k",
            str(k),
            "--output-dir",
            str(run_dir),
            "--run-id",
            run_id,
            "--softmax-temperature",
            str(args.softmax_temperature),
            "--gate-hidden",
            str(args.gate_hidden),
            "--freeze-retrieval-below-epoch",
            str(args.freeze_retrieval_below_epoch),
            "--retrieval-consistency-weight",
            str(args.retrieval_consistency_weight),
            "--opp115-root",
            str(args.opp115_root),
            "--input-set",
            str(args.input_set),
            "--polisis-input-set",
            str(args.polisis_input_set),
            "--seed",
            str(args.seed),
            "--random-state",
            str(args.random_state),
            "--privacybert-model-name",
            str(args.privacybert_model_name),
            "--privacybert-epochs",
            str(args.privacybert_epochs),
            "--privacybert-batch-size",
            str(args.privacybert_batch_size),
            "--privacybert-learning-rate",
            str(args.privacybert_learning_rate),
            "--privacybert-max-length",
            str(args.privacybert_max_length),
            "--privacybert-loss-type",
            str(args.privacybert_loss_type),
            "--privacybert-focal-gamma",
            str(args.privacybert_focal_gamma),
            "--privacybert-label-smoothing",
            str(args.privacybert_label_smoothing),
            "--privacybert-weight-decay",
            str(args.privacybert_weight_decay),
            "--privacybert-warmup-steps",
            str(args.privacybert_warmup_steps),
            "--privacybert-early-stopping-patience",
            str(args.privacybert_early_stopping_patience),
            "--bayesian-top-k",
            str(args.bayesian_top_k),
            "--calibration-bins",
            str(args.calibration_bins),
            "--bootstrap-resamples",
            str(args.bootstrap_resamples),
        ]
        if args.source_dir is not None:
            cmd += ["--source-dir", str(args.source_dir)]
        if args.polisis_root is not None:
            cmd += ["--polisis-root", str(args.polisis_root)]
        if args.polisis_source_dir is not None:
            cmd += ["--polisis-source-dir", str(args.polisis_source_dir)]
        if args.labeled_input_path is not None:
            cmd += ["--labeled-input-path", str(args.labeled_input_path)]
        if args.auxiliary_labeled_input_path is not None:
            cmd += [
                "--auxiliary-labeled-input-path",
                str(args.auxiliary_labeled_input_path),
            ]
        if args.disable_bayesian_scoring:
            cmd += ["--disable-bayesian-scoring"]
        if args.bayesian_priors_path is not None:
            cmd += ["--bayesian-priors-path", str(args.bayesian_priors_path)]
        if args.max_rows is not None:
            cmd += ["--max-rows", str(args.max_rows)]
        if args.trace_output is not None:
            cmd += ["--trace-output", str(args.trace_output)]

        if not args.no_confirm:
            print(f"--- Pre-run GPU status before k={k}: {_gpu_status()}")
            if fold_index > 0 and not _confirm_continue(
                f"About to start fold k={k} ({fold_index + 1}/{len(k_values)}). Proceed?"
            ):
                print("[CNMv2] Ablation aborted by user.")
                break

        _LOGGER.info("=== CNMv2 ablation: k=%d, output=%s (subprocess) ===", k, run_dir)
        proc = subprocess.run(cmd)

        if proc.returncode != 0:
            print(
                f"[CNMv2][ERROR] fold k={k} exited with code {proc.returncode} "
                f"(GPU status: {_gpu_status()})"
            )
            if args.no_confirm:
                raise SystemExit(proc.returncode)
            if not _confirm_continue(f"Fold k={k} failed. Skip it and continue?"):
                print("[CNMv2] Ablation aborted by user.")
                break
            continue

        manifest_path = run_dir / "phase3_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as handle:
            results[k] = json.load(handle)

        if not args.no_confirm:
            time.sleep(2)
            print(f"--- Post-run GPU status after k={k}: {_gpu_status()}")

    print("=== CNMv2 ablation summary ===")
    print(f"{'k':>4}  {'val_F1':>7}  {'test_F1':>8}  {'test_acc':>9}  {'primary':>9}")
    for k, mnf in results.items():
        m = mnf["metrics"]
        prim = m.get("bayesian_primary_score", "n/a")
        print(
            f"{k:>4}  {m['validation_macro_f1']:>7.4f}  "
            f"{m['test_macro_f1']:>8.4f}  {m['test_accuracy']:>9.4f}  {str(prim):>9}"
        )
    summary_path = args.output_dir / "ablation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as h:
        json.dump(
            {
                "k_values": k_values,
                "results": {
                    str(k): {
                        "validation_macro_f1": mnf["metrics"]["validation_macro_f1"],
                        "test_macro_f1": mnf["metrics"]["test_macro_f1"],
                        "test_accuracy": mnf["metrics"]["test_accuracy"],
                        "bayesian_primary_score": mnf["metrics"].get(
                            "bayesian_primary_score"
                        ),
                        "model_type": mnf["metrics"].get("model_type"),
                    }
                    for k, mnf in results.items()
                },
            },
            h,
            indent=2,
        )
        h.write("\n")
    print(f"Wrote {summary_path}")
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    load_dotenv_if_available(None)

    parser = argparse.ArgumentParser(
        description="CNMv2 (auxiliary-head retrieval) CLIs."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build-chroma", help="Build the ChromaDB memory index.")
    p_build.add_argument("--controls-path", type=Path, required=True)
    p_build.add_argument("--output-dir", type=Path, required=True)
    p_build.add_argument("--encoder", type=str, default=DEFAULT_EMBEDDING_MODEL)
    p_build.add_argument("--batch-size", type=int, default=64)
    p_build.add_argument("--device", type=str, default=None)
    p_build.set_defaults(func=_cmd_build_chroma)

    p_phase3 = sub.add_parser("phase3", help="Run Phase 3 with CNMv2.")
    p_phase3.add_argument("--cnm-index", type=Path, required=True)
    p_phase3.add_argument("--top-k", type=int, default=5)
    _add_phase3_args(p_phase3)
    p_phase3.set_defaults(func=_cmd_phase3)

    p_ab = sub.add_parser("ablation", help="Run CNMv2 top-k ablation.")
    p_ab.add_argument("--cnm-index", type=Path, required=True)
    p_ab.add_argument("--k-values", type=str, default="0,1,3,5,10")
    p_ab.add_argument(
        "--no-confirm",
        action="store_true",
        help=(
            "Skip the y/N confirmation gate + GPU status check between folds "
            "and run unattended (fails fast on the first fold error)."
        ),
    )
    _add_phase3_args(p_ab)
    p_ab.set_defaults(func=_cmd_ablation)

    args = parser.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    main()
