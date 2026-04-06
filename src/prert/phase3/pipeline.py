"""End-to-end Phase 3 baseline pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from prert.phase3.classifier import train_classifier
from prert.phase3.dataset import (
    LABELS,
    build_dataset_manifest,
    build_opp115_clause_examples,
    load_labeled_examples,
    split_examples_by_policy,
)
from prert.phase3.evaluation import evaluate_classifier
from prert.phase3.io import write_json, write_jsonl
from prert.phase3.risk import compute_bayesian_risk, load_bayesian_priors


def run_phase3_pipeline(
    output_dir: Path,
    opp115_root: Optional[Path] = None,
    input_set: str = "consolidation-0.75",
    source_dir: Optional[Path] = None,
    labeled_input_path: Optional[Path] = None,
    model_type: str = "naive_bayes",
    random_state: int = 42,
    max_features: int = 20000,
    ngram_max: int = 2,
    min_df: int = 2,
    max_df: float = 0.95,
    c: float = 1.0,
    max_iter: int = 1000,
    privacybert_model_name: str = "bert-base-uncased",
    privacybert_epochs: float = 2.0,
    privacybert_batch_size: int = 8,
    privacybert_learning_rate: float = 5e-5,
    privacybert_max_length: int = 256,
    enable_bayesian_scoring: bool = True,
    bayesian_priors_path: Optional[Path] = None,
    bayesian_top_k: int = 5,
    seed: int = 42,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if labeled_input_path is not None:
        examples = load_labeled_examples(labeled_input_path)
        source_name = f"labeled::{labeled_input_path.name}"
    else:
        if opp115_root is None:
            raise ValueError("opp115_root is required when labeled_input_path is not provided")
        examples = build_opp115_clause_examples(
            opp115_root=opp115_root,
            input_set=input_set,
            source_dir=source_dir,
            max_rows=max_rows,
        )
        source_name = f"opp115::{input_set}"

    if not examples:
        raise ValueError("No training examples were produced for Phase 3")

    splits = split_examples_by_policy(examples=examples, seed=seed)
    dataset_manifest = build_dataset_manifest(
        splits=splits,
        seed=seed,
        source=source_name,
        input_set=input_set,
    )

    training_rows = [row.as_dict() for row in splits["train"]]
    validation_rows = [row.as_dict() for row in splits["validation"]]
    test_rows = [row.as_dict() for row in splits["test"]]

    write_jsonl(output_dir / "training_dataset.jsonl", training_rows)
    write_jsonl(output_dir / "validation_dataset.jsonl", validation_rows)
    write_jsonl(output_dir / "test_dataset.jsonl", test_rows)
    write_json(output_dir / "dataset_manifest.json", dataset_manifest)

    selected_model = model_type.strip().lower()
    checkpoint_path = output_dir / "classifier_checkpoint" / "model.json"
    if selected_model in {"logreg_tfidf", "logistic_regression", "lr_tfidf"}:
        checkpoint_path = output_dir / "classifier_checkpoint" / "model.pkl"
    elif selected_model in {"privacybert", "privacy_bert", "bert_privacy"}:
        checkpoint_path = output_dir / "classifier_checkpoint" / "privacybert"

    model, training_summary = train_classifier(
        examples=splits["train"],
        labels=LABELS,
        output_path=checkpoint_path,
        model_type=model_type,
        random_state=random_state,
        max_features=max_features,
        ngram_max=ngram_max,
        min_df=min_df,
        max_df=max_df,
        c=c,
        max_iter=max_iter,
        privacybert_model_name=privacybert_model_name,
        privacybert_epochs=privacybert_epochs,
        privacybert_batch_size=privacybert_batch_size,
        privacybert_learning_rate=privacybert_learning_rate,
        privacybert_max_length=privacybert_max_length,
    )

    validation_metrics = evaluate_classifier(model, splits["validation"], LABELS)
    test_metrics = evaluate_classifier(model, splits["test"], LABELS)

    validation_predictions = validation_metrics.pop("predictions")
    test_predictions = test_metrics.pop("predictions")

    write_jsonl(output_dir / "validation_predictions.jsonl", validation_predictions)
    write_jsonl(output_dir / "test_predictions.jsonl", test_predictions)

    bayesian_payload: Dict[str, Any] = {
        "enabled": False,
        "validation": {},
        "test": {},
        "primary_score": None,
    }
    if enable_bayesian_scoring:
        priors = load_bayesian_priors(bayesian_priors_path)
        validation_risk = compute_bayesian_risk(validation_predictions, priors=priors, top_k=bayesian_top_k)
        test_risk = compute_bayesian_risk(test_predictions, priors=priors, top_k=bayesian_top_k)
        write_json(output_dir / "bayesian_risk_validation.json", validation_risk)
        write_json(output_dir / "bayesian_risk_test.json", test_risk)
        bayesian_payload = {
            "enabled": True,
            "validation": validation_risk,
            "test": test_risk,
            "primary_score": test_risk["overall"]["primary_score"],
            "priors_source": str(bayesian_priors_path) if bayesian_priors_path else "default",
        }

    metrics_payload = {
        "phase": "phase-3",
        "model_type": training_summary["model_type"],
        "labels": list(LABELS),
        "training": {
            "rows": len(splits["train"]),
            "vocabulary_size": int(training_summary["vocabulary_size"]),
            "config": {
                "seed": random_state,
                "max_features": max_features,
                "ngram_max": ngram_max,
                "min_df": min_df,
                "max_df": max_df,
                "c": c,
                "max_iter": max_iter,
                "privacybert_model_name": privacybert_model_name,
                "privacybert_epochs": privacybert_epochs,
                "privacybert_batch_size": privacybert_batch_size,
                "privacybert_learning_rate": privacybert_learning_rate,
                "privacybert_max_length": privacybert_max_length,
            },
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "bayesian": bayesian_payload,
    }

    write_json(output_dir / "classifier_metrics.json", metrics_payload)
    write_jsonl(
        output_dir / "classifier_metrics.jsonl",
        [
            {
                "split": "validation",
                "rows": validation_metrics["rows"],
                "accuracy": validation_metrics["accuracy"],
                "macro_precision": validation_metrics["macro_precision"],
                "macro_recall": validation_metrics["macro_recall"],
                "macro_f1": validation_metrics["macro_f1"],
            },
            {
                "split": "test",
                "rows": test_metrics["rows"],
                "accuracy": test_metrics["accuracy"],
                "macro_precision": test_metrics["macro_precision"],
                "macro_recall": test_metrics["macro_recall"],
                "macro_f1": test_metrics["macro_f1"],
            },
        ],
    )

    _write_model_card(output_dir / "model_card.md", metrics_payload, source_name)
    _write_scoring_spec(output_dir / "scoring_spec.md", bayesian_enabled=enable_bayesian_scoring)
    _write_prototype_demo(output_dir / "prototype_demo.md")

    manifest = {
        "phase": "phase-3",
        "seed": seed,
        "inputs": {
            "opp115_root": str(opp115_root) if opp115_root else "",
            "input_set": input_set,
            "source_dir": str(source_dir) if source_dir else "",
            "labeled_input_path": str(labeled_input_path) if labeled_input_path else "",
            "model_type": model_type,
            "enable_bayesian_scoring": enable_bayesian_scoring,
        },
        "dataset_manifest": {
            "total_rows": dataset_manifest["total_rows"],
            "class_distribution": dataset_manifest["class_distribution"],
            "splits": dataset_manifest["splits"],
            "policy_overlap": dataset_manifest["policy_overlap"],
        },
        "model_summary": {
            "model_type": metrics_payload["model_type"],
            "labels": list(LABELS),
            "vocabulary_size": metrics_payload["training"]["vocabulary_size"],
            "checkpoint_path": str(checkpoint_path),
            "training_config": metrics_payload["training"]["config"],
        },
        "metrics": {
            "validation_macro_f1": metrics_payload["validation"]["macro_f1"],
            "test_macro_f1": metrics_payload["test"]["macro_f1"],
            "validation_accuracy": metrics_payload["validation"]["accuracy"],
            "test_accuracy": metrics_payload["test"]["accuracy"],
            "bayesian_primary_score": bayesian_payload["primary_score"],
        },
        "primary_metric_surface": "bayesian_posterior" if enable_bayesian_scoring else "classifier_metrics",
        "output_files": {
            "training_dataset": "training_dataset.jsonl",
            "validation_dataset": "validation_dataset.jsonl",
            "test_dataset": "test_dataset.jsonl",
            "dataset_manifest": "dataset_manifest.json",
            "classifier_checkpoint": str(checkpoint_path.relative_to(output_dir)),
            "classifier_metrics": "classifier_metrics.json",
            "classifier_metrics_rows": "classifier_metrics.jsonl",
            "validation_predictions": "validation_predictions.jsonl",
            "test_predictions": "test_predictions.jsonl",
            "bayesian_validation": "bayesian_risk_validation.json" if enable_bayesian_scoring else "",
            "bayesian_test": "bayesian_risk_test.json" if enable_bayesian_scoring else "",
            "model_card": "model_card.md",
            "scoring_spec": "scoring_spec.md",
            "prototype_demo": "prototype_demo.md",
        },
    }
    write_json(output_dir / "phase3_manifest.json", manifest)
    return manifest


def _write_model_card(path: Path, metrics_payload: Dict[str, Any], source_name: str) -> None:
    text = f"""# Phase 3 Baseline Model Card

## Model

- Type: {metrics_payload['model_type']}
- Labels: {", ".join(metrics_payload['labels'])}
- Training rows: {metrics_payload['training']['rows']}
- Vocabulary size: {metrics_payload['training']['vocabulary_size']}

## Dataset Source

- {source_name}

## Held-Out Metrics

Validation:

- Accuracy: {metrics_payload['validation']['accuracy']}
- Macro precision: {metrics_payload['validation']['macro_precision']}
- Macro recall: {metrics_payload['validation']['macro_recall']}
- Macro F1: {metrics_payload['validation']['macro_f1']}

Test:

- Accuracy: {metrics_payload['test']['accuracy']}
- Macro precision: {metrics_payload['test']['macro_precision']}
- Macro recall: {metrics_payload['test']['macro_recall']}
- Macro F1: {metrics_payload['test']['macro_f1']}

## Notes

- Classifier metrics are retained for diagnostics and benchmark comparison.
- Bayesian posterior risk outputs are emitted when Bayesian scoring is enabled.
"""
    path.write_text(text, encoding="utf-8")


def _write_scoring_spec(path: Path, bayesian_enabled: bool) -> None:
    bayesian_section = """
## Bayesian Risk Outputs

- bayesian_risk_validation.json
- bayesian_risk_test.json

Each Bayesian output includes:

- per-level posterior alpha/beta
- posterior mean risk and interval bounds
- top contributing clauses for each level
""" if bayesian_enabled else ""

    text = """# Phase 3 Baseline Scoring Specification

## Output Schema

- actual_label: ground-truth label from held-out set.
- predicted_label: model prediction in {user, system, organization}.
- confidence: predicted class probability.

## Metrics

- accuracy
- macro_precision
- macro_recall
- macro_f1
- per-class precision/recall/f1/support
""" + bayesian_section + """

## Constraints

- All metric values are in [0, 1] except support counts.
- Dataset splits are deterministic for a fixed seed.
"""
    path.write_text(text, encoding="utf-8")


def _write_prototype_demo(path: Path) -> None:
    text = """# Phase 3 Baseline Prototype Demo

Run the baseline pipeline:

```bash
PYTHONPATH=src python scripts/run_phase3_baseline.py
```

Run with a custom labeled dataset:

```bash
PYTHONPATH=src python scripts/run_phase3_baseline.py \
  --labeled-input-path path/to/labeled_dataset.jsonl \
  --output-dir artifacts/phase-3
```

Inspect outputs:

- artifacts/phase-3/phase3_manifest.json
- artifacts/phase-3/classifier_metrics.json
- artifacts/phase-3/model_card.md
"""
    path.write_text(text, encoding="utf-8")
