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


def run_phase3_pipeline(
    output_dir: Path,
    opp115_root: Optional[Path] = None,
    input_set: str = "consolidation-0.75",
    source_dir: Optional[Path] = None,
    labeled_input_path: Optional[Path] = None,
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

    checkpoint_path = output_dir / "classifier_checkpoint" / "model.json"
    model, training_summary = train_classifier(
        examples=splits["train"],
        labels=LABELS,
        output_path=checkpoint_path,
    )

    validation_metrics = evaluate_classifier(model, splits["validation"], LABELS)
    test_metrics = evaluate_classifier(model, splits["test"], LABELS)

    write_jsonl(output_dir / "validation_predictions.jsonl", validation_metrics.pop("predictions"))
    write_jsonl(output_dir / "test_predictions.jsonl", test_metrics.pop("predictions"))

    metrics_payload = {
        "phase": "phase-3",
        "model_type": "multinomial_naive_bayes",
        "labels": list(LABELS),
        "training": {
            "rows": len(splits["train"]),
            "vocabulary_size": int(training_summary["vocabulary_size"]),
        },
        "validation": validation_metrics,
        "test": test_metrics,
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
    _write_scoring_spec(output_dir / "scoring_spec.md")
    _write_prototype_demo(output_dir / "prototype_demo.md")

    manifest = {
        "phase": "phase-3",
        "seed": seed,
        "inputs": {
            "opp115_root": str(opp115_root) if opp115_root else "",
            "input_set": input_set,
            "source_dir": str(source_dir) if source_dir else "",
            "labeled_input_path": str(labeled_input_path) if labeled_input_path else "",
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
        },
        "metrics": {
            "validation_macro_f1": metrics_payload["validation"]["macro_f1"],
            "test_macro_f1": metrics_payload["test"]["macro_f1"],
            "validation_accuracy": metrics_payload["validation"]["accuracy"],
            "test_accuracy": metrics_payload["test"]["accuracy"],
        },
        "output_files": {
            "training_dataset": "training_dataset.jsonl",
            "validation_dataset": "validation_dataset.jsonl",
            "test_dataset": "test_dataset.jsonl",
            "dataset_manifest": "dataset_manifest.json",
            "classifier_checkpoint": "classifier_checkpoint/model.json",
            "classifier_metrics": "classifier_metrics.json",
            "classifier_metrics_rows": "classifier_metrics.jsonl",
            "validation_predictions": "validation_predictions.jsonl",
            "test_predictions": "test_predictions.jsonl",
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

- This is a deterministic baseline for Phase 3 acceptance and reproducibility.
- Bayesian scoring integration is intentionally deferred to the next increment.
"""
    path.write_text(text, encoding="utf-8")


def _write_scoring_spec(path: Path) -> None:
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

## Constraints

- All metric values are in [0, 1] except support counts.
- Dataset splits are deterministic for a fixed seed.
"""
    path.write_text(text, encoding="utf-8")


def _write_prototype_demo(path: Path) -> None:
    text = """# Phase 3 Baseline Prototype Demo

Run the baseline pipeline:

```bash
PYTHONPATH=src python3 scripts/run_phase3_baseline.py
```

Run with a custom labeled dataset:

```bash
PYTHONPATH=src python3 scripts/run_phase3_baseline.py \
  --labeled-input-path path/to/labeled_dataset.jsonl \
  --output-dir artifacts/phase-3
```

Inspect outputs:

- artifacts/phase-3/phase3_manifest.json
- artifacts/phase-3/classifier_metrics.json
- artifacts/phase-3/model_card.md
"""
    path.write_text(text, encoding="utf-8")
