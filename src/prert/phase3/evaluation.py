"""Evaluation helpers for Phase 3 classifier outputs."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence

from prert.phase3.classifier import TextClassifier
from prert.phase3.types import ClauseExample


def evaluate_classifier(
    model: TextClassifier,
    examples: Sequence[ClauseExample],
    labels: Sequence[str],
) -> Dict[str, Any]:
    if not examples:
        return {
            "rows": 0,
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "per_class": {label: _empty_metrics() for label in labels},
            "confusion": _empty_confusion(labels),
        }

    predictions: List[Dict[str, Any]] = []
    confusion: Dict[str, Dict[str, int]] = {
        actual: {predicted: 0 for predicted in labels}
        for actual in labels
    }

    correct = 0
    for example in examples:
        predicted_label = model.predict(example.text)
        proba = model.predict_proba(example.text)
        confidence = float(proba.get(predicted_label, 0.0))
        actual_label = example.label

        if actual_label in confusion and predicted_label in confusion[actual_label]:
            confusion[actual_label][predicted_label] += 1

        if predicted_label == actual_label:
            correct += 1

        predictions.append(
            {
                "example_id": example.example_id,
                "policy_uid": example.policy_uid,
                "actual_label": actual_label,
                "predicted_label": predicted_label,
                "confidence": round(confidence, 6),
                "text": example.text,
            }
        )

    per_class = {}
    precision_values: List[float] = []
    recall_values: List[float] = []
    f1_values: List[float] = []

    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[actual][label] for actual in labels if actual != label)
        fn = sum(confusion[label][pred] for pred in labels if pred != label)
        support = sum(confusion[label].values())

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0

        metrics = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }
        per_class[label] = metrics

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    row_count = len(examples)
    accuracy = correct / row_count if row_count else 0.0

    return {
        "rows": row_count,
        "accuracy": round(accuracy, 6),
        "macro_precision": round(_mean(precision_values), 6),
        "macro_recall": round(_mean(recall_values), 6),
        "macro_f1": round(_mean(f1_values), 6),
        "per_class": per_class,
        "confusion": confusion,
        "predictions": predictions,
    }


def _mean(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)


def _empty_metrics() -> Dict[str, Any]:
    return {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 0,
    }


def _empty_confusion(labels: Sequence[str]) -> Dict[str, Dict[str, int]]:
    return {
        actual: {predicted: 0 for predicted in labels}
        for actual in labels
    }
