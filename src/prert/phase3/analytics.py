"""Analytics helpers for Phase 3 measurement targets."""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def compute_calibration_report(
    predictions: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    num_bins: int = 10,
) -> Dict[str, Any]:
    label_list = [str(label) for label in labels]
    overall = _calibration_for_target(predictions, label_list, num_bins=num_bins, target_label=None)
    per_label = {
        label: _calibration_for_target(predictions, label_list, num_bins=num_bins, target_label=label)
        for label in label_list
    }
    macro_ece = _mean(float(report["ece"]) for report in per_label.values())

    return {
        "num_rows": len(predictions),
        "num_bins": num_bins,
        "overall": overall,
        "per_label": per_label,
        "macro_ece": round(macro_ece, 6),
    }


def compute_threshold_sweep(
    predictions: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    focus_labels: Sequence[str] = ("user", "system"),
    thresholds: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    label_list = [str(label) for label in labels]
    focus = [label for label in focus_labels if label in label_list]
    threshold_values = [round(float(value), 4) for value in (thresholds or _default_thresholds())]

    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for label in focus:
        rows: List[Dict[str, Any]] = []
        for threshold in threshold_values:
            tp = fp = fn = tn = 0
            for prediction in predictions:
                probabilities = _extract_probabilities(prediction, label_list)
                score = float(probabilities.get(label, 0.0))
                is_positive = score >= threshold
                is_actual_positive = str(prediction.get("actual_label", "")) == label

                if is_positive and is_actual_positive:
                    tp += 1
                elif is_positive and not is_actual_positive:
                    fp += 1
                elif (not is_positive) and is_actual_positive:
                    fn += 1
                else:
                    tn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            rows.append(
                {
                    "threshold": threshold,
                    "precision": round(precision, 6),
                    "recall": round(recall, 6),
                    "f1": round(f1, 6),
                    "support_positive": tp + fn,
                    "predicted_positive": tp + fp,
                    "confusion": {
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "tn": tn,
                    },
                }
            )
        by_label[label] = rows

    return {
        "num_rows": len(predictions),
        "focus_labels": focus,
        "thresholds": threshold_values,
        "by_label": by_label,
    }


def compute_bootstrap_confidence_intervals(
    predictions: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    label_list = [str(label) for label in labels]
    if not predictions:
        return {
            "n_rows": 0,
            "n_resamples": n_resamples,
            "seed": seed,
            "metrics": {},
        }

    baseline = _classification_metrics_from_predictions(predictions, label_list)
    metric_series: Dict[str, List[float]] = {
        "accuracy": [],
        "macro_f1": [],
    }
    for label in label_list:
        metric_series[f"f1_{label}"] = []

    rnd = random.Random(seed)
    row_count = len(predictions)
    for _ in range(max(1, n_resamples)):
        sample = [predictions[rnd.randrange(row_count)] for _ in range(row_count)]
        metrics = _classification_metrics_from_predictions(sample, label_list)
        metric_series["accuracy"].append(float(metrics["accuracy"]))
        metric_series["macro_f1"].append(float(metrics["macro_f1"]))
        for label in label_list:
            metric_series[f"f1_{label}"].append(float(metrics["per_class_f1"][label]))

    summary_metrics: Dict[str, Dict[str, Any]] = {}
    for key, series in metric_series.items():
        summary_metrics[key] = {
            "baseline": round(float(_baseline_metric_value(baseline, key)), 6),
            "mean": round(_mean(series), 6),
            "interval_95": {
                "lower": round(_percentile(series, 2.5), 6),
                "upper": round(_percentile(series, 97.5), 6),
            },
        }

    return {
        "n_rows": len(predictions),
        "n_resamples": max(1, n_resamples),
        "seed": seed,
        "metrics": summary_metrics,
    }


def _calibration_for_target(
    predictions: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    num_bins: int,
    target_label: Optional[str],
) -> Dict[str, Any]:
    bins: List[Dict[str, float]] = [
        {
            "count": 0.0,
            "confidence_sum": 0.0,
            "outcome_sum": 0.0,
            "squared_error_sum": 0.0,
        }
        for _ in range(max(1, num_bins))
    ]

    total = 0
    for prediction in predictions:
        probabilities = _extract_probabilities(prediction, labels)
        actual_label = str(prediction.get("actual_label", ""))

        if target_label is None:
            predicted_label = str(prediction.get("predicted_label", ""))
            if predicted_label not in labels:
                predicted_label = max(probabilities.items(), key=lambda item: item[1])[0]
            confidence = float(probabilities.get(predicted_label, 0.0))
            outcome = 1.0 if predicted_label == actual_label else 0.0
        else:
            confidence = float(probabilities.get(target_label, 0.0))
            outcome = 1.0 if actual_label == target_label else 0.0

        confidence = max(0.0, min(1.0, confidence))
        bin_index = min(len(bins) - 1, int(confidence * len(bins)))
        bucket = bins[bin_index]
        bucket["count"] += 1.0
        bucket["confidence_sum"] += confidence
        bucket["outcome_sum"] += outcome
        bucket["squared_error_sum"] += (outcome - confidence) ** 2
        total += 1

    bin_reports: List[Dict[str, Any]] = []
    ece = 0.0
    brier = 0.0
    for index, bucket in enumerate(bins):
        count = int(bucket["count"])
        lower = index / len(bins)
        upper = (index + 1) / len(bins)
        if count == 0:
            avg_confidence = 0.0
            accuracy = 0.0
            gap = 0.0
        else:
            avg_confidence = bucket["confidence_sum"] / bucket["count"]
            accuracy = bucket["outcome_sum"] / bucket["count"]
            gap = accuracy - avg_confidence
            weight = bucket["count"] / max(total, 1)
            ece += weight * abs(gap)
            brier += bucket["squared_error_sum"]

        bin_reports.append(
            {
                "bin_index": index,
                "lower": round(lower, 6),
                "upper": round(upper, 6),
                "count": count,
                "avg_confidence": round(avg_confidence, 6),
                "accuracy": round(accuracy, 6),
                "gap": round(gap, 6),
            }
        )

    brier_score = (brier / max(total, 1)) if total else 0.0
    return {
        "target": target_label or "top_label",
        "num_rows": total,
        "ece": round(ece, 6),
        "brier": round(brier_score, 6),
        "bins": bin_reports,
    }


def _classification_metrics_from_predictions(
    predictions: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
) -> Dict[str, Any]:
    confusion: Dict[str, Dict[str, int]] = {
        actual: {predicted: 0 for predicted in labels}
        for actual in labels
    }

    correct = 0
    for prediction in predictions:
        actual = str(prediction.get("actual_label", ""))
        predicted = str(prediction.get("predicted_label", ""))
        if actual in confusion and predicted in confusion[actual]:
            confusion[actual][predicted] += 1
            if actual == predicted:
                correct += 1

    per_class_f1: Dict[str, float] = {}
    f1_values: List[float] = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[actual][label] for actual in labels if actual != label)
        fn = sum(confusion[label][predicted] for predicted in labels if predicted != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class_f1[label] = f1
        f1_values.append(f1)

    count = len(predictions)
    accuracy = correct / count if count else 0.0
    macro_f1 = _mean(f1_values)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
    }


def _extract_probabilities(prediction: Mapping[str, Any], labels: Sequence[str]) -> Dict[str, float]:
    raw = prediction.get("probabilities")
    if isinstance(raw, Mapping):
        values = {label: max(0.0, float(raw.get(label, 0.0))) for label in labels}
        total = sum(values.values())
        if total > 0:
            return {label: values[label] / total for label in labels}

    predicted_label = str(prediction.get("predicted_label", ""))
    confidence = max(0.0, min(1.0, float(prediction.get("confidence", 0.0))))

    fallback = {label: 0.0 for label in labels}
    if predicted_label in fallback:
        remainder = max(0.0, 1.0 - confidence)
        others = [label for label in labels if label != predicted_label]
        shared = (remainder / len(others)) if others else 0.0
        for label in others:
            fallback[label] = shared
        fallback[predicted_label] = confidence
    else:
        uniform = 1.0 / max(len(labels), 1)
        fallback = {label: uniform for label in labels}

    return fallback


def _baseline_metric_value(metrics: Mapping[str, Any], key: str) -> float:
    if key == "accuracy":
        return float(metrics.get("accuracy", 0.0))
    if key == "macro_f1":
        return float(metrics.get("macro_f1", 0.0))
    if key.startswith("f1_"):
        label = key[3:]
        return float(metrics.get("per_class_f1", {}).get(label, 0.0))
    return 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])

    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * weight)


def _mean(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return float(sum(values_list) / len(values_list))


def _default_thresholds() -> List[float]:
    return [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
