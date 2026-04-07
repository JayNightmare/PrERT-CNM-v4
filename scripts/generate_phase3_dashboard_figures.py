"""Generate static Phase 3 dashboard figures from artifact data."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from prert.phase3.analytics import (
    compute_bootstrap_confidence_intervals,
    compute_calibration_report,
    compute_threshold_sweep,
)

plt = importlib.import_module("matplotlib.pyplot")
colors = importlib.import_module("matplotlib.colors")


MODEL_CONFIG = [
    {
        "key": "nb",
        "label": "Naive Bayes",
        "artifact_dir": "phase-3-nb",
    },
    {
        "key": "logreg",
        "label": "LogReg TF-IDF",
        "artifact_dir": "phase-3-logreg",
    },
    {
        "key": "no_bayes",
        "label": "LogReg TF-IDF (No Bayes)",
        "artifact_dir": "phase-3-no-bayes",
    },
    {
        "key": "privacybert",
        "label": "PrivacyBERT",
        "artifact_dir": "phase-3-privacybert",
    },
]


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_json_if_exists(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    return _load_json(path)


def _load_jsonl_if_exists(path: Path) -> Optional[List[Dict[str, Any]]]:
    if not path.exists():
        return None

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _validate_metrics(metrics: Dict, labels: List[str], model_label: str) -> None:
    for split in ["validation", "test"]:
        if split not in metrics:
            raise ValueError(f"Missing split '{split}' for model {model_label}")
        for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "per_class", "confusion"]:
            if key not in metrics[split]:
                raise ValueError(f"Missing key '{split}.{key}' for model {model_label}")
        for label in labels:
            if label not in metrics[split]["per_class"]:
                raise ValueError(f"Missing per-class label '{label}' in split '{split}' for model {model_label}")
            if label not in metrics[split]["confusion"]:
                raise ValueError(f"Missing confusion row '{label}' in split '{split}' for model {model_label}")


def _plot_pie(values: Dict[str, int], title: str, out_path: Path) -> None:
    labels = list(values.keys())
    numbers = [int(values[label]) for label in labels]

    plt.figure(figsize=(7.5, 6.5))
    _, _, autotexts = plt.pie(
        numbers,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct * sum(numbers) / 100.0))})",
        startangle=120,
        counterclock=False,
        wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
        textprops={"fontsize": 10},
    )
    for autotext in autotexts:
        autotext.set_color("black")
        autotext.set_fontsize(9)

    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()


def _plot_aggregate_comparison(models: List[Dict], title: str, out_path: Path) -> None:
    metric_labels = [
        "Validation Accuracy",
        "Validation Macro F1",
        "Test Accuracy",
        "Test Macro F1",
    ]
    metric_extractors = [
        lambda m: float(m["metrics"]["validation"]["accuracy"]),
        lambda m: float(m["metrics"]["validation"]["macro_f1"]),
        lambda m: float(m["metrics"]["test"]["accuracy"]),
        lambda m: float(m["metrics"]["test"]["macro_f1"]),
    ]

    x_positions = list(range(len(metric_labels)))
    width = 0.18
    offsets = [(-1.5 + index) * width for index in range(len(models))]

    plt.figure(figsize=(10.5, 6.8))
    for index, model in enumerate(models):
        values = [extractor(model) for extractor in metric_extractors]
        bars = plt.bar(
            [x + offsets[index] for x in x_positions],
            values,
            width,
            label=model["label"],
        )
        for bar in bars:
            value = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.006,
                f"{value * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.ylim(0, 1.02)
    plt.ylabel("Score")
    plt.xticks(x_positions, metric_labels, rotation=15)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()


def _plot_macro_metric_heatmap(models: List[Dict], title: str, out_path: Path) -> None:
    columns = [
        ("validation", "macro_precision", "Val Macro Precision"),
        ("validation", "macro_recall", "Val Macro Recall"),
        ("validation", "macro_f1", "Val Macro F1"),
        ("test", "macro_precision", "Test Macro Precision"),
        ("test", "macro_recall", "Test Macro Recall"),
        ("test", "macro_f1", "Test Macro F1"),
    ]
    matrix = [
        [float(model["metrics"][split][metric]) for split, metric, _ in columns]
        for model in models
    ]

    plt.figure(figsize=(12.2, 6.2))
    image = plt.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(image, fraction=0.046, pad=0.04, label="Score")
    plt.title(title)
    plt.xticks(range(len(columns)), [label for _, _, label in columns], rotation=20, ha="right")
    plt.yticks(range(len(models)), [model["label"] for model in models])

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            color = "white" if value >= 0.78 else "black"
            plt.text(col_index, row_index, f"{value * 100:.1f}%", ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()


def _plot_per_class_f1_comparison(models: List[Dict], labels: List[str], title: str, out_path: Path) -> None:
    x_positions = list(range(len(labels)))
    width = 0.18
    offsets = [(-1.5 + index) * width for index in range(len(models))]

    plt.figure(figsize=(9.8, 6.8))
    for index, model in enumerate(models):
        values = [float(model["metrics"]["test"]["per_class"][label]["f1"]) for label in labels]
        bars = plt.bar(
            [x + offsets[index] for x in x_positions],
            values,
            width,
            label=model["label"],
        )
        for bar in bars:
            value = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.006,
                f"{value * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.ylim(0, 1.02)
    plt.ylabel("F1")
    plt.xticks(x_positions, labels)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()


def _plot_confusion_small_multiples(models: List[Dict], labels: List[str], title: str, out_path: Path) -> None:
    matrices = []
    for model in models:
        confusion = model["metrics"]["test"]["confusion"]
        matrix = [[int(confusion[actual][predicted]) for predicted in labels] for actual in labels]
        matrices.append(matrix)

    global_max = max(max(max(row) for row in matrix) for matrix in matrices)
    threshold = global_max * 0.45

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 10.0), constrained_layout=True)
    figure.suptitle(title, fontsize=13)

    for axis, model, matrix in zip(axes.flat, models, matrices):
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=global_max)
        axis.set_title(model["label"], fontsize=11)
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=15)
        axis.set_yticks(range(len(labels)))
        axis.set_yticklabels(labels)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                color = "white" if value >= threshold else "black"
                axis.text(j, i, str(value), ha="center", va="center", color=color, fontsize=9)

    figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="Count")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=170)
    plt.close(figure)


def _plot_delta_vs_nb_heatmap(models: List[Dict], labels: List[str], title: str, out_path: Path) -> None:
    baseline = next((model for model in models if model["key"] == "nb"), models[0])
    delta_columns = [
        ("test", "accuracy", "Test Accuracy"),
        ("test", "macro_f1", "Test Macro F1"),
        ("test", "per_class", labels[0], "F1 user"),
        ("test", "per_class", labels[1], "F1 system"),
        ("test", "per_class", labels[2], "F1 organization"),
    ]

    matrix = []
    row_labels = []
    for model in models:
        row = []
        for column in delta_columns:
            if len(column) == 3:
                split, metric, _ = column
                value = float(model["metrics"][split][metric])
                baseline_value = float(baseline["metrics"][split][metric])
            else:
                split, metric, class_label, _ = column
                value = float(model["metrics"][split][metric][class_label]["f1"])
                baseline_value = float(baseline["metrics"][split][metric][class_label]["f1"])
            row.append(value - baseline_value)
        matrix.append(row)
        row_labels.append(model["label"])

    flatten = [value for row in matrix for value in row]
    abs_max = max(abs(value) for value in flatten)
    norm = colors.TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

    plt.figure(figsize=(10.2, 6.3))
    image = plt.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")
    plt.colorbar(image, fraction=0.046, pad=0.04, label="Delta vs NB")
    plt.title(title)
    plt.xticks(range(len(delta_columns)), [column[-1] for column in delta_columns], rotation=20, ha="right")
    plt.yticks(range(len(row_labels)), row_labels)

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            color = "white" if abs(value) >= (abs_max * 0.55) else "black"
            plt.text(col_index, row_index, f"{value:+.3f}", ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()


def _plot_bayesian_intervals(models: List[Dict], level_order: List[str], title: str, out_path: Path) -> None:
    bayes_models = [model for model in models if model.get("bayesian_test") is not None]
    if not bayes_models:
        return

    x_positions = list(range(len(level_order)))
    width = 0.18
    offsets = [(-1 + index) * width for index in range(len(bayes_models))]

    plt.figure(figsize=(9.5, 6.8))
    for index, model in enumerate(bayes_models):
        bayesian_levels = model["bayesian_test"]["levels"]
        means = [float(bayesian_levels[level]["posterior"]["mean"]) for level in level_order]
        lowers = [float(bayesian_levels[level]["posterior"]["interval_95"]["lower"]) for level in level_order]
        uppers = [float(bayesian_levels[level]["posterior"]["interval_95"]["upper"]) for level in level_order]
        lower_err = [mean - lower for mean, lower in zip(means, lowers)]
        upper_err = [upper - mean for mean, upper in zip(means, uppers)]

        plt.errorbar(
            [x + offsets[index] for x in x_positions],
            means,
            yerr=[lower_err, upper_err],
            fmt="o",
            capsize=4,
            markersize=6,
            linewidth=1.5,
            label=model["label"],
        )

    na_models = [model["label"] for model in models if model.get("bayesian_test") is None]
    if na_models:
        plt.figtext(0.02, 0.01, f"Bayesian unavailable (N/A): {', '.join(na_models)}", fontsize=9)

    plt.ylim(0.6, 1.01)
    plt.ylabel("Posterior Mean")
    plt.xticks(x_positions, level_order)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()


def _resolve_calibration_payload(model: Dict, labels: List[str]) -> Optional[Dict[str, Any]]:
    payload = model.get("calibration_test")
    if payload is not None:
        return payload
    predictions = model.get("predictions_test")
    if not predictions:
        return None
    return compute_calibration_report(predictions, labels=labels, num_bins=10)


def _resolve_threshold_payload(model: Dict, labels: List[str]) -> Optional[Dict[str, Any]]:
    payload = model.get("threshold_test")
    if payload is not None:
        return payload
    predictions = model.get("predictions_test")
    if not predictions:
        return None
    return compute_threshold_sweep(predictions, labels=labels)


def _resolve_bootstrap_payload(model: Dict, labels: List[str]) -> Optional[Dict[str, Any]]:
    payload = model.get("bootstrap_test")
    if payload is not None:
        return payload
    predictions = model.get("predictions_test")
    if not predictions:
        return None
    return compute_bootstrap_confidence_intervals(predictions, labels=labels, n_resamples=500, seed=42)


def _plot_reliability_curves(models: List[Dict], labels: List[str], title: str, out_path: Path) -> None:
    plt.figure(figsize=(9.8, 6.8))
    plotted = False
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.2, color="gray", label="Perfect calibration")

    for model in models:
        payload = _resolve_calibration_payload(model, labels)
        if payload is None:
            continue
        bins = payload.get("overall", {}).get("bins", [])
        x_values = [float(bin_row["avg_confidence"]) for bin_row in bins if int(bin_row.get("count", 0)) > 0]
        y_values = [float(bin_row["accuracy"]) for bin_row in bins if int(bin_row.get("count", 0)) > 0]
        if not x_values:
            continue
        plotted = True
        plt.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=f"{model['label']} (ECE={float(payload['overall']['ece']):.3f})",
        )

    if not plotted:
        plt.text(0.5, 0.5, "Calibration data unavailable", ha="center", va="center", fontsize=12)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Predicted confidence")
    plt.ylabel("Observed accuracy")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()


def _plot_ece_summary(models: List[Dict], labels: List[str], title: str, out_path: Path) -> None:
    model_labels: List[str] = []
    ece_values: List[float] = []
    macro_ece_values: List[float] = []

    for model in models:
        payload = _resolve_calibration_payload(model, labels)
        if payload is None:
            continue
        model_labels.append(model["label"])
        ece_values.append(float(payload.get("overall", {}).get("ece", 0.0)))
        macro_ece_values.append(float(payload.get("macro_ece", 0.0)))

    plt.figure(figsize=(9.8, 6.5))
    if not model_labels:
        plt.text(0.5, 0.5, "ECE data unavailable", ha="center", va="center", fontsize=12)
    else:
        x_positions = list(range(len(model_labels)))
        width = 0.35
        bars_overall = plt.bar([x - (width / 2) for x in x_positions], ece_values, width, label="Overall ECE")
        bars_macro = plt.bar([x + (width / 2) for x in x_positions], macro_ece_values, width, label="Macro ECE")

        for bars in (bars_overall, bars_macro):
            for bar in bars:
                value = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.005,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.xticks(x_positions, model_labels, rotation=15)
        plt.ylim(0, max(0.05, max(ece_values + macro_ece_values) + 0.03))
        plt.ylabel("ECE")
        plt.legend(loc="upper left")

    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()


def _plot_threshold_sensitivity(models: List[Dict], labels: List[str], title: str, out_path: Path) -> None:
    payloads = {
        model["key"]: _resolve_threshold_payload(model, labels)
        for model in models
    }
    focus_labels = ["user", "system"]

    figure, axes = plt.subplots(1, len(focus_labels), figsize=(12.0, 5.8), constrained_layout=True)
    if len(focus_labels) == 1:
        axes = [axes]

    for axis, focus_label in zip(axes, focus_labels):
        has_data = False
        for model in models:
            payload = payloads.get(model["key"])
            if payload is None:
                continue
            series = payload.get("by_label", {}).get(focus_label, [])
            if not series:
                continue

            has_data = True
            recalls = [float(row.get("recall", 0.0)) for row in series]
            precisions = [float(row.get("precision", 0.0)) for row in series]
            axis.plot(recalls, precisions, marker="o", linewidth=1.6, markersize=4, label=model["label"])

            best_row = max(series, key=lambda row: float(row.get("f1", 0.0)))
            axis.text(
                float(best_row.get("recall", 0.0)),
                float(best_row.get("precision", 0.0)),
                f"t={float(best_row.get('threshold', 0.0)):.2f}",
                fontsize=8,
            )

        axis.set_title(f"{focus_label.title()} Threshold Sweep")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.grid(alpha=0.25)
        if not has_data:
            axis.text(0.5, 0.5, "No threshold data", ha="center", va="center", fontsize=11)

    handles, labels_text = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels_text, loc="lower center", ncol=2, frameon=False)
    figure.suptitle(title, fontsize=13)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=170)
    plt.close(figure)


def _plot_bootstrap_confidence(models: List[Dict], labels: List[str], title: str, out_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.8), constrained_layout=True)
    metric_keys = ["accuracy", "macro_f1"]
    metric_titles = ["Test Accuracy 95% CI", "Test Macro F1 95% CI"]

    for axis, metric_key, metric_title in zip(axes, metric_keys, metric_titles):
        x_positions: List[int] = []
        labels_text: List[str] = []
        for index, model in enumerate(models):
            payload = _resolve_bootstrap_payload(model, labels)
            if payload is None:
                continue
            metric_row = payload.get("metrics", {}).get(metric_key)
            if not metric_row:
                continue

            baseline = float(metric_row.get("baseline", metric_row.get("mean", 0.0)))
            lower = float(metric_row.get("interval_95", {}).get("lower", baseline))
            upper = float(metric_row.get("interval_95", {}).get("upper", baseline))

            x_positions.append(index)
            labels_text.append(model["label"])
            axis.errorbar(
                index,
                baseline,
                yerr=[[max(0.0, baseline - lower)], [max(0.0, upper - baseline)]],
                fmt="o",
                capsize=4,
                markersize=6,
                linewidth=1.4,
            )
            axis.text(index, min(1.0, upper + 0.01), f"[{lower:.3f}, {upper:.3f}]", ha="center", fontsize=8)

        axis.set_title(metric_title)
        axis.set_ylim(0, 1.02)
        axis.set_ylabel("Score")
        axis.set_xticks(x_positions)
        axis.set_xticklabels(labels_text, rotation=15)
        axis.grid(alpha=0.25)

    figure.suptitle(title, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=170)
    plt.close(figure)


def _load_run_history(path: Path) -> List[Dict[str, Any]]:
    rows = _load_jsonl_if_exists(path) or []
    return sorted(rows, key=lambda row: str(row.get("executed_at", "")))


def _plot_run_trends(models: List[Dict], history_rows: List[Dict[str, Any]], title: str, out_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 6.0), constrained_layout=True)
    metric_keys = ["test_accuracy", "test_macro_f1"]
    metric_titles = ["Test Accuracy Trend", "Test Macro F1 Trend"]

    for axis, metric_key, metric_title in zip(axes, metric_keys, metric_titles):
        plotted = False
        for model in models:
            model_rows = [
                row for row in history_rows
                if str(row.get("artifact_dir", "")) == str(model.get("artifact_dir", ""))
            ]
            if not model_rows:
                continue

            plotted = True
            x_positions = list(range(len(model_rows)))
            y_values = [float(row.get("metrics", {}).get(metric_key, 0.0)) for row in model_rows]
            x_labels = [str(row.get("executed_at", ""))[:10] for row in model_rows]

            axis.plot(x_positions, y_values, marker="o", linewidth=1.7, markersize=5, label=model["label"])
            axis.set_xticks(x_positions)
            axis.set_xticklabels(x_labels, rotation=30)

        axis.set_title(metric_title)
        axis.set_ylim(0, 1.02)
        axis.set_ylabel("Score")
        axis.grid(alpha=0.25)
        if not plotted:
            axis.text(0.5, 0.5, "No run history yet", ha="center", va="center", fontsize=11)

    handles, labels_text = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels_text, loc="lower center", ncol=2, frameon=False)
    figure.suptitle(title, fontsize=13)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=170)
    plt.close(figure)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifacts_root = repo_root / "artifacts"
    output_dir = repo_root / "docs" / "Project" / "Execution-Playbook" / "figures"

    models = []
    for config in MODEL_CONFIG:
        artifact_dir = artifacts_root / config["artifact_dir"]
        manifest_path = artifact_dir / "dataset_manifest.json"
        metrics_path = artifact_dir / "classifier_metrics.json"
        if not manifest_path.exists() or not metrics_path.exists():
            print(f"Skipping {config['label']}: missing required artifacts in {artifact_dir}")
            continue

        manifest = _load_json(manifest_path)
        metrics = _load_json(metrics_path)
        bayesian_test = _load_json_if_exists(artifact_dir / "bayesian_risk_test.json")
        calibration_test = _load_json_if_exists(artifact_dir / "calibration_test.json")
        threshold_test = _load_json_if_exists(artifact_dir / "threshold_sweep_test.json")
        bootstrap_test = _load_json_if_exists(artifact_dir / "bootstrap_ci_test.json")
        predictions_test = _load_jsonl_if_exists(artifact_dir / "test_predictions.jsonl")
        models.append(
            {
                "key": config["key"],
                "label": config["label"],
                "artifact_dir": config["artifact_dir"],
                "manifest": manifest,
                "metrics": metrics,
                "bayesian_test": bayesian_test,
                "calibration_test": calibration_test,
                "threshold_test": threshold_test,
                "bootstrap_test": bootstrap_test,
                "predictions_test": predictions_test,
            }
        )

    if not models:
        print("No complete Phase 3 model artifacts found. Run the Phase 3 pipeline variants before generating dashboard figures.")
        return

    baseline_manifest = models[0]["manifest"]
    labels = [str(label) for label in baseline_manifest.get("labels", ["user", "system", "organization"])]

    for model in models:
        if int(model["manifest"].get("total_rows", -1)) != int(baseline_manifest.get("total_rows", -1)):
            raise ValueError("All model variants must use comparable input data for dashboard charts")
        _validate_metrics(model["metrics"], labels, model["label"])

    class_distribution = {
        "organization": int(baseline_manifest["class_distribution"]["organization"]),
        "user": int(baseline_manifest["class_distribution"]["user"]),
        "system": int(baseline_manifest["class_distribution"]["system"]),
    }
    split_sizes = {
        "train": int(baseline_manifest["splits"]["train"]["rows"]),
        "validation": int(baseline_manifest["splits"]["validation"]["rows"]),
        "test": int(baseline_manifest["splits"]["test"]["rows"]),
    }

    _plot_pie(
        class_distribution,
        "Figure 5: Phase 3 Dataset Class Distribution",
        output_dir / "fig-05-phase3-class-distribution.png",
    )
    _plot_pie(
        split_sizes,
        "Figure 6: Phase 3 Split Sizes",
        output_dir / "fig-06-phase3-split-sizes.png",
    )
    _plot_aggregate_comparison(
        models,
        "Figure 7: Phase 3 Four-Model Aggregate Comparison",
        output_dir / "fig-07-phase3-model-comparison-overall-4way.png",
    )
    _plot_macro_metric_heatmap(
        models,
        "Figure 8: Phase 3 Macro Metric Heatmap by Model",
        output_dir / "fig-08-phase3-macro-metric-heatmap-4way.png",
    )
    _plot_per_class_f1_comparison(
        models,
        labels,
        "Figure 9: Phase 3 Test Per-Class F1 by Model",
        output_dir / "fig-09-phase3-per-class-f1-4way.png",
    )
    _plot_confusion_small_multiples(
        models,
        labels,
        "Figure 10: Phase 3 Test Confusion Matrices (All Models)",
        output_dir / "fig-10-phase3-confusion-matrix-small-multiples-4way.png",
    )
    _plot_bayesian_intervals(
        models,
        labels,
        "Figure 11: Phase 3 Bayesian Posterior Means with 95% Intervals",
        output_dir / "fig-11-phase3-bayesian-posterior-intervals.png",
    )
    _plot_delta_vs_nb_heatmap(
        models,
        labels,
        "Figure 12: Delta vs Naive Bayes (Test Metrics)",
        output_dir / "fig-12-phase3-delta-vs-nb-heatmap.png",
    )
    _plot_reliability_curves(
        models,
        labels,
        "Figure 13: Reliability Curves (All Models)",
        output_dir / "fig-13-phase3-calibration-reliability-curves-4way.png",
    )
    _plot_ece_summary(
        models,
        labels,
        "Figure 14: Expected Calibration Error Summary",
        output_dir / "fig-14-phase3-expected-calibration-error-summary.png",
    )
    _plot_threshold_sensitivity(
        models,
        labels,
        "Figure 15: Threshold Sensitivity (User/System)",
        output_dir / "fig-15-phase3-threshold-sensitivity-user-system-4way.png",
    )
    _plot_bootstrap_confidence(
        models,
        labels,
        "Figure 16: Bootstrap Confidence Intervals (Held-Out Metrics)",
        output_dir / "fig-16-phase3-bootstrap-confidence-intervals-metrics.png",
    )

    run_history_rows = _load_run_history(artifacts_root / "phase3_run_history.jsonl")
    _plot_run_trends(
        models,
        run_history_rows,
        "Figure 17: Dated Run Trend Snapshots",
        output_dir / "fig-17-phase3-run-trend-snapshots-timeline.png",
    )

    generated_names = [
        "fig-05-phase3-class-distribution.png",
        "fig-06-phase3-split-sizes.png",
        "fig-07-phase3-model-comparison-overall-4way.png",
        "fig-08-phase3-macro-metric-heatmap-4way.png",
        "fig-09-phase3-per-class-f1-4way.png",
        "fig-10-phase3-confusion-matrix-small-multiples-4way.png",
        "fig-11-phase3-bayesian-posterior-intervals.png",
        "fig-12-phase3-delta-vs-nb-heatmap.png",
        "fig-13-phase3-calibration-reliability-curves-4way.png",
        "fig-14-phase3-expected-calibration-error-summary.png",
        "fig-15-phase3-threshold-sensitivity-user-system-4way.png",
        "fig-16-phase3-bootstrap-confidence-intervals-metrics.png",
        "fig-17-phase3-run-trend-snapshots-timeline.png",
    ]

    print("Generated Phase 3 figures:")
    for name in generated_names:
        print((output_dir / name).relative_to(repo_root))


if __name__ == "__main__":
    main()
