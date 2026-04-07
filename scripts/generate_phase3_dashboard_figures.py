"""Generate static Phase 3 dashboard figures from artifact data."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Dict, List, Optional

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
    baseline = next(model for model in models if model["key"] == "nb")
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


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifacts_root = repo_root / "artifacts"
    output_dir = repo_root / "docs" / "Project" / "Execution-Playbook" / "figures"

    models = []
    for config in MODEL_CONFIG:
        artifact_dir = artifacts_root / config["artifact_dir"]
        manifest = _load_json(artifact_dir / "dataset_manifest.json")
        metrics = _load_json(artifact_dir / "classifier_metrics.json")
        bayesian_test = _load_json_if_exists(artifact_dir / "bayesian_risk_test.json")
        models.append(
            {
                "key": config["key"],
                "label": config["label"],
                "manifest": manifest,
                "metrics": metrics,
                "bayesian_test": bayesian_test,
            }
        )

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

    generated_names = [
        "fig-05-phase3-class-distribution.png",
        "fig-06-phase3-split-sizes.png",
        "fig-07-phase3-model-comparison-overall-4way.png",
        "fig-08-phase3-macro-metric-heatmap-4way.png",
        "fig-09-phase3-per-class-f1-4way.png",
        "fig-10-phase3-confusion-matrix-small-multiples-4way.png",
        "fig-11-phase3-bayesian-posterior-intervals.png",
        "fig-12-phase3-delta-vs-nb-heatmap.png",
    ]

    print("Generated Phase 3 figures:")
    for name in generated_names:
        print((output_dir / name).relative_to(repo_root))


if __name__ == "__main__":
    main()
