"""Generate static Phase 3 dashboard figures from artifact data."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Dict, List

plt = importlib.import_module("matplotlib.pyplot")


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _plot_pie(values: Dict[str, int], title: str, out_path: Path) -> None:
    labels = list(values.keys())
    numbers = [int(values[label]) for label in labels]

    plt.figure(figsize=(7.5, 6.5))
    wedges, texts, autotexts = plt.pie(
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


def _plot_confusion_matrix(confusion: Dict[str, Dict[str, int]], labels: List[str], title: str, out_path: Path) -> None:
    matrix = [[int(confusion[actual][predicted]) for predicted in labels] for actual in labels]

    plt.figure(figsize=(7.5, 6.5))
    image = plt.imshow(matrix, cmap="Blues")
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)

    max_value = max(max(row) for row in matrix) if matrix else 1
    threshold = max_value * 0.45
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            color = "white" if value >= threshold else "black"
            plt.text(j, i, str(value), ha="center", va="center", color=color, fontsize=10)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()


def _plot_aggregate_comparison(
    baseline_metrics: Dict,
    upgraded_metrics: Dict,
    title: str,
    out_path: Path,
) -> None:
    labels = [
        "Validation Accuracy",
        "Validation Macro F1",
        "Test Accuracy",
        "Test Macro F1",
    ]
    baseline_values = [
        float(baseline_metrics["validation"]["accuracy"]),
        float(baseline_metrics["validation"]["macro_f1"]),
        float(baseline_metrics["test"]["accuracy"]),
        float(baseline_metrics["test"]["macro_f1"]),
    ]
    upgraded_values = [
        float(upgraded_metrics["validation"]["accuracy"]),
        float(upgraded_metrics["validation"]["macro_f1"]),
        float(upgraded_metrics["test"]["accuracy"]),
        float(upgraded_metrics["test"]["macro_f1"]),
    ]

    x_positions = list(range(len(labels)))
    width = 0.35

    plt.figure(figsize=(9, 6.5))
    baseline_bars = plt.bar(
        [x - width / 2 for x in x_positions],
        baseline_values,
        width,
        label="naive_bayes",
    )
    upgraded_bars = plt.bar(
        [x + width / 2 for x in x_positions],
        upgraded_values,
        width,
        label="logreg_tfidf",
    )

    for bars in (baseline_bars, upgraded_bars):
        for bar in bars:
            value = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.008,
                f"{value * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.ylim(0, 1.02)
    plt.ylabel("Score")
    plt.xticks(x_positions, labels, rotation=15)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()


def _plot_per_class_f1_comparison(
    baseline_metrics: Dict,
    upgraded_metrics: Dict,
    labels: List[str],
    title: str,
    out_path: Path,
) -> None:
    baseline_values = [float(baseline_metrics["test"]["per_class"][label]["f1"]) for label in labels]
    upgraded_values = [float(upgraded_metrics["test"]["per_class"][label]["f1"]) for label in labels]

    x_positions = list(range(len(labels)))
    width = 0.35

    plt.figure(figsize=(8.5, 6.5))
    baseline_bars = plt.bar(
        [x - width / 2 for x in x_positions],
        baseline_values,
        width,
        label="naive_bayes",
    )
    upgraded_bars = plt.bar(
        [x + width / 2 for x in x_positions],
        upgraded_values,
        width,
        label="logreg_tfidf",
    )

    for bars in (baseline_bars, upgraded_bars):
        for bar in bars:
            value = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.008,
                f"{value * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.ylim(0, 1.02)
    plt.ylabel("F1")
    plt.xticks(x_positions, labels)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    baseline_dir = repo_root / "artifacts" / "phase-3-nb"
    upgraded_dir = repo_root / "artifacts" / "phase-3-logreg"
    output_dir = repo_root / "docs" / "Project" / "Execution-Playbook" / "figures"

    baseline_manifest = _load_json(baseline_dir / "dataset_manifest.json")
    baseline_metrics = _load_json(baseline_dir / "classifier_metrics.json")
    upgraded_manifest = _load_json(upgraded_dir / "dataset_manifest.json")
    upgraded_metrics = _load_json(upgraded_dir / "classifier_metrics.json")

    if (
        int(baseline_manifest.get("total_rows", -1))
        != int(upgraded_manifest.get("total_rows", -1))
    ):
        raise ValueError("Baseline and upgraded runs must use comparable input data for dashboard charts")

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

    labels = ["user", "system", "organization"]
    baseline_test_confusion = baseline_metrics["test"]["confusion"]
    upgraded_test_confusion = upgraded_metrics["test"]["confusion"]

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
    _plot_confusion_matrix(
        baseline_test_confusion,
        labels,
        "Figure 7: Phase 3 Naive Bayes Test Confusion Matrix",
        output_dir / "fig-07-phase3-test-confusion-matrix.png",
    )
    _plot_aggregate_comparison(
        baseline_metrics,
        upgraded_metrics,
        "Figure 8: Phase 3 Baseline vs Upgraded Aggregate Metrics",
        output_dir / "fig-08-phase3-model-comparison-overall.png",
    )
    _plot_per_class_f1_comparison(
        baseline_metrics,
        upgraded_metrics,
        labels,
        "Figure 9: Phase 3 Test Per-Class F1 Comparison",
        output_dir / "fig-09-phase3-model-comparison-per-class-f1.png",
    )
    _plot_confusion_matrix(
        upgraded_test_confusion,
        labels,
        "Figure 10: Phase 3 LogReg TF-IDF Test Confusion Matrix",
        output_dir / "fig-10-phase3-logreg-test-confusion-matrix.png",
    )

    print("Generated Phase 3 figures:")
    for name in [
        "fig-05-phase3-class-distribution.png",
        "fig-06-phase3-split-sizes.png",
        "fig-07-phase3-test-confusion-matrix.png",
        "fig-08-phase3-model-comparison-overall.png",
        "fig-09-phase3-model-comparison-per-class-f1.png",
        "fig-10-phase3-logreg-test-confusion-matrix.png",
    ]:
        print((output_dir / name).relative_to(repo_root))


if __name__ == "__main__":
    main()
