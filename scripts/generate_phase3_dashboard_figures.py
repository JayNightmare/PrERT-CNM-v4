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


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    phase3_dir = repo_root / "artifacts" / "phase-3"
    output_dir = repo_root / "docs" / "Project" / "Execution-Playbook" / "figures"

    dataset_manifest = _load_json(phase3_dir / "dataset_manifest.json")
    metrics = _load_json(phase3_dir / "classifier_metrics.json")

    class_distribution = {
        "organization": int(dataset_manifest["class_distribution"]["organization"]),
        "user": int(dataset_manifest["class_distribution"]["user"]),
        "system": int(dataset_manifest["class_distribution"]["system"]),
    }

    split_sizes = {
        "train": int(dataset_manifest["splits"]["train"]["rows"]),
        "validation": int(dataset_manifest["splits"]["validation"]["rows"]),
        "test": int(dataset_manifest["splits"]["test"]["rows"]),
    }

    labels = ["user", "system", "organization"]
    test_confusion = metrics["test"]["confusion"]

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
        test_confusion,
        labels,
        "Figure 7: Phase 3 Test Confusion Matrix",
        output_dir / "fig-07-phase3-test-confusion-matrix.png",
    )

    print("Generated Phase 3 figures:")
    for name in [
        "fig-05-phase3-class-distribution.png",
        "fig-06-phase3-split-sizes.png",
        "fig-07-phase3-test-confusion-matrix.png",
    ]:
        print((output_dir / name).relative_to(repo_root))


if __name__ == "__main__":
    main()
