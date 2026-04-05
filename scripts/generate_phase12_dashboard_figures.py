"""Generate static Phase 1/2 dashboard figures from artifact data."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt


def _count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _load_phase2_manifest(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_risk_band_counts(path: Path) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("row_type") != "metric":
                continue
            counts[row.get("risk_band", "unknown")] += 1

    ordered_keys = ["low", "medium", "high", "unknown"]
    return {key: counts[key] for key in ordered_keys if counts[key] > 0}


def _plot_bar(values: Dict[str, int], title: str, ylabel: str, out_path: Path) -> None:
    labels = list(values.keys())
    numbers = list(values.values())

    plt.figure(figsize=(9, 5.5))
    bars = plt.bar(labels, numbers, color=["#2A9D8F", "#457B9D", "#E76F51", "#6C757D"][: len(labels)])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.25)

    for bar, value in zip(bars, numbers):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    phase1_dir = repo_root / "artifacts" / "phase-1"
    phase2_dir = repo_root / "artifacts" / "phase-2"
    output_dir = repo_root / "docs" / "Project" / "Execution-Playbook" / "figures"

    controls = {
        "GDPR": _count_jsonl_rows(phase1_dir / "controls_gdpr.jsonl"),
        "ISO 27001": _count_jsonl_rows(phase1_dir / "controls_iso27001.jsonl"),
        "NIST PF 1.1": _count_jsonl_rows(phase1_dir / "controls_nistpf.jsonl"),
    }

    chunks = {
        "GDPR": _count_jsonl_rows(phase1_dir / "chunks_gdpr.jsonl"),
        "ISO 27001": _count_jsonl_rows(phase1_dir / "chunks_iso27001.jsonl"),
        "NIST PF 1.1": _count_jsonl_rows(phase1_dir / "chunks_nistpf.jsonl"),
    }

    manifest = _load_phase2_manifest(phase2_dir / "phase2_manifest.json")
    levels = {
        "User": int(manifest["coverage_summary"]["level_counts"]["user"]),
        "System": int(manifest["coverage_summary"]["level_counts"]["system"]),
        "Organization": int(manifest["coverage_summary"]["level_counts"]["organization"]),
    }

    risk_bands = _load_risk_band_counts(phase2_dir / "baseline_scores.jsonl")

    _plot_bar(
        controls,
        "Figure 1: Phase 1 Controls by Regulation",
        "Control Count",
        output_dir / "fig-01-phase1-controls-by-regulation.png",
    )
    _plot_bar(
        chunks,
        "Figure 2: Phase 1 Chunks by Regulation",
        "Chunk Count",
        output_dir / "fig-02-phase1-chunks-by-regulation.png",
    )
    _plot_bar(
        levels,
        "Figure 3: Phase 2 Metric Levels",
        "Metric Count",
        output_dir / "fig-03-phase2-metric-levels.png",
    )
    _plot_bar(
        risk_bands,
        "Figure 4: Phase 2 Risk Bands",
        "Metric Row Count",
        output_dir / "fig-04-phase2-risk-bands.png",
    )

    print("Generated figures:")
    for path in sorted(output_dir.glob("fig-*.png")):
        print(path.relative_to(repo_root))


if __name__ == "__main__":
    main()
