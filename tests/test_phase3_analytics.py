import math

from prert.phase3.analytics import (
    compute_bootstrap_confidence_intervals,
    compute_calibration_report,
    compute_threshold_sweep,
)


def _sample_predictions() -> list[dict[str, object]]:
    return [
        {
            "actual_label": "user",
            "predicted_label": "user",
            "confidence": 0.92,
            "probabilities": {"user": 0.92, "system": 0.04, "organization": 0.04},
        },
        {
            "actual_label": "user",
            "predicted_label": "system",
            "confidence": 0.54,
            "probabilities": {"user": 0.38, "system": 0.54, "organization": 0.08},
        },
        {
            "actual_label": "system",
            "predicted_label": "system",
            "confidence": 0.88,
            "probabilities": {"user": 0.07, "system": 0.88, "organization": 0.05},
        },
        {
            "actual_label": "organization",
            "predicted_label": "organization",
            "confidence": 0.97,
            "probabilities": {"user": 0.02, "system": 0.01, "organization": 0.97},
        },
        {
            "actual_label": "organization",
            "predicted_label": "user",
            "confidence": 0.62,
            "probabilities": {"user": 0.62, "system": 0.18, "organization": 0.20},
        },
        {
            "actual_label": "system",
            "predicted_label": "organization",
            "confidence": 0.51,
            "probabilities": {"user": 0.19, "system": 0.30, "organization": 0.51},
        },
    ]


def test_compute_calibration_report_ranges() -> None:
    labels = ["user", "system", "organization"]
    report = compute_calibration_report(_sample_predictions(), labels=labels, num_bins=5)

    assert report["num_rows"] == 6
    assert report["num_bins"] == 5
    assert 0.0 <= float(report["overall"]["ece"]) <= 1.0
    assert 0.0 <= float(report["overall"]["brier"]) <= 1.0
    assert 0.0 <= float(report["macro_ece"]) <= 1.0
    assert len(report["overall"]["bins"]) == 5



def test_threshold_sweep_is_ordered_and_bounded() -> None:
    labels = ["user", "system", "organization"]
    sweep = compute_threshold_sweep(_sample_predictions(), labels=labels)

    assert sweep["focus_labels"] == ["user", "system"]
    for label in sweep["focus_labels"]:
        series = sweep["by_label"][label]
        thresholds = [float(row["threshold"]) for row in series]
        assert thresholds == sorted(thresholds)
        for row in series:
            assert 0.0 <= float(row["precision"]) <= 1.0
            assert 0.0 <= float(row["recall"]) <= 1.0
            assert 0.0 <= float(row["f1"]) <= 1.0



def test_bootstrap_intervals_are_valid() -> None:
    labels = ["user", "system", "organization"]
    report = compute_bootstrap_confidence_intervals(
        _sample_predictions(),
        labels=labels,
        n_resamples=200,
        seed=9,
    )

    assert report["n_rows"] == 6
    assert report["n_resamples"] == 200

    for key in ["accuracy", "macro_f1", "f1_user", "f1_system", "f1_organization"]:
        metric = report["metrics"][key]
        lower = float(metric["interval_95"]["lower"])
        mean = float(metric["mean"])
        upper = float(metric["interval_95"]["upper"])
        baseline = float(metric["baseline"])
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= mean <= 1.0
        assert 0.0 <= upper <= 1.0
        assert lower <= upper
        assert 0.0 <= baseline <= 1.0

        # Keep baseline reasonably near the bootstrap mean for this synthetic fixture.
        assert math.fabs(baseline - mean) <= 0.35
