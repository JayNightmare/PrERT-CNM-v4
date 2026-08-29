from prert.phase3.analytics import compute_bootstrap_confidence_intervals

# Test with empty predictions
predictions_empty = []
result_empty = compute_bootstrap_confidence_intervals(
    predictions_empty,
    labels=["user", "system", "organization"],
    n_resamples=10,
    seed=42,
)
print("Empty result:", result_empty)

# Test with predictions from different labels
predictions_mixed = [
    {
        "actual_label": "user",
        "predicted_label": "user",
        "confidence": 0.9,
        "probabilities": {"user": 0.9, "system": 0.05, "organization": 0.05},
    },
    {
        "actual_label": "system",
        "predicted_label": "system",
        "confidence": 0.8,
        "probabilities": {"user": 0.1, "system": 0.8, "organization": 0.1},
    },
    {
        "actual_label": "organization",
        "predicted_label": "organization",
        "confidence": 0.7,
        "probabilities": {"user": 0.2, "system": 0.1, "organization": 0.7},
    },
]
result_mixed = compute_bootstrap_confidence_intervals(
    predictions_mixed,
    labels=["user", "system", "organization"],
    n_resamples=1000,
    seed=42,
)
print("Mixed result rows:", result_mixed["n_rows"])

# Test with unlabeled predictions (unstratified)
predictions_unlabeled = [
    {
        "actual_label": "unknown",
        "predicted_label": "user",
        "confidence": 0.9,
        "probabilities": {"user": 0.9, "system": 0.05, "organization": 0.05},
    },
    {
        "actual_label": "unknown",
        "predicted_label": "system",
        "confidence": 0.8,
        "probabilities": {"user": 0.1, "system": 0.8, "organization": 0.1},
    },
]
result_unlabeled = compute_bootstrap_confidence_intervals(
    predictions_unlabeled,
    labels=["user", "system", "organization"],
    n_resamples=1000,
    seed=42,
)
print("Unlabeled result rows:", result_unlabeled["n_rows"])
