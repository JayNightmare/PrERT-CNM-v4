from prert.phase3.analytics import compute_bootstrap_confidence_intervals

predictions = [
    {
        "actual_label": "user",
        "predicted_label": "user",
        "confidence": 0.9,
        "probabilities": {"user": 0.9, "system": 0.05, "organization": 0.05},
    },
    {
        "actual_label": "user",
        "predicted_label": "user",
        "confidence": 0.85,
        "probabilities": {"user": 0.85, "system": 0.1, "organization": 0.05},
    },
]
try:
    result = compute_bootstrap_confidence_intervals(
        predictions, labels=["user", "system", "organization"], n_resamples=10, seed=42
    )
    print("Success:", result["n_rows"])
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
