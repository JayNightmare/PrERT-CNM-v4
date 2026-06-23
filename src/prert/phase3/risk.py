"""Bayesian risk scoring helpers for Phase 3 outputs.

The classifier predicts clause-level labels and confidences. This module maps those
predictions into a lightweight Bayesian posterior risk summary across user/system/
organization levels, including uncertainty intervals and top contributing clauses.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

_LOGGER = logging.getLogger(__name__)


LEVELS: tuple[str, str, str] = ("user", "system", "organization")

DEFAULT_PRIORS: Dict[str, Dict[str, float]] = {
    "user": {"alpha": 1.5, "beta": 2.5},
    "system": {"alpha": 1.4, "beta": 2.6},
    "organization": {"alpha": 1.2, "beta": 2.8},
}

LEVEL_INDICATORS: Dict[str, Sequence[str]] = {
    "user": (
        "consent_control",
        "user_access_deletion",
    ),
    "system": (
        "security_controls",
        "tracking_controls",
    ),
    "organization": (
        "collection_use",
        "third_party_sharing",
        "retention_governance",
    ),
}


def load_bayesian_priors(path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    if path is None:
        return {level: dict(values) for level, values in DEFAULT_PRIORS.items()}

    payload = json.loads(path.read_text(encoding="utf-8"))
    priors: Dict[str, Dict[str, float]] = {}

    for level in LEVELS:
        source = payload.get(level, {})
        alpha = _positive_float(source.get("alpha"), DEFAULT_PRIORS[level]["alpha"])
        beta = _positive_float(source.get("beta"), DEFAULT_PRIORS[level]["beta"])
        priors[level] = {"alpha": alpha, "beta": beta}

    return priors


def compute_bayesian_risk(
    predictions: Sequence[Mapping[str, Any]],
    priors: Optional[Mapping[str, Mapping[str, float]]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    selected_priors = _normalize_priors(priors)
    _validate_priors_against_observations(selected_priors, predictions)

    # Aggregate posteriors using the model's predicted_label / confidence.
    predicted_view = _aggregate_levels(predictions, selected_priors, top_k, key="predicted_label")

    # C6: also produce a parallel "risk on actual labels" view so we can
    # detect divergence between the model's posterior and the ground truth
    # without re-running the whole pipeline.
    actual_view = _aggregate_levels(predictions, selected_priors, top_k, key="actual_label")

    predicted_view["actual_label_view"] = {
        "levels": actual_view["levels"],
        "overall": actual_view["overall"],
        "description": (
            "Posteriors recomputed using actual_label with confidence=1.0 for "
            "each clause; useful as a ground-truth reference for the predicted view."
        ),
    }
    return predicted_view


def _aggregate_levels(
    predictions: Sequence[Mapping[str, Any]],
    selected_priors: Mapping[str, Mapping[str, float]],
    top_k: int,
    key: str,
) -> Dict[str, Any]:
    level_stats: Dict[str, Dict[str, Any]] = {
        level: {
            "level": level,
            "prior": dict(selected_priors[level]),
            "alpha": float(selected_priors[level]["alpha"]),
            "beta": float(selected_priors[level]["beta"]),
            "evidence_count": 0,
            "contributors": [],
        }
        for level in LEVELS
    }

    for row in predictions:
        label = str(row.get(key, "")).strip().lower()
        if label not in level_stats:
            continue

        if key == "actual_label":
            confidence = 1.0
        else:
            confidence = _bounded_probability(row.get("confidence", 0.0))
        stats = level_stats[label]
        stats["alpha"] += confidence
        stats["beta"] += (1.0 - confidence)
        stats["evidence_count"] += 1
        stats["contributors"].append(
            {
                "example_id": row.get("example_id", ""),
                "policy_uid": row.get("policy_uid", ""),
                "actual_label": row.get("actual_label", ""),
                "predicted_label": row.get("predicted_label", ""),
                "confidence": round(confidence, 6),
                "text": row.get("text", ""),
            }
        )

    by_level: Dict[str, Any] = {}
    overall_weighted_sum = 0.0
    overall_weight = 0

    for level in LEVELS:
        stats = level_stats[level]
        alpha = float(stats["alpha"])
        beta = float(stats["beta"])
        mean = _beta_mean(alpha, beta)
        lower, upper = _beta_interval(alpha, beta)
        evidence_count = int(stats["evidence_count"])

        top_contributors = sorted(
            stats["contributors"],
            key=lambda item: float(item["confidence"]),
            reverse=True,
        )[: max(top_k, 1)]

        by_level[level] = {
            "level": level,
            "indicators": list(LEVEL_INDICATORS[level]),
            "prior": stats["prior"],
            "posterior": {
                "alpha": round(alpha, 6),
                "beta": round(beta, 6),
                "mean": round(mean, 6),
                "interval_95": {
                    "lower": round(lower, 6),
                    "upper": round(upper, 6),
                },
            },
            "evidence_count": evidence_count,
            "top_contributors": top_contributors,
        }

        if evidence_count > 0:
            overall_weighted_sum += mean * evidence_count
            overall_weight += evidence_count

    if overall_weight > 0:
        overall_mean = overall_weighted_sum / overall_weight
    else:
        overall_mean = _mean([_beta_mean(by_level[level]["posterior"]["alpha"], by_level[level]["posterior"]["beta"]) for level in LEVELS])

    return {
        "method": "beta_posterior_evidence_aggregation",
        "levels": by_level,
        "overall": {
            "posterior_mean": round(float(overall_mean), 6),
            "evidence_count": int(overall_weight),
            "primary_score": round(float(overall_mean), 6),
        },
        "primary_metric": "overall.posterior_mean",
    }


def _normalize_priors(
    priors: Optional[Mapping[str, Mapping[str, float]]],
) -> Dict[str, Dict[str, float]]:
    if priors is None:
        return {level: dict(values) for level, values in DEFAULT_PRIORS.items()}

    normalized: Dict[str, Dict[str, float]] = {}
    for level in LEVELS:
        source = priors.get(level, {})
        alpha = _positive_float(source.get("alpha"), DEFAULT_PRIORS[level]["alpha"])
        beta = _positive_float(source.get("beta"), DEFAULT_PRIORS[level]["beta"])
        normalized[level] = {"alpha": alpha, "beta": beta}
    return normalized


def _validate_priors_against_observations(
    priors: Mapping[str, Mapping[str, float]],
    predictions: Sequence[Mapping[str, Any]],
) -> None:
    """Warn when priors look incompatible with the observed label mix.

    C6: the freeze artifacts use a fixed Beta(alpha, beta) prior per level.
    If the supplied priors imply a posterior mean that's wildly different
    from the empirical actual-label rate (>0.4 absolute gap) we surface a
    warning so reviewers can re-tune the priors instead of silently
    publishing a posterior dominated by the prior.
    """
    if not predictions:
        return

    counts: Dict[str, int] = {level: 0 for level in LEVELS}
    total = 0
    for row in predictions:
        actual = str(row.get("actual_label", "")).strip().lower()
        if actual in counts:
            counts[actual] += 1
            total += 1
    if total == 0:
        return

    for level, params in priors.items():
        alpha = float(params.get("alpha", 0.0))
        beta = float(params.get("beta", 0.0))
        if alpha + beta <= 0:
            continue
        prior_mean = alpha / (alpha + beta)
        empirical = counts[level] / total
        if abs(prior_mean - empirical) > 0.4:
            _LOGGER.warning(
                "Bayesian prior for level=%s (mean=%.3f) deviates from empirical actual-label rate (%.3f) by %.3f; "
                "consider retuning priors before freezing.",
                level,
                prior_mean,
                empirical,
                abs(prior_mean - empirical),
            )


def _positive_float(value: Any, fallback: float) -> float:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    if candidate <= 0:
        return float(fallback)
    return candidate


def _bounded_probability(value: Any) -> float:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, candidate))


def _beta_mean(alpha: float, beta: float) -> float:
    denominator = alpha + beta
    if denominator <= 0:
        return 0.0
    return alpha / denominator


def _beta_interval(alpha: float, beta: float, z: float = 1.959964) -> tuple[float, float]:
    denominator = alpha + beta
    if denominator <= 0:
        return 0.0, 1.0

    mean = _beta_mean(alpha, beta)
    variance = (alpha * beta) / ((denominator ** 2) * (denominator + 1.0))
    std = math.sqrt(max(variance, 0.0))

    lower = max(0.0, mean - (z * std))
    upper = min(1.0, mean + (z * std))
    return lower, upper


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
