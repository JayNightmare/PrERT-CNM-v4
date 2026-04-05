"""Lightweight baseline text classifier for Phase 3."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from prert.phase3.types import ClauseExample


TOKEN_PATTERN = re.compile(r"[a-z0-9]{2,}")


class NaiveBayesTextClassifier:
    def __init__(self, labels: Sequence[str], alpha: float = 1.0) -> None:
        self.labels = list(labels)
        self.alpha = alpha
        self.class_doc_counts: Dict[str, int] = {label: 0 for label in self.labels}
        self.class_token_totals: Dict[str, int] = {label: 0 for label in self.labels}
        self.class_token_counts: Dict[str, Counter[str]] = {label: Counter() for label in self.labels}
        self.vocabulary: set[str] = set()
        self.total_docs = 0

    def fit(self, examples: Iterable[ClauseExample]) -> None:
        for example in examples:
            if example.label not in self.class_doc_counts:
                continue
            tokens = tokenize(example.text)
            if not tokens:
                continue
            self.total_docs += 1
            self.class_doc_counts[example.label] += 1
            self.class_token_counts[example.label].update(tokens)
            self.class_token_totals[example.label] += len(tokens)
            self.vocabulary.update(tokens)

    def predict(self, text: str) -> str:
        scores = self._class_log_scores(text)
        return max(scores.items(), key=lambda item: item[1])[0]

    def predict_proba(self, text: str) -> Dict[str, float]:
        log_scores = self._class_log_scores(text)
        max_log = max(log_scores.values())
        shifted = {label: math.exp(score - max_log) for label, score in log_scores.items()}
        total = sum(shifted.values())
        if total <= 0:
            uniform = 1.0 / max(len(self.labels), 1)
            return {label: uniform for label in self.labels}
        return {label: shifted[label] / total for label in self.labels}

    def save(self, path: Path) -> None:
        payload = {
            "labels": self.labels,
            "alpha": self.alpha,
            "class_doc_counts": self.class_doc_counts,
            "class_token_totals": self.class_token_totals,
            "class_token_counts": {label: dict(counter) for label, counter in self.class_token_counts.items()},
            "vocabulary": sorted(self.vocabulary),
            "total_docs": self.total_docs,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    @classmethod
    def load(cls, path: Path) -> "NaiveBayesTextClassifier":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        model = cls(labels=payload["labels"], alpha=float(payload["alpha"]))
        model.class_doc_counts = {k: int(v) for k, v in payload["class_doc_counts"].items()}
        model.class_token_totals = {k: int(v) for k, v in payload["class_token_totals"].items()}
        model.class_token_counts = {
            label: Counter({token: int(count) for token, count in counts.items()})
            for label, counts in payload["class_token_counts"].items()
        }
        model.vocabulary = set(payload.get("vocabulary", []))
        model.total_docs = int(payload.get("total_docs", sum(model.class_doc_counts.values())))
        return model

    def _class_log_scores(self, text: str) -> Dict[str, float]:
        tokens = tokenize(text)
        vocab_size = max(len(self.vocabulary), 1)
        denominator_by_class = {
            label: self.class_token_totals[label] + (self.alpha * vocab_size)
            for label in self.labels
        }

        log_scores: Dict[str, float] = {}
        for label in self.labels:
            prior_numerator = self.class_doc_counts[label] + self.alpha
            prior_denominator = self.total_docs + (self.alpha * len(self.labels))
            score = math.log(prior_numerator / max(prior_denominator, 1e-9))

            token_counter = self.class_token_counts[label]
            denominator = denominator_by_class[label]
            for token in tokens:
                numerator = token_counter[token] + self.alpha
                score += math.log(numerator / max(denominator, 1e-9))
            log_scores[label] = score

        return log_scores


def train_classifier(
    examples: Sequence[ClauseExample],
    labels: Sequence[str],
    output_path: Path,
) -> Tuple[NaiveBayesTextClassifier, Dict[str, float]]:
    model = NaiveBayesTextClassifier(labels=labels)
    model.fit(examples)
    model.save(output_path)

    summary = {
        "training_examples": float(len(examples)),
        "vocabulary_size": float(len(model.vocabulary)),
        "labels": float(len(labels)),
    }
    return model, summary


def tokenize(text: str) -> List[str]:
    lowered = text.lower()
    return TOKEN_PATTERN.findall(lowered)
