"""Lightweight baseline text classifier for Phase 3."""

from __future__ import annotations

import json
import importlib
import math
import pickle
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Protocol, Sequence, Tuple

from prert.phase3.types import ClauseExample


TOKEN_PATTERN = re.compile(r"[a-z0-9]{2,}")
TFIDF_TOKEN_PATTERN = r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9']+\b"


class TextClassifier(Protocol):
    def fit(self, examples: Iterable[ClauseExample]) -> None: ...

    def predict(self, text: str) -> str: ...

    def predict_proba(self, text: str) -> Dict[str, float]: ...

    def save(self, path: Path) -> None: ...


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


class TfidfLogisticRegressionClassifier:
    """TF-IDF + class-weighted logistic regression model.

    This model is designed to improve minority-class discrimination compared to
    the Naive Bayes baseline while preserving deterministic behavior.
    """

    def __init__(
        self,
        labels: Sequence[str],
        random_state: int = 42,
        max_features: int = 20000,
        ngram_max: int = 2,
        min_df: int = 2,
        max_df: float = 0.95,
        c: float = 1.0,
        max_iter: int = 1000,
        class_weight: str | Dict[str, float] = "balanced",
    ) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Model type 'logreg_tfidf' requires scikit-learn. "
                "Install dependencies and re-run."
            ) from exc

        self.labels = list(labels)
        self.random_state = random_state
        self.max_features = max_features
        self.ngram_max = ngram_max
        self.min_df = min_df
        self.max_df = max_df
        self.c = c
        self.max_iter = max_iter
        self.class_weight = class_weight

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=TFIDF_TOKEN_PATTERN,
            max_features=max_features,
            ngram_range=(1, ngram_max),
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,
        )
        self.model = LogisticRegression(
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
            C=c,
            solver="lbfgs",
        )
        self.is_fit = False

    def fit(self, examples: Iterable[ClauseExample]) -> None:
        texts: List[str] = []
        labels: List[str] = []
        for example in examples:
            if example.label not in self.labels:
                continue
            text = (example.text or "").strip()
            if not text:
                continue
            texts.append(text)
            labels.append(example.label)

        if not texts:
            raise ValueError("No training examples available for logistic regression model")

        x_train = self.vectorizer.fit_transform(texts)
        self.model.fit(x_train, labels)
        self.is_fit = True

    def predict(self, text: str) -> str:
        if not self.is_fit:
            raise RuntimeError("Model is not fit")
        features = self.vectorizer.transform([text])
        prediction = self.model.predict(features)[0]
        return str(prediction)

    def predict_proba(self, text: str) -> Dict[str, float]:
        if not self.is_fit:
            raise RuntimeError("Model is not fit")
        features = self.vectorizer.transform([text])
        proba = self.model.predict_proba(features)[0]
        probabilities = {label: 0.0 for label in self.labels}
        for label, value in zip(self.model.classes_, proba):
            probabilities[str(label)] = float(value)
        total = sum(probabilities.values())
        if total <= 0:
            uniform = 1.0 / max(len(self.labels), 1)
            return {label: uniform for label in self.labels}
        return {label: probabilities[label] / total for label in self.labels}

    def save(self, path: Path) -> None:
        payload: Dict[str, Any] = {
            "labels": self.labels,
            "random_state": self.random_state,
            "max_features": self.max_features,
            "ngram_max": self.ngram_max,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "c": self.c,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
            "vectorizer": self.vectorizer,
            "model": self.model,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(payload, handle)


class PrivacyBertClassifier:
    """PrivacyBERT-style transformer classifier backend.

    This backend is optional and only activated when model_type=privacybert.
    It requires transformers, datasets, and torch to be installed.
    """

    def __init__(
        self,
        labels: Sequence[str],
        model_name: str = "bert-base-uncased",
        random_state: int = 42,
        num_train_epochs: float = 2.0,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        max_length: int = 256,
    ) -> None:
        try:
            torch = importlib.import_module("torch")
            datasets_module = importlib.import_module("datasets")
            transformers_module = importlib.import_module("transformers")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Model type 'privacybert' requires torch, datasets, and transformers. "
                "Install dependencies and re-run."
            ) from exc

        self.labels = list(labels)
        self.model_name = model_name
        self.random_state = random_state
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length

        self._torch = torch
        self._dataset_cls = datasets_module.Dataset
        self._trainer_cls = transformers_module.Trainer
        self._training_args_cls = transformers_module.TrainingArguments
        self._auto_tokenizer_cls = transformers_module.AutoTokenizer
        self._auto_model_cls = transformers_module.AutoModelForSequenceClassification

        self.label_to_id = {label: idx for idx, label in enumerate(self.labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

        self.tokenizer = self._auto_tokenizer_cls.from_pretrained(model_name)
        self.model = self._auto_model_cls.from_pretrained(
            model_name,
            num_labels=len(self.labels),
            label2id=self.label_to_id,
            id2label=self.id_to_label,
        )
        self.is_fit = False

    def fit(self, examples: Iterable[ClauseExample]) -> None:
        texts: List[str] = []
        labels: List[int] = []
        for example in examples:
            label = str(example.label).strip().lower()
            if label not in self.label_to_id:
                continue
            text = (example.text or "").strip()
            if not text:
                continue
            texts.append(text)
            labels.append(self.label_to_id[label])

        if not texts:
            raise ValueError("No training examples available for privacybert model")

        dataset = self._dataset_cls.from_dict({"text": texts, "label": labels})

        def _tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
            return self.tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )

        dataset = dataset.map(_tokenize, batched=True)
        dataset = dataset.remove_columns(["text"])
        dataset.set_format(type="torch")

        with tempfile.TemporaryDirectory(prefix="phase3-privacybert-") as tmpdir:
            training_args = self._training_args_cls(
                output_dir=tmpdir,
                # overwrite_output_dir=True,
                learning_rate=self.learning_rate,
                per_device_train_batch_size=self.batch_size,
                num_train_epochs=self.num_train_epochs,
                save_strategy="no",
                logging_strategy="no",
                report_to=[],
                seed=self.random_state,
                data_seed=self.random_state,
            )
            trainer = self._trainer_cls(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                # tokenizer=self.tokenizer,
            )
            trainer.train()
            self.model = trainer.model

        self.model.eval()
        self.is_fit = True

    def predict(self, text: str) -> str:
        probabilities = self.predict_proba(text)
        return max(probabilities.items(), key=lambda item: item[1])[0]

    def predict_proba(self, text: str) -> Dict[str, float]:
        if not self.is_fit:
            raise RuntimeError("Model is not fit")

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        with self._torch.no_grad():
            logits = self.model(**encoded).logits[0]
            probs = self._torch.softmax(logits, dim=-1).cpu().tolist()

        output = {label: 0.0 for label in self.labels}
        for idx, value in enumerate(probs):
            label = self.id_to_label.get(idx)
            if label is None:
                continue
            output[label] = float(value)

        total = sum(output.values())
        if total <= 0:
            uniform = 1.0 / max(len(self.labels), 1)
            return {label: uniform for label in self.labels}
        return {label: output[label] / total for label in self.labels}

    def save(self, path: Path) -> None:
        save_dir = path if path.suffix == "" else (path.parent / path.stem)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        metadata = {
            "labels": self.labels,
            "model_name": self.model_name,
            "random_state": self.random_state,
            "num_train_epochs": self.num_train_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_length": self.max_length,
        }
        with (save_dir / "training_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
            handle.write("\n")


def train_classifier(
    examples: Sequence[ClauseExample],
    labels: Sequence[str],
    output_path: Path,
    model_type: str = "naive_bayes",
    random_state: int = 42,
    max_features: int = 20000,
    ngram_max: int = 2,
    min_df: int = 2,
    max_df: float = 0.95,
    c: float = 1.0,
    max_iter: int = 1000,
    privacybert_model_name: str = "bert-base-uncased",
    privacybert_epochs: float = 2.0,
    privacybert_batch_size: int = 8,
    privacybert_learning_rate: float = 5e-5,
    privacybert_max_length: int = 256,
) -> Tuple[TextClassifier, Dict[str, float]]:
    selected_model = model_type.strip().lower()
    if selected_model in {"naive_bayes", "nb"}:
        model: TextClassifier = NaiveBayesTextClassifier(labels=labels)
        model_name = "multinomial_naive_bayes"
    elif selected_model in {"logreg_tfidf", "logistic_regression", "lr_tfidf"}:
        model = TfidfLogisticRegressionClassifier(
            labels=labels,
            random_state=random_state,
            max_features=max_features,
            ngram_max=ngram_max,
            min_df=min_df,
            max_df=max_df,
            c=c,
            max_iter=max_iter,
        )
        model_name = "logreg_tfidf"
    elif selected_model in {"privacybert", "privacy_bert", "bert_privacy"}:
        model = PrivacyBertClassifier(
            labels=labels,
            model_name=privacybert_model_name,
            random_state=random_state,
            num_train_epochs=privacybert_epochs,
            batch_size=privacybert_batch_size,
            learning_rate=privacybert_learning_rate,
            max_length=privacybert_max_length,
        )
        model_name = "privacybert"
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'")

    model.fit(examples)
    model.save(output_path)

    vocabulary_size = 0.0
    if isinstance(model, NaiveBayesTextClassifier):
        vocabulary_size = float(len(model.vocabulary))
    elif isinstance(model, TfidfLogisticRegressionClassifier):
        vocabulary_size = float(len(model.vectorizer.vocabulary_))

    summary = {
        "model_type": model_name,
        "training_examples": float(len(examples)),
        "vocabulary_size": vocabulary_size,
        "labels": float(len(labels)),
    }
    return model, summary


def tokenize(text: str) -> List[str]:
    lowered = text.lower()
    return TOKEN_PATTERN.findall(lowered)
