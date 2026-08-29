"""CNM-augmented PrivBERT classifier.

Wraps `PrivacyBertClassifier` and prepends top-k retrieved controls to each
clause before tokenisation. The retrieval context is joined into a single
augmented string using `[SEP]` separators; token truncation is left to the
tokeniser (max_length preserved from the base classifier).

Retrieval traces are recorded per-example under `predict_proba_trace` so the
Bayesian aggregator can log which controls fired for each clause. The base
classifier interface (fit, predict, predict_proba, save) is preserved.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from prert.phase3.classifier import (
    DEFAULT_PRIVACYBERT_MODEL_NAME,
    PrivacyBertClassifier,
)
from prert.phase3.cnm.memory import MemoryIndex, load_memory_index
from prert.phase3.cnm.retriever import Retriever, RetrievedControl
from prert.phase3.types import ClauseExample

_LOGGER = logging.getLogger(__name__)


@dataclass
class CNMConfig:
    """CNM inference configuration.

    top_k=0 disables retrieval (equivalent to plain PrivBERT baseline).
    """

    index_dir: Path
    top_k: int = 5
    include_source_tag: bool = True
    max_control_chars: int = 300
    separator: str = " [SEP] "
    encoder_name_override: str | None = None
    # If true, retrieval happens for both fit and predict. If false, only predict.
    augment_training: bool = True
    trace_output_path: Path | None = None

    def as_metadata_dict(self) -> Dict[str, Any]:
        return {
            "index_dir": str(self.index_dir),
            "top_k": self.top_k,
            "include_source_tag": self.include_source_tag,
            "max_control_chars": self.max_control_chars,
            "encoder_name_override": self.encoder_name_override or "",
            "augment_training": self.augment_training,
        }


def _format_controls(retrieved: Sequence[RetrievedControl], config: CNMConfig) -> str:
    if not retrieved:
        return ""
    chunks: List[str] = []
    for r in retrieved:
        text = r.entry.text.strip().replace("\n", " ")
        if config.max_control_chars > 0 and len(text) > config.max_control_chars:
            text = text[: config.max_control_chars].rstrip() + "..."
        if config.include_source_tag and r.entry.source:
            chunks.append(f"[{r.entry.source}] {text}")
        else:
            chunks.append(text)
    return config.separator.join(chunks)


def augment_clause_text(
    clause_text: str,
    retrieved: Sequence[RetrievedControl],
    config: CNMConfig,
) -> str:
    context = _format_controls(retrieved, config)
    if not context:
        return clause_text
    return f"{clause_text}{config.separator}{context}"


class CNMPrivacyBertClassifier(PrivacyBertClassifier):
    """PrivBERT classifier with retrieval-augmented input.

    Behaves identically to the base class when config.top_k == 0.
    """

    def __init__(
        self,
        labels: Sequence[str],
        cnm_config: CNMConfig,
        model_name: str = DEFAULT_PRIVACYBERT_MODEL_NAME,
        random_state: int = 42,
        num_train_epochs: float = 2.0,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        max_length: int = 256,
        loss_type: str = "focal",
        focal_gamma: float = 2.0,
        label_smoothing_factor: float = 0.05,
        weight_decay: float = 0.01,
        warmup_steps: float = 0.1,
        early_stopping_patience: int = 1,
    ) -> None:
        super().__init__(
            labels=labels,
            model_name=model_name,
            random_state=random_state,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            label_smoothing_factor=label_smoothing_factor,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            early_stopping_patience=early_stopping_patience,
        )
        self.cnm_config = cnm_config
        self._index: MemoryIndex | None = None
        self._retriever: Retriever | None = None
        self._trace_records: List[Dict[str, Any]] = []
        self._traces_enabled = cnm_config.trace_output_path is not None

    # ---------------- Retrieval plumbing ----------------

    def _ensure_retriever(self) -> Retriever:
        if self._retriever is not None:
            return self._retriever
        _LOGGER.info("Loading CNM memory index from %s", self.cnm_config.index_dir)
        self._index = load_memory_index(self.cnm_config.index_dir)
        encoder_name = self.cnm_config.encoder_name_override or self._index.encoder_name
        if encoder_name != self._index.encoder_name:
            _LOGGER.warning(
                "Query encoder %s differs from index encoder %s; geometries may not align.",
                encoder_name,
                self._index.encoder_name,
            )
        from prert.phase3.cnm.embed import TextEncoder

        encoder = TextEncoder(model_name=encoder_name)
        self._retriever = Retriever(self._index, query_encoder=encoder)
        return self._retriever

    def _augment(self, text: str, example_id: str = "", split: str = "predict") -> str:
        if self.cnm_config.top_k <= 0:
            return text
        retriever = self._ensure_retriever()
        retrieved = retriever.retrieve(text, self.cnm_config.top_k)
        if self._traces_enabled:
            self._trace_records.append(
                {
                    "example_id": example_id,
                    "split": split,
                    "clause_text": text,
                    "k": self.cnm_config.top_k,
                    "retrieved": [
                        {
                            "control_id": r.entry.control_id,
                            "source": r.entry.source,
                            "section": r.entry.section,
                            "score": r.score,
                            "rank": r.rank,
                        }
                        for r in retrieved
                    ],
                }
            )
        return augment_clause_text(text, retrieved, self.cnm_config)

    # ---------------- fit / predict overrides ----------------

    def fit(
        self,
        examples: Iterable[ClauseExample],
        validation_examples: Iterable[ClauseExample] | None = None,
    ) -> None:
        examples = list(examples)
        val_examples = (
            list(validation_examples) if validation_examples is not None else None
        )

        if self.cnm_config.augment_training and self.cnm_config.top_k > 0:
            _LOGGER.info(
                "Augmenting %d training examples with top-%d retrieval",
                len(examples),
                self.cnm_config.top_k,
            )
            augmented_train: List[ClauseExample] = []
            for ex in examples:
                augmented_train.append(
                    ClauseExample(
                        example_id=ex.example_id,
                        text=self._augment(
                            ex.text, example_id=ex.example_id, split="train"
                        ),
                        label=ex.label,
                        source=ex.source,
                        policy_uid=ex.policy_uid,
                        category=ex.category,
                        metadata={**ex.metadata, "cnm_augmented": True},
                    )
                )
            examples = augmented_train

            if val_examples is not None:
                augmented_val: List[ClauseExample] = []
                for ex in val_examples:
                    augmented_val.append(
                        ClauseExample(
                            example_id=ex.example_id,
                            text=self._augment(
                                ex.text, example_id=ex.example_id, split="validation"
                            ),
                            label=ex.label,
                            source=ex.source,
                            policy_uid=ex.policy_uid,
                            category=ex.category,
                            metadata={**ex.metadata, "cnm_augmented": True},
                        )
                    )
                val_examples = augmented_val

        super().fit(examples, validation_examples=val_examples)
        self._flush_traces()

    def predict(self, text: str) -> str:  # type: ignore[override]
        return super().predict(self._augment(text, split="predict"))

    def predict_proba(self, text: str) -> Dict[str, float]:  # type: ignore[override]
        return super().predict_proba(self._augment(text, split="predict"))

    def predict_with_trace(self, text: str, example_id: str = "") -> Dict[str, Any]:
        """Return probabilities alongside the retrieved control ids."""
        retriever = self._ensure_retriever()
        retrieved: List[RetrievedControl] = []
        if self.cnm_config.top_k > 0:
            retrieved = retriever.retrieve(text, self.cnm_config.top_k)
        augmented = augment_clause_text(text, retrieved, self.cnm_config)
        probs = super().predict_proba(augmented)
        return {
            "example_id": example_id,
            "probabilities": probs,
            "retrieved": [
                {
                    "control_id": r.entry.control_id,
                    "source": r.entry.source,
                    "section": r.entry.section,
                    "score": r.score,
                    "rank": r.rank,
                }
                for r in retrieved
            ],
        }

    def _flush_traces(self) -> None:
        if not self._traces_enabled or not self._trace_records:
            return
        path = self.cnm_config.trace_output_path
        assert path is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in self._trace_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        _LOGGER.info("Wrote %d CNM trace records to %s", len(self._trace_records), path)

    def save(self, path: Path) -> None:  # type: ignore[override]
        super().save(path)
        save_dir = path if path.suffix == "" else (path.parent / path.stem)
        with (save_dir / "cnm_config.json").open("w", encoding="utf-8") as handle:
            json.dump(
                self.cnm_config.as_metadata_dict(), handle, indent=2, ensure_ascii=False
            )
            handle.write("\n")
