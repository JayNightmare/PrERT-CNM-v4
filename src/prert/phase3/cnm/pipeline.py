"""CNM-enabled Phase 3 pipeline.

Reuses `run_phase3_pipeline` from the base package but swaps the classifier
factory so the CNM wrapper is used for training + inference. The full
freeze-artifact set (predictions, calibration, bootstrap, Bayesian scoring)
is produced unchanged, so downstream analysis scripts do not need to know
CNM was involved.

Implementation note: `prert.phase3.pipeline` binds `train_classifier` as a
module-level name at import time (`from prert.phase3.classifier import
train_classifier`). Patching only the source module (`prert.phase3.classifier`)
is a no-op because the pipeline module already holds its own reference. This
shim therefore patches the local binding inside `prert.phase3.pipeline`
directly, and restores it in a `finally` block.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from prert.phase3 import classifier as _base_classifier
from prert.phase3 import pipeline as _base_pipeline
from prert.phase3.classifier import (
    DEFAULT_PRIVACYBERT_MODEL_NAME,
    TextClassifier,
)
from prert.phase3.cnm.classifier import CNMConfig, CNMPrivacyBertClassifier
from prert.phase3.pipeline import run_phase3_pipeline as _run_phase3_pipeline
from prert.phase3.types import ClauseExample

_LOGGER = logging.getLogger(__name__)


def run_phase3_cnm_pipeline(
    output_dir: Path,
    cnm_config: CNMConfig,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the Phase 3 pipeline with the CNM classifier.

    All kwargs are forwarded to `run_phase3_pipeline`. `model_type` is forced
    to `privacybert` (the CNM wrapper subclasses that backend); we override
    the classifier factory for the duration of the call so the wrapper is
    trained instead of the plain PrivBERT model.
    """
    kwargs["model_type"] = "privacybert"

    original_train_classifier = _base_classifier.train_classifier

    def _cnm_train_classifier(
        examples: Sequence[ClauseExample],
        labels: Sequence[str],
        output_path: Path,
        model_type: str = "privacybert",
        random_state: int = 42,
        max_features: int = 20000,
        ngram_max: int = 2,
        min_df: int = 2,
        max_df: float = 0.95,
        c: float = 1.0,
        max_iter: int = 1000,
        privacybert_model_name: str = DEFAULT_PRIVACYBERT_MODEL_NAME,
        privacybert_epochs: float = 2.0,
        privacybert_batch_size: int = 8,
        privacybert_learning_rate: float = 5e-5,
        privacybert_max_length: int = 256,
        privacybert_loss_type: str = "focal",
        privacybert_focal_gamma: float = 2.0,
        privacybert_label_smoothing: float = 0.05,
        privacybert_weight_decay: float = 0.01,
        privacybert_warmup_steps: float = 0,
        privacybert_early_stopping_patience: int = 1,
        validation_examples: Sequence[ClauseExample] | None = None,
    ) -> Tuple[TextClassifier, Dict[str, float]]:
        _LOGGER.info(
            "Training CNMPrivacyBertClassifier (top_k=%d, index=%s)",
            cnm_config.top_k,
            cnm_config.index_dir,
        )
        model = CNMPrivacyBertClassifier(
            labels=labels,
            cnm_config=cnm_config,
            model_name=privacybert_model_name,
            random_state=random_state,
            num_train_epochs=privacybert_epochs,
            batch_size=privacybert_batch_size,
            learning_rate=privacybert_learning_rate,
            max_length=privacybert_max_length,
            loss_type=privacybert_loss_type,
            focal_gamma=privacybert_focal_gamma,
            label_smoothing_factor=privacybert_label_smoothing,
            weight_decay=privacybert_weight_decay,
            warmup_steps=privacybert_warmup_steps,
            early_stopping_patience=privacybert_early_stopping_patience,
        )
        model.fit(examples, validation_examples=validation_examples)
        model.save(output_path)
        summary = {
            "model_type": "cnm_privacybert",
            "training_examples": float(len(examples)),
            "vocabulary_size": 0.0,
            "labels": float(len(labels)),
            "backbone_model_name": privacybert_model_name,
            "cnm_index_dir": str(cnm_config.index_dir),
            "cnm_top_k": float(cnm_config.top_k),
        }
        return model, summary

    # Patch BOTH the source module attribute AND the local binding inside
    # `prert.phase3.pipeline`. The pipeline module holds its own reference
    # to `train_classifier` (bound at import time), so patching only the
    # source module is a silent no-op that leaves the CNM inactive.
    original_base_train = _base_classifier.train_classifier
    original_pipeline_train = _base_pipeline.train_classifier
    _base_classifier.train_classifier = _cnm_train_classifier  # type: ignore[assignment]
    _base_pipeline.train_classifier = _cnm_train_classifier  # type: ignore[assignment]
    _LOGGER.info(
        "Monkey-patched train_classifier in prert.phase3.pipeline (id=%s -> id=%s)",
        id(original_pipeline_train),
        id(_cnm_train_classifier),
    )
    try:
        manifest = _run_phase3_pipeline(output_dir=output_dir, **kwargs)
    finally:
        _base_classifier.train_classifier = original_base_train  # type: ignore[assignment]
        _base_pipeline.train_classifier = original_pipeline_train  # type: ignore[assignment]

    manifest.setdefault("cnm", {}).update(
        {
            "enabled": cnm_config.top_k > 0,
            "config": cnm_config.as_metadata_dict(),
        }
    )
    return manifest


def run_cnm_ablation(
    output_root: Path,
    cnm_index_dir: Path,
    k_values: Sequence[int],
    **kwargs: Any,
) -> Dict[int, Dict[str, Any]]:
    """Run the pipeline across a sweep of top_k values.

    For each k in k_values, writes to output_root/k={k}/ and returns a dict
    of k -> manifest. k=0 is treated as the plain-PrivBERT ablation baseline.
    """
    results: Dict[int, Dict[str, Any]] = {}
    for k in k_values:
        run_dir = output_root / f"k={k}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cnm_config = CNMConfig(index_dir=cnm_index_dir, top_k=int(k))
        _LOGGER.info("=== CNM ablation: k=%d, output=%s ===", k, run_dir)
        manifest = run_phase3_cnm_pipeline(
            output_dir=run_dir, cnm_config=cnm_config, **kwargs
        )
        results[k] = manifest
    return results
