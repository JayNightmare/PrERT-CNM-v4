"""CNMv2-enabled Phase 3 pipeline shim.

Same design as CNMv1's pipeline.py: patch both `prert.phase3.classifier.
train_classifier` AND `prert.phase3.pipeline.train_classifier` (local
binding) so the CNMv2 wrapper is trained instead of the plain PrivBERT
model. Restores originals in a `finally` block.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

from prert.phase3 import classifier as _base_classifier
from prert.phase3 import pipeline as _base_pipeline
from prert.phase3.classifier import (
    DEFAULT_PRIVACYBERT_MODEL_NAME,
    TextClassifier,
)
from prert.phase3.cnm_v2.classifier_v2 import CNMv2Classifier, CNMv2TrainingConfig
from prert.phase3.pipeline import run_phase3_pipeline as _run_phase3_pipeline
from prert.phase3.types import ClauseExample

_LOGGER = logging.getLogger("prert.phase3.cnm_v2.pipeline_v2")


def run_phase3_cnmv2_pipeline(
    output_dir: Path,
    cnm_config: CNMv2TrainingConfig,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the Phase 3 pipeline with the CNMv2 classifier."""
    kwargs["model_type"] = "privacybert"

    def _cnmv2_train_classifier(
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
        privacybert_epochs: float = 3.0,
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
            "Training CNMv2Classifier (top_k=%d, index=%s, freeze_below_epoch=%d)",
            cnm_config.top_k,
            cnm_config.index_dir,
            cnm_config.freeze_retrieval_below_epoch,
        )
        model = CNMv2Classifier(
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
            "model_type": "cnmv2_privacybert",
            "training_examples": float(len(examples)),
            "vocabulary_size": 0.0,
            "labels": float(len(labels)),
            "backbone_model_name": privacybert_model_name,
            "cnm_index_dir": str(cnm_config.index_dir),
            "cnm_top_k": float(cnm_config.top_k),
            "cnm_freeze_below_epoch": float(cnm_config.freeze_retrieval_below_epoch),
            "cnm_consistency_weight": float(cnm_config.retrieval_consistency_weight),
        }
        return model, summary

    original_base_train = _base_classifier.train_classifier
    original_pipeline_train = _base_pipeline.train_classifier
    _base_classifier.train_classifier = _cnmv2_train_classifier  # type: ignore[assignment]
    _base_pipeline.train_classifier = _cnmv2_train_classifier  # type: ignore[assignment]
    _LOGGER.info(
        "Monkey-patched train_classifier in prert.phase3.pipeline for CNMv2 "
        "(id=%s -> id=%s)",
        id(original_pipeline_train),
        id(_cnmv2_train_classifier),
    )
    try:
        manifest = _run_phase3_pipeline(output_dir=output_dir, **kwargs)
    finally:
        _base_classifier.train_classifier = original_base_train  # type: ignore[assignment]
        _base_pipeline.train_classifier = original_pipeline_train  # type: ignore[assignment]

    manifest.setdefault("cnm", {}).update(
        {
            "enabled": cnm_config.top_k > 0,
            "config": cnm_config.as_dict(),
            "architecture": "cnmv2_auxiliary_head",
        }
    )
    return manifest


def run_cnmv2_ablation(
    output_root: Path,
    cnm_index_dir: Path,
    k_values: Sequence[int],
    common_cnm_args: Dict[str, Any] | None = None,
    **kwargs: Any,
) -> Dict[int, Dict[str, Any]]:
    import gc

    import torch

    common_cnm_args = dict(common_cnm_args or {})
    results: Dict[int, Dict[str, Any]] = {}
    for k in k_values:
        run_dir = output_root / f"k={k}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cnm_config = CNMv2TrainingConfig(
            index_dir=cnm_index_dir, top_k=int(k), **common_cnm_args
        )
        _LOGGER.info("=== CNMv2 ablation: k=%d, output=%s ===", k, run_dir)
        manifest = run_phase3_cnmv2_pipeline(
            output_dir=run_dir,
            cnm_config=cnm_config,
            **kwargs,
        )
        results[k] = manifest
        # Release GPU memory before loading the next fold's model; without
        # this, CUDA allocator fragmentation across folds has caused segfaults.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    return results
