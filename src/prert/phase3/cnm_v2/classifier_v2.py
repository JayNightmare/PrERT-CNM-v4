"""CNMv2 classifier: encoder + text head + retrieval head + gate.

Drop-in replacement for CNMv1's `CNMPrivacyBertClassifier`. Same public
API (`fit`, `predict`, `predict_proba`, `save`) so the existing Phase 3
pipeline shim reuses this transparently.

Training-time retrieval precompute:
    1. Load ChromaMemoryIndex → get the (M, D) embedding matrix
    2. Encode all training clauses with the same sentence-transformer once
    3. Batched cosine top-k → (N, k) index tensor + (N, k) score tensor
    4. Pass those tensors through the Dataset so each training batch
       carries its own retrieval slice; no per-step retrieval calls

This decouples training throughput from Chroma latency. Chroma is used
at inference time and for the traceability trace file.
"""

from __future__ import annotations

import importlib
import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from prert.phase3.classifier import (
    DEFAULT_PRIVACYBERT_MODEL_NAME,
    _PrivacyBertTrainingDataset,
    _patch_multiprocess_resource_tracker,
)
from prert.phase3.cnm_v2.memory_chroma import ChromaMemoryIndex
from prert.phase3.cnm_v2.model import CNMv2Config, CNMv2Model
from prert.phase3.types import ClauseExample

_LOGGER = logging.getLogger(__name__)


@dataclass
class CNMv2TrainingConfig:
    """CNMv2-specific hyperparameters (in addition to the base PrivBERT ones)."""

    index_dir: Path
    top_k: int = 5
    softmax_temperature: float = 1.0
    gate_hidden: int = 128
    freeze_retrieval_below_epoch: int = 0
    # If > 0, freezes retrieval head + gate for the first N epochs so the
    # text head can train cleanly (=baseline) before retrieval joins in.
    # 0 means retrieval participates from step 0 (safe due to gate init).

    retrieval_consistency_weight: float = 0.1
    # Weight on the auxiliary loss that pushes L[control] toward the
    # empirical class distribution of clauses that retrieved that control
    # in the top-k. Small value → gentle regularisation, not the main
    # training signal. 0 disables.

    encoder_batch_size: int = 32
    encoder_device: Optional[str] = None
    encoder_name_override: Optional[str] = None

    trace_output_path: Optional[Path] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index_dir": str(self.index_dir),
            "top_k": int(self.top_k),
            "softmax_temperature": float(self.softmax_temperature),
            "gate_hidden": int(self.gate_hidden),
            "freeze_retrieval_below_epoch": int(self.freeze_retrieval_below_epoch),
            "retrieval_consistency_weight": float(self.retrieval_consistency_weight),
            "encoder_batch_size": int(self.encoder_batch_size),
            "encoder_name_override": self.encoder_name_override or "",
        }


class _CNMv2TrainingDataset:
    """Wraps _PrivacyBertTrainingDataset so each item also carries retrieval
    indices + scores for its example."""

    def __init__(
        self,
        base_dataset: _PrivacyBertTrainingDataset,
        retrieved_indices: np.ndarray,  # (N, k) int64
        retrieved_scores: np.ndarray,  # (N, k) float32
    ) -> None:
        assert len(base_dataset) == retrieved_indices.shape[0]
        assert retrieved_indices.shape == retrieved_scores.shape
        self.base = base_dataset
        self.retrieved_indices = retrieved_indices
        self.retrieved_scores = retrieved_scores

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.base[index]
        # The HF collator will stack these tensors along dim 0.
        import torch

        item["retrieved_indices"] = torch.tensor(
            self.retrieved_indices[index], dtype=torch.long
        )
        item["retrieved_scores"] = torch.tensor(
            self.retrieved_scores[index], dtype=torch.float32
        )
        return item


class CNMv2Classifier:
    """PrivBERT + auxiliary retrieval head with gated combination.

    Public API mirrors PrivacyBertClassifier so the pipeline shim needs no
    changes beyond swapping which class it instantiates.
    """

    def __init__(
        self,
        labels: Sequence[str],
        cnm_config: CNMv2TrainingConfig,
        model_name: str = DEFAULT_PRIVACYBERT_MODEL_NAME,
        random_state: int = 42,
        num_train_epochs: float = 3.0,
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
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "CNMv2 requires torch and transformers. Install and retry."
            ) from exc

        _patch_multiprocess_resource_tracker()

        self._torch = torch
        self._transformers = transformers
        self.labels = list(labels)
        self.label_to_id = {l: i for i, l in enumerate(self.labels)}
        self.id_to_label = {i: l for l, i in self.label_to_id.items()}

        self.cnm_config = cnm_config
        self.model_name = model_name
        self.random_state = random_state
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.loss_type = loss_type
        self.focal_gamma = float(focal_gamma)
        self.label_smoothing_factor = float(label_smoothing_factor)
        self.weight_decay = float(weight_decay)
        self.warmup_steps = float(warmup_steps)
        self.early_stopping_patience = int(early_stopping_patience)

        # We use the RobertaModel *backbone* (no classification head). Our
        # CNMv2Model provides its own text head.
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.encoder_backbone = transformers.AutoModel.from_pretrained(model_name)

        self._memory: Optional[ChromaMemoryIndex] = None
        self._retrieval_encoder = None
        self._model: Optional[CNMv2Model] = None
        self.is_fit = False

    # ------------------------------------------------------------------
    # Retrieval precompute
    # ------------------------------------------------------------------
    def _ensure_memory(self) -> ChromaMemoryIndex:
        if self._memory is None:
            self._memory = ChromaMemoryIndex(persist_dir=self.cnm_config.index_dir)
            self._memory.load()
        return self._memory

    def _ensure_retrieval_encoder(self):
        if self._retrieval_encoder is None:
            from prert.phase3.cnm.embed import TextEncoder

            encoder_name = (
                self.cnm_config.encoder_name_override
                or self._ensure_memory().encoder_name
            )
            self._retrieval_encoder = TextEncoder(
                model_name=encoder_name,
                device=self.cnm_config.encoder_device,
            )
        return self._retrieval_encoder

    def _precompute_retrievals(
        self,
        texts: Sequence[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode `texts` and batched-top-k against the memory index."""
        memory = self._ensure_memory()
        encoder = self._ensure_retrieval_encoder()
        _LOGGER.info(
            "Precomputing top-%d retrievals for %d clauses (memory=%d)",
            self.cnm_config.top_k,
            len(texts),
            len(memory),
        )
        query_matrix = encoder.encode(
            texts,
            batch_size=self.cnm_config.encoder_batch_size,
            normalise=True,
        )
        indices, scores = memory.batch_top_k(query_matrix, k=self.cnm_config.top_k)
        return indices, scores

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(
        self,
        examples: Iterable[ClauseExample],
        validation_examples: Optional[Iterable[ClauseExample]] = None,
    ) -> None:
        torch = self._torch
        transformers = self._transformers

        # --- extract text/label
        train_texts: List[str] = []
        train_labels: List[int] = []
        for ex in examples:
            label = str(ex.label).strip().lower()
            if label not in self.label_to_id:
                continue
            text = (ex.text or "").strip()
            if not text:
                continue
            train_texts.append(text)
            train_labels.append(self.label_to_id[label])
        if not train_texts:
            raise ValueError("No training examples")

        # --- precompute retrievals for training set
        train_ret_idx, train_ret_scores = self._precompute_retrievals(train_texts)

        # --- tokenise
        train_enc = self.tokenizer(
            list(train_texts),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None,
        )
        train_base = _PrivacyBertTrainingDataset(
            encodings=train_enc, labels=train_labels
        )
        train_ds = _CNMv2TrainingDataset(train_base, train_ret_idx, train_ret_scores)

        # --- validation set
        eval_ds = None
        if validation_examples is not None:
            val_texts: List[str] = []
            val_labels: List[int] = []
            for ex in validation_examples:
                label = str(ex.label).strip().lower()
                if label not in self.label_to_id:
                    continue
                text = (ex.text or "").strip()
                if not text:
                    continue
                val_texts.append(text)
                val_labels.append(self.label_to_id[label])
            if val_texts:
                val_ret_idx, val_ret_scores = self._precompute_retrievals(val_texts)
                val_enc = self.tokenizer(
                    list(val_texts),
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors=None,
                )
                val_base = _PrivacyBertTrainingDataset(
                    encodings=val_enc, labels=val_labels
                )
                eval_ds = _CNMv2TrainingDataset(val_base, val_ret_idx, val_ret_scores)

        # --- build CNMv2 model wrapping the backbone
        memory = self._ensure_memory()
        cnm_arch_config = CNMv2Config(
            hidden_size=self.encoder_backbone.config.hidden_size,
            num_labels=len(self.labels),
            memory_size=len(memory),
            top_k=self.cnm_config.top_k,
            softmax_temperature=self.cnm_config.softmax_temperature,
            gate_hidden=self.cnm_config.gate_hidden,
            freeze_retrieval_below_epoch=self.cnm_config.freeze_retrieval_below_epoch,
        )
        self._model = CNMv2Model(encoder=self.encoder_backbone, config=cnm_arch_config)

        # --- class weights for focal / weighted CE
        class_counts = [0] * len(self.labels)
        for i in train_labels:
            class_counts[i] += 1
        total = sum(class_counts)
        n_classes = len(self.labels)
        class_weights = [total / (n_classes * max(c, 1)) for c in class_counts]
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32)

        # --- empirical class distribution per retrieved control
        # For the retrieval consistency loss: for each control j, compute the
        # empirical class distribution over training clauses that retrieved
        # j in their top-k. This gives us a target for L[j]. Precomputed
        # once here from train_ret_idx + train_labels.
        empirical_control_dist = self._compute_control_class_distribution(
            train_ret_idx,
            np.array(train_labels),
            n_controls=len(memory),
            n_classes=n_classes,
        )
        empirical_control_dist_t = torch.tensor(
            empirical_control_dist, dtype=torch.float32
        )

        loss_type = self.loss_type
        focal_gamma = self.focal_gamma
        retrieval_consistency_weight = self.cnm_config.retrieval_consistency_weight
        model_ref = self._model

        # --- custom Trainer subclass with dual loss
        class CNMv2Trainer(transformers.Trainer):  # type: ignore[misc]
            def compute_loss(self, model, inputs, return_outputs=False, **_kwargs):  # type: ignore[no-untyped-def]
                labels_ = inputs.pop("labels")
                out = model(**inputs, labels=labels_)
                logits = out.logits

                weights = class_weights_t.to(logits.device)

                # Primary task loss on the combined logits.
                if loss_type == "focal":
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    probs = log_probs.exp()
                    target_idx = labels_.long().unsqueeze(1)
                    true_log = log_probs.gather(1, target_idx).squeeze(1)
                    true_p = probs.gather(1, target_idx).squeeze(1)
                    alpha_t = weights[labels_.long()]
                    focal = (1.0 - true_p).pow(focal_gamma)
                    task_loss = -(alpha_t * focal * true_log).mean()
                else:
                    loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
                    task_loss = loss_fct(
                        logits.view(-1, logits.size(-1)), labels_.view(-1)
                    )

                loss = task_loss

                # Retrieval consistency loss: KL from softmax(L[j]) to the
                # empirical class distribution of control j. Applied to
                # controls that appear in the current batch's retrievals.
                if (
                    retrieval_consistency_weight > 0
                    and not model_ref.retrieval_frozen()
                ):
                    retrieved_idx = inputs.get("retrieved_indices")
                    if retrieved_idx is not None:
                        # Get unique control indices in this batch
                        flat_idx = retrieved_idx.reshape(-1).unique()
                        target = empirical_control_dist_t.to(logits.device)[
                            flat_idx
                        ]  # (U, C)
                        pred_logits = model_ref.control_logits[flat_idx]  # (U, C)
                        pred_log_probs = torch.nn.functional.log_softmax(
                            pred_logits, dim=-1
                        )
                        # KL(target || pred): target is fixed empirical, pred is learned.
                        # KL = Σ target * (log target - log pred)
                        target_safe = target.clamp(min=1e-8)
                        kl = (target_safe * (target_safe.log() - pred_log_probs)).sum(
                            dim=-1
                        )
                        # Mean over unique controls
                        loss = loss + retrieval_consistency_weight * kl.mean()

                # Restore labels for HF (some versions expect them)
                inputs["labels"] = labels_
                return (loss, out) if return_outputs else loss

        # --- HF Trainer plumbing (mirrors your existing setup)
        def _compute_metrics(eval_pred):  # type: ignore[no-untyped-def]
            from sklearn.metrics import f1_score

            preds, label_ids = eval_pred
            if isinstance(preds, tuple):
                preds = preds[0]
            arg = preds.argmax(axis=-1)
            return {
                "macro_f1": float(
                    f1_score(label_ids, arg, average="macro", zero_division=0)
                )
            }

        with tempfile.TemporaryDirectory(prefix="cnmv2-") as tmpdir:
            training_args_kwargs: Dict[str, Any] = {
                "output_dir": tmpdir,
                "learning_rate": self.learning_rate,
                "per_device_train_batch_size": self.batch_size,
                "num_train_epochs": self.num_train_epochs,
                "dataloader_pin_memory": self._has_accelerator(),
                "logging_strategy": "no",
                "report_to": [],
                "seed": self.random_state,
                "data_seed": self.random_state,
                "weight_decay": self.weight_decay,
                "warmup_steps": self.warmup_steps,
                "remove_unused_columns": False,  # keep retrieved_indices/scores
            }
            if eval_ds is not None:
                training_args_kwargs.update(
                    {
                        "eval_strategy": "epoch",
                        "save_strategy": "epoch",
                        "save_total_limit": 1,
                        "per_device_eval_batch_size": self.batch_size,
                        "load_best_model_at_end": True,
                        "metric_for_best_model": "macro_f1",
                        "greater_is_better": True,
                    }
                )
            else:
                training_args_kwargs["save_strategy"] = "no"

            targs = transformers.TrainingArguments(**training_args_kwargs)

            trainer_kwargs: Dict[str, Any] = {
                "model": self._model,
                "args": targs,
                "train_dataset": train_ds,
            }
            if eval_ds is not None:
                trainer_kwargs["eval_dataset"] = eval_ds
                trainer_kwargs["compute_metrics"] = _compute_metrics
                if self.early_stopping_patience > 0 and hasattr(
                    transformers, "EarlyStoppingCallback"
                ):
                    trainer_kwargs["callbacks"] = [
                        transformers.EarlyStoppingCallback(
                            early_stopping_patience=self.early_stopping_patience
                        )
                    ]

            # Epoch tracker so model.set_epoch fires each epoch (for
            # retrieval-freeze warm-up).
            model_holder = self._model

            class _EpochTracker(transformers.TrainerCallback):  # type: ignore[misc]
                def on_epoch_begin(self, args, state, control, **kwargs):
                    model_holder.set_epoch(state.epoch or 0.0)

            trainer_kwargs.setdefault("callbacks", []).append(_EpochTracker())

            trainer = CNMv2Trainer(**trainer_kwargs)
            trainer.train()
            self._model = trainer.model

        self._model.eval()
        self.is_fit = True

    def _has_accelerator(self) -> bool:
        torch = self._torch
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and callable(getattr(cuda, "is_available", None)):
            return bool(cuda.is_available())
        return False

    # ------------------------------------------------------------------
    # Empirical control → class distribution (for retrieval consistency loss)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_control_class_distribution(
        retrieved_idx: np.ndarray,  # (N, k) int64
        labels: np.ndarray,  # (N,) int64
        n_controls: int,
        n_classes: int,
    ) -> np.ndarray:
        """For each control j, return P(class | control j retrieved in top-k).

        Uniform prior [1/C, ...] for controls that are never retrieved during
        training.
        """
        counts = np.zeros((n_controls, n_classes), dtype=np.float64)
        for i in range(retrieved_idx.shape[0]):
            for j in retrieved_idx[i]:
                counts[j, labels[i]] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        never = row_sums.squeeze(-1) == 0
        # Laplace smoothing + uniform for zero-retrieval controls
        counts = counts + 0.5
        row_sums = counts.sum(axis=1, keepdims=True)
        dist = counts / row_sums
        dist[never] = 1.0 / n_classes
        return dist.astype(np.float32)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_proba(self, text: str) -> Dict[str, float]:
        if not self.is_fit or self._model is None:
            raise RuntimeError("Model is not fit")
        torch = self._torch

        # Retrieve top-k for this clause
        encoder = self._ensure_retrieval_encoder()
        memory = self._ensure_memory()
        q = encoder.encode([text], normalise=True)
        indices, scores = memory.batch_top_k(q, k=self.cnm_config.top_k)

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = next(self._model.parameters()).device
        with torch.no_grad():
            out = self._model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
                retrieved_indices=torch.tensor(
                    indices, dtype=torch.long, device=device
                ),
                retrieved_scores=torch.tensor(
                    scores, dtype=torch.float32, device=device
                ),
            )
            probs = torch.softmax(out.logits[0], dim=-1).cpu().tolist()

        return {self.id_to_label[i]: float(p) for i, p in enumerate(probs)}

    def predict(self, text: str) -> str:
        p = self.predict_proba(text)
        return max(p.items(), key=lambda kv: kv[1])[0]

    def predict_with_trace(self, text: str, example_id: str = "") -> Dict[str, Any]:
        """Return probabilities + gate value + retrieved controls with metadata."""
        if not self.is_fit or self._model is None:
            raise RuntimeError("Model is not fit")
        torch = self._torch
        memory = self._ensure_memory()
        encoder = self._ensure_retrieval_encoder()
        q = encoder.encode([text], normalise=True)
        indices, scores = memory.batch_top_k(q, k=self.cnm_config.top_k)

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = next(self._model.parameters()).device
        with torch.no_grad():
            out = self._model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
                retrieved_indices=torch.tensor(
                    indices, dtype=torch.long, device=device
                ),
                retrieved_scores=torch.tensor(
                    scores, dtype=torch.float32, device=device
                ),
            )
            probs = torch.softmax(out.logits[0], dim=-1).cpu().tolist()
            gate = float(out.gate[0].cpu().item())

        records = memory.records()
        retrieved = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            rec = records[int(idx)]
            retrieved.append(
                {
                    "rank": rank,
                    "control_id": rec.control_id,
                    "source": rec.source,
                    "section": rec.section,
                    "score": float(score),
                    "text": rec.text[:200],
                }
            )
        return {
            "example_id": example_id,
            "probabilities": {
                self.id_to_label[i]: float(p) for i, p in enumerate(probs)
            },
            "gate_alpha": gate,
            "retrieved": retrieved,
        }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        save_dir = path if path.suffix == "" else (path.parent / path.stem)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch = self._torch

        # Save the encoder backbone + tokenizer via HF's native path so the
        # checkpoint is portable.
        self.encoder_backbone.save_pretrained(save_dir / "encoder")
        self.tokenizer.save_pretrained(save_dir / "encoder")

        # Save the CNMv2 heads (text head, retrieval-head control_logits, gate).
        torch.save(
            {
                "text_dense": self._model.text_dense.state_dict(),
                "text_out": self._model.text_out.state_dict(),
                "control_logits": self._model.control_logits.data.cpu(),
                "gate_dense": self._model.gate_dense.state_dict(),
                "gate_out": self._model.gate_out.state_dict(),
                "cnm_config": self._model.config.__dict__,
            },
            save_dir / "cnm_heads.pt",
        )

        # Save the training config for reproducibility.
        (save_dir / "cnm_config.json").write_text(
            json.dumps(
                {
                    "labels": self.labels,
                    "model_name": self.model_name,
                    "random_state": self.random_state,
                    "num_train_epochs": self.num_train_epochs,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "max_length": self.max_length,
                    "loss_type": self.loss_type,
                    "focal_gamma": self.focal_gamma,
                    "weight_decay": self.weight_decay,
                    "warmup_steps": self.warmup_steps,
                    "cnm": self.cnm_config.as_dict(),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
