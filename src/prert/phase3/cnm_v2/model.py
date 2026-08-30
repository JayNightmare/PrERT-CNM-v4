"""CNMv2: gated auxiliary-head retrieval-augmented classifier.

Architecture (as approved 2026-08-29):

    clause x
        │
        ▼
    PrivBERT encoder  ────► h(x) ∈ ℝ^H
        │                       │
        ├─ text-head ──────────►  logits_T = W_T · h(x)   ∈ ℝ^C
        │                       │
        ├─ retrieval-head ─────► logits_R = Σ_j softmax(sim(q,c_j)/τ)_j · L[r_j]
        │                       │   where L ∈ ℝ^{|M|×C} is per-control learned logits
        │                       │
        └─ gate ───────────────► α = σ(w_g · h(x) + b_g)  ∈ [0,1]
                                │
    logits = α · logits_T + (1-α) · logits_R
    p = softmax(logits)

The text head trains on raw clauses (no retrieval concatenation), preventing
the class-discrimination collapse we observed with input-side augmentation.
The retrieval head is a learned per-control class-prior over the memory:
each control c_j owns a learned 3-vector L[j] that says "if this control is
retrieved for a clause, how does it shift class probabilities?" These are
initialised at zero (no prior effect) and learned end-to-end.

Retrieval is precomputed once per example (before training) using the
frozen sentence-transformer encoder and stored as a per-example list of
(control_index, similarity) pairs. This decouples ChromaDB from the
training loop and eliminates the retrieval bottleneck the previous design
had.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

_LOGGER = logging.getLogger(__name__)


@dataclass
class CNMv2Config:
    """Static hyperparameters for the CNMv2 architecture."""

    hidden_size: int  # encoder output dim (768 for PrivBERT)
    num_labels: int  # 3 (user/system/organization)
    memory_size: int  # |M| — number of controls in the index
    top_k: int = 5  # retrieval budget per clause
    softmax_temperature: float = 1.0
    gate_hidden: int = 128  # width of the gate MLP hidden layer
    dropout: float = 0.1
    freeze_retrieval_below_epoch: int = 0
    # If > 0, keeps the retrieval-head + gate frozen for the first N epochs
    # so the text head can converge first before the retrieval head starts
    # contributing. Safeguard against early-training gate-collapse.


class CNMv2Model(nn.Module):
    """Wraps a HuggingFace encoder with a text head, retrieval head, and gate.

    Compatible with HuggingFace Trainer: `forward()` accepts the standard
    `input_ids`, `attention_mask`, `labels` kwargs and returns an object
    with a `.logits` attribute and (optionally) a `.loss` attribute so the
    Trainer's default plumbing works. Additionally accepts per-example
    retrieval tensors `retrieved_indices` (LongTensor of shape [B, k]) and
    `retrieved_scores` (FloatTensor of shape [B, k], pre-softmax cosine
    similarities).
    """

    def __init__(self, encoder: nn.Module, config: CNMv2Config) -> None:
        super().__init__()
        self.encoder = encoder
        self.config = config

        H = config.hidden_size
        C = config.num_labels
        M = config.memory_size

        # ----- text head: exactly the existing HF classification head shape
        # (Linear + tanh + Linear on the [CLS] token). We rebuild it locally
        # so we can gate its output cleanly. The encoder's built-in head is
        # NOT used; we drop AutoModelForSequenceClassification's head and
        # keep only the RobertaModel backbone.
        self.text_dense = nn.Linear(H, H)
        self.text_dropout = nn.Dropout(config.dropout)
        self.text_out = nn.Linear(H, C)

        # ----- retrieval head: per-control learned logits.
        # Initialised at zero so the retrieval head produces logits_R = 0 at
        # the start of training. Only the gate can bring retrieval into play
        # once the text head has learned.
        self.control_logits = nn.Parameter(torch.zeros(M, C))

        # ----- gate: σ(w · h + b), scalar per example.
        # Bias initialised to +2.0 → σ(2.0) ≈ 0.88 at start. This means at
        # step 0 the model behaves almost identically to the plain text
        # classifier, so we cannot regress below baseline during warm-up.
        self.gate_dense = nn.Linear(H, config.gate_hidden)
        self.gate_out = nn.Linear(config.gate_hidden, 1)
        nn.init.zeros_(self.gate_out.weight)
        nn.init.constant_(self.gate_out.bias, 2.0)

        self._current_epoch: float = 0.0

    # ------------------------------------------------------------------
    # Public API used by the training callback
    # ------------------------------------------------------------------
    def set_epoch(self, epoch: float) -> None:
        self._current_epoch = float(epoch)

    def retrieval_frozen(self) -> bool:
        return self._current_epoch < self.config.freeze_retrieval_below_epoch

    # ------------------------------------------------------------------
    # Head computations
    # ------------------------------------------------------------------
    def _pooled(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Use the [CLS] / <s> token (position 0), same as
        # RobertaClassificationHead.
        return hidden_states[:, 0, :]

    def _text_logits(self, h_cls: torch.Tensor) -> torch.Tensor:
        x = self.text_dropout(h_cls)
        x = self.text_dense(x)
        x = torch.tanh(x)
        x = self.text_dropout(x)
        return self.text_out(x)  # (B, C)

    def _retrieval_logits(
        self,
        retrieved_indices: torch.Tensor,  # (B, k) long
        retrieved_scores: torch.Tensor,  # (B, k) float, cosine sims
    ) -> torch.Tensor:
        # Softmax over the top-k similarities to get per-retrieved-control
        # attention weights. Higher-similarity controls contribute more of
        # their learned class-prior.
        weights = F.softmax(
            retrieved_scores / self.config.softmax_temperature, dim=-1
        )  # (B, k)
        control_prior = self.control_logits[retrieved_indices]  # (B, k, C)
        logits_r = (weights.unsqueeze(-1) * control_prior).sum(dim=1)  # (B, C)
        return logits_r

    def _gate(self, h_cls: torch.Tensor) -> torch.Tensor:
        g = torch.tanh(self.gate_dense(h_cls))
        g = self.gate_out(g).squeeze(-1)  # (B,)
        return torch.sigmoid(g)  # (B,) in [0,1]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        retrieved_indices: Optional[torch.Tensor] = None,
        retrieved_scores: Optional[torch.Tensor] = None,
        **_kwargs,
    ):
        encoder_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = encoder_out.last_hidden_state  # (B, T, H)
        h_cls = self._pooled(hidden)  # (B, H)

        logits_t = self._text_logits(h_cls)  # (B, C)

        logits_r: Optional[torch.Tensor] = None
        retrieval_active = (
            retrieved_indices is not None
            and retrieved_scores is not None
            and retrieved_indices.numel() > 0  # k=0 arrives here as (B, 0) tensors
            and not self.retrieval_frozen()
        )
        if retrieval_active:
            logits_r = self._retrieval_logits(retrieved_indices, retrieved_scores)
            alpha = self._gate(h_cls).unsqueeze(-1)  # (B, 1)
            logits = alpha * logits_t + (1.0 - alpha) * logits_r
        else:
            # Text-head-only mode: identical geometry to the plain PrivBERT
            # baseline. Used at k=0 in the ablation and during retrieval-freeze
            # warm-up.
            logits = logits_t
            alpha = torch.ones(input_ids.shape[0], 1, device=input_ids.device)

        # Return a namespace that mimics HF ModelOutput. Loss is computed
        # externally by the CNMv2Trainer (needs class weights + focal loss +
        # retrieval-consistency term).
        return CNMv2Output(
            logits=logits,
            logits_text=logits_t,
            logits_retrieval=logits_r,
            gate=alpha.squeeze(-1),
            hidden=h_cls,
        )


@dataclass
class CNMv2Output(ModelOutput):
    logits: torch.Tensor = None
    logits_text: torch.Tensor = None
    logits_retrieval: Optional[torch.Tensor] = None
    gate: torch.Tensor = None
    hidden: torch.Tensor = None
    loss: Optional[torch.Tensor] = None
