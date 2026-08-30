"""Unit tests for CNMv2Model architecture.

Tests architectural invariants that the training loop depends on:
- Forward pass produces correct output shape (B, C)
- Gate value in [0,1]
- Text-head-only mode (no retrieval) matches gate=1 case exactly
- retrieval_frozen() controls whether retrieval participates
- Gradient flow: control_logits get gradients when unfrozen
- Gate init: at step 0, α ≈ 0.88 (so we cannot regress below baseline early)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from prert.phase3.cnm.model import CNMv2Config, CNMv2Model


class _StubEncoder(nn.Module):
    """Mimics HF encoder output: has `.last_hidden_state` shape (B, T, H)."""

    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(1000, hidden_size)

    class _Out:
        def __init__(self, last_hidden_state):
            self.last_hidden_state = last_hidden_state

    def forward(self, input_ids, attention_mask=None, return_dict=True):
        h = self.embed(input_ids)  # (B, T, H)
        return self._Out(h)


def _make_model(memory_size: int = 20, num_labels: int = 3) -> CNMv2Model:
    torch.manual_seed(0)
    encoder = _StubEncoder(hidden_size=32)
    cfg = CNMv2Config(
        hidden_size=32, num_labels=num_labels, memory_size=memory_size, top_k=5
    )
    return CNMv2Model(encoder=encoder, config=cfg)


def test_forward_output_shape():
    model = _make_model()
    B, T = 4, 10
    input_ids = torch.randint(0, 1000, (B, T))
    ret_idx = torch.randint(0, 20, (B, 5))
    ret_scores = torch.rand(B, 5)
    out = model(
        input_ids=input_ids, retrieved_indices=ret_idx, retrieved_scores=ret_scores
    )
    assert out.logits.shape == (B, 3)
    assert out.logits_text.shape == (B, 3)
    assert out.gate.shape == (B,)
    assert (out.gate >= 0).all() and (out.gate <= 1).all()


def test_gate_init_biases_toward_text_head():
    """Gate output at step 0 must be biased toward text head (α > 0.8)."""
    model = _make_model()
    B, T = 8, 10
    input_ids = torch.randint(0, 1000, (B, T))
    ret_idx = torch.randint(0, 20, (B, 5))
    ret_scores = torch.rand(B, 5)
    with torch.no_grad():
        out = model(
            input_ids=input_ids, retrieved_indices=ret_idx, retrieved_scores=ret_scores
        )
    # Bias initialised to +2.0, gate_out.weight is zero, so gate = σ(2.0) ≈ 0.88.
    assert (out.gate > 0.85).all(), f"Gate too low at init: {out.gate}"


def test_control_logits_init_zero():
    """Retrieval head must produce zero logits at initialisation."""
    model = _make_model(memory_size=20, num_labels=3)
    B = 4
    input_ids = torch.randint(0, 1000, (B, 8))
    ret_idx = torch.randint(0, 20, (B, 5))
    ret_scores = torch.rand(B, 5)
    with torch.no_grad():
        out = model(
            input_ids=input_ids, retrieved_indices=ret_idx, retrieved_scores=ret_scores
        )
    # control_logits zero → logits_R zero → logits ≈ α · logits_T
    assert torch.allclose(
        out.logits_retrieval, torch.zeros_like(out.logits_retrieval), atol=1e-6
    )
    # Consequently logits == α · logits_T (up to fp noise)
    expected = out.gate.unsqueeze(-1) * out.logits_text
    assert torch.allclose(out.logits, expected, atol=1e-5)


def test_retrieval_frozen_skips_head():
    """When retrieval_frozen(), model bypasses retrieval + gate entirely."""
    model = _make_model()
    model.config.freeze_retrieval_below_epoch = 1
    model.set_epoch(0.0)
    assert model.retrieval_frozen()

    B = 4
    input_ids = torch.randint(0, 1000, (B, 8))
    ret_idx = torch.randint(0, 20, (B, 5))
    ret_scores = torch.rand(B, 5)
    with torch.no_grad():
        out = model(
            input_ids=input_ids, retrieved_indices=ret_idx, retrieved_scores=ret_scores
        )
    # In frozen mode logits should equal logits_text (α=1)
    assert torch.allclose(out.logits, out.logits_text)
    assert torch.allclose(out.gate, torch.ones(B))


def test_control_logits_receive_gradient_when_unfrozen():
    """After unfreezing, control_logits participate in backward pass."""
    model = _make_model()
    model.set_epoch(0.0)
    B = 4
    input_ids = torch.randint(0, 1000, (B, 8))
    ret_idx = torch.randint(0, 20, (B, 5))
    ret_scores = torch.rand(B, 5)
    out = model(
        input_ids=input_ids, retrieved_indices=ret_idx, retrieved_scores=ret_scores
    )
    loss = out.logits.sum()
    loss.backward()
    # control_logits should have a non-None gradient for rows that were retrieved
    grad = model.control_logits.grad
    assert grad is not None
    retrieved_rows = ret_idx.unique()
    for row in retrieved_rows:
        assert torch.any(grad[row] != 0), f"No gradient on retrieved control row {row}"


def test_no_retrieval_input_uses_text_only():
    """When retrieved_indices=None, model falls back to text head only."""
    model = _make_model()
    B = 4
    input_ids = torch.randint(0, 1000, (B, 8))
    out = model(input_ids=input_ids, retrieved_indices=None, retrieved_scores=None)
    assert torch.allclose(out.logits, out.logits_text)
    assert torch.allclose(out.gate, torch.ones(B))
    assert out.logits_retrieval is None


def test_empty_retrieval_uses_text_only():
    """Ablation baseline k=0 arrives as (B, 0) tensors, not None. The model
    must treat that as text-head-only mode without NaN/undefined behaviour.
    """
    model = _make_model()
    B = 4
    input_ids = torch.randint(0, 1000, (B, 8))
    ret_idx = torch.zeros((B, 0), dtype=torch.long)
    ret_scores = torch.zeros((B, 0), dtype=torch.float32)
    out = model(
        input_ids=input_ids, retrieved_indices=ret_idx, retrieved_scores=ret_scores
    )
    # Same guarantees as retrieval=None: logits == logits_text, gate ones.
    assert torch.allclose(out.logits, out.logits_text)
    assert torch.allclose(out.gate, torch.ones(B))
    assert out.logits_retrieval is None
    # No NaN anywhere in the output
    assert not torch.isnan(out.logits).any()
    assert not torch.isnan(out.gate).any()


def test_softmax_temperature_affects_retrieval_weights():
    """Lower temperature → sharper retrieval weight distribution."""
    torch.manual_seed(0)
    encoder = _StubEncoder(hidden_size=32)

    cfg_hot = CNMv2Config(
        hidden_size=32, num_labels=3, memory_size=20, top_k=5, softmax_temperature=10.0
    )
    cfg_cold = CNMv2Config(
        hidden_size=32, num_labels=3, memory_size=20, top_k=5, softmax_temperature=0.1
    )
    model_hot = CNMv2Model(encoder=encoder, config=cfg_hot)
    model_cold = CNMv2Model(encoder=encoder, config=cfg_cold)
    # Set control_logits to a diverse pattern so retrieval output actually varies
    with torch.no_grad():
        model_hot.control_logits.copy_(torch.randn(20, 3))
        model_cold.control_logits.copy_(model_hot.control_logits.clone())

    B = 4
    input_ids = torch.randint(0, 1000, (B, 8))
    ret_idx = torch.randint(0, 20, (B, 5))
    ret_scores = torch.randn(B, 5) * 2.0
    with torch.no_grad():
        hot_r = model_hot._retrieval_logits(ret_idx, ret_scores)
        cold_r = model_cold._retrieval_logits(ret_idx, ret_scores)
    # Cold (low T) should have more extreme values than hot (high T).
    assert cold_r.abs().max() > hot_r.abs().max()
