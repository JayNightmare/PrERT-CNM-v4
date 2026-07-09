"""Neural layer introspection helpers for the Phase 4 Visual Layers GUI.

This module provides PrivacyBERT-focused analysis with layer activations,
attention summaries, math breakdown snapshots, and architecture map payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import sqrt
from pathlib import Path
import importlib
import json
import tempfile
import uuid
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from prert.phase4.compliance_assessor import split_policy_clauses


DEFAULT_MAX_CLAUSES = 16
DEFAULT_TOP_TOKENS = 8
BERT_LAYER_COUNT = 12
BERT_HEAD_COUNT = 12
HEATMAP_MAX_TOKENS = 64


@dataclass(frozen=True)
class _ModelBundle:
    tokenizer: Any
    model: Any
    torch: Any
    model_id: str
    revision: str


def _normalize_model_id(model_id: str) -> str:
    return str(model_id or "").strip()


def _normalize_revision(revision: str) -> str:
    value = str(revision or "").strip()
    return value or "main"


def _load_kwargs(revision: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"revision": revision}
    token = (
        importlib.import_module("os").getenv("HF_TOKEN")
        or importlib.import_module("os").getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    if token:
        kwargs["token"] = token
    return kwargs


@lru_cache(maxsize=3)
def _get_model_bundle(model_id: str, revision: str) -> _ModelBundle:
    normalized_model_id = _normalize_model_id(model_id)
    if not normalized_model_id:
        raise ValueError("MODEL_ID is required for Visual Layers analysis.")

    normalized_revision = _normalize_revision(revision)
    torch_module = importlib.import_module("torch")
    transformers_module = importlib.import_module("transformers")

    load_kwargs = _load_kwargs(normalized_revision)
    tokenizer = transformers_module.AutoTokenizer.from_pretrained(normalized_model_id, **load_kwargs)
    model = transformers_module.AutoModelForSequenceClassification.from_pretrained(normalized_model_id, **load_kwargs)

    if getattr(model, "config", None) is not None:
        setattr(model.config, "output_hidden_states", True)
        setattr(model.config, "output_attentions", True)

    model.eval()
    return _ModelBundle(
        tokenizer=tokenizer,
        model=model,
        torch=torch_module,
        model_id=normalized_model_id,
        revision=normalized_revision,
    )


def _coerce_indices(selected: Optional[Sequence[Any]], maximum: int, default_count: int) -> List[int]:
    if not selected:
        return list(range(min(default_count, maximum)))

    parsed: List[int] = []
    for value in selected:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= index < maximum:
            parsed.append(index)

    unique = sorted(set(parsed))
    if unique:
        return unique
    return list(range(min(default_count, maximum)))


def _sigmoid(value: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-value))


def _clean_tokens(tokens: Sequence[str]) -> List[str]:
    cleaned: List[str] = []
    for token in tokens:
        normalized = str(token)
        if normalized in {"[PAD]", "[CLS]", "[SEP]"}:
            cleaned.append(normalized)
            continue
        cleaned.append(normalized.replace("##", ""))
    return cleaned


def _sequence_length_from_mask(mask: Any) -> int:
    try:
        return int(mask.sum().item())
    except Exception:
        try:
            return int(mask.sum())
        except Exception:
            return int(len(mask))


def _build_architecture_map(layer_scores: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = [
        {"id": "input", "label": "Input Tokens", "group": "io", "level": 0, "trigger": 1.0},
        {"id": "embedding", "label": "Embeddings", "group": "embedding", "level": 1, "trigger": 1.0},
    ]
    edges: List[Dict[str, Any]] = [
        {"src": "input", "dst": "embedding", "weight": 1.0},
    ]

    for index, score in enumerate(layer_scores, start=1):
        node_id = f"layer_{index}"
        trigger = float(score.get("mean_norm", 0.0))
        nodes.append(
            {
                "id": node_id,
                "label": f"Layer {index}",
                "group": "transformer",
                "level": index + 1,
                "trigger": trigger,
            }
        )
        previous = "embedding" if index == 1 else f"layer_{index - 1}"
        edges.append({"src": previous, "dst": node_id, "weight": trigger})

    nodes.extend(
        [
            {"id": "pooler", "label": "Pooler", "group": "head", "level": len(layer_scores) + 2, "trigger": 1.0},
            {"id": "classifier", "label": "Classifier", "group": "head", "level": len(layer_scores) + 3, "trigger": 1.0},
            {"id": "softmax", "label": "Softmax", "group": "head", "level": len(layer_scores) + 4, "trigger": 1.0},
        ]
    )

    last_layer = f"layer_{len(layer_scores)}" if layer_scores else "embedding"
    edges.extend(
        [
            {"src": last_layer, "dst": "pooler", "weight": 1.0},
            {"src": "pooler", "dst": "classifier", "weight": 1.0},
            {"src": "classifier", "dst": "softmax", "weight": 1.0},
        ]
    )

    return {
        "nodes": nodes,
        "edges": edges,
        "triggered_node_ids": [str(node["id"]) for node in nodes],
    }


def _summarize_layer_activations(torch_module: Any, hidden_states: Sequence[Any], token_count: int) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    usable = list(hidden_states[1:])
    for index, tensor in enumerate(usable, start=1):
        layer_tensor = tensor[0, :token_count, :]
        norms = torch_module.linalg.vector_norm(layer_tensor, dim=-1)
        summaries.append(
            {
                "layer": index,
                "mean_norm": float(norms.mean().item()),
                "max_norm": float(norms.max().item()),
                "mean_abs": float(layer_tensor.abs().mean().item()),
                "activation_score": float(_sigmoid(float(layer_tensor.mean().item()))),
            }
        )
    return summaries


def _attention_entropy(row: Sequence[float]) -> float:
    import math

    epsilon = 1e-12
    return float(-sum(float(p) * math.log(float(p) + epsilon) for p in row))


def _summarize_attention(
    torch_module: Any,
    attentions: Sequence[Any],
    tokens: Sequence[str],
    token_count: int,
    selected_layers: Sequence[int],
    selected_heads: Sequence[int],
    top_tokens: int,
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for layer_index in selected_layers:
        if layer_index < 0 or layer_index >= len(attentions):
            continue
        layer_attention = attentions[layer_index][0, :, :token_count, :token_count]
        heads = min(int(layer_attention.shape[0]), BERT_HEAD_COUNT)
        for head_index in selected_heads:
            if head_index < 0 or head_index >= heads:
                continue
            head_matrix = layer_attention[head_index]
            cls_row = head_matrix[0]
            values, indices = torch_module.topk(cls_row, k=min(top_tokens, token_count))
            top_focus = [
                {
                    "token": str(tokens[int(idx)]),
                    "score": float(val),
                }
                for val, idx in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist())
            ]

            summaries.append(
                {
                    "layer": layer_index + 1,
                    "head": head_index,
                    "mean": float(head_matrix.mean().item()),
                    "max": float(head_matrix.max().item()),
                    "entropy": _attention_entropy(cls_row.detach().cpu().tolist()),
                    "cls_focus": top_focus,
                    "tokens": [str(token) for token in tokens[: min(token_count, HEATMAP_MAX_TOKENS)]],
                    "matrix": head_matrix[
                        : min(token_count, HEATMAP_MAX_TOKENS),
                        : min(token_count, HEATMAP_MAX_TOKENS),
                    ]
                    .detach()
                    .cpu()
                    .tolist(),
                }
            )
    return summaries


def _math_breakdown(
    bundle: _ModelBundle,
    hidden_states: Sequence[Any],
    attentions: Sequence[Any],
    token_count: int,
    selected_layer_index: int,
) -> Dict[str, Any]:
    if selected_layer_index < 0:
        selected_layer_index = 0
    if selected_layer_index >= len(attentions):
        selected_layer_index = max(0, len(attentions) - 1)

    details: Dict[str, Any] = {
        "layer": selected_layer_index + 1,
        "supported": False,
        "note": "Detailed Q/K/V extraction unavailable for this architecture.",
    }

    base_model = getattr(bundle.model, "base_model", None)
    encoder = getattr(base_model, "encoder", None) if base_model is not None else None
    layers = getattr(encoder, "layer", None) if encoder is not None else None
    if layers is None or selected_layer_index >= len(layers):
        return details

    layer_module = layers[selected_layer_index]
    attention_self = getattr(getattr(layer_module, "attention", None), "self", None)
    intermediate = getattr(layer_module, "intermediate", None)
    output_dense = getattr(getattr(layer_module, "output", None), "dense", None)
    if attention_self is None or intermediate is None or output_dense is None:
        return details

    torch_module = bundle.torch
    hidden_in = hidden_states[selected_layer_index][0, :token_count, :]

    query_projection = attention_self.query(hidden_in)
    key_projection = attention_self.key(hidden_in)
    value_projection = attention_self.value(hidden_in)

    num_heads = int(getattr(attention_self, "num_attention_heads", BERT_HEAD_COUNT))
    head_dim = int(getattr(attention_self, "attention_head_size", int(query_projection.shape[-1] / max(num_heads, 1))))

    q = query_projection.view(token_count, num_heads, head_dim).transpose(0, 1)
    k = key_projection.view(token_count, num_heads, head_dim).transpose(0, 1)
    v = value_projection.view(token_count, num_heads, head_dim).transpose(0, 1)

    scale = 1.0 / sqrt(float(head_dim))
    scores = torch_module.matmul(q, k.transpose(-1, -2)) * scale
    probs = torch_module.softmax(scores, dim=-1)
    context = torch_module.matmul(probs, v)

    ffn_intermediate = intermediate(hidden_in)
    ffn_output = output_dense(ffn_intermediate)

    observed_attention = attentions[selected_layer_index][0, :, :token_count, :token_count]
    divergence = float((probs - observed_attention).abs().mean().item())

    details = {
        "layer": selected_layer_index + 1,
        "supported": True,
        "q_mean_norm": float(torch_module.linalg.vector_norm(query_projection, dim=-1).mean().item()),
        "k_mean_norm": float(torch_module.linalg.vector_norm(key_projection, dim=-1).mean().item()),
        "v_mean_norm": float(torch_module.linalg.vector_norm(value_projection, dim=-1).mean().item()),
        "attention_score_mean": float(scores.mean().item()),
        "attention_prob_mean": float(probs.mean().item()),
        "context_mean_abs": float(context.abs().mean().item()),
        "ffn_intermediate_mean": float(ffn_intermediate.mean().item()),
        "ffn_output_mean": float(ffn_output.mean().item()),
        "attention_reconstruction_mae": divergence,
    }
    return details


def _step_trace(
    text: str,
    token_count: int,
    predictions: Sequence[Mapping[str, Any]],
    selected_layers: Sequence[int],
) -> List[Dict[str, Any]]:
    trace: List[Dict[str, Any]] = [
        {
            "step": "input",
            "detail": f"Received {len(text)} characters.",
        },
        {
            "step": "tokenize",
            "detail": f"Generated {token_count} tokens after truncation and padding.",
        },
    ]

    for layer_index in selected_layers:
        trace.append(
            {
                "step": f"transformer_layer_{layer_index + 1}",
                "detail": "Self-attention + feed-forward block executed.",
            }
        )

    trace.extend(
        [
            {
                "step": "classifier",
                "detail": "Linear classification head projected pooled representation into logits.",
            },
            {
                "step": "softmax",
                "detail": f"Normalized logits across {len(predictions)} output classes.",
            },
        ]
    )
    return trace


def _analyze_clause(
    bundle: _ModelBundle,
    text: str,
    max_length: int,
    selected_layers: Sequence[int],
    selected_heads: Sequence[int],
    top_tokens: int,
) -> Dict[str, Any]:
    if not text.strip():
        raise ValueError("Input text is empty.")

    encoded = bundle.tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    with bundle.torch.no_grad():
        outputs = bundle.model(
            **encoded,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states
    attentions = outputs.attentions
    if hidden_states is None or attentions is None:
        raise RuntimeError("Model did not return hidden_states and attentions. Use a BERT-compatible checkpoint.")

    token_count = _sequence_length_from_mask(encoded["attention_mask"][0])
    input_ids = encoded["input_ids"][0, :token_count]
    raw_tokens = bundle.tokenizer.convert_ids_to_tokens(input_ids)
    tokens = _clean_tokens(raw_tokens)

    logits = outputs.logits[0]
    probabilities = bundle.torch.softmax(logits, dim=-1).detach().cpu().tolist()

    labels = []
    id2label = getattr(bundle.model.config, "id2label", None)
    for index in range(len(probabilities)):
        label = str(id2label.get(index, str(index))) if isinstance(id2label, dict) else str(index)
        labels.append(label)

    predictions = [
        {
            "label": labels[index],
            "score": float(probabilities[index]),
        }
        for index in range(len(probabilities))
    ]
    predictions.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)

    layer_activation = _summarize_layer_activations(bundle.torch, hidden_states, token_count)
    architecture = _build_architecture_map(layer_activation)
    attention = _summarize_attention(
        bundle.torch,
        attentions,
        tokens,
        token_count,
        selected_layers,
        selected_heads,
        top_tokens,
    )
    math = _math_breakdown(bundle, hidden_states, attentions, token_count, selected_layers[0] if selected_layers else 0)
    trace = _step_trace(text, token_count, predictions, selected_layers)

    return {
        "text": text,
        "token_count": token_count,
        "tokens": tokens,
        "predictions": predictions,
        "layer_activation": layer_activation,
        "attention": attention,
        "math_breakdown": math,
        "step_trace": trace,
        "architecture": architecture,
    }


def _aggregate_policy_results(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "clauses": 0,
            "avg_token_count": 0.0,
            "layer_activation": [],
            "attention": [],
            "label_distribution": {},
        }

    clause_count = len(results)
    avg_tokens = sum(int(item.get("token_count", 0)) for item in results) / max(clause_count, 1)

    layer_table: Dict[int, Dict[str, float]] = {}
    for item in results:
        for layer in item.get("layer_activation", []):
            layer_idx = int(layer.get("layer", 0))
            bucket = layer_table.setdefault(layer_idx, {"mean_norm": 0.0, "max_norm": 0.0, "mean_abs": 0.0, "count": 0.0})
            bucket["mean_norm"] += float(layer.get("mean_norm", 0.0))
            bucket["max_norm"] += float(layer.get("max_norm", 0.0))
            bucket["mean_abs"] += float(layer.get("mean_abs", 0.0))
            bucket["count"] += 1.0

    layer_activation: List[Dict[str, Any]] = []
    for layer_idx in sorted(layer_table):
        bucket = layer_table[layer_idx]
        count = max(float(bucket.get("count", 1.0)), 1.0)
        layer_activation.append(
            {
                "layer": layer_idx,
                "mean_norm": float(bucket.get("mean_norm", 0.0) / count),
                "max_norm": float(bucket.get("max_norm", 0.0) / count),
                "mean_abs": float(bucket.get("mean_abs", 0.0) / count),
            }
        )

    attention_rows: List[Dict[str, Any]] = []
    attention_table: Dict[Tuple[int, int], Dict[str, float]] = {}
    for item in results:
        for row in item.get("attention", []):
            key = (int(row.get("layer", 0)), int(row.get("head", 0)))
            bucket = attention_table.setdefault(key, {"mean": 0.0, "max": 0.0, "entropy": 0.0, "count": 0.0})
            bucket["mean"] += float(row.get("mean", 0.0))
            bucket["max"] += float(row.get("max", 0.0))
            bucket["entropy"] += float(row.get("entropy", 0.0))
            bucket["count"] += 1.0

    for (layer, head), bucket in sorted(attention_table.items()):
        count = max(float(bucket.get("count", 1.0)), 1.0)
        attention_rows.append(
            {
                "layer": layer,
                "head": head,
                "mean": float(bucket.get("mean", 0.0) / count),
                "max": float(bucket.get("max", 0.0) / count),
                "entropy": float(bucket.get("entropy", 0.0) / count),
            }
        )

    label_counts: Dict[str, int] = {}
    for item in results:
        predictions = item.get("predictions", [])
        if not predictions:
            continue
        top_label = str(predictions[0].get("label", "unknown"))
        label_counts[top_label] = label_counts.get(top_label, 0) + 1

    return {
        "clauses": clause_count,
        "avg_token_count": avg_tokens,
        "layer_activation": layer_activation,
        "attention": attention_rows,
        "label_distribution": label_counts,
    }


def run_visual_layers_analysis(
    *,
    mode: str,
    clause_text: str,
    policy_text: str,
    model_id: str,
    model_revision: str,
    max_length: int,
    max_clauses: int,
    selected_layers: Optional[Sequence[Any]] = None,
    selected_heads: Optional[Sequence[Any]] = None,
    top_tokens: int = DEFAULT_TOP_TOKENS,
) -> Dict[str, Any]:
    normalized_mode = str(mode or "single_clause").strip().lower()
    if normalized_mode not in {"single_clause", "full_policy"}:
        raise ValueError("Mode must be either 'single_clause' or 'full_policy'.")

    selected_layer_indices = _coerce_indices(selected_layers, BERT_LAYER_COUNT, default_count=4)
    selected_head_indices = _coerce_indices(selected_heads, BERT_HEAD_COUNT, default_count=3)

    requested_max_length = max(32, min(int(max_length), 512))
    requested_max_clauses = max(1, min(int(max_clauses), 64))
    requested_top_tokens = max(1, min(int(top_tokens), 24))

    bundle = _get_model_bundle(_normalize_model_id(model_id), _normalize_revision(model_revision))

    if normalized_mode == "single_clause":
        clause = str(clause_text or "").strip()
        if not clause:
            raise ValueError("Single-clause mode requires input text.")
        clause_result = _analyze_clause(
            bundle,
            clause,
            requested_max_length,
            selected_layer_indices,
            selected_head_indices,
            requested_top_tokens,
        )
        return {
            "mode": "single_clause",
            "model": {"model_id": bundle.model_id, "revision": bundle.revision},
            "selected_layers": selected_layer_indices,
            "selected_heads": selected_head_indices,
            "max_length": requested_max_length,
            "result": clause_result,
        }

    clauses = split_policy_clauses(policy_text)
    if not clauses:
        raise ValueError("Full-policy mode requires valid policy text.")

    limited_clauses = clauses[:requested_max_clauses]
    clause_results: List[Dict[str, Any]] = []
    for clause in limited_clauses:
        clause_results.append(
            _analyze_clause(
                bundle,
                clause,
                requested_max_length,
                selected_layer_indices,
                selected_head_indices,
                requested_top_tokens,
            )
        )

    aggregate = _aggregate_policy_results(clause_results)
    architecture = _build_architecture_map(aggregate.get("layer_activation", []))

    return {
        "mode": "full_policy",
        "model": {"model_id": bundle.model_id, "revision": bundle.revision},
        "selected_layers": selected_layer_indices,
        "selected_heads": selected_head_indices,
        "max_length": requested_max_length,
        "requested_clauses": requested_max_clauses,
        "result": {
            "clauses": clause_results,
            "aggregate": aggregate,
            "architecture": architecture,
        },
    }


def render_visual_layers_svg(analysis: Mapping[str, Any]) -> str:
    mode = str(analysis.get("mode", "single_clause"))
    result = analysis.get("result")
    architecture = {}
    if mode == "full_policy":
        architecture = dict((result or {}).get("architecture", {})) if isinstance(result, dict) else {}
    else:
        architecture = dict((result or {}).get("architecture", {})) if isinstance(result, dict) else {}

    nodes = architecture.get("nodes", []) if isinstance(architecture, dict) else []
    edges = architecture.get("edges", []) if isinstance(architecture, dict) else []

    if not nodes:
        return "<div style='padding:12px;border:1px solid #d6d6d6;border-radius:10px;background:#fafaf8;'>No neural layer nodes are available for visualization.</div>"

    width = 1260
    height = 360
    margin_x = 56.0
    margin_y = 70.0

    max_level = max(int(node.get("level", 0)) for node in nodes)
    spacing = (width - 2 * margin_x) / max(max_level, 1)

    positions: Dict[str, Tuple[float, float]] = {}
    for node in nodes:
        node_id = str(node.get("id", ""))
        level = int(node.get("level", 0))
        x = margin_x + (spacing * level)
        y = height / 2.0
        positions[node_id] = (x, y)

    node_colors = {
        "io": "#0ca678",
        "embedding": "#1971c2",
        "transformer": "#6741d9",
        "head": "#f08c00",
    }

    frame_id = f"viz_{uuid.uuid4().hex}"
    svg: List[str] = [
        f"<div id='{frame_id}' style='border:1px solid #d6d6d6;border-radius:12px;background:#f8fafc;padding:8px;overflow:auto;'>",
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>",
        "<defs><style>.ttl{font:700 14px sans-serif;fill:#1f2a37}.lbl{font:11px sans-serif;fill:#213547}.sub{font:10px sans-serif;fill:#5b6b72}.hint{font:10px sans-serif;fill:#495057}</style></defs>",
        "<rect x='0' y='0' width='100%' height='100%' fill='#f8fafc' rx='14' />",
        "<text class='ttl' x='24' y='28'>Visual Layers Node Map</text>",
        "<text class='hint' x='24' y='44'>Scroll to zoom. Shift+scroll to pan horizontally.</text>",
        f"<g id='{frame_id}_viewport' transform='translate(0,0) scale(1)'>",
    ]

    for edge in edges:
        src = str(edge.get("src", ""))
        dst = str(edge.get("dst", ""))
        if src not in positions or dst not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        weight = float(edge.get("weight", 0.0))
        opacity = 0.35 + min(max(weight, 0.0), 1.0) * 0.5
        svg.append(
            f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='#2d3748' stroke-width='1.8' stroke-opacity='{opacity:.3f}' />"
        )

    for node in nodes:
        node_id = str(node.get("id", ""))
        if node_id not in positions:
            continue
        x, y = positions[node_id]
        group = str(node.get("group", "transformer"))
        color = node_colors.get(group, "#334155")
        trigger = float(node.get("trigger", 0.0))
        radius = 10.0 if group != "transformer" else 8.0 + min(max(trigger, 0.0), 16.0)
        label = str(node.get("label", node_id))
        safe_label = label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        svg.append(f"<circle cx='{x}' cy='{y}' r='{radius:.2f}' fill='{color}' fill-opacity='0.88'><title>{safe_label} | trigger={trigger:.3f}</title></circle>")
        svg.append(f"<text class='lbl' x='{x - 24}' y='{y + 28}'>{label}</text>")
        svg.append(f"<text class='sub' x='{x - 22}' y='{y + 42}'>trigger={trigger:.3f}</text>")

    svg.append("</g>")
    svg.append("</svg>")
    svg.append("</div>")
    svg.append(
        """
<script>
(function() {
  var root = document.getElementById('"""
        + frame_id
        + """');
  if (!root) return;
  var viewport = document.getElementById('"""
        + frame_id
        + """_viewport');
  if (!viewport) return;
  var scale = 1.0;
  var tx = 0.0;
  var ty = 0.0;
  function apply() {
    viewport.setAttribute('transform', 'translate(' + tx.toFixed(2) + ',' + ty.toFixed(2) + ') scale(' + scale.toFixed(3) + ')');
  }
  root.addEventListener('wheel', function(event) {
    event.preventDefault();
    if (event.shiftKey) {
      tx -= event.deltaY * 0.35;
      apply();
      return;
    }
    var zoomStep = event.deltaY < 0 ? 1.08 : 0.92;
    scale = Math.max(0.45, Math.min(3.2, scale * zoomStep));
    apply();
  }, { passive: false });
})();
</script>
"""
    )
    return "".join(svg)


def render_attention_heatmap_png(
    analysis: Mapping[str, Any],
    *,
    layer: int,
    head: int,
) -> str:
    mode = str(analysis.get("mode", "single_clause"))
    result = _as_dict(analysis.get("result"))
    attention_rows = _as_list(result.get("attention"))
    if mode == "full_policy":
        clauses = _as_list(result.get("clauses"))
        if clauses:
            attention_rows = _as_list(_as_dict(clauses[0]).get("attention"))

    match = {}
    for item in attention_rows:
        row = _as_dict(item)
        if int(row.get("layer", -1)) == int(layer) and int(row.get("head", -1)) == int(head):
            match = row
            break

    matrix = _as_list(match.get("matrix")) if match else []
    tokens = [str(token) for token in _as_list(match.get("tokens"))] if match else []
    if not matrix:
        raise ValueError("No heatmap matrix available for selected layer/head in the current view.")

    matplotlib_module = importlib.import_module("matplotlib")
    matplotlib_module.use("Agg")
    pyplot = importlib.import_module("matplotlib.pyplot")

    output_dir = Path(tempfile.gettempdir()) / "prert-phase4-visual-layers"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"attention_heatmap_l{int(layer)}_h{int(head)}_{uuid.uuid4().hex}.png"

    fig, ax = pyplot.subplots(figsize=(7.2, 6.4), dpi=160)
    image = ax.imshow(matrix, cmap="viridis", interpolation="nearest", aspect="auto")
    ax.set_title(f"Attention Heatmap | Layer {int(layer)} Head {int(head)}")
    ax.set_xlabel("Key Token Index")
    ax.set_ylabel("Query Token Index")

    token_cap = min(len(tokens), 12)
    if token_cap > 0:
        ticks = list(range(token_cap))
        short_labels = [tokens[index][:8] for index in ticks]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(short_labels, fontsize=7)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, format="png")
    pyplot.close(fig)
    return str(path)


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _render_png_from_architecture(analysis: Mapping[str, Any], target_path: Path) -> None:
    matplotlib_module = importlib.import_module("matplotlib")
    matplotlib_module.use("Agg")
    pyplot = importlib.import_module("matplotlib.pyplot")

    mode = str(analysis.get("mode", "single_clause"))
    result = analysis.get("result")
    architecture = {}
    if mode == "full_policy":
        architecture = dict((result or {}).get("architecture", {})) if isinstance(result, dict) else {}
    else:
        architecture = dict((result or {}).get("architecture", {})) if isinstance(result, dict) else {}

    nodes = architecture.get("nodes", []) if isinstance(architecture, dict) else []
    edges = architecture.get("edges", []) if isinstance(architecture, dict) else []
    if not nodes:
        raise ValueError("No architecture nodes available for PNG export.")

    max_level = max(int(node.get("level", 0)) for node in nodes)
    positions: Dict[str, Tuple[float, float]] = {}
    for node in nodes:
        node_id = str(node.get("id", ""))
        level = int(node.get("level", 0))
        positions[node_id] = (float(level), 0.0)

    fig, ax = pyplot.subplots(figsize=(16, 3.6), dpi=160)
    ax.set_axis_off()

    for edge in edges:
        src = str(edge.get("src", ""))
        dst = str(edge.get("dst", ""))
        if src not in positions or dst not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        weight = float(edge.get("weight", 0.0))
        alpha = 0.35 + min(max(weight, 0.0), 1.0) * 0.5
        ax.plot([x1, x2], [y1, y2], color="#1f2937", linewidth=1.5, alpha=alpha)

    for node in nodes:
        node_id = str(node.get("id", ""))
        if node_id not in positions:
            continue
        x, y = positions[node_id]
        trigger = float(node.get("trigger", 0.0))
        size = 160 if str(node.get("group", "")) != "transformer" else 80 + (trigger * 22)
        ax.scatter([x], [y], s=size, color="#2563eb", alpha=0.85)
        ax.text(x, y - 0.14, str(node.get("label", node_id)), ha="center", va="top", fontsize=8)

    ax.set_xlim(-0.6, float(max_level) + 0.8)
    ax.set_ylim(-0.4, 0.45)
    fig.tight_layout()
    fig.savefig(target_path, format="png")
    pyplot.close(fig)


def export_visual_layers_map(analysis: Mapping[str, Any], file_format: str) -> str:
    normalized = str(file_format or "svg").strip().lower()
    output_dir = Path(tempfile.gettempdir()) / "prert-phase4-visual-layers"
    output_dir.mkdir(parents=True, exist_ok=True)

    if normalized == "svg":
        path = output_dir / f"visual_layers_{uuid.uuid4().hex}.svg"
        path.write_text(render_visual_layers_svg(analysis), encoding="utf-8")
        return str(path)

    if normalized == "png":
        path = output_dir / f"visual_layers_{uuid.uuid4().hex}.png"
        _render_png_from_architecture(analysis, path)
        return str(path)

    raise ValueError("Unsupported export format. Use svg or png.")


def build_visual_layers_markdown(analysis: Mapping[str, Any]) -> str:
    mode = str(analysis.get("mode", "single_clause"))
    model = analysis.get("model", {})
    lines = [
        "### Visual Layers Summary",
        f"- Mode: {mode.replace('_', ' ')}",
        f"- Model: {str(model.get('model_id', 'n/a'))}",
        f"- Revision: {str(model.get('revision', 'n/a'))}",
    ]

    if mode == "single_clause":
        result = analysis.get("result", {})
        token_count = int(result.get("token_count", 0)) if isinstance(result, dict) else 0
        predictions = result.get("predictions", []) if isinstance(result, dict) else []
        lines.append(f"- Tokens analyzed: {token_count}")
        if predictions:
            top = predictions[0]
            lines.append(f"- Top label: {top.get('label', 'n/a')} ({float(top.get('score', 0.0)):.4f})")
    else:
        result = analysis.get("result", {})
        aggregate = result.get("aggregate", {}) if isinstance(result, dict) else {}
        lines.append(f"- Clauses analyzed: {int(aggregate.get('clauses', 0))}")
        lines.append(f"- Avg tokens per clause: {float(aggregate.get('avg_token_count', 0.0)):.2f}")
        distribution = aggregate.get("label_distribution", {})
        if isinstance(distribution, dict) and distribution:
            joined = ", ".join(f"{key}:{value}" for key, value in sorted(distribution.items()))
            lines.append(f"- Dominant label distribution: {joined}")

    return "\n".join(lines)


def write_visual_layers_json(analysis: Mapping[str, Any]) -> str:
    output_dir = Path(tempfile.gettempdir()) / "prert-phase4-visual-layers"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"visual_layers_{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)
