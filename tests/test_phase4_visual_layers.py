from pathlib import Path

from prert.phase4 import visual_layers


class _DummyBundle:
    def __init__(self) -> None:
        self.model_id = "dummy/model"
        self.revision = "main"


class _DummyConfig:
    def __init__(self) -> None:
        self.output_hidden_states = False
        self.output_attentions = False
        self.attn_implementation = "sdpa"
        self._attn_implementation = "sdpa"


class _DummyModelWithSetter:
    def __init__(self) -> None:
        self.config = _DummyConfig()
        self.calls = []

    def set_attn_implementation(self, mode: str) -> None:
        self.calls.append(mode)


def test_run_visual_layers_single_clause_with_stubbed_backend(monkeypatch) -> None:
    monkeypatch.setattr(visual_layers, "_get_model_bundle", lambda model_id, revision: _DummyBundle())

    def _fake_analyze(_bundle, text, _max_length, _selected_layers, _selected_heads, _top_tokens):
        return {
            "text": text,
            "token_count": 7,
            "tokens": ["[CLS]", "we", "collect", "data", "for", "support", "[SEP]"],
            "predictions": [{"label": "organization", "score": 0.92}],
            "layer_activation": [
                {"layer": 1, "mean_norm": 1.2, "max_norm": 2.3, "mean_abs": 0.21, "activation_score": 0.53}
            ],
            "attention": [
                {"layer": 1, "head": 0, "mean": 0.11, "max": 0.45, "entropy": 1.23, "cls_focus": [{"token": "collect", "score": 0.25}]}
            ],
            "math_breakdown": {"layer": 1, "supported": False, "note": "stub"},
            "step_trace": [{"step": "tokenize", "detail": "stub"}],
            "architecture": {
                "nodes": [{"id": "input", "label": "Input Tokens", "group": "io", "level": 0, "trigger": 1.0}],
                "edges": [],
                "triggered_node_ids": ["input"],
            },
        }

    monkeypatch.setattr(visual_layers, "_analyze_clause", _fake_analyze)

    result = visual_layers.run_visual_layers_analysis(
        mode="single_clause",
        clause_text="We collect data for support.",
        policy_text="",
        model_id="dummy/model",
        model_revision="main",
        max_length=128,
        max_clauses=8,
        selected_layers=["0", "1"],
        selected_heads=["0"],
        top_tokens=4,
    )

    assert result["mode"] == "single_clause"
    assert result["model"]["model_id"] == "dummy/model"
    assert result["result"]["token_count"] == 7
    assert result["result"]["predictions"][0]["label"] == "organization"


def test_run_visual_layers_full_policy_aggregates(monkeypatch) -> None:
    monkeypatch.setattr(visual_layers, "_get_model_bundle", lambda model_id, revision: _DummyBundle())
    monkeypatch.setattr(
        visual_layers,
        "split_policy_clauses",
        lambda _text: [
            "Clause one about consent and transparency.",
            "Clause two about retention and security.",
        ],
    )

    def _fake_analyze(_bundle, text, _max_length, _selected_layers, _selected_heads, _top_tokens):
        return {
            "text": text,
            "token_count": 10,
            "tokens": ["[CLS]", "clause", "token", "[SEP]"],
            "predictions": [{"label": "organization", "score": 0.88}],
            "layer_activation": [
                {"layer": 1, "mean_norm": 1.0, "max_norm": 2.0, "mean_abs": 0.2},
                {"layer": 2, "mean_norm": 1.1, "max_norm": 2.1, "mean_abs": 0.21},
            ],
            "attention": [
                {"layer": 1, "head": 0, "mean": 0.12, "max": 0.41, "entropy": 1.1, "cls_focus": []}
            ],
            "math_breakdown": {"layer": 1, "supported": False, "note": "stub"},
            "step_trace": [{"step": "tokenize", "detail": "stub"}],
            "architecture": {
                "nodes": [{"id": "input", "label": "Input Tokens", "group": "io", "level": 0, "trigger": 1.0}],
                "edges": [],
                "triggered_node_ids": ["input"],
            },
        }

    monkeypatch.setattr(visual_layers, "_analyze_clause", _fake_analyze)

    result = visual_layers.run_visual_layers_analysis(
        mode="full_policy",
        clause_text="",
        policy_text="Long policy text",
        model_id="dummy/model",
        model_revision="main",
        max_length=128,
        max_clauses=8,
        selected_layers=["0", "1"],
        selected_heads=["0"],
        top_tokens=4,
    )

    aggregate = result["result"]["aggregate"]
    assert result["mode"] == "full_policy"
    assert aggregate["clauses"] == 2
    assert aggregate["label_distribution"]["organization"] == 2
    assert len(result["result"]["architecture"]["nodes"]) >= 4


def test_render_and_export_svg(tmp_path: Path) -> None:
    analysis = {
        "mode": "single_clause",
        "model": {"model_id": "dummy/model", "revision": "main"},
        "result": {
            "architecture": {
                "nodes": [
                    {"id": "input", "label": "Input Tokens", "group": "io", "level": 0, "trigger": 1.0},
                    {"id": "embedding", "label": "Embeddings", "group": "embedding", "level": 1, "trigger": 1.0},
                    {"id": "layer_1", "label": "Layer 1", "group": "transformer", "level": 2, "trigger": 0.7},
                    {"id": "softmax", "label": "Softmax", "group": "head", "level": 3, "trigger": 1.0},
                ],
                "edges": [
                    {"src": "input", "dst": "embedding", "weight": 1.0},
                    {"src": "embedding", "dst": "layer_1", "weight": 0.7},
                    {"src": "layer_1", "dst": "softmax", "weight": 1.0},
                ],
                "triggered_node_ids": ["input", "embedding", "layer_1", "softmax"],
            }
        },
    }

    svg = visual_layers.render_visual_layers_svg(analysis)
    assert "<svg" in svg
    assert "Visual Layers Node Map" in svg

    export_path = Path(visual_layers.export_visual_layers_map(analysis, "svg"))
    assert export_path.exists()
    assert export_path.suffix == ".svg"
    assert "<svg" in export_path.read_text(encoding="utf-8")


def test_render_attention_heatmap_png() -> None:
    analysis = {
        "mode": "single_clause",
        "result": {
            "attention": [
                {
                    "layer": 1,
                    "head": 0,
                    "tokens": ["[CLS]", "we", "collect", "data"],
                    "matrix": [
                        [0.6, 0.2, 0.1, 0.1],
                        [0.3, 0.4, 0.2, 0.1],
                        [0.2, 0.2, 0.4, 0.2],
                        [0.1, 0.2, 0.3, 0.4],
                    ],
                }
            ]
        },
    }

    path = Path(visual_layers.render_attention_heatmap_png(analysis, layer=1, head=0))
    assert path.exists()
    assert path.suffix == ".png"


def test_force_eager_attention_sets_model_and_config() -> None:
    model = _DummyModelWithSetter()

    visual_layers._force_eager_attention(model)

    assert model.config.output_hidden_states is True
    assert model.config.output_attentions is True
    assert model.config.attn_implementation == "eager"
    assert model.config._attn_implementation == "eager"
    assert model.calls == ["eager"]
