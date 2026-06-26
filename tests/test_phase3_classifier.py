from __future__ import annotations

from types import SimpleNamespace

from prert.phase3.classifier import PrivacyBertClassifier
from prert.phase3.types import ClauseExample



class _FakeTokenizer:
    @classmethod
    def from_pretrained(cls, _model_name: str) -> "_FakeTokenizer":
        return cls()

    def __call__(self, text, **_kwargs):  # type: ignore[no-untyped-def]
        if isinstance(text, list):
            return {"input_ids": [[1] for _ in text], "attention_mask": [[1] for _ in text]}
        return {"input_ids": [[1]], "attention_mask": [[1]]}


class _FakeModel:
    @classmethod
    def from_pretrained(cls, _model_name: str, **_kwargs) -> "_FakeModel":
        return cls()

    def eval(self) -> None:
        return None


class _FakeTrainingArguments:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        type(self).calls.append(kwargs)
        self.kwargs = kwargs


class _FakeTrainer:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        type(self).calls.append(kwargs)
        self.model = kwargs.get("model")
        self.args = kwargs.get("args")
        self.train_dataset = kwargs.get("train_dataset")

    def train(self) -> None:
        return None


class _FakeEarlyStoppingCallback:
    instances: list[dict[str, object]] = []

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        type(self).instances.append(kwargs)


def _make_torch_stub():
    class _FakeTensor:
        def __init__(self, values):
            self.values = list(values)

        def to(self, _device):
            return self

    def _tensor(values, dtype=None):  # noqa: ARG001
        return _FakeTensor(values)

    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        xpu=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        tensor=_tensor,
        float32="float32",
        nn=SimpleNamespace(
            CrossEntropyLoss=lambda **_kwargs: None,
            functional=SimpleNamespace(log_softmax=lambda x, dim=-1: x),  # noqa: ARG005
        ),
    )


def _make_transformers_stub(include_early_stopping: bool = True):
    namespace = SimpleNamespace(
        Trainer=_FakeTrainer,
        TrainingArguments=_FakeTrainingArguments,
        AutoTokenizer=_FakeTokenizer,
        AutoModelForSequenceClassification=_FakeModel,
    )
    if include_early_stopping:
        namespace.EarlyStoppingCallback = _FakeEarlyStoppingCallback
    return namespace


def _install_import_stubs(monkeypatch, torch_stub, transformers_stub) -> None:
    def _import_module(name: str):
        if name == "torch":
            return torch_stub
        if name == "multiprocess.resource_tracker":
            raise ModuleNotFoundError(name)
        if name == "datasets":
            raise ImportError("datasets module is not required.")
        if name == "transformers":
            return transformers_stub
        raise AssertionError(name)

    monkeypatch.setattr("prert.phase3.classifier.importlib.import_module", _import_module)


def _make_examples():
    return [
        ClauseExample(
            example_id="e1",
            text="Users can opt out.",
            label="user",
            source="opp115",
            policy_uid="p1",
            category="User Choice/Control",
        )
    ]


def test_privacybert_fit_disables_pin_memory_without_accelerator(monkeypatch) -> None:
    _FakeTrainingArguments.calls.clear()
    _FakeTrainer.calls.clear()
    _install_import_stubs(monkeypatch, _make_torch_stub(), _make_transformers_stub())

    classifier = PrivacyBertClassifier(labels=["user", "system", "organization"])
    classifier.fit(_make_examples())

    assert _FakeTrainingArguments.calls
    assert _FakeTrainingArguments.calls[-1]["dataloader_pin_memory"] is False


def test_privacybert_fit_enables_eval_and_best_model_with_validation(monkeypatch) -> None:
    _FakeTrainingArguments.calls.clear()
    _FakeTrainer.calls.clear()
    _FakeEarlyStoppingCallback.instances.clear()
    _install_import_stubs(monkeypatch, _make_torch_stub(), _make_transformers_stub())

    classifier = PrivacyBertClassifier(
        labels=["user", "system", "organization"],
        loss_type="focal",
        focal_gamma=2.0,
        weight_decay=0.01,
        warmup_steps=0.1,
        early_stopping_patience=2,
    )
    classifier.fit(
        _make_examples(),
        validation_examples=[
            ClauseExample(
                example_id="v1",
                text="The service collects data.",
                label="organization",
                source="opp115",
                policy_uid="p2",
                category="First Party Collection/Use",
            )
        ],
    )

    args = _FakeTrainingArguments.calls[-1]
    assert args["eval_strategy"] == "epoch"
    assert args["save_strategy"] == "epoch"
    assert args["load_best_model_at_end"] is True
    assert args["metric_for_best_model"] == "macro_f1"
    assert args["weight_decay"] == 0.01
    assert args["warmup_steps"] == 0.1
    # label_smoothing_factor must NOT appear under focal loss
    assert "label_smoothing_factor" not in args

    trainer_call = _FakeTrainer.calls[-1]
    assert trainer_call.get("eval_dataset") is not None
    assert callable(trainer_call.get("compute_metrics"))
    assert _FakeEarlyStoppingCallback.instances
    assert _FakeEarlyStoppingCallback.instances[-1]["early_stopping_patience"] == 2


def test_privacybert_label_smoothing_applied_only_for_plain_ce(monkeypatch) -> None:
    _FakeTrainingArguments.calls.clear()
    _FakeTrainer.calls.clear()
    _install_import_stubs(monkeypatch, _make_torch_stub(), _make_transformers_stub())

    classifier = PrivacyBertClassifier(
        labels=["user", "system", "organization"],
        loss_type="ce",
        label_smoothing_factor=0.1,
    )
    classifier.fit(_make_examples())

    args = _FakeTrainingArguments.calls[-1]
    assert args["label_smoothing_factor"] == 0.1


def test_privacybert_rejects_unsupported_loss_type() -> None:
    import pytest

    with pytest.raises(ValueError, match="Unsupported loss_type"):
        PrivacyBertClassifier(
            labels=["user", "system", "organization"],
            loss_type="hinge",
        )
