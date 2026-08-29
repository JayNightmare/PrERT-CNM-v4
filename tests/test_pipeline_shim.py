"""Verify that `run_phase3_cnm_pipeline` actually swaps the classifier factory.

The shim patches the `train_classifier` name in TWO places:
  1. `prert.phase3.classifier.train_classifier` (source module attribute)
  2. `prert.phase3.pipeline.train_classifier` (local binding imported at
      module-import time)

Patching only (1) is a silent no-op because `prert.phase3.pipeline` has
already bound the name locally. This test guarantees that both bindings
get swapped and restored.

The test does NOT run the full pipeline (which requires PrivBERT + a real
Phase 1 index). It only verifies the monkey-patch swap by capturing the
`train_classifier` reference from inside a fake pipeline call.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple


def _install_stub_pipeline_module() -> None:
    """Install stub modules for prert.phase3.pipeline and .classifier before
    importing prert.phase3.cnm.pipeline, so we can inspect what the shim
    swaps without needing PrivBERT/torch in the test environment.
    """
    import importlib

    importlib.import_module("prert.phase3.cnm_v2")

    # Stub prert.config.load_dotenv_if_available (imported by cli).
    if "prert.config" not in sys.modules:
        config_mod = types.ModuleType("prert.config")
        config_mod.load_dotenv_if_available = lambda _: None
        sys.modules["prert.config"] = config_mod

    # Stub prert.phase3.classifier with a sentinel train_classifier.
    classifier_mod = types.ModuleType("prert.phase3.classifier")

    def _real_train_classifier(
        examples: Sequence[Any] = (),
        labels: Sequence[str] = (),
        output_path: Path | None = None,
        **kwargs: Any,
    ) -> Tuple[Any, Dict[str, Any]]:
        return object(), {
            "model_type": "SENTINEL_REAL",
            "training_examples": 0.0,
            "vocabulary_size": 0.0,
            "labels": 0.0,
            "backbone_model_name": "",
        }

    classifier_mod.train_classifier = _real_train_classifier
    classifier_mod.DEFAULT_PRIVACYBERT_MODEL_NAME = "stub/privbert"
    # Placeholder for TextClassifier protocol.
    classifier_mod.TextClassifier = object
    sys.modules["prert.phase3.classifier"] = classifier_mod
    setattr(sys.modules["prert.phase3"], "classifier", classifier_mod)

    # Stub prert.phase3.pipeline that imports train_classifier the same way
    # the real module does, so the local binding lives here.
    from prert.phase3.classifier import train_classifier as _pipe_train  # noqa: F401

    pipeline_mod = types.ModuleType("prert.phase3.pipeline")
    pipeline_mod.train_classifier = _pipe_train

    def _run_phase3_pipeline(output_dir: Path, **kwargs: Any) -> Dict[str, Any]:
        # Deliberately call the LOCAL binding, mirroring how the real
        # pipeline uses `train_classifier` after `from ... import ...`.
        # Pass the same signature the real pipeline uses so the shim's
        # `_cnm_train_classifier` accepts the call.
        model, summary = pipeline_mod.train_classifier(
            examples=[],
            labels=["user", "system", "organization"],
            output_path=Path(output_dir) / "checkpoint",
        )
        return {
            "metrics": {"model_type": summary["model_type"]},
            "dataset_manifest": {},
        }

    pipeline_mod.run_phase3_pipeline = _run_phase3_pipeline
    sys.modules["prert.phase3.pipeline"] = pipeline_mod
    setattr(sys.modules["prert.phase3"], "pipeline", pipeline_mod)

    # Stub types module so the CNM classifier import works.
    types_mod = types.ModuleType("prert.phase3.types")
    from dataclasses import dataclass, field  # noqa: WPS433

    @dataclass
    class ClauseExample:  # minimal stand-in
        example_id: str = ""
        text: str = ""
        label: str = ""
        source: str = ""
        policy_uid: str = ""
        category: str = ""
        metadata: Dict[str, Any] = field(default_factory=dict)

    types_mod.ClauseExample = ClauseExample
    sys.modules["prert.phase3.types"] = types_mod


def test_shim_swaps_pipeline_local_binding(monkeypatch, tmp_path):
    _install_stub_pipeline_module()

    # Re-import cnm.pipeline so it picks up the stubs.
    for mod in list(sys.modules):
        if (
            mod.startswith("prert.phase3.cnm.pipeline")
            or mod == "prert.phase3.cnm.classifier"
        ):
            del sys.modules[mod]

    # Provide a stub CNMPrivacyBertClassifier so the shim can instantiate it.
    cnm_classifier_mod = types.ModuleType("prert.phase3.cnm.classifier")

    class _StubCNMClassifier:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def fit(self, examples, validation_examples=None):  # noqa: D401
            self.fitted = True

        def save(self, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("stub")

    from dataclasses import dataclass, field  # noqa: WPS433

    @dataclass
    class CNMConfig:
        index_dir: Path
        top_k: int = 5
        augment_training: bool = True

        def as_metadata_dict(self) -> Dict[str, Any]:
            return {"top_k": self.top_k}

    cnm_classifier_mod.CNMConfig = CNMConfig
    cnm_classifier_mod.CNMPrivacyBertClassifier = _StubCNMClassifier
    sys.modules["prert.phase3.cnm.classifier"] = cnm_classifier_mod

    # Now import the real shim, which will bind its stubbed dependencies.
    from prert.phase3.cnm import pipeline as cnm_pipeline  # noqa: WPS433

    # Sanity: before the shim runs, pipeline.train_classifier returns SENTINEL_REAL.
    import prert.phase3.pipeline as base_pipeline

    _, pre_summary = base_pipeline.train_classifier(
        examples=[], labels=[], output_path=tmp_path / "ck"
    )
    assert pre_summary["model_type"] == "SENTINEL_REAL"

    manifest = cnm_pipeline.run_phase3_cnm_pipeline(
        output_dir=tmp_path,
        cnm_config=CNMConfig(index_dir=tmp_path / "index", top_k=5),
    )

    # The shim's _cnm_train_classifier must have replaced the pipeline's local
    # binding at call-time, so the manifest.metrics.model_type reflects the
    # shim's return value, not SENTINEL_REAL.
    assert manifest["metrics"]["model_type"] == "cnm_privacybert", (
        "Shim did NOT replace pipeline.train_classifier local binding. "
        f"Got model_type={manifest['metrics']['model_type']!r}, "
        "expected 'cnm_privacybert'."
    )

    # After the shim returns, the original binding must be restored.
    _, post_summary = base_pipeline.train_classifier(
        examples=[], labels=[], output_path=tmp_path / "ck2"
    )
    assert post_summary["model_type"] == "SENTINEL_REAL", (
        "Shim did not restore prert.phase3.pipeline.train_classifier " "after the run."
    )
