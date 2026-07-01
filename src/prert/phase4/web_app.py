"""Gradio GUI entry point for Phase 4 workflows.

The Gradio app source lives in ``huggingface/space/app.py`` so the local CLI and
the deployable Hugging Face Space use the same interface.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Dict


def _default_space_app_path() -> Path:
    return Path(__file__).resolve().parents[3] / "huggingface" / "space" / "app.py"


def _load_space_app(app_path: Path | None = None) -> ModuleType:
    resolved_path = app_path or _default_space_app_path()
    if not resolved_path.exists():
        raise RuntimeError(f"Gradio Space app not found: {resolved_path}")

    spec = importlib.util.spec_from_file_location("prert_phase4_gradio_space_app", resolved_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Gradio Space app: {resolved_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_SPACE_APP_MODULE = _load_space_app()
demo = getattr(_SPACE_APP_MODULE, "demo")


def launch_defaults() -> Dict[str, Any]:
    builder = getattr(_SPACE_APP_MODULE, "_launch_kwargs", None)
    if callable(builder):
        return dict(builder())
    return {}


if __name__ == "__main__":
    demo.queue().launch(**launch_defaults())
