"""PrERT Phase 1 implementation package."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
import tomllib

try:
    __version__ = importlib.metadata.version("prert-cnm-v4")
except importlib.metadata.PackageNotFoundError:
    _pyproject = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
    if _pyproject.exists():
        with _pyproject.open("rb") as _f:
            __version__ = tomllib.load(_f).get("project", {}).get("version", "0.0.0")
    else:
        __version__ = "0.0.0"

__all__ = [
    "__version__",
    "config",
    "extract",
    "chunking",
    "chroma",
    "phase2",
    "phase3",
    "phase4",
]
