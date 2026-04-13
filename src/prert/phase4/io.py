"""I/O helpers for Phase 4 validation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from prert.phase3.io import read_json, read_jsonl


def resolve_manifest_path(artifact_dir: Path, manifest_path: Optional[Path] = None) -> Path:
    if manifest_path is not None:
        return manifest_path
    return artifact_dir / "phase3_manifest.json"


def load_phase3_manifest(artifact_dir: Path, manifest_path: Optional[Path] = None) -> Dict[str, Any]:
    resolved = resolve_manifest_path(artifact_dir, manifest_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Phase 3 manifest not found: {resolved}")
    return read_json(resolved)


def resolve_output_path(artifact_dir: Path, output_files: Dict[str, Any], key: str, fallback: str) -> Path:
    rel = str(output_files.get(key, "")).strip()
    if rel:
        return artifact_dir / rel
    return artifact_dir / fallback


def load_optional_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return read_json(path)


def load_optional_jsonl_rows(path: Path) -> Optional[list[Dict[str, Any]]]:
    if not path.exists():
        return None
    return read_jsonl(path)
