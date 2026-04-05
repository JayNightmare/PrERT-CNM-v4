"""Runtime configuration for local scripts and CLIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Optional


@dataclass(frozen=True)
class ChromaSettings:
    host: str
    api_key: str
    tenant: str
    database: str
    default_collection_name: str = "ground_truth"

    @classmethod
    def from_env(cls, env_path: Optional[Path] = None) -> "ChromaSettings":
        load_dotenv_if_available(env_path)

        host = os.getenv("CHROMA_HOST", "api.trychroma.com").strip()
        api_key = os.getenv("CHROMA_API_KEY", "").strip()
        tenant = os.getenv("CHROMA_TENANT", "").strip()
        database = os.getenv("CHROMA_DATABASE", "").strip()
        default_collection = os.getenv("CHROMA_COLLECTION_NAME", "ground_truth").strip() or "ground_truth"

        missing = []
        if not api_key:
            missing.append("CHROMA_API_KEY")
        if not tenant:
            missing.append("CHROMA_TENANT")
        if not database:
            missing.append("CHROMA_DATABASE")

        if missing:
            missing_csv = ", ".join(missing)
            raise ValueError(f"Missing required environment variables: {missing_csv}")

        return cls(
            host=host,
            api_key=api_key,
            tenant=tenant,
            database=database,
            default_collection_name=default_collection,
        )


def load_dotenv_if_available(env_path: Optional[Path]) -> None:
    """Load .env values if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
    except Exception:
        return

    if env_path is None:
        env_path = Path.cwd() / ".env"

    if env_path.exists():
        load_dotenv(env_path, override=False)
