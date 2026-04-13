"""CLI launcher for the Phase 4 Streamlit web application."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> None:
    args = _parse_args()

    try:
        from streamlit.web import cli as stcli
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency guard
        raise SystemExit(
            "Streamlit is not installed. Install dependencies and retry: python -m pip install -e ."
        ) from exc

    app_path = Path(__file__).resolve().parents[1] / "phase4" / "web_app.py"
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
    ]

    if args.headless:
        sys.argv.extend(["--server.headless", "true"])

    raise SystemExit(stcli.main())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Phase 4 compliance web GUI")
    parser.add_argument("--port", type=int, default=8501, help="Port for the web app.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Streamlit in headless mode.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
