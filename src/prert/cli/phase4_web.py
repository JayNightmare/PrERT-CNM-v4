"""CLI launcher for the Phase 4 Gradio web application."""

from __future__ import annotations

import argparse
from typing import Any, Dict


DEFAULT_PORT = 7860
PUBLIC_HOST = "0.0.0.0"


def main() -> None:
    args = _parse_args()

    try:
        from prert.phase4 import web_app
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency guard
        raise SystemExit(
            "Gradio web app dependencies are not installed. Install dependencies and retry: python -m pip install -e ."
        ) from exc
    except Exception as exc:  # pragma: no cover - runtime deployment guard
        raise SystemExit(f"Unable to load the Phase 4 Gradio web app: {exc}") from exc

    launch_kwargs: Dict[str, Any] = {}
    launch_defaults = getattr(web_app, "launch_defaults", None)
    if callable(launch_defaults):
        default_launch_kwargs = launch_defaults()
        if isinstance(default_launch_kwargs, dict):
            launch_kwargs.update(default_launch_kwargs)

    launch_kwargs.update(_build_launch_kwargs(args))

    web_app.demo.queue().launch(**launch_kwargs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Phase 4 Gradio compliance web GUI")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port for the web app.")
    parser.add_argument(
        "--host",
        help="Host/address for the web app. Use 0.0.0.0 to accept remote connections from a reachable host.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Bind to 0.0.0.0 and create a temporary Gradio public share URL.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a temporary Gradio public share URL.",
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        help="Open the app in a browser when it launches.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _build_launch_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    host = args.host or (PUBLIC_HOST if args.public else None)
    launch_kwargs: Dict[str, Any] = {"server_port": int(args.port)}

    if host:
        launch_kwargs["server_name"] = host

    if args.share or args.public:
        launch_kwargs["share"] = True

    if args.inbrowser:
        launch_kwargs["inbrowser"] = True

    return launch_kwargs


if __name__ == "__main__":
    main()
