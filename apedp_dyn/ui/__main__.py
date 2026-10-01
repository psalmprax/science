"""CLI entrypoint for running the APEDP Dyn Visual Studio."""
from __future__ import annotations

import argparse

from apedp_dyn.ui.server import run_server


def main() -> None:
    parser = argparse.ArgumentParser(description="APEDP Dyn Visual Studio Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
