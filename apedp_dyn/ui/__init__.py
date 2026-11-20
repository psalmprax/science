"""APEDP Dyn Web Studio and Visual Adjudication Interface."""
from __future__ import annotations

from typing import Any

__all__ = ["app", "run_server"]


def __getattr__(name: str) -> Any:
    if name in ("app", "run_server"):
        from apedp_dyn.ui import server
        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
