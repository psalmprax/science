"""Runtime half of the proposer/referee boundary (AGENTS.md §1.1).

The lint rule in ``pyproject.toml`` stops a proposer *importing* the referee
kinematics. That is a build-time control and it cannot see a dynamic import, an
``importlib`` call, or code passed through ``exec``. This module is the runtime
companion: every public referee checker asks who called it and refuses when the
answer is a proposer or a model adapter.

Two things make it more than a name check:

* the **whole stack** is inspected, not just the immediate caller, so a proposer
  helper that bounces through an innocently named module is still refused;
* each frame's **source path** is checked as well as its module name, so rewriting
  ``__name__`` does not launder a call — the file the code actually lives in is
  the file the frame reports.

It is a boundary, not a sandbox. Code that copies referee logic into a new file
under a different path can still compute numbers; nothing short of process
isolation prevents that. What this stops is the accident AGENTS.md §1.1 names and
the casual circumvention of it.
"""

from __future__ import annotations

import sys
from types import FrameType

#: Packages whose code must never adjudicate a candidate.
FORBIDDEN_PREFIXES = ("mmlm_matter_001.proposers", "mmlm_matter_001.models")

#: Source-path fragments that identify the same packages, independent of `__name__`.
FORBIDDEN_PATH_PARTS = (
    "mmlm_matter_001/proposers/",
    "mmlm_matter_001/models/",
)

#: Bound on the stack walk: deep enough to see a real caller, short enough to stay cheap.
MAX_STACK_FRAMES = 64


class BoundaryError(RuntimeError):
    """A proposer or model adapter tried to adjudicate a candidate."""


def _module_name(frame: FrameType | None) -> str:
    if frame is None:
        return ""
    return str(frame.f_globals.get("__name__", ""))


def _source_path(frame: FrameType) -> str:
    return str(frame.f_globals.get("__file__", "")).replace("\\", "/")


def _frame_is_forbidden(frame: FrameType) -> bool:
    module = _module_name(frame)
    if any(module == prefix or module.startswith(prefix + ".")
           for prefix in FORBIDDEN_PREFIXES):
        return True
    path = _source_path(frame)
    return any(part in path for part in FORBIDDEN_PATH_PARTS)


def caller_module(depth: int = 1) -> str:
    """The module `depth` frames above the caller of this function.

    ``depth=0`` is the caller itself, ``depth=1`` its caller, and so on.
    """
    try:
        frame = sys._getframe(depth + 1)
    except ValueError:
        return ""
    return _module_name(frame)


def assert_referee_caller(action: str) -> None:
    """Refuse when any frame in the calling stack is proposer-side code."""
    frame = sys._getframe(1)
    seen = 0
    while frame is not None and seen < MAX_STACK_FRAMES:
        if _frame_is_forbidden(frame):
            origin = _module_name(frame) or _source_path(frame)
            raise BoundaryError(
                f"{origin} may not call {action}: a proposer must never adjudicate its "
                "own candidate (AGENTS.md §1.1). Emit the candidate and let the gate "
                "score it.")
        frame = frame.f_back
        seen += 1


__all__ = [
    "FORBIDDEN_PATH_PARTS",
    "FORBIDDEN_PREFIXES",
    "MAX_STACK_FRAMES",
    "BoundaryError",
    "assert_referee_caller",
    "caller_module",
]
