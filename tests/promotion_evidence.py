"""The repository's tests use the evidence helpers that ship with apedp-referee's tests.

The helpers live in ``apedp_referee/tests/referee_evidence.py`` so the package's
own tests can import them where nothing else is installed. This module loads
that one file and re-exports it, so there is one copy.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "apedp_referee" / "tests" / "referee_evidence.py"
_NAME = "referee_evidence"

if _NAME in sys.modules:
    _module = sys.modules[_NAME]
else:
    _spec = importlib.util.spec_from_file_location(_NAME, _PATH)
    assert _spec is not None and _spec.loader is not None, _PATH
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_NAME] = _module
    _spec.loader.exec_module(_module)

globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
__all__ = [k for k in vars(_module) if not k.startswith("_")]
