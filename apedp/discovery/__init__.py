"""APEDP Discovery Engine - Symbolic regression and scientific discovery.

Split into focused modules for maintainability while preserving
full backward compatibility through re-exports.
"""

from __future__ import annotations

# Re-export everything from core implementation
from ._core import *  # noqa: F401,F403

# Ensure all names are properly exported
import sys as _sys
from . import _core as _core_module

_sys.modules[__name__].__dict__.update(_core_module.__dict__)
