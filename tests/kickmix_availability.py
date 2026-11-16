"""Whether this interpreter can actually build circuits with kickmix.

``import kickmix`` succeeding is not that. The extension lives in
``apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix/``,
a directory that exists in every checkout whether or not the extension has
been built into it. As soon as its parent is on ``sys.path`` -- and several
tests and modules put it there -- ``import kickmix`` returns an empty
*namespace package* for that directory: no ``CircuitBuilder``, nothing to
build with, but no ``ImportError`` either.

Six tests guarded themselves with ``try: import kickmix / except: skip``.
Alone they skipped. In the full run, after another test had extended
``sys.path``, the import succeeded, the guard passed, and they failed with
``module 'kickmix' has no attribute 'CircuitBuilder'`` or ``Kickmix is
required to build physical circuits``. The first real CI run of the test
matrix stopped on exactly one of them
(docs/SYSTEM_FLAWS_AUDIT.md, 2026-11-16 round, F-07).

The guard is therefore the capability, not the import.
"""

from __future__ import annotations

from types import ModuleType

import pytest

__all__ = ["require_kickmix", "usable_kickmix"]


def usable_kickmix() -> ModuleType | None:
    """The kickmix module if it can build circuits here, otherwise ``None``."""
    try:
        import kickmix
    except Exception:  # noqa: BLE001 - ImportError, or OSError from an ABI mismatch: either way this interpreter cannot use the extension
        return None
    return kickmix if hasattr(kickmix, "CircuitBuilder") else None


def require_kickmix() -> ModuleType:
    """Skip the calling test unless kickmix can build circuits here."""
    module = usable_kickmix()
    if module is None:
        pytest.skip("kickmix cannot build circuits in this interpreter (extension not built or not loadable)")
    return module
