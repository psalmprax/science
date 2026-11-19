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

from pathlib import Path
from types import ModuleType

import pytest

#: The verifier's own kickmix build. `*.so` is git-ignored, so this file exists
#: only where someone has built it (scripts/setup_verifier_env.py); it is not
#: in a clone and CI does not build it.
SHIPPED_EXTENSION = (
    Path(__file__).resolve().parents[1]
    / "apedp_dyn"
    / "sandbox"
    / "google-verifiable-quantum-circuit-compilation"
    / "kickmix"
    / "kickmix.so"
)

__all__ = ["SHIPPED_EXTENSION", "require_kickmix", "require_shipped_extension", "usable_kickmix"]


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
        pytest.skip(
            "kickmix cannot build circuits in this interpreter (extension not built or not loadable)"
        )
    return module


def require_shipped_extension() -> Path:
    """Skip the calling test unless the verifier's kickmix extension is on disk.

    Contracts are locked to that file's bytes
    (``apedp_dyn.quantum_compilation.contracts``), so a test that asserts a
    lock, compiles a FAROS candidate or checks the build pin has nothing to
    check in a checkout where the extension was never built. There it used to
    fail with ``FileNotFoundError``, which reads as a broken contract and is
    not one.
    """
    if not SHIPPED_EXTENSION.is_file():
        pytest.skip(
            "the verifier's kickmix extension is not built in this checkout "
            f"({SHIPPED_EXTENSION.name} is git-ignored; run scripts/setup_verifier_env.py)"
        )
    return SHIPPED_EXTENSION
