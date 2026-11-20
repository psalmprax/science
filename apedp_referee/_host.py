"""Where the referee meets the full ``apedp`` distribution, and only where.

The referee decides with what it is given: explicit ledger paths, explicit
evidence. Three conveniences belong to the host application instead -- the
default ledger location, the scratch directory a remote ledger is cached in,
and the remote storage backend named by ``APEDP_STORAGE_BACKEND``. They live in
``apedp_dyn``, so they are imported here, inside functions, when a caller asks
for them. With ``apedp-referee`` installed alone the referee still works; those
three calls say what to pass instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_ALONE = (
    "{what} comes from the full apedp distribution (apedp_dyn), which is not "
    "installed; with apedp-referee alone, {instead}"
)


def ledger_path(filename: str) -> Path:
    """The host's default location for a ledger file (``apedp_dyn.env``)."""
    try:
        from apedp_dyn.env import get_ledger_path
    except ImportError as exc:
        raise RuntimeError(_ALONE.format(
            what="A default ledger location",
            instead="construct the ledger with persistence_path=<file>")) from exc
    return Path(get_ledger_path(filename))


def temp_storage_dir() -> Path:
    """The host's scratch directory for a remote ledger's local cache."""
    try:
        from apedp_dyn.env import get_temp_storage_dir
    except ImportError as exc:
        raise RuntimeError(_ALONE.format(
            what="A cache directory for a remote ledger",
            instead="pass persistence_path=<file> alongside storage_backend")) from exc
    return Path(get_temp_storage_dir())


def storage_backend_from_env() -> Any:
    """The remote backend named by ``APEDP_STORAGE_BACKEND``."""
    try:
        from apedp_dyn.epistemic.remote_storage import get_storage_backend
    except ImportError as exc:
        raise RuntimeError(_ALONE.format(
            what="APEDP_STORAGE_BACKEND",
            instead="pass a storage_backend object to the ledger")) from exc
    return get_storage_backend()
