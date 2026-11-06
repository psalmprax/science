"""Operational and Lifecycle Infrastructure for APEDP-X.

Provides storage auditing, automated cache lifecycle pruning, and host
filesystem headroom guards.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from apedp_dyn.operational.lifecycle_pruner import (
        LifecyclePolicy,
        PruneResult,
        StorageFootprint,
        audit_storage_footprint,
        auto_guard_storage,
        prune_lifecycle,
    )

__all__ = [
    "LifecyclePolicy",
    "PruneResult",
    "StorageFootprint",
    "audit_storage_footprint",
    "auto_guard_storage",
    "prune_lifecycle",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        import apedp_dyn.operational.lifecycle_pruner as lp
        return getattr(lp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
