"""APEDP Experimental Harness (AEH).

External experimental-control, capability isolation, and audit layer for APEDP-X.
"""
from __future__ import annotations

from apedp_dyn.harness.core import AEH, HarnessResultPackage
from apedp_dyn.harness.ledger import AuditEvent, EventLedger
from apedp_dyn.harness.manifest import ExperimentManifest
from apedp_dyn.harness.roles import (
    CAPABILITY_MATRIX,
    ComponentRole,
    DataFirewall,
    FirewallViolationError,
    IsolatedDataVault,
    RolePermissions,
)

__all__ = [
    "AEH",
    "CAPABILITY_MATRIX",
    "AuditEvent",
    "ComponentRole",
    "DataFirewall",
    "EventLedger",
    "ExperimentManifest",
    "FirewallViolationError",
    "HarnessResultPackage",
    "IsolatedDataVault",
    "RolePermissions",
]
