r"""Deterministic Quantum Photonics and Fault-Tolerant Quantum Computing Modules.

Status: experimental
"""
from __future__ import annotations

from apedp_dyn.quantum.sparrow_photonics import (
    DeterministicVsProbabilisticScaling,
    PassiveFusionEngine,
    SFFCCLatticeArchitect,
    SparrowFaultToleranceReferee,
    SparrowFaultToleranceThresholds,
    SparrowHardwareSpec,
)

__all__ = [
    "DeterministicVsProbabilisticScaling",
    "PassiveFusionEngine",
    "SFFCCLatticeArchitect",
    "SparrowFaultToleranceReferee",
    "SparrowFaultToleranceThresholds",
    "SparrowHardwareSpec",
]
