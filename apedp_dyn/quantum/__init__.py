r"""Deterministic Quantum Photonics, Fault-Tolerant QC, and DMRG Modules.

Status: experimental
"""
from __future__ import annotations

from apedp_dyn.quantum.dmrg import (
    THERMODYNAMIC_ENERGY_PER_SITE_HEISENBERG,
    DmrgResult,
    MatrixProductState,
    dmrg_ground_state,
    heisenberg_exact_ground_energy,
    heisenberg_mpo,
    heisenberg_sparse_hamiltonian,
    mps_energy_expectation,
)
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
    "DmrgResult",
    "MatrixProductState",
    "PassiveFusionEngine",
    "SFFCCLatticeArchitect",
    "SparrowFaultToleranceReferee",
    "SparrowFaultToleranceThresholds",
    "SparrowHardwareSpec",
    "THERMODYNAMIC_ENERGY_PER_SITE_HEISENBERG",
    "dmrg_ground_state",
    "heisenberg_exact_ground_energy",
    "heisenberg_mpo",
    "heisenberg_sparse_hamiltonian",
    "mps_energy_expectation",
]
