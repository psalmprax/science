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
from apedp_dyn.quantum.magic_state import (
    DistillationProtocol,
    DistillationRoundResult,
    DistillationScheduleResult,
    MagicStateCultivationProposer,
    compute_15_to_1_round,
    compute_willow_cultivation_round,
    simulate_distillation_cascade,
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
    "THERMODYNAMIC_ENERGY_PER_SITE_HEISENBERG",
    "DeterministicVsProbabilisticScaling",
    "DistillationProtocol",
    "DistillationRoundResult",
    "DistillationScheduleResult",
    "DmrgResult",
    "MagicStateCultivationProposer",
    "MatrixProductState",
    "PassiveFusionEngine",
    "SFFCCLatticeArchitect",
    "SparrowFaultToleranceReferee",
    "SparrowFaultToleranceThresholds",
    "SparrowHardwareSpec",
    "compute_15_to_1_round",
    "compute_willow_cultivation_round",
    "dmrg_ground_state",
    "heisenberg_exact_ground_energy",
    "heisenberg_mpo",
    "heisenberg_sparse_hamiltonian",
    "mps_energy_expectation",
    "simulate_distillation_cascade",
]
