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
from apedp_dyn.quantum.dqi_photonic_resource import (
    DQI_GLUZDOV_V6_TARGET_WORKLOAD,
    DQI_REFERENCE_WORKLOAD,
    DQIPhotonicMappingAssumptions,
    DQIPhotonicResourceMapper,
    DQIWorkload,
    SuperconductingBaseline,
    map_reference_workload,
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
    "DQI_GLUZDOV_V6_TARGET_WORKLOAD",
    "DQI_REFERENCE_WORKLOAD",
    "THERMODYNAMIC_ENERGY_PER_SITE_HEISENBERG",
    "DQIPhotonicMappingAssumptions",
    "DQIPhotonicResourceMapper",
    "DQIWorkload",
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
    "SuperconductingBaseline",
    "compute_15_to_1_round",
    "compute_willow_cultivation_round",
    "dmrg_ground_state",
    "heisenberg_exact_ground_energy",
    "heisenberg_mpo",
    "heisenberg_sparse_hamiltonian",
    "map_reference_workload",
    "mps_energy_expectation",
    "simulate_distillation_cascade",
]
