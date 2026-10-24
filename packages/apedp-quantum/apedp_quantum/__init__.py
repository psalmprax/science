"""APEDP-Quantum: Autonomous Quantum Discovery, DMRG & Fault-Tolerant Photonics."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure root package can be imported if running within monorepo
_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

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

__version__ = "0.1.0"

__all__ = [
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
    "THERMODYNAMIC_ENERGY_PER_SITE_HEISENBERG",
    "compute_15_to_1_round",
    "compute_willow_cultivation_round",
    "dmrg_ground_state",
    "heisenberg_exact_ground_energy",
    "heisenberg_mpo",
    "heisenberg_sparse_hamiltonian",
    "mps_energy_expectation",
    "simulate_distillation_cascade",
]
