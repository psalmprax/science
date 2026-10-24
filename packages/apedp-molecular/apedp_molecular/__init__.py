"""APEDP-Molecular: Autonomous Molecular SE(3)-Equivariant Flow & Relaxation."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure root package can be imported if running within monorepo
_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from apedp_dyn.molecular.se3_flow import (
    MolecularForceField,
    RelaxationResult,
    bond_lengths,
    center_of_mass,
    energy,
    energy_gradient,
    equivariance_error,
    random_rotation,
    reflection_matrix,
    relax,
    rigid_transform,
    velocity_field,
)

__version__ = "0.1.0"

__all__ = [
    "MolecularForceField",
    "RelaxationResult",
    "bond_lengths",
    "center_of_mass",
    "energy",
    "energy_gradient",
    "equivariance_error",
    "random_rotation",
    "reflection_matrix",
    "relax",
    "rigid_transform",
    "velocity_field",
]
