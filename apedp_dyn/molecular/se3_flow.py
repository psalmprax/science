r"""Molecular SE(3)-Equivariant Conformer Flow: distance-field relaxation.

Status: experimental

Scope & honesty ceiling:
- Implements an E(3)-equivariant gradient flow on 3D atom clouds driven by an
  invariant molecular force field (harmonic bonds plus a soft pairwise
  repulsion). Because the energy depends only on interatomic distances, the
  induced velocity field ``-grad E`` transforms *equivariantly* under all
  orthogonal transforms (rotations, translations, reflections), which is exactly
  the SE(3)/O(3) symmetry structure exploited by modern equivariant networks.
- The force field is intentionally classical (no electronic structure, no
  learned message passing, no torsion/angle terms). This module demonstrates and
  verifies the *symmetry and energy-descent machinery*; it does not attempt
  protein folding, docking, or quantum-chemistry accuracy.
- Known honest limitation: distance-only energies are achiral - the flow
  transports a chiral configuration to its own handedness by equivariance, but
  cannot energetically distinguish enantiomers. This is stated, not hidden.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_SE3_TOL = 1e-10


@dataclass(frozen=True)
class MolecularForceField:
    """Classical force field: harmonic bonds + soft pair repulsion.

    ``E = sum_(i,j) in bonds k/2 (r_ij - r0)^2 + sum_(i<j) eps (sigma / r_ij)^12``
    """

    bonds: tuple[tuple[int, int], ...]
    spring_constant: float = 1.0
    target_length: float = 1.0
    repulsion_strength: float = 1e-3
    repulsion_sigma: float = 0.5
    pair_cutoff_power: int = 12

    def __post_init__(self) -> None:
        if not self.bonds:
            raise ValueError("force field requires at least one bond")
        if self.target_length <= 0.0:
            raise ValueError("target_length must be positive")


@dataclass
class RelaxationResult:
    """Trajectory of an equivariant gradient-flow relaxation."""

    coordinates: np.ndarray
    trajectory: list[np.ndarray] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)


def _pairwise(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    deltas = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    return deltas, distances


# Divisions against pair distances are clamped to this floor, mirroring the
# CUDA steppers' `.clamp(min=1e-8)` so coincident atoms yield finite forces
# instead of silent inf/NaN propagation.
_MIN_PAIR_DISTANCE = 1e-8

# Soft-core width (Angstrom): repulsion is evaluated at the smoothed distance
# r_eff = sqrt(r^2 + delta^2) instead of the raw pair distance. The hard
# r^-12 wall at r -> 0 produced energies ~1e90 and gradient magnitudes ~1e96
# (float64 overflow two steps after any near-coincident start) and, worse, a
# fake equilibrium: at exact coincidence the direction vector is the zero
# vector, so a hard-wall potential simultaneously had E ~ 1e90 and grad E = 0,
# which relaxers registered as a stable minimum. The soft core saturates both
# energy and force at small r while keeping the physical r^-12 regime for all
# r >> delta.
_SOFT_CORE_WIDTH = 0.1


def _non_bonded_pairs(force_field: MolecularForceField, n: int) -> list[tuple[int, int]]:
    """Pair list for the non-bonded core repulsion: all pairs minus 1-2 bonds.

    Pairwise r^-12 repulsion applied to covalently bonded atoms is a modeling
    error, not a numerical one: when the equilibrium bond length r0 <= sigma the
    non-bonded wall dominates the harmonic spring and distorts every relaxed
    geometry. Bonded terms already carry their own exchange repulsion; the
    non-bonded term exists to keep *non-bonded* atoms apart.
    """
    bonded = {(min(u, v), max(u, v)) for u, v in force_field.bonds}
    return [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if (i, j) not in bonded
    ]


def energy(coords: np.ndarray, force_field: MolecularForceField) -> float:
    """Invariant potential energy of a configuration."""
    coords = np.asarray(coords, dtype=np.float64)
    _, distances = _pairwise(coords)
    total = 0.0
    for i, j in force_field.bonds:
        total += 0.5 * force_field.spring_constant * (distances[i, j] - force_field.target_length) ** 2
    n = coords.shape[0]
    sigma = force_field.repulsion_sigma
    strength = force_field.repulsion_strength
    delta_sq = _SOFT_CORE_WIDTH * _SOFT_CORE_WIDTH
    for i, j in _non_bonded_pairs(force_field, n):
        r2 = float(distances[i, j]) ** 2 + delta_sq
        r_eff = math.sqrt(r2)
        total += strength * (sigma / r_eff) ** force_field.pair_cutoff_power
    return float(total)


def energy_gradient(coords: np.ndarray, force_field: MolecularForceField) -> np.ndarray:
    """Analytic Cartesian gradient ``d E / d x_i`` (pairwise antisymmetric)."""
    coords = np.asarray(coords, dtype=np.float64)
    deltas, distances = _pairwise(coords)
    n = coords.shape[0]
    gradient = np.zeros_like(coords)
    for i, j in force_field.bonds:
        d = max(float(distances[i, j]), _MIN_PAIR_DISTANCE)
        direction = deltas[i, j] / d
        force = force_field.spring_constant * (distances[i, j] - force_field.target_length)
        gradient[i] += force * direction
        gradient[j] -= force * direction
    sigma = force_field.repulsion_sigma
    strength = force_field.repulsion_strength
    power = force_field.pair_cutoff_power
    delta_sq = _SOFT_CORE_WIDTH * _SOFT_CORE_WIDTH
    for i, j in _non_bonded_pairs(force_field, n):
        r2_eff = float(distances[i, j]) ** 2 + delta_sq
        r_eff = math.sqrt(r2_eff)
        # d E / d x_i for E = strength*(sigma/r_eff)^p is
        # (dE/dr) * u with dE/dr = -p*strength*sigma^p * r / r_eff^(p+2) and
        # u = deltas/r, which collapses to the clean vector form below; the
        # |contribution| ~ r / delta^(p+2) -> 0 for r -> 0, so coincident starts
        # yield bounded energies and a force that is repulsive but finite at any
        # accessible separation instead of ~1e96 N.
        coefficient = power * strength * sigma ** power
        contribution = -coefficient * deltas[i, j] / r_eff ** (power + 2)
        gradient[i] += contribution
        gradient[j] -= contribution
    return gradient


def velocity_field(coords: np.ndarray, force_field: MolecularForceField) -> np.ndarray:
    """Equivariant drift ``v(x) = -grad E(x)`` in Angstrom-like units per time."""
    return -energy_gradient(coords, force_field)


def rigid_transform(coords: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Apply ``x -> R x + t`` to every atom."""
    coords = np.asarray(coords, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    return coords @ rotation.T + translation


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation via QR of a Gaussian matrix."""
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def reflection_matrix() -> np.ndarray:
    """Householder reflection across the xy-plane (improper orthogonal transform)."""
    return np.diag([1.0, 1.0, -1.0])


def center_of_mass(coords: np.ndarray) -> np.ndarray:
    return np.asarray(coords, dtype=np.float64).mean(axis=0)


def bond_lengths(coords: np.ndarray, bonds: tuple[tuple[int, int], ...]) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    return np.array([np.linalg.norm(coords[i] - coords[j]) for i, j in bonds])


def relax(
    coords: np.ndarray,
    force_field: MolecularForceField,
    step_size: float = 0.01,
    n_steps: int = 400,
    order: int = 1,
) -> RelaxationResult:
    """Relax a configuration by integrating ``dx/dt = -grad E`` with JetFlowStepper."""
    from apedp_dyn.video.jet_flow_stepper import JetFlowConfig, JetFlowStepper

    stepper = JetFlowStepper(JetFlowConfig(num_steps=1, order=order))
    current = np.asarray(coords, dtype=np.float64).copy()
    trajectory = [current.copy()]
    energies = [energy(current, force_field)]
    for _ in range(n_steps):
        current, _, _ = stepper.step(
            lambda z, _t: velocity_field(z, force_field), current, 0.0, step_size
        )
        trajectory.append(current.copy())
        energies.append(energy(current, force_field))
    return RelaxationResult(coordinates=current, trajectory=trajectory, energies=energies)


def equivariance_error(
    coords: np.ndarray,
    force_field: MolecularForceField,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> float:
    """``||v(R x + t) - R v(x)||`` relative to ``||v(x)||``."""
    coords = np.asarray(coords, dtype=np.float64)
    transformed = rigid_transform(coords, rotation, translation)
    lhs = velocity_field(transformed, force_field)
    rhs = velocity_field(coords, force_field) @ rotation.T
    scale = float(np.linalg.norm(rhs))
    denominator = scale if scale > 0 else 1.0
    return float(np.linalg.norm(lhs - rhs) / denominator)
