r"""Molecular SE(3)-equivariant flow: invariance, equivariance, relaxation guaranteed.

Status: experimental
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.molecular.se3_flow import (
    MolecularForceField,
    bond_lengths,
    center_of_mass,
    energy,
    energy_gradient,
    equivariance_error,
    random_rotation,
    reflection_matrix,
    relax,
    rigid_transform,
)

FORCE_FIELD = MolecularForceField(
    bonds=((0, 1), (0, 2), (0, 3)),
    spring_constant=2.0,
    target_length=1.0,
)

_INITIAL_COORDS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.15, 0.1, 0.2],
        [0.1, 1.2, -0.25],
        [-0.15, -0.2, 1.1],
    ]
)


def test_energy_invariant_under_rigid_and_reflection_transforms():
    rng = np.random.default_rng(3)
    rotation = random_rotation(rng)
    translation = np.array([0.7, -1.3, 2.1])
    base = energy(_INITIAL_COORDS, FORCE_FIELD)
    rotated = energy(rigid_transform(_INITIAL_COORDS, rotation, translation), FORCE_FIELD)
    mirrored = energy(rigid_transform(_INITIAL_COORDS, reflection_matrix(), np.zeros(3)), FORCE_FIELD)
    assert abs(rotated - base) < 1e-12
    assert abs(mirrored - base) < 1e-12


def test_analytic_gradient_matches_finite_differences():
    step = 1e-6
    numerical = np.zeros_like(_INITIAL_COORDS)
    for k in range(_INITIAL_COORDS.size):
        direction = np.zeros_like(_INITIAL_COORDS)
        direction.ravel()[k] = step
        plus = energy(_INITIAL_COORDS + direction, FORCE_FIELD)
        minus = energy(_INITIAL_COORDS - direction, FORCE_FIELD)
        numerical.ravel()[k] = (plus - minus) / (2.0 * step)
    analytic = energy_gradient(_INITIAL_COORDS, FORCE_FIELD)
    assert float(np.max(np.abs(numerical - analytic))) < 1e-8


def test_velocity_field_equivariant_under_rotations_and_mirrors():
    rng = np.random.default_rng(5)
    rotation = random_rotation(rng)
    translation = np.array([-0.4, 0.9, 1.7])
    assert equivariance_error(_INITIAL_COORDS, FORCE_FIELD, rotation, translation) < 1e-12
    assert equivariance_error(_INITIAL_COORDS, FORCE_FIELD, reflection_matrix(), np.zeros(3)) < 1e-12
    assert equivariance_error(_INITIAL_COORDS, FORCE_FIELD, np.eye(3), translation) < 1e-12


@pytest.fixture(scope="module")
def relaxation_bundle():
    original = relax(_INITIAL_COORDS, FORCE_FIELD, step_size=0.01, n_steps=500)
    mirrored = relax(_INITIAL_COORDS @ reflection_matrix().T, FORCE_FIELD, step_size=0.01, n_steps=500)
    return original, mirrored


def test_relaxation_decreases_energy_and_reaches_target_bonds(relaxation_bundle):
    original, _ = relaxation_bundle
    steps = np.diff(original.energies)
    assert np.all(steps < 0.0), float(steps.max())
    final_lengths = bond_lengths(original.coordinates, FORCE_FIELD.bonds)
    assert float(np.max(np.abs(final_lengths - FORCE_FIELD.target_length))) < 1e-2, final_lengths
    assert original.energies[-1] < 1e-4, original.energies[-1]


def test_center_of_mass_conserved_during_relaxation(relaxation_bundle):
    original, _ = relaxation_bundle
    drift = np.max(np.abs(center_of_mass(original.coordinates) - center_of_mass(_INITIAL_COORDS)))
    assert drift < 1e-12, drift


def test_mirror_image_relaxation_is_exact_mirror(relaxation_bundle):
    original, mirrored = relaxation_bundle
    mirror = reflection_matrix()
    gap = float(np.max(np.abs(mirrored.coordinates - original.coordinates @ mirror.T)))
    assert gap < 1e-12, gap
    handedness_gap = float(np.max(np.abs(original.coordinates - original.coordinates @ mirror.T)))
    assert handedness_gap > 0.5, handedness_gap
