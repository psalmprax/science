"""Tests for pocket-conditioned SE(3) equivariant flow and analytical torsion gradients."""
from __future__ import annotations

import math

import numpy as np
import pytest

from apedp_dyn.molecular.chemistry_bridge import parse_smiles
from apedp_dyn.molecular.equivariant_pocket_flow import (
    BindingPocket,
    PocketAtom,
    PocketConditionedForceField,
    TorsionTerm,
    build_pocket_force_field,
    intramolecular_energy_and_grad,
    pocket_equivariance_error,
    relax_ligand_in_pocket,
    torsion_energy_and_grad,
)


def _torsion_only_ff(periodicity: int, phase_rad: float, barrier_kcal: float) -> PocketConditionedForceField:
    return PocketConditionedForceField(
        bonds=(),
        angles=(),
        torsions=(
            TorsionTerm(
                i=0,
                j=1,
                k=2,
                atom_l=3,
                periodicity=periodicity,
                phase_rad=phase_rad,
                barrier_kcal=barrier_kcal,
            ),
        ),
        chirality_terms=(),
        ligand_elements=("C", "C", "C", "C"),
        ligand_charges=(0.0, 0.0, 0.0, 0.0),
    )


def test_torsion_energy_canonical_dihedrals() -> None:
    ff = _torsion_only_ff(periodicity=3, phase_rad=0.0, barrier_kcal=1.4)
    # Eclipsed (phi = 0): V_n * (1 + cos(0)) = 2 * V_n
    c_eclipsed = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.5], [1.0, 0.0, 1.5]],
        dtype=np.float64,
    )
    e_ec, g_ec = torsion_energy_and_grad(c_eclipsed, ff.torsions)
    assert e_ec == pytest.approx(2.8, rel=1e-12, abs=1e-12)
    assert np.allclose(g_ec, 0.0, atol=1e-12)

    # Trans (phi = pi) with n=3, gamma=0: 1 + cos(3*pi) = 0
    c_trans = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.5], [-1.0, 0.0, 1.5]],
        dtype=np.float64,
    )
    e_tr, g_tr = torsion_energy_and_grad(c_trans, ff.torsions)
    assert e_tr == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(g_tr, 0.0, atol=1e-12)

    # Orthogonal (phi = +pi/2) with n=1, gamma=pi/3
    ff_n1 = _torsion_only_ff(periodicity=1, phase_rad=math.pi / 3.0, barrier_kcal=2.0)
    c_perp = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.5], [0.0, 1.0, 1.5]],
        dtype=np.float64,
    )
    e_perp, _ = torsion_energy_and_grad(c_perp, ff_n1.torsions)
    expected = 2.0 * (1.0 + math.cos(math.pi / 2.0 - math.pi / 3.0))
    assert e_perp == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    ("periodicity", "phase_rad", "barrier_kcal"),
    [
        (1, 0.25, 1.8),
        (2, math.pi, 3.0),
        (3, 0.4, 1.4),
        (3, 0.0, 2.2),
    ],
)
def test_torsion_gradients_match_central_finite_differences(
    periodicity: int, phase_rad: float, barrier_kcal: float
) -> None:
    ff = _torsion_only_ff(periodicity=periodicity, phase_rad=phase_rad, barrier_kcal=barrier_kcal)
    # Non-orthogonal 3D conformation (arbitrary bond lengths and oblique bond angles)
    coords = np.array(
        [
            [1.15, -0.42, -0.63],
            [0.12, 0.18, 0.05],
            [-0.25, 0.87, 1.32],
            [0.74, 1.93, 1.78],
        ],
        dtype=np.float64,
    )
    _, analytic = torsion_energy_and_grad(coords, ff.torsions)

    h = 1e-6
    fd = np.zeros_like(coords)
    for a in range(coords.shape[0]):
        for d in range(3):
            cp = coords.copy()
            cm = coords.copy()
            cp[a, d] += h
            cm[a, d] -= h
            ep, _ = torsion_energy_and_grad(cp, ff.torsions)
            em, _ = torsion_energy_and_grad(cm, ff.torsions)
            fd[a, d] = (ep - em) / (2.0 * h)

    assert np.allclose(analytic, fd, atol=1e-7, rtol=1e-7)
    # Exact Newton third-law translational and rotational invariance
    assert np.allclose(analytic.sum(axis=0), 0.0, atol=1e-12)
    torques = np.cross(coords, analytic).sum(axis=0)
    assert np.allclose(torques, 0.0, atol=1e-12)


def test_intramolecular_energy_and_grad_with_torsions_matches_fd() -> None:
    graph = parse_smiles("CCCCO")
    ff = build_pocket_force_field(graph)
    rng = np.random.default_rng(11)
    coords = rng.normal(0.0, 1.4, size=(len(graph.atoms), 3))
    _, analytic = intramolecular_energy_and_grad(coords, ff)
    h = 1e-6
    fd = np.zeros_like(coords)
    for a in range(coords.shape[0]):
        for d in range(3):
            cp = coords.copy()
            cm = coords.copy()
            cp[a, d] += h
            cm[a, d] -= h
            ep, _ = intramolecular_energy_and_grad(cp, ff)
            em, _ = intramolecular_energy_and_grad(cm, ff)
            fd[a, d] = (ep - em) / (2.0 * h)
    assert np.allclose(analytic, fd, atol=1e-6, rtol=1e-6)
    # Implicit 1-2 and 1-3 exclusion check even with bonds=(), angles=() and zero barrier
    zero_barrier = _torsion_only_ff(periodicity=3, phase_rad=0.0, barrier_kcal=0.0)
    far_14 = np.array(
        [[0.0, 0.0, -50.0], [0.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 1.5, 50.0]],
        dtype=np.float64,
    )
    e_13_excluded, _ = intramolecular_energy_and_grad(far_14, zero_barrier)
    assert e_13_excluded < 1e-12


def test_torsion_collinear_and_zero_bond_guards() -> None:
    ff = _torsion_only_ff(periodicity=3, phase_rad=0.3, barrier_kcal=1.4)
    collinear = np.array(
        [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.5], [1.0, 0.2, 2.0]],
        dtype=np.float64,
    )
    e_col, g_col = intramolecular_energy_and_grad(collinear, ff)
    assert math.isfinite(e_col)
    assert np.all(np.isfinite(g_col))

    zero_bond = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    e_zb, g_zb = intramolecular_energy_and_grad(zero_bond, ff)
    assert math.isfinite(e_zb)
    assert np.all(np.isfinite(g_zb))


def test_pocket_equivariance_and_relaxation_with_torsions() -> None:
    graph = parse_smiles("CCCCO")
    ff = build_pocket_force_field(graph)
    assert len(ff.torsions) >= 2

    pocket = BindingPocket(
        pocket_id="test_pocket",
        target_name="kinase_mock",
        atoms=(
            PocketAtom(0, "N", (3.8, 0.5, 0.0), partial_charge=-0.3),
            PocketAtom(1, "O", (-3.5, 2.2, 1.2), partial_charge=-0.4),
            PocketAtom(2, "C", (0.2, -3.9, -0.8), partial_charge=0.05),
        ),
    )

    rng = np.random.default_rng(7)
    lig_coords = rng.normal(0.0, 1.2, size=(len(graph.atoms), 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    t = np.array([4.5, -2.1, 3.3], dtype=np.float64)

    err = pocket_equivariance_error(lig_coords, ff, pocket, q, t)
    assert err < 1e-10

    dock = relax_ligand_in_pocket(graph, pocket, n_steps=120, seed=19)
    assert math.isfinite(dock.final_total_energy)
    assert dock.final_total_energy <= dock.initial_total_energy + 1e-6
    assert np.all(np.isfinite(dock.ligand_coordinates))
