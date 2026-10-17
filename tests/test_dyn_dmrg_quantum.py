r"""DMRG verification: MPO correctness, exact-diagonalization agreement, size limits.

Status: experimental
"""
from __future__ import annotations

import numpy as np

from apedp_dyn.quantum.dmrg import (
    THERMODYNAMIC_ENERGY_PER_SITE_HEISENBERG,
    MatrixProductState,
    dmrg_ground_state,
    heisenberg_exact_ground_energy,
    heisenberg_mpo,
    heisenberg_sparse_hamiltonian,
    mps_energy_expectation,
)


def _random_mps(n_sites: int, bond_dim: int, seed: int) -> MatrixProductState:
    rng = np.random.default_rng(seed)
    tensors = []
    for site in range(n_sites):
        d_l = 1 if site == 0 else bond_dim
        d_r = 1 if site == n_sites - 1 else bond_dim
        tensors.append(rng.standard_normal((d_l, 2, d_r)))
    return MatrixProductState(tensors)


def test_mpo_matches_sparse_hamiltonian_expectation():
    n_sites = 4
    j_coupling = 1.0
    field = 0.37
    mps = _random_mps(n_sites, bond_dim=3, seed=11)
    psi = mps.to_statevector()
    psi = psi / np.linalg.norm(psi)

    h_sparse = heisenberg_sparse_hamiltonian(n_sites, j_coupling=j_coupling, field=field)
    expected = float(psi @ (h_sparse @ psi))

    mpo = heisenberg_mpo(n_sites, j_coupling=j_coupling, field=field)
    got = mps_energy_expectation(mps, mpo)
    assert abs(got - expected) < 1e-10, f"{got} vs {expected}"


def test_mpo_matches_sparse_expectation_n2_j_and_field():
    n_sites = 2
    mps = _random_mps(n_sites, bond_dim=2, seed=7)
    psi = mps.to_statevector()
    psi = psi / np.linalg.norm(psi)
    h_sparse = heisenberg_sparse_hamiltonian(n_sites, field=0.5)
    expected = float(psi @ (h_sparse @ psi))
    got = mps_energy_expectation(mps, heisenberg_mpo(n_sites, field=0.5))
    assert abs(got - expected) < 1e-12


def test_dmrg_matches_exact_diagonalization_n8():
    result, _ = dmrg_ground_state(8, max_bond_dim=32, max_sweeps=4, energy_tol=1e-10)
    ed = heisenberg_exact_ground_energy(8)
    assert result.converged
    assert abs(result.energy - ed) < 1e-8, f"dmrg={result.energy} ed={ed}"


def test_dmrg_residual_is_eigenstate_n8():
    result, mps = dmrg_ground_state(8, max_bond_dim=32, max_sweeps=4, energy_tol=1e-10)
    psi = mps.to_statevector()
    psi = psi / np.linalg.norm(psi)
    h_sparse = heisenberg_sparse_hamiltonian(8)
    residual = h_sparse @ psi - result.energy * psi
    assert np.linalg.norm(residual) < 1e-7, f"residual norm {np.linalg.norm(residual)}"


def test_dmrg_energy_expectation_consistency_n10():
    result, mps = dmrg_ground_state(10, max_bond_dim=32, max_sweeps=6, energy_tol=1e-10)
    mpo = heisenberg_mpo(10)
    expectation = mps_energy_expectation(mps, mpo)
    assert abs(expectation - result.energy) < 1e-7 * max(1.0, abs(result.energy))


def test_dmrg_zeeman_field_matches_ed():
    # h = 0.8 exceeds the N=8 singlet-triplet gap, so the field genuinely
    # selects a polarized ground state rather than leaving the singlet intact.
    result, _ = dmrg_ground_state(8, field=0.8, max_bond_dim=32, max_sweeps=4, energy_tol=1e-10)
    ed = heisenberg_exact_ground_energy(8, field=0.8)
    assert abs(result.energy - ed) < 1e-8, f"dmrg={result.energy} ed={ed}"


def test_dmrg_long_chain_thermodynamic_limit():
    result, _ = dmrg_ground_state(
        24, max_bond_dim=20, max_sweeps=2, energy_tol=0.0, solver_tol=1e-7
    )
    energy_per_site = result.energy_per_site
    assert abs(energy_per_site - THERMODYNAMIC_ENERGY_PER_SITE_HEISENBERG) < 0.01, energy_per_site


def test_dmrg_energy_improves_or_plateaus_across_sweeps():
    result, _ = dmrg_ground_state(12, max_bond_dim=32, max_sweeps=3, energy_tol=0.0, solver_tol=1e-8)
    sweeps = result.sweep_energies
    assert len(sweeps) >= 2
    assert sweeps[-1] <= sweeps[0] + 1e-10
    ed = heisenberg_exact_ground_energy(12)
    assert abs(sweeps[-1] - ed) < 1e-6 * max(1.0, abs(ed))
