r"""Density Matrix Renormalization Group (DMRG) for 1D Quantum Spin Chains.

Status: experimental

Scope & honesty ceiling:
- Implements a real two-site DMRG in the Matrix Product State (MPS) / Matrix
  Product Operator (MPO) representation for the open spin-1/2 Heisenberg chain
  ``H = J * sum_i S_i . S_{i+1} - h * sum_i S^z_i``.
- Ground states are *verified against exact diagonalization* for small chains
  and against the known thermodynamic-limit energy per site ``1/4 - ln 2`` for
  long chains. This module does not implement time evolution, finite
  temperature, or frustrated 2D lattice DMRG.
- The MPS machinery here is numerically independent of the Tensor-Train
  discovery path in ``apedp_dyn.epistemic.high_dim_pde`` (which stores PDE
  fields as TT cores rather than variationally minimizing a quantum energy).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix, eye, kron
from scipy.sparse.linalg import LinearOperator, eigsh

_SITE_DIM = 2
_BOND_DIM_MPO = 5

_SZ = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=np.float64)
_SP = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
_SM = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
_ID = np.eye(2, dtype=np.float64)


@dataclass
class DmrgResult:
    """Outcome of a DMRG ground-state search."""

    energy: float
    energy_per_site: float
    n_sites: int
    max_bond_dim_used: int
    sweep_energies: list[float] = field(default_factory=list)
    converged: bool = False
    sweeps_run: int = 0


class MatrixProductState:
    """Open-boundary MPS with per-site ``(D_l, d, D_r)`` tensors."""

    def __init__(self, tensors: list[np.ndarray]) -> None:
        if not tensors:
            raise ValueError("MPS requires at least one tensor")
        self.tensors = [np.asarray(t, dtype=np.float64) for t in tensors]
        self._check_shapes()

    def _check_shapes(self) -> None:
        n = len(self.tensors)
        for idx, tensor in enumerate(self.tensors):
            if tensor.ndim != 3:
                raise ValueError(f"site {idx} tensor must be rank-3, got shape {tensor.shape}")
            if tensor.shape[1] != _SITE_DIM:
                raise ValueError(f"site {idx} physical dim must be {_SITE_DIM}")
        for idx in range(n - 1):
            if self.tensors[idx].shape[2] != self.tensors[idx + 1].shape[0]:
                raise ValueError(f"bond mismatch between sites {idx} and {idx + 1}")

    @property
    def n_sites(self) -> int:
        return len(self.tensors)

    @property
    def max_bond_dim(self) -> int:
        return max(int(t.shape[0]) for t in self.tensors)

    def to_statevector(self) -> np.ndarray:
        """Contract the MPS into a dense ``2**n`` statevector."""
        psi = self.tensors[0][0]  # (d, D)
        for tensor in self.tensors[1:]:
            psi = np.tensordot(psi, tensor, axes=([1], [0]))
            psi = psi.reshape(-1, tensor.shape[2])
        return psi.reshape(-1)

    def norm(self) -> float:
        return float(np.linalg.norm(self.to_statevector()))


_CH_START = 0
_CH_DONE = 4


def heisenberg_mpo(n_sites: int, j_coupling: float = 1.0, field: float = 0.0) -> list[np.ndarray]:
    r"""Five-channel MPO for ``H = J * sum S_i . S_{i+1} - h * sum S^z_i`` (open chain).

    Each MPO path represents exactly one term of the Hamiltonian sum.
    Channels: 0 = no term emitted yet, 1..3 = open two-site halves
    (S+, Sz, S-; must close at the immediately following site),
    4 = term completed (identity thereafter). The Zeeman term is a single-site
    term that terminates the path immediately, so no spurious ``h^2`` products
    can be generated.
    """
    if n_sites < 2:
        raise ValueError("Heisenberg MPO requires n_sites >= 2")
    m = _BOND_DIM_MPO
    w = np.zeros((m, m, _SITE_DIM, _SITE_DIM), dtype=np.float64)

    w[0, 0] = _ID
    w[0, 1] = _SP
    w[0, 2] = _SZ
    w[0, 3] = _SM
    w[0, 4] = -field * _SZ

    w[1, 4] = 0.5 * j_coupling * _SM
    w[2, 4] = j_coupling * _SZ
    w[3, 4] = 0.5 * j_coupling * _SP

    w[4, 4] = _ID
    return [w.copy() for _ in range(n_sites)]


def _left_boundary() -> np.ndarray:
    l0 = np.zeros((1, _BOND_DIM_MPO, 1), dtype=np.float64)
    l0[0, 0, 0] = 1.0
    return l0


def _right_boundary() -> np.ndarray:
    rn = np.zeros((1, _BOND_DIM_MPO, 1), dtype=np.float64)
    rn[0, _CH_DONE, 0] = 1.0
    return rn


def _extend_left_env(env: np.ndarray, a: np.ndarray, w: np.ndarray) -> np.ndarray:
    t1 = np.einsum("abc,asd->bcsd", env, a)
    t2 = np.einsum("bcsd,bets->cetd", t1, w)
    return np.einsum("cetd,ctf->def", t2, a)


def _extend_right_env(env: np.ndarray, a: np.ndarray, w: np.ndarray) -> np.ndarray:
    t1 = np.einsum("def,asd->aesf", env, a)
    t2 = np.einsum("aesf,bets->abtf", t1, w)
    return np.einsum("abtf,ctf->abc", t2, a)


def _effective_matvec(
    theta: np.ndarray,
    l_env: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    r_env: np.ndarray,
) -> np.ndarray:
    r"""Matrix-free two-site effective Hamiltonian application ``H_eff @ theta``.

    Contracts ``L_env . W1 . W2 . R_env`` against ``theta`` without building the
    dense ``(D_l d^2 D_r)^2`` matrix.
    """
    v1 = np.einsum("cijg,abc->baijg", theta, l_env)
    v2 = np.einsum("baijg,bcis->acsjg", v1, w1)
    v3 = np.einsum("acsjg,cdjr->adsrg", v2, w2)
    return np.einsum("adsrg,fdg->asrf", v3, r_env)


def _solve_two_site(
    theta0: np.ndarray,
    l_env: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    r_env: np.ndarray,
    tol: float = 1e-11,
    maxiter: int = 4000,
) -> tuple[float, np.ndarray]:
    """Ground eigenpair of the matrix-free two-site operator via Lanczos."""
    shape = theta0.shape
    size = int(np.prod(shape))

    def matvec(x: np.ndarray) -> np.ndarray:
        return _effective_matvec(x.reshape(shape), l_env, w1, w2, r_env).ravel()

    op = LinearOperator((size, size), matvec=matvec, dtype=np.float64)
    v0 = theta0.ravel()
    norm0 = float(np.linalg.norm(v0))
    if norm0 > 0:
        v0 = v0 / norm0
    ncv = min(size, 40)
    vals, vecs = eigsh(op, k=1, which="SA", v0=v0, tol=tol, maxiter=maxiter, ncv=ncv)
    return float(vals[0]), vecs[:, 0].reshape(shape)


def _build_left_environments(mps: MatrixProductState, mpo: list[np.ndarray]) -> list[np.ndarray]:
    envs = [_left_boundary()]
    env = envs[0]
    for site in range(mps.n_sites):
        env = _extend_left_env(env, mps.tensors[site], mpo[site])
        envs.append(env)
    return envs


def _build_right_environments(mps: MatrixProductState, mpo: list[np.ndarray]) -> list[np.ndarray]:
    envs: list[np.ndarray | None] = [None] * (mps.n_sites + 1)
    env = _right_boundary()
    envs[mps.n_sites] = env
    for site in range(mps.n_sites - 1, -1, -1):
        env = _extend_right_env(env, mps.tensors[site], mpo[site])
        envs[site] = env
    return [e for e in envs if e is not None]


def _split_theta(
    theta: np.ndarray,
    max_bond_dim: int,
    cutoff: float,
    left_canonical: bool,
) -> tuple[np.ndarray, np.ndarray]:
    d_l, d_p1, d_p2, d_r = theta.shape
    matrix = theta.reshape(d_l * d_p1, d_p2 * d_r)
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    keep = int(np.sum(s > cutoff))
    keep = min(max(keep, 1), max_bond_dim, s.shape[0])
    u = u[:, :keep]
    s = s[:keep]
    vh = vh[:keep, :]
    norm_s = float(np.linalg.norm(s))
    if norm_s > 1e-15:
        s = s / norm_s
    if left_canonical:
        tensor_left = u.reshape(d_l, d_p1, keep)
        tensor_right = (s[:, None] * vh).reshape(keep, d_p2, d_r)
    else:
        tensor_left = (u * s[None, :]).reshape(d_l, d_p1, keep)
        tensor_right = vh.reshape(keep, d_p2, d_r)
    return tensor_left, tensor_right


def _initial_mps(n_sites: int, max_bond_dim: int, rng: np.random.Generator) -> MatrixProductState:
    tensors: list[np.ndarray] = []
    for site in range(n_sites):
        d_l = 1 if site == 0 else min(max_bond_dim, 2 ** site)
        d_r = 1 if site == n_sites - 1 else min(max_bond_dim, 2 ** (site + 1))
        tensors.append(rng.standard_normal((d_l, _SITE_DIM, d_r)) * 0.1)
    # Right-canonicalize so that right environments are orthonormal from sweep 0
    for site in range(n_sites - 1, 0, -1):
        t = tensors[site]
        dl, dp, dr = t.shape
        mat = t.reshape(dl, dp * dr)
        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        tensors[site] = vh.reshape(vh.shape[0], dp, dr)
        tensors[site - 1] = np.tensordot(tensors[site - 1], u * s, axes=([2], [0]))
    norm0 = float(np.linalg.norm(tensors[0]))
    if norm0 > 1e-15:
        tensors[0] = tensors[0] / norm0
    return MatrixProductState(tensors)


def dmrg_ground_state(
    n_sites: int,
    j_coupling: float = 1.0,
    field: float = 0.0,
    max_bond_dim: int = 64,
    max_sweeps: int = 12,
    energy_tol: float = 1e-12,
    svd_cutoff: float = 1e-12,
    solver_tol: float = 1e-11,
    seed: int = 2026,
) -> tuple[DmrgResult, MatrixProductState]:
    """Run two-site DMRG sweeps until energy convergence or sweep budget."""
    if n_sites < 2:
        raise ValueError("DMRG requires n_sites >= 2")
    rng = np.random.default_rng(seed)
    mpo = heisenberg_mpo(n_sites, j_coupling=j_coupling, field=field)
    mps = _initial_mps(n_sites, max_bond_dim, rng)

    sweep_energies: list[float] = []
    converged = False
    sweeps_run = 0
    energy = float("nan")

    for _ in range(max_sweeps):
        sweeps_run += 1
        left_envs = _build_left_environments(mps, mpo)
        right_envs = _build_right_environments(mps, mpo)

        for site in range(n_sites - 1):
            theta0 = np.tensordot(mps.tensors[site], mps.tensors[site + 1], axes=([2], [0]))
            energy, theta = _solve_two_site(
                theta0, left_envs[site], mpo[site], mpo[site + 1], right_envs[site + 2], tol=solver_tol
            )
            a_left, a_right = _split_theta(theta, max_bond_dim, svd_cutoff, left_canonical=True)
            mps.tensors[site] = a_left
            mps.tensors[site + 1] = a_right
            left_envs[site + 1] = _extend_left_env(left_envs[site], a_left, mpo[site])

        right_envs = _build_right_environments(mps, mpo)
        for site in range(n_sites - 2, -1, -1):
            theta0 = np.tensordot(mps.tensors[site], mps.tensors[site + 1], axes=([2], [0]))
            energy, theta = _solve_two_site(
                theta0, left_envs[site], mpo[site], mpo[site + 1], right_envs[site + 2], tol=solver_tol
            )
            a_left, a_right = _split_theta(theta, max_bond_dim, svd_cutoff, left_canonical=False)
            mps.tensors[site] = a_left
            mps.tensors[site + 1] = a_right
            right_envs[site + 1] = _extend_right_env(right_envs[site + 2], a_right, mpo[site + 1])

        sweep_energies.append(float(energy))
        if len(sweep_energies) >= 2 and abs(sweep_energies[-1] - sweep_energies[-2]) < energy_tol:
            converged = True
            break

    result = DmrgResult(
        energy=float(energy),
        energy_per_site=float(energy) / n_sites,
        n_sites=n_sites,
        max_bond_dim_used=mps.max_bond_dim,
        sweep_energies=sweep_energies,
        converged=converged,
        sweeps_run=sweeps_run,
    )
    return result, mps


def heisenberg_sparse_hamiltonian(
    n_sites: int,
    j_coupling: float = 1.0,
    field: float = 0.0,
) -> csr_matrix:
    """Dense-kron sparse Hamiltonian for the open spin-1/2 Heisenberg chain."""
    if n_sites < 2:
        raise ValueError("exact diagonalization requires n_sites >= 2")
    if n_sites > 14:
        raise ValueError("exact diagonalization is limited to n_sites <= 14")

    def kron_chain(ops: list[csr_matrix]) -> csr_matrix:
        out = ops[0]
        for op in ops[1:]:
            out = kron(out, op, format="csr")
        return out

    def site_op(op: np.ndarray, site: int) -> csr_matrix:
        ops = [csr_matrix(eye(2)) for _ in range(n_sites)]
        ops[site] = csr_matrix(op)
        return kron_chain(ops)

    h = csr_matrix((2 ** n_sites, 2 ** n_sites), dtype=np.float64)
    sz_ops = [site_op(_SZ, s) for s in range(n_sites)]
    sp_ops = [site_op(_SP, s) for s in range(n_sites)]
    sm_ops = [site_op(_SM, s) for s in range(n_sites)]
    for s in range(n_sites - 1):
        h = h + j_coupling * (sz_ops[s] @ sz_ops[s + 1])
        h = h + 0.5 * j_coupling * (sp_ops[s] @ sm_ops[s + 1])
        h = h + 0.5 * j_coupling * (sm_ops[s] @ sp_ops[s + 1])
    for s in range(n_sites):
        h = h - field * sz_ops[s]
    return h


def heisenberg_exact_ground_energy(
    n_sites: int,
    j_coupling: float = 1.0,
    field: float = 0.0,
) -> float:
    """Exact-diagonalization referee energy for the open Heisenberg chain."""
    h = heisenberg_sparse_hamiltonian(n_sites, j_coupling=j_coupling, field=field)
    vals, _ = eigsh(h, k=1, which="SA", maxiter=10000, tol=1e-12)
    return float(vals[0])


def mps_energy_expectation(mps: MatrixProductState, mpo: list[np.ndarray]) -> float:
    """``<psi|H|psi> / <psi|psi>`` via full environment contraction."""
    left = _left_boundary()
    for site in range(mps.n_sites):
        left = _extend_left_env(left, mps.tensors[site], mpo[site])
    right = _right_boundary()
    numerator = float(np.einsum("abc,abc->", left, right))
    norm = mps.norm()
    return numerator / (norm * norm)


THERMODYNAMIC_ENERGY_PER_SITE_HEISENBERG: float = 0.25 - np.log(2.0)
