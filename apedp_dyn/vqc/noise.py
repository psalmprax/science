"""Density matrices and noise channels: the exact ensemble, not a trajectory.

Status: experimental

:class:`~apedp_dyn.vqc.backends.StatevectorBackend` samples Pauli errors, which
gives the right ensemble only in the limit of many shots and cannot represent a
non-Pauli channel at all. This module is the exact counterpart for small systems:

    rho -> sum_k K_k rho K_k^dagger

with the channels the VQC milestone names -- bit flip, phase flip, depolarising,
amplitude damping and pure dephasing -- and the checks that make an exact
simulation falsifiable rather than decorative: trace preservation, Hermiticity,
positivity (`min eigenvalue >= -1e-9`), purity, and fidelity against a pure
reference.

The channels are given as Kraus operators, so a valid density matrix is the
*result* of the arithmetic rather than an assertion: `is_valid()` recomputes it,
and the tests apply every channel on a grid of probabilities.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = [
    "CHANNELS",
    "DensityMatrix",
    "amplitude_damping",
    "bit_flip",
    "density_matrix_after",
    "dephasing",
    "depolarizing",
    "phase_flip",
    "purity_of_bell_after_depolarizing",
]

_I: np.ndarray = np.eye(2, dtype=np.complex128)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def _check(p: float, name: str) -> float:
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} probability {p} is not in [0, 1]")
    return p


def bit_flip(p: float) -> list[np.ndarray]:
    """`X` with probability p."""
    p = _check(p, "bit_flip")
    return [math.sqrt(1.0 - p) * _I, math.sqrt(p) * _X]


def phase_flip(p: float) -> list[np.ndarray]:
    """`Z` with probability p."""
    p = _check(p, "phase_flip")
    return [math.sqrt(1.0 - p) * _I, math.sqrt(p) * _Z]


def depolarizing(p: float) -> list[np.ndarray]:
    """A uniformly random Pauli error with total probability p."""
    p = _check(p, "depolarizing")
    return [math.sqrt(1.0 - p) * _I,
            math.sqrt(p / 3.0) * _X,
            math.sqrt(p / 3.0) * _Y,
            math.sqrt(p / 3.0) * _Z]


def amplitude_damping(gamma: float) -> list[np.ndarray]:
    """`|1> -> |0>` with probability gamma; the T1 channel."""
    gamma = _check(gamma, "amplitude_damping")
    return [np.array([[1.0, 0.0], [0.0, math.sqrt(1.0 - gamma)]], dtype=np.complex128),
            np.array([[0.0, math.sqrt(gamma)], [0.0, 0.0]], dtype=np.complex128)]


def dephasing(p: float) -> list[np.ndarray]:
    """Loss of phase coherence with probability p; the T2 channel."""
    p = _check(p, "dephasing")
    return [np.array([[1.0, 0.0], [0.0, math.sqrt(1.0 - p)]], dtype=np.complex128),
            np.array([[0.0, 0.0], [0.0, math.sqrt(p)]], dtype=np.complex128)]


CHANNELS = {
    "bit_flip": bit_flip,
    "phase_flip": phase_flip,
    "depolarizing": depolarizing,
    "amplitude_damping": amplitude_damping,
    "dephasing": dephasing,
}


class DensityMatrix:
    """An exact density matrix for a small register."""

    def __init__(self, num_qubits: int) -> None:
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be positive, got {num_qubits}")
        if num_qubits > 12:
            raise ValueError(f"{num_qubits} qubits is a {2 ** num_qubits}^2 matrix; "
                             f"this tier is for demonstrations, not for hardware-sized states")
        self.num_qubits = int(num_qubits)
        dim = 1 << self.num_qubits
        self.rho = np.zeros((dim, dim), dtype=np.complex128)
        self.rho[0, 0] = 1.0

    # -- construction ------------------------------------------------------
    def set_pure(self, vector) -> None:
        """Set a pure state from amplitudes, checking normalisation."""
        psi = np.asarray(vector, dtype=np.complex128).reshape(-1)
        if psi.size != self.rho.shape[0]:
            raise ValueError(f"expected {self.rho.shape[0]} amplitudes, got {psi.size}")
        norm = float(np.sqrt(np.vdot(psi, psi).real))
        if abs(norm - 1.0) > 1e-9:
            raise ValueError(f"state is not normalised (norm {norm})")
        self.rho = np.outer(psi, psi.conj())

    def set_basis_state(self, value: int) -> None:
        self.rho[:] = 0.0
        self.rho[int(value), int(value)] = 1.0

    # -- evolution ---------------------------------------------------------
    def apply_unitary(self, qubit: int, unitary) -> None:
        """`rho -> U rho U^dagger`, with `U` acting on one qubit."""
        u = np.asarray(unitary, dtype=np.complex128)
        if u.shape != (2, 2):
            raise ValueError(f"expected a 2x2 unitary, got {u.shape}")
        # A unitary is the single-Kraus case, so there is one code path for both.
        self.apply_kraus(qubit, [u])

    def apply_kraus(self, qubit: int, kraus) -> None:
        """`rho -> sum_k K_k rho K_k^dagger` with every `K_k` acting on `qubit`."""
        dim = self.rho.shape[0]
        hi, lo = dim >> (qubit + 1), 1 << qubit
        # Axes: (row high, row bit, row low, column high, column bit, column low).
        # The labels are case-distinct on purpose: `q` and `v` are the bit indices,
        # and mixing them up is exactly the kind of bug this reshape invites.
        view = self.rho.reshape(hi, 2, lo, hi, 2, lo)
        out = np.zeros_like(view)
        for operator in kraus:
            k = np.asarray(operator, dtype=np.complex128)
            if k.shape != (2, 2):
                raise ValueError(f"expected a 2x2 Kraus operator, got {k.shape}")
            # K on the row index...
            half = np.einsum("qb,pbrstu->pqrstu", k, view, optimize=True)
            # ...then K^dagger on the column index.
            out += np.einsum("pqrstu,vt->pqrsvu", half, k.conj(), optimize=True)
        self.rho = out.reshape(dim, dim)

    def apply_channel(self, qubit: int, name: str, probability: float) -> None:
        """Apply one of `CHANNELS` by name."""
        try:
            kraus = CHANNELS[name](probability)
        except KeyError as error:
            raise ValueError(f"unknown channel {name!r}; known: {sorted(CHANNELS)}") from error
        self.apply_kraus(qubit, kraus)

    # -- measurement and metrics ------------------------------------------
    def probabilities(self, qubit: int) -> tuple[float, float]:
        diagonal = np.real(np.diag(self.rho))
        index = np.arange(diagonal.size)
        one = ((index >> qubit) & 1).astype(bool)
        return float(diagonal[~one].sum()), float(diagonal[one].sum())

    def measurement_distribution(self, qubits) -> dict[str, float]:
        """Z-basis distribution over `qubits`, bit order as given."""
        diagonal = np.real(np.diag(self.rho))
        index = np.arange(diagonal.size)
        totals: dict[str, float] = {}
        for basis, weight in zip(index, diagonal, strict=True):
            if weight <= 1e-15:
                continue
            key = "".join("1" if (basis >> qubit) & 1 else "0" for qubit in qubits)
            totals[key] = totals.get(key, 0.0) + float(weight)
        return dict(sorted(totals.items()))

    def trace(self) -> float:
        return float(np.real(np.trace(self.rho)))

    def purity(self) -> float:
        """`Tr(rho^2)`: 1.0 for a pure state, 1/d for the maximally mixed one."""
        return float(np.real(np.trace(self.rho @ self.rho)))

    def min_eigenvalue(self) -> float:
        return float(np.min(np.linalg.eigvalsh(self.rho)))

    def fidelity_with(self, vector) -> float:
        """`<psi|rho|psi>` for a pure reference state."""
        psi = np.asarray(vector, dtype=np.complex128).reshape(-1)
        return float(np.real(psi.conj() @ self.rho @ psi))

    def is_valid(self, *, tolerance: float = 1e-9) -> bool:
        """Hermitian, unit trace, positive semidefinite -- recomputed, not assumed."""
        hermitian = np.allclose(self.rho, self.rho.conj().T, atol=tolerance)
        return bool(hermitian
                    and abs(self.trace() - 1.0) <= tolerance
                    and self.min_eigenvalue() >= -tolerance)

    def as_dict(self) -> dict[str, Any]:
        return {"trace": self.trace(), "purity": self.purity(),
                "min_eigenvalue": self.min_eigenvalue(), "num_qubits": self.num_qubits}


def density_matrix_after(circuit_gates, *, num_qubits: int, channels=()) -> DensityMatrix:
    """Apply gates then channels, in order, to |0...0>.

    `circuit_gates` is a sequence of `(name, targets, params)` handled through
    :func:`apedp_dyn.vqc.gates.matrix_for` for one-qubit gates and rejected for
    anything else -- this tier is a channel laboratory, not a second executor.
    Use the statevector backend to *run* circuits; this function exists so a
    noise claim can be checked against an independently constructed density
    matrix.
    """
    from apedp_dyn.vqc.gates import GATE_SPECS, matrix_for

    state = DensityMatrix(num_qubits)
    for name, targets, params in circuit_gates:
        spec = GATE_SPECS[name]
        if spec.arity != 1:
            raise ValueError(f"{name!r} is not a one-qubit gate; "
                             f"use a circuit backend for multi-qubit gates")
        state.apply_unitary(targets[0], matrix_for(name, tuple(params)))
    for qubit, channel, probability in channels:
        state.apply_channel(qubit, channel, probability)
    return state


def purity_of_bell_after_depolarizing(p: float) -> float:
    """`(1 - p)^2 + p^2 / 3` for one depolarised half of a Bell pair.

    A Bell state under `depolarizing(p)` on one qubit becomes a uniform mixture
    over the four Bell states with weights `(1-p, p/3, p/3, p/3)`; the fidelity
    to the original is therefore `1-p` and the purity is the sum of squared
    weights. Both are analytic, so both are checkable.
    """
    return (1.0 - p) ** 2 + p ** 2 / 3.0
