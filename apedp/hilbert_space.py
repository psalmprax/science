"""Hilbert-space ontology: machine-checkable semantics for finite-dimensional
quantum states, observables, measurement, and composition.

Scope (honest): finite-dimensional systems only. No QFT, no continuous
variables, no open-system dynamics. Measurement is Born-rule bookkeeping
with seeded sampling, not a theory of collapse.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QuantumState:
    """A state vector with dimension and normalization checks."""
    amplitudes: list
    label: str = ""

    def __post_init__(self) -> None:
        import numpy as np
        self._vec = np.asarray(self.amplitudes, dtype=complex).ravel()

    @property
    def dim(self) -> int:
        return int(self._vec.size)

    def norm(self) -> float:
        import numpy as np
        return float(np.linalg.norm(self._vec))

    def is_normalized(self, tol: float = 1e-9) -> bool:
        return abs(self.norm() - 1.0) <= tol

    def density_matrix(self):
        v = self._vec.reshape(-1, 1)
        return v @ v.conj().T


@dataclass
class QuantumOperator:
    """A square matrix with Hermitian/unitary checks and spectra."""
    matrix: list
    label: str = ""

    def __post_init__(self) -> None:
        import numpy as np
        self._mat = np.asarray(self.matrix, dtype=complex)

    def is_square(self) -> bool:
        return self._mat.ndim == 2 and self._mat.shape[0] == self._mat.shape[1]

    def is_hermitian(self, tol: float = 1e-9) -> bool:
        import numpy as np
        if not self.is_square():
            return False
        return bool(np.allclose(self._mat, self._mat.conj().T, atol=tol))

    def is_unitary(self, tol: float = 1e-9) -> bool:
        import numpy as np
        if not self.is_square():
            return False
        eye = np.eye(self._mat.shape[0])
        return bool(np.allclose(self._mat @ self._mat.conj().T, eye, atol=tol))

    def eigenvalues(self) -> list:
        import numpy as np
        return [complex(v) for v in np.linalg.eigvalsh(self._mat)
                ] if self.is_hermitian() else [
                    complex(v) for v in np.linalg.eigvals(self._mat)]


@dataclass
class Observable:
    """A Hermitian operator read as a measurable quantity."""
    operator: QuantumOperator
    name: str = ""
    unit: str = ""

    def is_valid(self, tol: float = 1e-9) -> bool:
        return self.operator.is_hermitian(tol)

    def expectation(self, state: QuantumState) -> float:
        import numpy as np
        if state.dim != self.operator._mat.shape[0]:
            raise ValueError("state/operator dimension mismatch")
        return float(np.real(state._vec.conj() @ self.operator._mat @ state._vec))

    def born_probabilities(self, state: QuantumState) -> list:
        """Probabilities over the eigenbasis (sorted eigenorder)."""
        import numpy as np
        if not self.is_valid():
            raise ValueError("observable must be Hermitian")
        vals, vecs = np.linalg.eigh(self.operator._mat)
        probs = [float(abs(vecs[:, i].conj() @ state._vec) ** 2)
                 for i in range(len(vals))]
        return probs


@dataclass
class QuantumMeasurement:
    """Seeded Born-rule sampling from an observable + state."""
    observable: Observable
    state: QuantumState
    seed: int = 42

    def sample(self, n: int = 1000) -> dict:
        import numpy as np
        probs = self.observable.born_probabilities(self.state)
        rng = np.random.default_rng(self.seed)
        outcomes = rng.choice(len(probs), size=n, p=np.asarray(probs) / sum(probs))
        vals = self.observable.operator.eigenvalues()
        counts = [int((outcomes == i).sum()) for i in range(len(probs))]
        return {"eigenvalues": [complex(v) for v in vals],
                "probabilities": probs, "counts": counts, "n": n}


def compose_states(a: QuantumState, b: QuantumState) -> QuantumState:
    """Tensor-product composition for composite (many-body, small) systems."""
    import numpy as np
    vec = np.kron(a._vec, b._vec)
    return QuantumState(amplitudes=[complex(v) for v in vec],
                        label=f"{a.label}x{b.label}")


def reduced_purity(state: QuantumState, dim_a: int) -> float:
    """Purity of the reduced state: 1.0 separable-pure, <1 entangled/mixed."""
    import numpy as np
    d = state.dim
    if d % dim_a != 0:
        raise ValueError("dimensions do not factor")
    dim_b = d // dim_a
    psi = state._vec.reshape(dim_a, dim_b)
    rho_a = psi @ psi.conj().T
    return float(np.real(np.trace(rho_a @ rho_a)))


def truncated_ladder(dim: int):
    """Finite Fock-basis ladder operators (a, a†, n, x, p quadratures).

    Honest truncation: infinite-dimensional CV algebra approximated in a
    dim-dimensional cutoff. Canonical [x, p] = i holds only approximately;
    the defect shrinks with dim and is reported, not hidden.
    """
    import numpy as np
    a = np.zeros((dim, dim), dtype=complex)
    for n in range(1, dim):
        a[n - 1, n] = np.sqrt(n)
    ad = a.conj().T
    x = (a + ad) / np.sqrt(2.0)
    p = (a - ad) / (1j * np.sqrt(2.0))
    comm = x @ p - p @ x
    err = comm - 1j * np.eye(dim)
    defect = float(np.max(np.abs(err)))
    # Truncation error lives on the cutoff boundary row/column; the
    # interior (dim-1)x(dim-1) block is exact. Report both, hide neither.
    bulk = float(np.max(np.abs(err[:dim - 1, :dim - 1]))) if dim > 1 else 0.0
    return {"a": a, "adag": ad, "x": x, "p": p, "n": ad @ a,
            "commutator_defect": defect, "bulk_defect": bulk, "dim": dim}


def lindblad_step(rho, hamiltonian, jump_ops: list, dt: float,
                  gamma_list: list | None = None):
    """One Lindblad dissipator step: drho = -i[H,rho] + sum γ(LρL† − ½{L†L,ρ}).

    Small dense systems only. Returns the stepped density matrix (trace
    renormalized; renormalization reported, not hidden).
    """
    import numpy as np
    rho = np.asarray(rho, dtype=complex)
    h = np.asarray(hamiltonian, dtype=complex)
    if gamma_list is None:
        gamma_list = [1.0] * len(jump_ops)
    drho = -1j * (h @ rho - rho @ h)
    for op, g in zip(jump_ops, gamma_list):
        lop = np.asarray(op, dtype=complex)
        ldl = lop.conj().T @ lop
        drho = drho + float(g) * (lop @ rho @ lop.conj().T
                                  - 0.5 * (ldl @ rho + rho @ ldl))
    out = rho + drho * dt
    tr = float(np.real(np.trace(out)))
    return {"rho": out / tr if tr != 0 else out,
            "trace_before_renorm": tr, "dt": dt}


@dataclass
class QuantumSystem:
    """A governed system: registers states/observables with dimensional
    checks delegated to the caller's unit strings (no Hilbert-unit conflation)."""
    states: dict = field(default_factory=dict)
    observables: dict = field(default_factory=dict)

    def add_state(self, name: str, state: QuantumState) -> None:
        if not state.is_normalized():
            raise ValueError(f"state '{name}' is not normalized")
        self.states[name] = state

    def add_observable(self, name: str, obs: Observable) -> None:
        if not obs.is_valid():
            raise ValueError(f"observable '{name}' is not Hermitian")
        self.observables[name] = obs
