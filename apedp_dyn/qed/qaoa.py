"""QAOA over the selection bits, on an explicitly approximate surrogate.

Status: experimental

The exact objective ``J(z)`` is *not* quadratic in ``z``: the coefficients are
re-fitted for every selection, so ``J`` is a minimum of quadratics. A quantum
phase separator needs a diagonal cost Hamiltonian, so this module builds the
quadratic pseudo-Boolean surrogate that the standard derivation uses (Champion
et al., 2019, take the same step to put SINDy on an annealer):

    Q(z) = J(0) + sum_i q_i z_i + sum_{i<j} Q_ij z_i z_j
    q_i  = J({i}) - J(0)
    Q_ij = J({i,j}) - J({i}) - J({j}) + J(0)

built from ``1 + m + m(m-1)/2`` exact fits of the real objective. The surrogate
cost is *paid out of the same budget* every other proposer pays, and the arm's
answer is always scored by the real objective. What the search actually
optimises is therefore an approximation, and the module never hides that: the
record reports the surrogate's own minimiser, whether it agrees with the best
selection the arm scored, and how much probability the circuit puts on it.

The circuit simulated here is the standard QAOA for a QUBO: uniform
superposition, then per layer a phase separator ``exp(-i gamma H_C)`` and a
mixer ``exp(-i beta sum_q X_q)``. ``H_C`` is diagonal in the computational basis,
which is exactly right for a pseudo-Boolean cost, but on hardware each quadratic
term is one ``RZZ``, so the rotation count is reported as the honest circuit
cost. Only the listed rotation types are counted; no encoding, routing or error
model is included in that number.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from apedp_dyn.qed.objective import Fit

# A dense diagonal cost over 2**n outcomes is what a statevector simulation
# needs; beyond this the object is a specification, not something to hold.
MAX_QUBITS = 20


class QuboRefusal(ValueError):
    """Raised when a surrogate cannot be built or simulated as declared."""


@dataclass(frozen=True)
class Qubo:
    """A quadratic pseudo-Boolean cost, with ``z_j = 1`` meaning "term j is in"."""

    offset: float
    linear: tuple[float, ...]
    quadratic: tuple[tuple[int, int, float], ...]

    @property
    def n(self) -> int:
        return len(self.linear)

    def energy(self, bits: Sequence[int]) -> float:
        if len(bits) != self.n:
            raise QuboRefusal(f"expected {self.n} bits, got {len(bits)}")
        total = self.offset
        for index, bit in enumerate(bits):
            if bit:
                total += self.linear[index]
        for i, j, value in self.quadratic:
            if bits[i] and bits[j]:
                total += value
        return float(total)

    def energies(self) -> np.ndarray:
        """Every ``2**n`` cost in one vector, indexed by the mask with bit ``i`` = ``z_i``."""
        if self.n > MAX_QUBITS:
            raise QuboRefusal(f"2**{self.n} outcomes is a specification, not a table")
        masks = np.arange(2 ** self.n, dtype=np.int64)
        bits = ((masks[:, None] >> np.arange(self.n)) & 1).astype(float)
        values = np.full(2 ** self.n, self.offset, dtype=float)
        values += bits @ np.asarray(self.linear, dtype=float)
        for i, j, value in self.quadratic:
            values += value * bits[:, i] * bits[:, j]
        _out: np.ndarray = values
        return _out

    def argmin(self) -> tuple[int, float]:
        values = self.energies()
        index = int(np.argmin(values))
        return index, float(values[index])

    def rotations(self, depth: int) -> dict[str, Any]:
        """The rotations one layer of the phase separator plus mixer costs."""
        quadratic = len(self.quadratic)
        return {
            "qubits": self.n,
            "layers": int(depth),
            "rz_per_layer": self.n,
            "rzz_per_layer": quadratic,
            "rx_per_layer": self.n,
            "rotations_per_layer": 2 * self.n + quadratic,
            "rotations_total": int(depth) * (2 * self.n + quadratic),
        }


def build_quadratic_surrogate(
    evaluate: Callable[[Sequence[int]], Fit], n: int
) -> tuple[Qubo, int]:
    """Fit the surrogate from exact singleton and pair fits of the real objective.

    Returns the surrogate and how many real evaluations it cost. Raises
    :class:`BudgetExhausted` from ``evaluate`` if the budget cannot cover the
    ``n(n+1)/2 + 1`` fits the expansion needs, so a proposer can abstain instead
    of building a surrogate out of a partial expansion.
    """
    if n < 1:
        raise QuboRefusal("a surrogate needs at least one selection bit")
    empty = evaluate(()).objective
    singles = [evaluate((index,)) for index in range(n)]
    linear = tuple(single.objective - empty for single in singles)
    quadratic = []
    for i in range(n):
        for j in range(i + 1, n):
            pair = evaluate((i, j))
            value = pair.objective - singles[i].objective - singles[j].objective + empty
            quadratic.append((i, j, float(value)))
    return Qubo(offset=float(empty), linear=linear, quadratic=tuple(quadratic)), n * (n + 1) // 2 + 1


def normalised(values: np.ndarray) -> np.ndarray:
    """Rescale a cost to ``[0, 1]`` for the phase separator.

    An affine rescaling of a QUBO does not move its minimiser, so the surrogate
    the circuit optimises is the surrogate that was fitted; only the angle scale
    changes.
    """
    values = np.asarray(values, dtype=float)
    spread = float(values.max() - values.min())
    if spread <= 0:
        _out: np.ndarray = np.zeros_like(values)
        return _out
    _out = (values - values.min()) / spread
    return _out


def _apply_mixer(state: np.ndarray, beta: float, n: int) -> np.ndarray:
    """``exp(-i beta sum_q X_q)``: the same single-qubit rotation on every axis."""
    unitary = np.array(
        [
            [np.cos(beta), -1j * np.sin(beta)],
            [-1j * np.sin(beta), np.cos(beta)],
        ],
        dtype=complex,
    )
    tensor = state.reshape((2,) * n)
    for axis in range(n):
        tensor = np.moveaxis(np.tensordot(unitary, tensor, axes=([1], [axis])), 0, axis)
    return tensor.reshape(-1)


def qaoa_state(costs: np.ndarray, depth: int, angles: Sequence[float]) -> np.ndarray:
    """The QAOA state for a diagonal cost, with ``2 * depth`` angles."""
    costs = np.asarray(costs, dtype=float)
    n = int(np.round(np.log2(costs.size)))
    if 2 ** n != costs.size or n < 1:
        raise QuboRefusal(f"a diagonal cost needs a power-of-two length, got {costs.size}")
    if len(angles) != 2 * depth:
        raise QuboRefusal(f"depth {depth} needs {2 * depth} angles, got {len(angles)}")
    state: np.ndarray = np.full(costs.size, 1.0 / np.sqrt(costs.size), dtype=complex)
    for layer in range(depth):
        gamma = float(angles[2 * layer])
        beta = float(angles[2 * layer + 1])
        state = state * np.exp(-1j * gamma * costs)
        state = _apply_mixer(state, beta, n)
    return state


def _expectation(costs: np.ndarray, depth: int, angles: Sequence[float]) -> float:
    state = qaoa_state(costs, depth, angles)
    return float(np.sum(np.abs(state) ** 2 * costs))


def optimise_angles(
    costs: np.ndarray,
    depth: int,
    *,
    rng: np.random.Generator,
    restarts: int = 3,
    rounds: int = 14,
) -> tuple[tuple[float, ...], float]:
    """Deterministic pattern search over the angles, best of several starts.

    No gradient, no library solver: a coordinate-wise step search with halving,
    so the same seed gives the same angles on every host.
    """
    best_angles: tuple[float, ...] = ()
    best_value = float("inf")
    for restart in range(restarts):
        base = [0.4 + 0.2 * rng.random(), 0.3 + 0.2 * rng.random()]
        current = np.array([base[0], base[1]] * max(depth, 1), dtype=float)
        if restart:
            current = current + 0.05 * rng.standard_normal(current.size)
        value = _expectation(costs, depth, current)
        step = 0.5
        for _ in range(rounds):
            improved = False
            for index in range(current.size):
                for delta in (step, -step):
                    trial = current.copy()
                    trial[index] += delta
                    candidate = _expectation(costs, depth, trial)
                    if candidate < value - 1e-12:
                        current, value, improved = trial, candidate, True
            if not improved:
                step *= 0.5
        if value < best_value:
            best_value = value
            best_angles = tuple(float(angle) for angle in current)
    return best_angles, best_value


@dataclass(frozen=True)
class QaoaRun:
    """One measured QAOA run over a surrogate."""

    depth: int
    shots: int
    angles: tuple[float, ...]
    expected_cost: float
    optimal_cost: float
    probabilities: tuple[float, ...]
    samples: tuple[int, ...]
    candidates: tuple[int, ...]
    rotations: dict[str, Any]

    def probability_of(self, mask: int) -> float:
        return float(self.probabilities[mask])


def run_qaoa(
    qubo: Qubo,
    *,
    depth: int,
    shots: int,
    rng: np.random.Generator,
    restarts: int = 3,
) -> QaoaRun:
    """Optimise the angles, measure the circuit, and order the candidates.

    The surrogate's own minimiser is put first, then the measured outcomes in
    descending sampled frequency, so a proposer that spends its budget scoring
    candidates spends it on the most probable ones.
    """
    if depth < 1:
        raise QuboRefusal(f"depth must be at least one layer, got {depth}")
    raw = qubo.energies()
    costs = normalised(raw)
    angles, _expected = optimise_angles(costs, depth, rng=rng, restarts=restarts)
    state = qaoa_state(costs, depth, angles)
    probabilities = np.abs(state) ** 2
    total = float(probabilities.sum())
    if total <= 0:
        raise QuboRefusal("the QAOA state has no probability mass")
    probabilities = probabilities / total

    draws = rng.choice(raw.size, size=shots, p=probabilities)
    counts: dict[int, int] = {}
    for draw in draws:
        counts[int(draw)] = counts.get(int(draw), 0) + 1
    ordered = sorted(counts, key=lambda mask: (-counts[mask], -probabilities[mask], mask))
    surrogate_mask, _ = qubo.argmin()
    candidates = [surrogate_mask] + [mask for mask in ordered if mask != surrogate_mask]
    return QaoaRun(
        depth=depth,
        shots=shots,
        angles=angles,
        expected_cost=float(np.sum(probabilities * raw)),
        optimal_cost=float(raw.min()),
        probabilities=tuple(float(value) for value in probabilities),
        samples=tuple(ordered),
        candidates=tuple(candidates),
        rotations=qubo.rotations(depth),
    )


def mask_to_support(mask: int, n: int) -> tuple[int, ...]:
    return tuple(index for index in range(n) if (mask >> index) & 1)


def support_to_mask(support: Sequence[int], n: int) -> int:
    mask = 0
    for index in support:
        if index < 0 or index >= n:
            raise QuboRefusal(f"index {index} is out of range for {n} bits")
        mask |= 1 << index
    return mask


__all__ = [
    "MAX_QUBITS",
    "QaoaRun",
    "Qubo",
    "QuboRefusal",
    "build_quadratic_surrogate",
    "mask_to_support",
    "normalised",
    "optimise_angles",
    "qaoa_state",
    "run_qaoa",
    "support_to_mask",
]
