"""An exact amplitude-level (statevector) tier for the DQI gadgets.

Status: experimental

Why this exists
---------------
Every count in this repository is measured on ``kickmix``, which stores
computational-basis states. That is the right engine for the decoder -- its input
is a basis state -- but it cannot answer the two questions the record's licence
depends on:

1. **Is a read-then-gate gadget the same map as the gate it replaces?** On a
   basis state the answer is yes, and ``kickmix`` can show that for one input at
   a time. What it cannot show is what the gadget does to a *superposition*, and
   that is exactly the boundary of the claim: `docs/EEA_SCHEDULE_CAMPAIGN.md` §2.1
   licenses the gates on P1 ("every register read holds a computational-basis
   value"), which is a statement about the input ensemble, not about the circuit.
2. **Is ``HMR`` a ``MR``?** It is one H short of one, and whether that H exists is
   a property of the builder, not of this tier -- but the algebra of why the
   substitution would work, and why dropping the H breaks it, needs amplitudes.

This module is that tier, and nothing else. It is deliberately small: a dense
statevector of ``n`` qubits, the gates the DQI gadgets use, exact branch
enumeration over measurement outcomes, and a fidelity. It is **not** a
competition engine and it is not a path to a receipt: the arm is 1,913 qubits, so
its amplitudes would be ``2**1913`` complex numbers (`dimension_report` says how
many bytes that is). What it can do is settle the licence at the sizes where
amplitudes are representable, which is what `scripts/measure_vqc_licence_boundary.py`
does.

Gadget semantics
----------------
A *read-then-gate* replaces a controlled operation with

    cx(control, ancilla); mr(ancilla -> bit); bit ? operation : nothing

and leaves ``control`` untouched, because ``mr`` clears the ancilla and reports
its value rather than consuming the source. That is the map this tier is asked to
compare against the controlled operation itself.

Example::

    sim = StatevectorSimulator(4)
    ops = [("set_basis", 0, 1), ("gate", "h", (1,)), ("measure", 2, 0)]
    for bits, probability, state in sim.branches(ops):
        ...
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

__all__ = [
    "Branch",
    "StatevectorSimulator",
    "dimension_report",
    "fidelity",
    "global_phase_free_equality",
]

_H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)


@dataclasses.dataclass(frozen=True)
class Branch:
    """One measurement outcome of one trajectory: its bits, weight, and state."""

    bits: tuple[bool, ...]
    probability: float
    state: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "bits": [bool(bit) for bit in self.bits],
            "probability": float(self.probability),
        }


class StatevectorSimulator:
    """A dense exact simulator for the gates the DQI gadgets are built from."""

    def __init__(self, num_qubits: int) -> None:
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be positive, got {num_qubits}")
        if num_qubits > 26:
            raise ValueError(
                f"{num_qubits} qubits is {2.0 ** num_qubits / 2.0 ** 20:.0f} Mi amplitudes; "
                f"this tier is for gadgets, not for the decoder"
            )
        self.num_qubits = int(num_qubits)
        self.state: np.ndarray = np.zeros(1 << self.num_qubits, dtype=np.complex128)
        self.state[0] = 1.0
        self.bits: dict[int, bool] = {}

    # -- state preparation -------------------------------------------------
    def reset(self) -> None:
        self.state[:] = 0.0
        self.state[0] = 1.0
        self.bits = {}

    def set_basis_state(self, value: int) -> None:
        """Prepare the computational basis state with index `value` (bit i = qubit i)."""
        self.state[:] = 0.0
        self.state[int(value)] = 1.0

    def set_state(self, amplitudes: Any) -> None:
        vector = np.asarray(amplitudes, dtype=np.complex128).reshape(-1)
        if vector.size != self.state.size:
            raise ValueError(f"expected {self.state.size} amplitudes, got {vector.size}")
        norm = float(np.sqrt(np.vdot(vector, vector).real))
        if abs(norm - 1.0) > 1e-9:
            raise ValueError(f"state is not normalised (norm {norm})")
        self.state[:] = vector

    # -- one-qubit and two-qubit gates ------------------------------------
    def _axis(self, qubit: int) -> Any:
        """The state as ``(higher qubits, this qubit, lower qubits)``.

        The basis index is ``sum(bit_i << i)``, so qubit ``q`` is the middle axis of
        this C-order view: ``index = high * 2**(q+1) + b * 2**q + low``.
        """
        return self.state.reshape(self.state.size >> (qubit + 1), 2, 1 << qubit)

    def apply_matrix(self, qubit: int, matrix: Any) -> None:
        """Apply a 2x2 unitary to `qubit`."""
        if not 0 <= qubit < self.num_qubits:
            raise IndexError(f"qubit {qubit} out of range")
        block = self._axis(qubit)
        m = np.asarray(matrix, dtype=np.complex128)
        zero = block[:, 0, :].copy()
        one = block[:, 1, :].copy()
        block[:, 0, :] = m[0, 0] * zero + m[0, 1] * one
        block[:, 1, :] = m[1, 0] * zero + m[1, 1] * one

    def x(self, qubit: int) -> None:
        self.apply_matrix(qubit, np.array([[0, 1], [1, 0]], dtype=np.complex128))

    def z(self, qubit: int) -> None:
        self.apply_matrix(qubit, np.array([[1, 0], [0, -1]], dtype=np.complex128))

    def h(self, qubit: int) -> None:
        self.apply_matrix(qubit, _H)

    def phase(self, qubit: int, angle: float) -> None:
        self.apply_matrix(qubit, np.array(
            [[1.0, 0.0], [0.0, np.exp(1j * angle)]], dtype=np.complex128))

    def cx(self, control: int, target: int) -> None:
        self._xor_into(control, target, [(control,)])

    def ccx(self, control0: int, control1: int, target: int) -> None:
        self._xor_into(control0, target, [(control0, control1)])

    def _xor_into(self, control: int, target: int, controls: Any) -> None:
        if control == target:
            raise ValueError("control and target must differ")
        index = np.arange(self.state.size)
        mask: np.ndarray = np.ones(self.state.size, dtype=bool)
        for group in controls:
            for c in group:
                mask &= ((index >> c) & 1).astype(bool)
        partner = index[mask] ^ (1 << target)
        self.state[mask], self.state[partner] = (
            self.state[partner].copy(), self.state[mask].copy())

    def cz(self, control: int, target: int) -> None:
        index = np.arange(self.state.size)
        mask = (((index >> control) & 1) & ((index >> target) & 1)).astype(bool)
        self.state[mask] *= -1.0

    def swap(self, first: int, second: int) -> None:
        self._conditioned_swap(1.0, first, second)

    def cswap(self, control: int, first: int, second: int) -> None:
        """Controlled swap, which is what the decoder emits as `cswap`."""
        self._conditioned_swap(1.0, first, second, control=control)

    def _conditioned_swap(self, scale: float, first: int, second: int, *,
                          control: int | None = None) -> None:
        index = np.arange(self.state.size)
        mask = (((index >> first) & 1) != ((index >> second) & 1))
        if control is not None:
            mask &= ((index >> control) & 1).astype(bool)
        partner = index[mask] ^ (1 << first) ^ (1 << second)
        low = self.state[mask].copy()
        high = self.state[partner].copy()
        self.state[mask] = scale * high
        self.state[partner] = scale * low

    # -- measurement -------------------------------------------------------
    def probabilities(self, qubit: int) -> tuple[float, float]:
        """``(p0, p1)`` for a Z-basis measurement of `qubit`."""
        index = np.arange(self.state.size)
        one = ((index >> qubit) & 1).astype(bool)
        p1 = float(np.sum(np.abs(self.state[one]) ** 2))
        return 1.0 - p1, p1

    def measure_and_reset(self, qubit: int, outcome: bool | None = None,
                          rng: np.random.Generator | None = None) -> bool:
        """Z-basis measurement followed by a reset to ``|0>``; returns the outcome.

        This is the pinned binary's ``MR``: the collapse is real (the coherence
        between the two outcomes is gone), the reset puts the qubit back so it can
        be reused, and the classical outcome is what a ``push_condition`` reads.
        Passing `outcome` forces a branch instead of sampling it.
        """
        _p0, p1 = self.probabilities(qubit)
        if outcome is None:
            rng = rng or np.random.default_rng()
            outcome = bool(rng.random() < p1)
        index = np.arange(self.state.size)
        one = ((index >> qubit) & 1).astype(bool)
        keep = one if outcome else ~one
        self.state[~keep] = 0.0
        if outcome:
            # Move the surviving amplitude from bit=1 to bit=0: the reset.
            self.state[index[~one]] = self.state[index[one]]
            self.state[index[one]] = 0.0
        norm = float(np.sqrt(np.sum(np.abs(self.state) ** 2)))
        if norm < 1e-12:
            raise ValueError("branch has zero probability; it cannot be executed")
        self.state /= norm
        return bool(outcome)

    # -- op-list execution with exact branching ---------------------------
    def apply_op(self, op: Any) -> bool | None:
        """Apply one op; return the measurement outcome if it measured."""
        kind = op[0]
        if kind == "gate":
            name, args = op[1], op[2]
            getattr(self, name)(*args)
            return None
        if kind == "measure":
            qubit, bit_index = op[1], op[2]
            outcome = self.measure_and_reset(qubit)
            self.bits[bit_index] = outcome
            return outcome
        if kind == "bit_store":
            bit_index, value = op[1], op[2]
            control = op[3] if len(op) > 3 else None
            if control is None:
                self.bits[bit_index] = bool(value)
            elif self.bits.get(control, False):
                self.bits[bit_index] = bool(value)
            return None
        if kind == "conditioned":
            bit_index, ops = op[1], op[2]
            if self.bits.get(bit_index, False):
                for inner in ops:
                    self.apply_op(inner)
            return None
        raise ValueError(f"unknown op {kind!r}")

    def branches(self, ops: Any, *, max_branches: int = 64) -> list[Branch]:
        """Every measurement outcome, with its weight and its post-state."""
        pending: list[tuple[tuple[bool, ...], float, np.ndarray, dict[int, bool], list[Any]]] = [
            ((), 1.0, self.state.copy(), dict(self.bits), list(ops))
        ]
        finished: list[Branch] = []
        while pending:
            bits, probability, state, classical, remaining = pending.pop()
            sim = StatevectorSimulator(self.num_qubits)
            sim.state = state.copy()
            sim.bits = dict(classical)
            measured = False
            for position, op in enumerate(remaining):
                if op[0] == "measure":
                    qubit, bit_index = op[1], op[2]
                    p0, p1 = sim.probabilities(qubit)
                    for outcome, weight in ((False, p0), (True, p1)):
                        if weight < 1e-12:
                            continue
                        child = StatevectorSimulator(self.num_qubits)
                        child.state = sim.state.copy()
                        child.bits = dict(sim.bits)
                        child.measure_and_reset(qubit, outcome=outcome)
                        child.bits[bit_index] = outcome
                        rest = remaining[position + 1:]
                        pending.append(((*bits, outcome), probability * weight,
                                        child.state.copy(), dict(child.bits),
                                        list(rest)))
                        if len(pending) + len(finished) > max_branches:
                            raise RuntimeError(
                                f"more than {max_branches} branches; this tier enumerates "
                                f"outcomes exactly and is not for long trajectories"
                            )
                    measured = True
                    break
                sim.apply_op(op)
            if not measured:
                finished.append(Branch(bits=bits, probability=probability,
                                       state=sim.state.copy()))
        finished.sort(key=lambda branch: (branch.probability, branch.bits), reverse=True)
        total = sum(branch.probability for branch in finished)
        if abs(total - 1.0) > 1e-9:
            raise RuntimeError(f"branch weights sum to {total}, not 1")
        return finished


def global_phase_free_equality(left: Any, right: Any, *, tolerance: float = 1e-9) -> bool:
    """Whether two statevectors are equal up to one global phase."""
    a = np.asarray(left, dtype=np.complex128).reshape(-1)
    b = np.asarray(right, dtype=np.complex128).reshape(-1)
    if a.size != b.size:
        return False
    overlap = np.vdot(a, b)
    magnitude = abs(overlap)
    if magnitude < 1e-12:
        return False
    return bool(np.allclose(b, overlap / magnitude * a, atol=tolerance, rtol=0.0))


def fidelity(reference: Any, branches: Any) -> float:
    """``sum_i p_i * |<reference|psi_i>|^2``: how much of `reference` the mixture keeps.

    The gated circuit is a channel, not a unitary: a measurement makes it a
    mixture of trajectories. This is the probability that an ideal state
    `reference` would still be found if the mixture were measured against it, so
    it is 1.0 exactly when every branch is the reference up to global phase.
    """
    reference = np.asarray(reference, dtype=np.complex128).reshape(-1)
    total = 0.0
    for branch in branches:
        overlap = abs(np.vdot(reference, branch.state)) ** 2
        total += branch.probability * float(overlap)
    return float(total)


def dimension_report(num_qubits: int, *, bytes_per_amplitude: int = 16) -> dict[str, Any]:
    """What an amplitude-level treatment of `num_qubits` would cost, in arithmetic.

    Reported on a log10 scale as well as a float, because the decoder's 1,913 qubits
    do not fit in a double: the point of the number is that it is 576 decimal digits
    of amplitudes, not that it has a value.
    """
    log10_amplitudes = num_qubits * math.log10(2.0)
    report: dict[str, Any] = {
        "num_qubits": int(num_qubits),
        "log10_amplitudes": log10_amplitudes,
        "log10_bytes_at_16_per_amplitude": log10_amplitudes + math.log10(bytes_per_amplitude),
        "digits_of_amplitudes": int(log10_amplitudes) + 1,
    }
    if log10_amplitudes < 308.0:  # a double can hold the count; do not overpromise
        amplitudes = 2.0 ** num_qubits
        report["amplitudes"] = amplitudes
        report["bytes_at_16_per_amplitude"] = amplitudes * bytes_per_amplitude
    else:
        report["amplitudes"] = "too large for a float; see log10_amplitudes"
        report["bytes_at_16_per_amplitude"] = (
            "too large for a float; see log10_bytes_at_16_per_amplitude")
    return report
