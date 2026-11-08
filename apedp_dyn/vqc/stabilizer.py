"""A stabilizer backend: Clifford circuits at chain lengths a dense state cannot reach.

Status: experimental

The dense amplitude tier in :mod:`apedp_dyn.vqc.backends` is exact and general, and
it stops at roughly 25 qubits because it stores ``2**n`` complex numbers. A
Clifford circuit never needs them: its state is always the simultaneous ``+1``
eigenspace of ``n`` commuting Pauli operators, so ``100`` qubits cost ``100``
Pauli strings of ``100`` letters. That is the tier this module adds, and it is
useful for the obvious reason -- a GHZ chain of 100 qubits is a state the
statevector tier cannot write down, and a state this module handles in
milliseconds.

What is stored
--------------
``n`` independent commuting Pauli strings, each with a phase ``i**p``. A Pauli
string is a tuple of ``I``/``X``/``Y``/``Z`` labels per qubit, and the phase is
tracked as an integer modulo four, so nothing is ever rounded: every product,
conjugation and sign is computed from exact tables built once at import time by
multiplying 2x2 and 4x4 matrices and matching the result against
``i**p * P``. If a gate is not Clifford its conjugated Pauli is not a Pauli, the
match fails, and the gate is refused rather than approximated.

Measurement
-----------
Measuring ``Z_q`` either anticommutes with one of the generators -- the outcome is
uniformly random, and the generators are rewritten so that ``+/-Z_q`` replaces the
one that anticommuted -- or it commutes with all of them, in which case the
outcome is already decided and is read off by expressing ``Z_q`` as a product of
generators (Gaussian elimination over GF(2) on the ``2n``-bit characteristic
vectors). Both cases are exact, including the sign, which is what makes the
100-qubit numbers below numbers and not sketches.

What this tier refuses
----------------------
Non-Clifford gates (``t``, ``ccx``, ``cswap``, the parameterised rotations, a
controlled phase), and every :class:`~apedp_dyn.vqc.circuit.Noise` operation. A
stochastic channel turns a stabilizer state into a mixture, and this module
represents pure states only. The dense tier remains the referee for those; this
one is a fast path with a narrow, stated domain, and it raises
:class:`StabilizerUnsupportedError` rather than guessing.

Verification
------------
The cross-check in ``tests/test_vqc_stabilizer.py`` runs the *same* circuit builder
on both tiers for chain lengths the dense tier can hold and compares every one of
the ``2**n`` outcome probabilities, a set of Pauli expectations, and sampled
counts. The 100-qubit run is then checked against the closed-form GHZ
predictions, which the dense tier has confirmed at small ``n``.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from apedp_dyn.vqc import native
from apedp_dyn.vqc.backends import RunResult, bitstring
from apedp_dyn.vqc.circuit import Circuit, CircuitValidationError, Gate, Measure, Noise, Reset
from apedp_dyn.vqc.gates import GATE_SPECS, GateSpec

__all__ = [
    "CLIFFORD_GATES",
    "StabilizerBackend",
    "StabilizerState",
    "StabilizerUnsupportedError",
    "clifford_pauli_images",
    "conjugate_pauli",
    "ghz_circuit",
    "ghz_even_parity_probability",
    "ghz_generators",
    "is_clifford_circuit",
    "pauli_string",
]

#: `I`, `X`, `Y`, `Z` as small integers; the row/column order of every table below.
#: The names are the physics, not one-letter throwaways, hence the lint exemption.
I, X, Y, Z = 0, 1, 2, 3  # noqa: E741
_LABELS = "IXYZ"

#: The gate names this tier executes. Anything else is refused by name.
CLIFFORD_GATES = frozenset({"h", "x", "y", "z", "s", "sdg", "cx", "cz", "swap"})

_PAULI_MATRICES: tuple[np.ndarray, ...] = (
    np.eye(2, dtype=np.complex128),
    np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
)


class StabilizerUnsupportedError(ValueError):
    """A circuit this tier cannot represent, named rather than approximated."""


def _snap(matrix: np.ndarray) -> tuple[int, int]:
    """``(phase, label)`` with ``matrix == i**phase * P``; refuses anything else."""
    for phase in range(4):
        for label in range(4):
            if np.allclose(matrix, (1j ** phase) * _PAULI_MATRICES[label], atol=1e-12):
                return phase, label
    raise StabilizerUnsupportedError(
        "the conjugated operator is not a Pauli matrix, so this operation is not "
        "Clifford")


def _single_qubit_rules() -> dict[str, tuple[tuple[int, int], ...]]:
    """For each one-qubit gate: how it maps each Pauli, with the exact phase."""
    from apedp_dyn.vqc.gates import matrix_for

    rules: dict[str, tuple[tuple[int, int], ...]] = {}
    for name in ("h", "x", "y", "z", "s", "sdg"):
        unitary = matrix_for(name)
        rule = [(_snap(unitary @ _PAULI_MATRICES[label] @ unitary.conj().T))
                for label in range(4)]
        if rule[I] != (0, I):
            raise StabilizerUnsupportedError(f"{name!r} does not fix the identity")
        rules[name] = tuple(rule)
    return rules


def _two_qubit_matrix(name: str) -> np.ndarray:
    """The 4x4 unitary of a two-qubit gate, first target as the first tensor factor."""
    matrix: np.ndarray = np.zeros((4, 4), dtype=np.complex128)
    for first, second in itertools.product((0, 1), repeat=2):
        column = 2 * first + second
        if name == "cx":
            row = 2 * first + (first ^ second)
        elif name == "swap":
            row = 2 * second + first
        elif name == "cz":
            row = column
        else:  # pragma: no cover - the caller only builds known names
            raise StabilizerUnsupportedError(name)
        matrix[row, column] = 1.0 if name != "cz" else (-1.0 if first and second else 1.0)
    return matrix


def _two_qubit_rules() -> dict[str, dict[tuple[int, int], tuple[int, int, int]]]:
    rules: dict[str, dict[tuple[int, int], tuple[int, int, int]]] = {}
    for name in ("cx", "cz", "swap"):
        unitary = _two_qubit_matrix(name)
        table = {}
        for left, right in itertools.product(range(4), repeat=2):
            operator = np.kron(_PAULI_MATRICES[left], _PAULI_MATRICES[right])
            conjugated = unitary @ operator @ unitary.conj().T
            # The two labels are not independent after conjugation, so match the
            # 4x4 product against all sixteen tensor products.
            for new_left, new_right in itertools.product(range(4), repeat=2):
                for phase in range(4):
                    candidate = (1j ** phase) * np.kron(_PAULI_MATRICES[new_left],
                                                        _PAULI_MATRICES[new_right])
                    if np.allclose(conjugated, candidate, atol=1e-12):
                        table[(left, right)] = (phase, new_left, new_right)
                        break
                else:
                    continue
                break
            else:  # pragma: no cover - all four two-qubit Cliffords are Clifford
                raise StabilizerUnsupportedError(f"{name!r} is not Clifford")
        rules[name] = table
    return rules


_SINGLE = _single_qubit_rules()
_PAIR = _two_qubit_rules()

#: ``(_phase, _label)`` with ``P_a P_b == i**_phase * P_label``, built once from matrices.
_PRODUCT: dict[tuple[int, int], tuple[int, int]] = {
    (a, b): _snap(_PAULI_MATRICES[a] @ _PAULI_MATRICES[b])
    for a in range(4) for b in range(4)
}


def _multiply(left: Sequence[int], left_phase: int,
              right: Sequence[int], right_phase: int) -> tuple[tuple[int, ...], int]:
    """The exact product of two Pauli strings, as ``(labels, phase)``."""
    labels = []
    phase = (left_phase + right_phase) % 4
    for a, b in zip(left, right, strict=True):
        # `X Y = iZ`, `Y X = -iZ`, `Z X = iY`, `X Z = -iY`, `Y Z = iX`, `Z Y = -iX`;
        # every product of two equal labels is the identity with phase 0.
        if a == b:
            labels.append(I)
        elif a == I:
            labels.append(b)
        elif b == I:
            labels.append(a)
        else:
            product = _PRODUCT[(a, b)]
            phase = (phase + product[0]) % 4
            labels.append(product[1])
    return tuple(labels), phase


def _labels_to_mask(labels: Sequence[int]) -> int:
    """The characteristic vector as one integer: ``x`` bits low, ``z`` bits high."""
    mask = 0
    for position, label in enumerate(labels):
        if label in (X, Y):
            mask |= 1 << position
        if label in (Z, Y):
            mask |= 1 << (position + len(labels))
    return mask


def _format_phase(phase: int) -> str:
    return {0: "+", 1: "+i", 2: "-", 3: "-i"}[phase % 4]


def pauli_string(labels: Sequence[int], phase: int = 0) -> str:
    """``'+XXI'``-style text for one signed Pauli string, for reports and tests."""
    return _format_phase(phase) + "".join(_LABELS[label] for label in labels)


def _product_phase(left_mask: int, right_mask: int, left_phase: int,
                   right_phase: int, num_qubits: int) -> int:
    """The phase of the product of two Pauli strings, from their masks and phases.

    Pauli multiplication is not vector addition. With ``X^a Z^b * X^c Z^d =
    (-1)^(b.c) X^(a+c) Z^(b+d)``, every reduction step carries a sign that depends on
    the Z bits of the accumulator and the X bits of the row it absorbs, so adding the
    two phases is only correct when one of those two sets is empty -- and which rows
    the elimination picks is an implementation detail. Without the correction, a
    sign depended on that choice: ``+YYX`` was read as a stabilizer of a GHZ state it
    has zero overlap with, and ``Y_0 Y_1`` was not found in the group of a Bell state.
    """
    x_bits = (1 << num_qubits) - 1
    crossing = bin((left_mask >> num_qubits) & right_mask & x_bits).count("1")
    return (left_phase + right_phase + 2 * crossing) % 4


class StabilizerState:
    """``n`` commuting signed Pauli generators: a pure stabilizer state.

    Mutable, because a circuit is a sequence of updates and copying an ``n``-row
    tableau per gate is wasted work. :meth:`clone` exists for the callers that do
    need a snapshot (measurement probabilities, sampling).
    """

    def __init__(self, num_qubits: int) -> None:
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be positive, got {num_qubits}")
        self.num_qubits = int(num_qubits)
        # `|0...0>` is the +1 eigenspace of every `Z_q`.
        self.generators: list[tuple[int, ...]] = []
        self.phases: list[int] = []
        for qubit in range(self.num_qubits):
            labels = [I] * self.num_qubits
            labels[qubit] = Z
            self.generators.append(tuple(labels))
            self.phases.append(0)
        self._basis: dict[int, tuple[int, int]] | None = None

    # ------------------------------------------------------------------ basics
    def clone(self) -> StabilizerState:
        copy = StabilizerState.__new__(StabilizerState)
        copy.num_qubits = self.num_qubits
        copy.generators = list(self.generators)
        copy.phases = list(self.phases)
        copy._basis = None
        return copy

    def generator_strings(self) -> tuple[str, ...]:
        """The generators as ``'+ZZI'``-style strings, for inspection and tests."""
        return tuple(pauli_string(labels, phase)
                     for labels, phase in zip(self.generators, self.phases, strict=True))

    def _invalidate(self) -> None:
        self._basis = None

    # ------------------------------------------------------------ group algebra
    def _echelon(self) -> dict[int, tuple[int, int]]:
        """A pivot-indexed GF(2) basis of the generator group, with its phases."""
        if self._basis is not None:
            return self._basis
        basis: dict[int, tuple[int, int]] = {}
        for labels, phase in zip(self.generators, self.phases, strict=True):
            mask = _labels_to_mask(labels)
            # The mask is ``X^a Z^b`` and the generator is a string of Pauli
            # matrices, so a row with ``k`` Y's is ``i**k`` times its mask: the
            # conversion has to happen here, once per row, or every later reduction
            # combines phases from two different conventions.
            current = (phase + sum(1 for label in labels if label == Y)) % 4
            while mask:
                pivot = (mask & -mask).bit_length() - 1
                existing = basis.get(pivot)
                if existing is None:
                    basis[pivot] = (mask, current)
                    break
                current = _product_phase(mask, existing[0], current, existing[1],
                                         self.num_qubits)
                mask ^= existing[0]
            else:
                if current % 2:
                    # A dependency whose phases do not cancel means these rows are
                    # not a valid stabilizer: the state would be its own negative.
                    raise StabilizerUnsupportedError(
                        "the generators are not a consistent stabilizer state")
        self._basis = basis
        return basis

    def phase_of(self, labels: Sequence[int]) -> int | None:
        """The phase ``i**p`` with which `labels` appears in the group, or ``None``.

        This is the group-membership test the state-level referee is built on:
        two pure stabilizer states are the same state exactly when every signed
        generator of one is in the other's group, and that is a phase comparison in
        this representation rather than a distance.

        Two conventions meet here and both have to be accounted for, because the
        mask represents ``X^a Z^b`` while the argument is a *string of Pauli
        matrices*. The letter ``Y`` is not the vector sum ``X + Z`` but ``i X Z``, so
        a string with ``k`` Y's is ``i**k X^a Z^b`` and the phase of the matrix
        product is ``p - k``. Dropping that term read ``+YYX`` as a stabilizer of a
        GHZ state it is orthogonal to, and dropping the multiplication phase in the
        reduction made the answer depend on which rows the elimination picked.
        """
        basis = self._echelon()
        mask = _labels_to_mask(labels)
        # The reduction picks the rows whose masks XOR to `mask`, and the product of
        # exactly those rows is *the* group element with this mask: a stabilizer group
        # holds one element per mask, so the accumulation is over the rows and not
        # over the target as if it were one of them.
        element_mask = 0
        element_phase = 0
        while mask:
            pivot = (mask & -mask).bit_length() - 1
            entry = basis.get(pivot)
            if entry is None:
                return None
            element_phase = _product_phase(element_mask, entry[0], element_phase,
                                           entry[1], self.num_qubits)
            element_mask ^= entry[0]
            mask ^= entry[0]
        twists = sum(1 for label in labels if label == Y)
        return (element_phase - twists) % 4

    def stabilizes(self, labels: Sequence[int], *, sign: int = 0) -> bool:
        """Whether the state is the ``(-1)**sign`` eigenstate of this Pauli string."""
        return self.phase_of(labels) == (2 * sign) % 4

    def validate(self) -> None:
        """Refuse a tableau that is not `n` independent, commuting generators.

        Two failure modes are silent in the arithmetic and loud here: generators that
        anticommute (there is no simultaneous eigenspace, so every measurement and
        amplitude computed from them is wrong), and generators that are linearly
        dependent (the group has fewer than ``2**n`` elements, so this is not a pure
        state). The phase consistency of a dependency is already refused by
        :meth:`_echelon`; this is the structural check on top of it.

        ``O(n**2)`` symplectic products, computed on the packed bit masks so the
        cost is ``n**2 / 64`` machine words per check: the gate path maintains the
        invariant exactly and does not pay for this per gate, so it is a check for
        hand-built tableaus and for tests, not a runtime guard.
        """
        n = self.num_qubits
        if len(self.generators) != n or len(self.phases) != n:
            raise StabilizerUnsupportedError(
                f"{n} qubits need {n} generators, got "
                f"{len(self.generators)} generators and {len(self.phases)} phases")
        low = (1 << n) - 1
        masks = [_labels_to_mask(labels) for labels in self.generators]
        xs = [mask & low for mask in masks]
        zs = [mask >> n for mask in masks]
        for first in range(n):
            x_first, z_first = xs[first], zs[first]
            for second in range(first + 1, n):
                # Symplectic product: one iff the two strings anticommute.
                if ((x_first & zs[second]) ^ (z_first & xs[second])).bit_count() & 1:
                    raise StabilizerUnsupportedError(
                        f"generators {first} and {second} anticommute "
                        f"({self.generator_strings()[first]}, "
                        f"{self.generator_strings()[second]}), so there is no "
                        f"simultaneous eigenspace and no state")
        basis = self._echelon()
        if len(basis) != n:
            raise StabilizerUnsupportedError(
                f"the {n} generators span only rank {len(basis)}, so the group has "
                f"fewer than 2**{n} elements and the state is not pure: "
                f"{self.generator_strings()}")

    def expectation(self, labels: Sequence[int]) -> float:
        """``<P>``: ``+1``/``-1`` if the state is an eigenstate of `P`, else ``0``."""
        if len(labels) != self.num_qubits:
            raise ValueError(f"expected {self.num_qubits} labels, got {len(labels)}")
        phase = self.phase_of(labels)
        if phase is None:
            return 0.0
        if phase % 2:
            raise StabilizerUnsupportedError(
                "a Hermitian operator cannot have an imaginary eigenvalue here")
        return 1.0 if phase == 0 else -1.0

    # ------------------------------------------------------------------- gates
    def apply_gate(self, name: str, targets: Sequence[int],
                   params: Sequence[float] = ()) -> None:
        """Conjugate every generator by one Clifford gate."""
        spec = _clifford_spec(name, targets, params, self.num_qubits)
        for index, labels in enumerate(self.generators):
            self.generators[index], self.phases[index] = _conjugated(
                labels, self.phases[index], name, targets, spec.arity)
        self._invalidate()

    # ------------------------------------------------------------- measurement
    def _random_qubit(self, qubit: int) -> int | None:
        """Index of a generator that anticommutes with ``Z_q``, if any."""
        for index, labels in enumerate(self.generators):
            if labels[qubit] in (X, Y):
                return index
        return None

    def outcome_of(self, qubit: int) -> float:
        """``P(measurement of ``Z_q`` returns 1)``: ``0.5`` or exactly ``0``/``1``."""
        if self._random_qubit(qubit) is not None:
            return 0.5
        phase = self.phase_of(_z_at(qubit, self.num_qubits))
        if phase is None:  # pragma: no cover - a stabilizer state always decides Z_q
            raise StabilizerUnsupportedError(
                f"Z_{qubit} commutes with every generator but is not in the group")
        return 1.0 if phase == 2 else 0.0

    def _collapse(self, qubit: int, outcome: bool) -> None:
        """Project onto ``Z_q = (-1)**outcome`` and rewrite the generators."""
        index = self._random_qubit(qubit)
        if index is None:
            return  # the outcome was already decided; the group is unchanged
        anticommuting = self.generators[index]
        for other in range(len(self.generators)):
            if other == index:
                continue
            if self.generators[other][qubit] in (X, Y):
                labels, phase = _multiply(self.generators[other], self.phases[other],
                                          anticommuting, self.phases[index])
                self.generators[other] = labels
                self.phases[other] = phase
        target = list(_z_at(qubit, self.num_qubits))
        self.generators[index] = tuple(target)
        self.phases[index] = 2 if outcome else 0
        self._invalidate()

    def measure(self, qubit: int, *, outcome: bool | None = None,
                rng: np.random.Generator | None = None) -> bool:
        """Measure ``Z_q`` without resetting, collapsing the state onto the branch.

        Passing `outcome` forces the branch; it raises :class:`ValueError` if that
        branch has probability zero, which is the same refusal the dense tier makes.
        """
        if not 0 <= qubit < self.num_qubits:
            raise ValueError(f"qubit {qubit} out of range")
        probability_one = self.outcome_of(qubit)
        if outcome is None:
            rng = rng or np.random.default_rng()
            outcome = bool(rng.random() < probability_one)
        elif probability_one == 0.0 and outcome:
            raise ValueError("branch has zero probability; it cannot be executed")
        elif probability_one == 1.0 and not outcome:
            raise ValueError("branch has zero probability; it cannot be executed")
        self._collapse(qubit, bool(outcome))
        return bool(outcome)

    def reset(self, qubit: int, *, rng: np.random.Generator | None = None) -> None:
        """`Measure` followed by a conditional `X`: the qubit ends in ``|0>``."""
        outcome = self.measure(qubit, rng=rng)
        if outcome:
            self.apply_gate("x", (qubit,))

    def sample(self, *, rng: np.random.Generator | None = None) -> tuple[bool, ...]:
        """One Z-basis outcome for every qubit, in order; destructively measures."""
        rng = rng or np.random.default_rng()
        return tuple(self.measure(qubit, rng=rng) for qubit in range(self.num_qubits))

    def probability_of_bitstring(self, bits: Sequence[bool]) -> float:
        """The exact probability of one Z-basis outcome, by chained measurement."""
        if len(bits) != self.num_qubits:
            raise ValueError(f"expected {self.num_qubits} bits, got {len(bits)}")
        state = self.clone()
        probability = 1.0
        for qubit, bit in enumerate(bits):
            one = state.outcome_of(qubit)
            if one == 0.5:
                probability *= 0.5
            elif bool(one == 1.0) != bool(bit):
                return 0.0
            state._collapse(qubit, bool(bit))
        return probability

    def distribution(self) -> dict[tuple[bool, ...], float]:
        """Every outcome with nonzero probability -- only for small ``n``."""
        if self.num_qubits > 12:
            raise StabilizerUnsupportedError(
                f"enumerating 2**{self.num_qubits} outcomes is not a report anyone reads; "
                f"use probability_of_bitstring or sample")
        outcomes: dict[tuple[bool, ...], float] = {}
        for bits in itertools.product((False, True), repeat=self.num_qubits):
            probability = self.probability_of_bitstring(bits)
            if probability:
                outcomes[bits] = probability
        return outcomes


def _clifford_spec(name: str, targets: Sequence[int], params: Sequence[float],
                   num_qubits: int) -> GateSpec:
    """The gate spec, after refusing anything this tier cannot conjugate exactly."""
    spec = GATE_SPECS.get(name)
    if spec is None:
        raise StabilizerUnsupportedError(f"unknown gate {name!r}")
    if name not in CLIFFORD_GATES:
        raise StabilizerUnsupportedError(
            f"{name!r} is not in this tier's Clifford set {sorted(CLIFFORD_GATES)}")
    if params:
        raise StabilizerUnsupportedError(
            f"{name!r} takes parameters; a parameterised rotation is not Clifford")
    if len(targets) != spec.arity:
        raise ValueError(f"{name!r} takes {spec.arity} qubits, got {len(targets)}")
    for target in targets:
        if not 0 <= target < num_qubits:
            raise ValueError(f"qubit {target} out of range")
    return spec


def _conjugated(labels: Sequence[int], phase: int, name: str, targets: Sequence[int],
                arity: int) -> tuple[tuple[int, ...], int]:
    """Conjugate one signed Pauli string by one Clifford gate, exactly."""
    if arity == 1:
        qubit = targets[0]
        delta, new_label = _SINGLE[name][labels[qubit]]
        if new_label == labels[qubit]:
            return tuple(labels), (phase + delta) % 4
        updated = list(labels)
        updated[qubit] = new_label
        return tuple(updated), (phase + delta) % 4
    first, second = targets
    delta, new_first, new_second = _PAIR[name][(labels[first], labels[second])]
    if (new_first, new_second) == (labels[first], labels[second]):
        return tuple(labels), (phase + delta) % 4
    updated = list(labels)
    updated[first] = new_first
    updated[second] = new_second
    return tuple(updated), (phase + delta) % 4


def conjugate_pauli(labels: Sequence[int], phase: int = 0, *,
                    ops: Sequence[object]) -> tuple[tuple[int, ...], int]:
    """``op_{last} ... op_1 P op_1† ... op_{last}†`` for one signed Pauli string.

    This is a Clifford circuit's *conjugation action*, which is a complete invariant
    of its unitary up to one global phase: if two Cliffords map every Pauli
    generator the same way, their product commutes with the whole Pauli group and is
    therefore a scalar. It is also the reason a circuit's unitary can be compared
    without ever writing down ``2**n`` amplitudes -- the question is a finite table
    of signed strings, not a matrix.
    """
    current = tuple(labels)
    current_phase = phase % 4
    for op in ops:
        if not isinstance(op, Gate) or op.condition:
            raise StabilizerUnsupportedError(
                f"{op!r} is not an unconditioned gate; conjugation is defined for "
                f"unitary Clifford gates only")
        spec = _clifford_spec(op.name, op.targets, op.params, len(current))
        current, current_phase = _conjugated(current, current_phase, op.name, op.targets,
                                             spec.arity)
    return current, current_phase


def clifford_pauli_images(circuit: Circuit) -> tuple[tuple[tuple[int, ...], int], ...]:
    """The circuit's images of ``X_0, Z_0, X_1, Z_1, ...`` as signed Pauli strings.

    ``2n`` entries, exactly computed, for any ``n``: this is the table two Clifford
    circuits are compared on. A relative phase anywhere in the circuit changes one of
    these entries; a global phase changes none, which is the same convention the
    dense referee uses when it minimizes over ``phi``.
    """
    circuit.validate()
    ops = tuple(circuit.ops)
    images = []
    for qubit in range(circuit.num_qubits):
        for label in (X, Z):
            labels = [I] * circuit.num_qubits
            labels[qubit] = label
            images.append(conjugate_pauli(labels, 0, ops=ops))
    return tuple(images)


def _z_at(qubit: int, num_qubits: int) -> tuple[int, ...]:
    labels = [I] * num_qubits
    labels[qubit] = Z
    return tuple(labels)


class StabilizerBackend:
    """Clifford execution with a stabilizer state per shot, no amplitudes.

    Rust-first, Python fallback: with `engine="auto"` (the default) the shot loop
    runs in :mod:`apedp_dyn.vqc.native` when that extension is importable and in
    this module otherwise, and the returned :class:`RunResult` says which one ran
    in its `engine` field. `engine="rust"` refuses rather than downgrading
    silently; `engine="python"` is the reference path, and it is the one the
    parity suite compares against.
    """

    name = "stabilizer"

    def run(self, circuit: Circuit, *, shots: int, seed: int | None = None,
            keep_states: bool = False, engine: str = "auto") -> RunResult:
        circuit.validate()
        if shots < 1:
            raise ValueError(f"shots must be positive, got {shots}")
        if keep_states:
            raise StabilizerUnsupportedError(
                "the stabilizer tier holds generators, not amplitudes; keep_states has "
                "nothing to return")
        if engine not in ("auto", "python", "rust"):
            raise ValueError(f"engine must be 'auto', 'python' or 'rust', got {engine!r}")
        if engine != "python":
            native_result = native.run(circuit, shots=shots, seed=seed)
            if native_result is not None:
                _out: RunResult = native_result
                return _out
            if engine == "rust":
                # A requested native run that did not happen is a refusal, not a
                # quiet downgrade: the caller asked for a provenance to cite.
                raise StabilizerUnsupportedError(
                    f"the native engine could not run this circuit: "
                    f"{native.last_failure() or native.unavailable_reason()}")
        rng = np.random.default_rng(seed)
        counts: dict[str, int] = {}
        samples: list[tuple[bool, ...]] = []
        excitation = dict.fromkeys(circuit.ancillas, 0.0)

        for _shot in range(shots):
            state = StabilizerState(circuit.num_qubits)
            bits: dict[int, bool] = {}
            for op in circuit.ops:
                if op.condition and not all(bits.get(bit, False) is value
                                            for bit, value in op.condition):
                    continue
                if isinstance(op, Gate):
                    state.apply_gate(op.name, op.targets, op.params)
                elif isinstance(op, Measure):
                    bits[op.bit] = state.measure(op.qubit, rng=rng)
                elif isinstance(op, Reset):
                    state.reset(op.qubit, rng=rng)
                elif isinstance(op, Noise):
                    raise StabilizerUnsupportedError(
                        f"noise op {op.kind!r} turns a pure stabilizer state into a "
                        f"mixture; use the statevector tier")
                else:  # pragma: no cover - Circuit.validate rejects other ops
                    raise CircuitValidationError(f"unknown op {op!r}")
            outcome = tuple(bool(bits.get(index, False)) for index in range(circuit.num_bits))
            key = bitstring(outcome)
            counts[key] = counts.get(key, 0) + 1
            samples.append(outcome)
            for ancilla in circuit.ancillas:
                excitation[ancilla] += state.outcome_of(ancilla)

        return RunResult(
            backend=self.name,
            shots=shots,
            seed=seed,
            counts=counts,
            samples=tuple(samples),
            ancilla_excitation={key: value / shots for key, value in excitation.items()},
            engine="python",
        )


def is_clifford_circuit(circuit: Circuit) -> bool:
    """Whether every operation in `circuit` is inside this tier's domain."""
    for op in circuit.ops:
        if isinstance(op, Gate):
            if op.name not in CLIFFORD_GATES or op.params:
                return False
        elif isinstance(op, Noise):
            return False
    return True


def ghz_circuit(num_qubits: int, *, measure: bool = True,
                basis: str = "z") -> Circuit:
    """`H` on qubit 0 then a `CX` ladder: ``(|0...0> + |1...1>)/sqrt(2)``.

    The one circuit both tiers can be asked for, which is why it is the
    cross-check: `basis="x"` appends `H` to every qubit, turning the state into a
    uniform superposition over even-parity strings, and `measure=False` leaves the
    state unmeasured for the amplitude-level tier to inspect.
    """
    if num_qubits < 2:
        raise ValueError(f"a chain needs at least two qubits, got {num_qubits}")
    if basis not in ("z", "x"):
        raise ValueError(f"basis must be 'z' or 'x', got {basis!r}")
    ops: list[Any] = [Gate("h", (0,))]
    ops.extend(Gate("cx", (qubit - 1, qubit)) for qubit in range(1, num_qubits))
    if basis == "x":
        ops.extend(Gate("h", (qubit,)) for qubit in range(num_qubits))
    num_bits = 0
    if measure:
        ops.extend(Measure(qubit, qubit) for qubit in range(num_qubits))
        num_bits = num_qubits
    return Circuit(
        ops=tuple(ops),
        num_qubits=num_qubits,
        num_bits=num_bits,
        metadata={"family": "ghz", "basis": basis, "qubits": str(num_qubits)},
    )


def ghz_generators(num_qubits: int) -> tuple[tuple[int, ...], ...]:
    """The closed-form stabilizers of the GHZ state, for asserting against a run."""
    if num_qubits < 2:
        raise ValueError(f"a chain needs at least two qubits, got {num_qubits}")
    generators: list[tuple[int, ...]] = [tuple([X] * num_qubits)]
    for qubit in range(num_qubits - 1):
        labels = [I] * num_qubits
        labels[qubit] = Z
        labels[qubit + 1] = Z
        generators.append(tuple(labels))
    return tuple(generators)


def ghz_even_parity_probability(num_qubits: int) -> float:
    """``2**-(n-1)``: the weight the X-basis GHZ state gives each even string."""
    return math.ldexp(1.0, -(num_qubits - 1))
