"""The five-qubit code: a perfect distance-3 code for arbitrary single-qubit errors.

Status: experimental

The repetition codes in :mod:`apedp_dyn.vqc.logical` protect one error axis each.
The `[[5, 1, 3]]` code protects **all three**: any single-qubit `X`, `Y` or `Z` is
identified and corrected, because its sixteen syndromes are exactly the sixteen
Pauli classes `{I, X_i, Y_i, Z_i}`. That is what "perfect" means here, and it is
checkable: the decoder table is built from the stabilizers by commutation, and the
module refuses to import if two classes collide.

Everything is a circuit in the IR: encoding by measuring the four stabilizers and
fixing the signs with a conditional Pauli frame, syndrome extraction with four
ancillas, sixteen classically conditioned corrections, and a logical readout.

Scope, stated because it is easy to overclaim:

* **perfect syndrome extraction.** Ancillas are noiseless and measurements ideal;
  errors are injected on the five data qubits only.
* **one observable.** The readout measures the logical `Z` (`Z^5`), so a residual
  logical `Z` flip is invisible to it. The exact prediction is the probability
  that the residual anticommutes with `Z_L`, and it is compared with exactly that.
* **no threshold.** A single distance cannot demonstrate one. No encoded gate is
  implemented and nothing here is fault tolerant.

The exact prediction is not a fitted curve: it enumerates all `4^5 = 1024` Pauli
error patterns with their probabilities under independent depolarising noise,
applies the same decoder the circuit applies, and asks whether the residual is a
logical `X`/`Y`. That is arithmetic over the code's own stabilizer table, computed
without running the circuit.
"""
from __future__ import annotations

import dataclasses
import itertools
import math
from functools import lru_cache
from typing import Any

import numpy as np

from apedp_dyn.vqc.backends import StatevectorBackend
from apedp_dyn.vqc.circuit import Circuit, Gate, Measure, Noise

__all__ = [
    "FIVE_QUBIT_STABILIZERS",
    "LOGICAL_X",
    "LOGICAL_Z",
    "FiveQubitRun",
    "encoded_state",
    "exact_failure_probability",
    "five_qubit_code_circuit",
    "five_qubit_encoder_circuit",
    "five_qubit_logical_error_rate",
    "syndrome_of",
]

#: The Laflamme stabilizer set, as Pauli strings over qubits 0..4.
FIVE_QUBIT_STABILIZERS = ("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ")
LOGICAL_X = "XXXXX"
LOGICAL_Z = "ZZZZZ"

DATA = 5
#: Encoder and syndrome ancillas: one per stabilizer, reset by every measurement.
ANCILLAS = (5, 6, 7, 8)
#: Bits: 0-3 encoder syndrome, 4-7 correction syndrome, 8-12 readout.
SYNDROME_BITS = (4, 5, 6, 7)
READOUT_BITS = (8, 9, 10, 11, 12)

_PRODUCT = {
    frozenset("IX"): "X", frozenset("IY"): "Y", frozenset("IZ"): "Z",
    frozenset("XY"): "Z", frozenset("XZ"): "Y", frozenset("YZ"): "X",
}
_ANTICOMMUTING = frozenset({frozenset("XY"), frozenset("XZ"), frozenset("YZ")})


def _multiply(left: str, right: str) -> str:
    """The phase-free Pauli product of two strings of equal length."""
    if len(left) != len(right):
        raise ValueError("Pauli strings must have the same length")
    out = []
    for a, b in zip(left, right, strict=True):
        if a == b:
            out.append("I")
        else:
            out.append(_PRODUCT[frozenset((a, b))])
    return "".join(out)


def _anticommutes(left: str, right: str) -> bool:
    """Symplectic inner product: an odd number of anticommuting positions."""
    parity = 0
    for a, b in zip(left, right, strict=True):
        if frozenset((a, b)) in _ANTICOMMUTING:
            parity ^= 1
    return bool(parity)


def syndrome_of(pauli: str) -> tuple[bool, bool, bool, bool]:
    """The measured syndrome of a Pauli error: bit `i` set iff it anticommutes with `g_i`."""
    if len(pauli) != DATA:
        raise ValueError(f"expected a {DATA}-qubit Pauli string, got {pauli!r}")
    return tuple(_anticommutes(pauli, stabilizer) for stabilizer in FIVE_QUBIT_STABILIZERS)


def _single_qubit_classes() -> dict[tuple[bool, bool, bool, bool], str]:
    classes: dict[tuple[bool, bool, bool, bool], str] = {}
    candidates = ["I" * DATA]
    for qubit in range(DATA):
        for pauli in "XYZ":
            string = ["I"] * DATA
            string[qubit] = pauli
            candidates.append("".join(string))
    for candidate in candidates:
        signature = syndrome_of(candidate)
        if signature in classes:
            raise AssertionError(
                f"the code is not perfect: {candidate!r} and {classes[signature]!r} share "
                f"the syndrome {signature}")
        classes[signature] = candidate
    if len(classes) != 16:
        raise AssertionError(f"expected 16 distinct syndromes, got {len(classes)}")
    return classes


#: Syndrome -> the single-qubit Pauli that produced it. Built once, asserted perfect.
DECODER: dict[tuple[bool, bool, bool, bool], str] = _single_qubit_classes()


@lru_cache(maxsize=32)
def _frame_correction(signature: tuple[bool, ...]) -> str:
    """A Pauli that flips exactly the stabilizer signs in `signature` and fixes `Z_L`.

    Encoding by measurement produces a random sign pattern; this is the Pauli frame
    that returns it to the `+1` eigenspace without changing the logical value (it
    must commute with `Z_L`, or a logical flip would be introduced silently).
    """
    signature = tuple(bool(bit) for bit in signature)
    if len(signature) != len(FIVE_QUBIT_STABILIZERS):
        raise ValueError(f"expected {len(FIVE_QUBIT_STABILIZERS)} signs, got {len(signature)}")
    best: str | None = None
    for candidate in map("".join, itertools.product("IXYZ", repeat=DATA)):
        if syndrome_of(candidate) != signature:
            continue
        if _anticommutes(candidate, LOGICAL_Z):
            continue
        # Prefer the lowest-weight frame (fewest gates), then a stable tie-break.
        if best is None or candidate.count("I") > best.count("I"):
            best = candidate
        elif candidate.count("I") == best.count("I") and candidate < best:
            best = candidate
    if best is None:  # pragma: no cover - the code's symmetry guarantees existence
        raise AssertionError(f"no frame correction for signature {signature}")
    return best


def _apply_pauli(ops: list[Any], pauli: str, condition=()) -> None:
    for qubit, letter in enumerate(pauli):
        if letter == "X":
            ops.append(Gate("x", (qubit,), condition=condition))
        elif letter == "Z":
            ops.append(Gate("z", (qubit,), condition=condition))
        elif letter == "Y":
            ops.append(Gate("x", (qubit,), condition=condition))
            ops.append(Gate("z", (qubit,), condition=condition))


def _measure_stabilizer(ops: list[Any], stabilizer: str, ancilla: int, bit: int) -> None:
    """Measure one code stabilizer into `bit` with one ancilla.

    Rotate each `X`/`Y` factor to `Z`, CNOT the support into the ancilla, measure
    it, then undo the rotation so the data are left where they were found. The
    ancilla is reset by the measurement, so the four stabilizers share four
    ancillas.
    """
    for qubit, letter in enumerate(stabilizer):
        if letter == "X":
            ops.append(Gate("h", (qubit,)))
        elif letter == "Y":
            ops.append(Gate("sdg", (qubit,)))
            ops.append(Gate("h", (qubit,)))
    for qubit, letter in enumerate(stabilizer):
        if letter != "I":
            ops.append(Gate("cx", (qubit, ancilla)))
    ops.append(Measure(ancilla, bit))
    for qubit, letter in reversed(list(enumerate(stabilizer))):
        if letter == "X":
            ops.append(Gate("h", (qubit,)))
        elif letter == "Y":
            ops.append(Gate("h", (qubit,)))
            ops.append(Gate("s", (qubit,)))


def _encoded_amplitudes(prepare_logical_one: bool) -> np.ndarray:
    """`|0_L>` (or `|1_L>`), built by projecting `|00000>` with the stabilizers."""
    state: np.ndarray = np.zeros(1 << DATA, dtype=np.complex128)
    state[0] = 1.0
    for stabilizer in FIVE_QUBIT_STABILIZERS:
        projected = state.copy()
        for basis in range(1 << DATA):
            amplitude = state[basis]
            if amplitude == 0.0:
                continue
            image = basis
            sign = 1.0
            for qubit, letter in enumerate(stabilizer):
                bit = (basis >> qubit) & 1
                if letter == "X":
                    image ^= 1 << qubit
                elif letter == "Z":
                    sign *= -1.0 if bit else 1.0
                elif letter == "Y":
                    image ^= 1 << qubit
                    sign *= -1.0 if bit else 1.0
            projected[image] += sign * amplitude
        norm = np.linalg.norm(projected)
        if norm < 1e-12:
            raise AssertionError(f"{stabilizer} annihilated the state")
        state = projected / norm
    if prepare_logical_one:
        state = state[::-1].copy()  # X on all five qubits: reverse the basis order
    return state


def encoded_state(prepare_logical_one: bool = False) -> np.ndarray:
    """The exact encoded statevector, independent of the encoder circuit."""
    return _encoded_amplitudes(prepare_logical_one)


def five_qubit_encoder_circuit(*, prepare_logical_one: bool = False,
                               metadata: dict[str, str] | None = None) -> Circuit:
    """The encoder alone: stabilizer projection plus the conditional Pauli frame."""
    ops: list[Any] = []
    for index, stabilizer in enumerate(FIVE_QUBIT_STABILIZERS):
        _measure_stabilizer(ops, stabilizer, ANCILLAS[index], index)
    for pattern in itertools.product((False, True), repeat=len(FIVE_QUBIT_STABILIZERS)):
        signature = tuple(bool(bit) for bit in pattern)
        if not any(signature):
            continue
        condition = tuple((bit, bool(value)) for bit, value in enumerate(signature))
        _apply_pauli(ops, _frame_correction(signature), condition)
    if prepare_logical_one:
        _apply_pauli(ops, LOGICAL_X)
    return Circuit(ops=tuple(ops), num_qubits=DATA + len(ANCILLAS),
                   num_bits=len(FIVE_QUBIT_STABILIZERS), ancillas=ANCILLAS,
                   metadata={"generator": "vqc.five_qubit.five_qubit_encoder_circuit",
                             "code": "five-qubit-perfect",
                             **(metadata or {})})


def five_qubit_code_circuit(*, prepare_logical_one: bool = False,
                            error_probability: float = 0.0,
                            injected_errors: tuple[tuple[int, str], ...] = (),
                            metadata: dict[str, str] | None = None) -> Circuit:
    """Encode, corrupt, extract the syndrome, correct, and read the logical `Z`."""
    ops: list[Any] = []
    # 1. Encode by projection: measure the four stabilizers, then fix the signs.
    for index, stabilizer in enumerate(FIVE_QUBIT_STABILIZERS):
        _measure_stabilizer(ops, stabilizer, ANCILLAS[index], index)
    for pattern in itertools.product((False, True), repeat=len(FIVE_QUBIT_STABILIZERS)):
        signature = tuple(bool(bit) for bit in pattern)
        if not any(signature):
            continue
        condition = tuple((bit, bool(value)) for bit, value in enumerate(signature))
        _apply_pauli(ops, _frame_correction(signature), condition)

    # 2. The logical value, then the errors the experiment is about.
    if prepare_logical_one:
        _apply_pauli(ops, LOGICAL_X)
    for qubit, pauli in injected_errors:
        _apply_pauli(ops, "I" * qubit + pauli + "I" * (DATA - qubit - 1))
    if error_probability > 0.0:
        ops += [Noise(qubit, "depolarizing", error_probability) for qubit in range(DATA)]

    # 3. Syndrome extraction and correction.
    for index, stabilizer in enumerate(FIVE_QUBIT_STABILIZERS):
        _measure_stabilizer(ops, stabilizer, ANCILLAS[index], SYNDROME_BITS[index])
    for pattern in itertools.product((False, True), repeat=len(FIVE_QUBIT_STABILIZERS)):
        signature = tuple(bool(bit) for bit in pattern)
        correction = DECODER[signature]
        if correction == "I" * DATA:
            continue
        condition = tuple((SYNDROME_BITS[index], bool(value))
                          for index, value in enumerate(signature))
        _apply_pauli(ops, correction, condition)

    # 4. Logical Z readout: the parity of the five data qubits.
    ops += [Measure(qubit, READOUT_BITS[qubit]) for qubit in range(DATA)]

    return Circuit(ops=tuple(ops), num_qubits=DATA + len(ANCILLAS),
                   num_bits=len(READOUT_BITS) + 8, ancillas=ANCILLAS,
                   metadata={"generator": "vqc.five_qubit.five_qubit_code_circuit",
                             "code": "five-qubit-perfect", "distance": "3",
                             "syndrome_extraction": "perfect-ancilla",
                             "observable": "Z",
                             **(metadata or {})})


def exact_failure_probability(probability: float) -> float:
    """Exact `P(logical Z flip)` under independent depolarising noise at `probability`.

    Enumerates all 1024 Pauli patterns: the decoder is applied to each, and the
    residual is counted as a failure when it anticommutes with `Z_L`. Nothing here
    runs the circuit, so it is an independent check on the experiment.
    """
    p = float(probability)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability {p} is not in [0, 1]")
    total = 0.0
    for pattern in itertools.product("IXYZ", repeat=DATA):
        weight = 1.0
        for letter in pattern:
            weight *= (1.0 - p) if letter == "I" else p / 3.0
        if weight == 0.0:
            continue
        error = "".join(pattern)
        residual = _multiply(error, DECODER[syndrome_of(error)])
        if _anticommutes(residual, LOGICAL_Z):
            total += weight
    return total


@dataclasses.dataclass(frozen=True)
class FiveQubitRun:
    """One measured logical error rate for the five-qubit code."""

    probability: float
    shots: int
    failures: int
    seed: int | None
    prepare_logical_one: bool
    circuit_sha256: str

    @property
    def rate(self) -> float:
        return self.failures / self.shots

    @property
    def standard_error(self) -> float:
        return math.sqrt(max(self.rate * (1.0 - self.rate), 1e-12) / self.shots)

    @property
    def exact(self) -> float:
        return exact_failure_probability(self.probability)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": "five-qubit-perfect",
            "distance": 3,
            "observable": "Z",
            "prepared": "1" if self.prepare_logical_one else "0",
            "physical_error_rate": self.probability,
            "shots": self.shots,
            "failures": self.failures,
            "logical_error_rate": self.rate,
            "standard_error": self.standard_error,
            "exact_failure_probability": self.exact,
            "circuit_sha256": self.circuit_sha256,
            "seed": self.seed,
            "syndrome_extraction": "perfect-ancilla",
            "counts_phase_errors": False,
        }


def five_qubit_logical_error_rate(probability: float, *, shots: int = 2000,
                                  seed: int | None = 0,
                                  prepare_logical_one: bool = False,
                                  backend=None) -> FiveQubitRun:
    """Run the code under depolarising noise and count logical `Z` failures."""
    if shots < 1:
        raise ValueError(f"shots must be positive, got {shots}")
    circuit = five_qubit_code_circuit(prepare_logical_one=prepare_logical_one,
                                      error_probability=probability)
    backend = backend or StatevectorBackend()
    result = backend.run(circuit, shots=shots, seed=seed)
    failures = 0
    for sample in result.samples:
        parity = sum(bool(sample[bit]) for bit in READOUT_BITS) % 2
        if bool(parity) != prepare_logical_one:
            failures += 1
    return FiveQubitRun(probability=float(probability), shots=shots, failures=failures,
                        seed=seed, prepare_logical_one=prepare_logical_one,
                        circuit_sha256=circuit.sha256())
