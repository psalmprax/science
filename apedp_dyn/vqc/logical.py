"""A logical qubit, built out of the IR: the three-qubit bit-flip code.

Status: experimental

This is the smallest honest logical-qubit experiment: a repetition code that
encodes one qubit in three, extracts two parity syndromes with two ancillas,
decodes, corrects, and then reports whether the encoded information survived.
Everything is a circuit in :mod:`apedp_dyn.vqc.circuit` -- encoding gates, noise
operations, ancilla measurements, classically conditioned corrections and a final
readout -- so the same object can be serialised, hashed, simulated and
adjudicated.

Scope, stated because it is easy to overclaim:

* **bit flips only.** The repetition code protects against `X`; a `Z` error is
  invisible to it. That is why the other half of the milestone is a code that
  protects arbitrary single-qubit errors, and why nothing here is called
  fault-tolerant: the syndrome extraction is assumed perfect (the ancillas are
  noiseless and the readout is ideal).
* **no threshold claim.** A single distance cannot demonstrate a threshold. The
  measured logical failure rate is compared against the exact combinatorial
  prediction for this code and against the uncorrected physical rate, and those
  are the only two claims made.

The exact prediction for distance 3 with independent bit-flip errors at rate `p`
is the probability of two or more errors, `3p^2(1-p) + p^3 = 3p^2 - 2p^3`: a
single error is corrected, a pair is *miscorrected* into a logical flip, and three
errors are left alone.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from apedp_dyn.vqc.backends import RunResult, StatevectorBackend
from apedp_dyn.vqc.circuit import Circuit, Gate, Measure, Noise

__all__ = [
    "DISTANCE",
    "RepetitionCodeRun",
    "RepetitionRun",
    "bit_flip_code_circuit",
    "decode",
    "logical_error_rate",
    "phase_flip_code_circuit",
    "repetition_code_circuit",
    "repetition_exact_failure_probability",
    "repetition_logical_error_rate",
    "single_error_syndrome",
]

DISTANCE = 3

#: Which syndromes each single-qubit error produces, for distance 3.
_SYNDROME_OF_ERROR = {
    0: (True, False),
    1: (True, True),
    2: (False, True),
}


def single_error_syndrome(qubit: int) -> tuple[bool, bool]:
    """The `(s0, s1)` syndrome a single `X` on `qubit` produces."""
    return _SYNDROME_OF_ERROR[int(qubit)]


def phase_flip_code_circuit(*, prepare_logical_one: bool = False,
                            error_probability: float = 0.0,
                            injected_errors: tuple[int, ...] = (),
                            metadata: dict[str, str] | None = None) -> Circuit:
    """The same repetition code in the X basis: it protects against `Z`.

    Encoding is the bit-flip encoder conjugated by Hadamards, so
    `|0_L> = |+++>` and `|1_L> = |--->`. A phase flip is turned into a bit flip by
    rotating all three qubits with `H` before the syndrome extraction, which is why
    the same decoder and the same exact prediction apply: single `Z` errors are
    corrected, a pair is miscorrected, and the failure rate is `3p^2 - 2p^3`.
    """
    ops: list[Any] = [Gate("x", (0,))] if prepare_logical_one else []
    # The bit-flip encoder, then `H` on all three: `|000>` becomes `|+++>` and
    # `|111>` becomes `|--->`. (`H` on the input as well would encode `|+>` instead
    # of `|0>` -- the conjugation is on the output side only.)
    ops += [Gate("cx", (0, 1)), Gate("cx", (0, 2))]
    ops += [Gate("h", (qubit,)) for qubit in range(DISTANCE)]
    ops += [Gate("z", (qubit,)) for qubit in injected_errors]
    if error_probability > 0.0:
        ops += [Noise(qubit, "phase_flip", error_probability) for qubit in range(DISTANCE)]

    # Rotate back to the Z basis: `Z` errors become `X` errors and the logical
    # states become |000> and |111>, so the bit-flip decoder applies unchanged.
    ops += [Gate("h", (qubit,)) for qubit in range(DISTANCE)]
    for j in range(DISTANCE - 1):
        ops += [Gate("cx", (j, DISTANCE + j)), Gate("cx", (j + 1, DISTANCE + j))]
        ops += [Measure(DISTANCE + j, j)]
    for qubit, (s0, s1) in _SYNDROME_OF_ERROR.items():
        ops.append(Gate("x", (qubit,), condition=((0, s0), (1, s1))))
    ops += [Measure(qubit, DISTANCE - 1 + qubit) for qubit in range(DISTANCE)]

    return Circuit(ops=tuple(ops), num_qubits=DISTANCE + 2, num_bits=2 * DISTANCE - 1,
                   ancillas=(DISTANCE, DISTANCE + 1),
                   metadata={"generator": "vqc.logical.phase_flip_code_circuit",
                             "code": "repetition-3-phaseflip",
                             "distance": str(DISTANCE),
                             "syndrome_extraction": "perfect-ancilla",
                             **(metadata or {})})


def bit_flip_code_circuit(*, prepare_logical_one: bool = False,
                          error_probability: float = 0.0,
                          injected_errors: tuple[int, ...] = (),
                          metadata: dict[str, str] | None = None) -> Circuit:
    """Encode, (optionally) corrupt, extract syndromes, correct, and read out.

    Qubits: 0, 1, 2 are the data qubits; 3 and 4 are the syndrome ancillas.
    Bits: 0 and 1 hold the syndrome, 2, 3 and 4 hold the final readout.

    `injected_errors` applies deterministic `X` errors after encoding (for
    syndrome tests); `error_probability` puts a `bit_flip` noise op on every data
    qubit instead, so a noisy run is the same circuit with a different parameter.
    """
    ops: list[Any] = [Gate("x", (0,))] if prepare_logical_one else []
    ops += [Gate("cx", (0, 1)), Gate("cx", (0, 2))]
    ops += [Gate("x", (qubit,)) for qubit in injected_errors]
    if error_probability > 0.0:
        ops += [Noise(qubit, "bit_flip", error_probability) for qubit in range(DISTANCE)]

    # s_j = q_j xor q_(j+1) into ancilla 3+j; the measurement resets the ancilla.
    for j in range(DISTANCE - 1):
        ops += [Gate("cx", (j, DISTANCE + j)), Gate("cx", (j + 1, DISTANCE + j))]
        ops += [Measure(DISTANCE + j, j)]

    # The decoder is a lookup on the two syndrome bits; each correction is a
    # classically conditioned `X` that only fires on its own syndrome.
    for qubit, (s0, s1) in _SYNDROME_OF_ERROR.items():
        condition = ((0, s0), (1, s1))
        ops.append(Gate("x", (qubit,), condition=condition))

    ops += [Measure(qubit, DISTANCE - 1 + qubit) for qubit in range(DISTANCE)]

    return Circuit(ops=tuple(ops), num_qubits=DISTANCE + 2, num_bits=2 * DISTANCE - 1,
                   ancillas=(DISTANCE, DISTANCE + 1),
                   metadata={"generator": "vqc.logical.bit_flip_code_circuit",
                             "code": "repetition-3-bitflip",
                             "distance": str(DISTANCE),
                             "syndrome_extraction": "perfect-ancilla",
                             **(metadata or {})})


def decode(readout: tuple[bool, ...] | str) -> bool:
    """Majority vote of the three readout bits: the logical value the decoder reports."""
    bits = [bit == "1" if isinstance(bit, str) else bool(bit) for bit in readout]
    if len(bits) != DISTANCE:
        raise ValueError(f"expected {DISTANCE} readout bits, got {len(bits)}")
    return sum(bits) > DISTANCE // 2


def analytic_failure_probability(probability: float) -> float:
    """`3p^2 - 2p^3`: the probability of an uncorrectable (or miscorrected) pattern."""
    p = float(probability)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability {p} is not in [0, 1]")
    return 3.0 * p * p - 2.0 * p ** 3


@dataclass(frozen=True)
class RepetitionCodeRun:
    """One measured logical error rate, with the uncertainty it was measured at."""

    probability: float
    shots: int
    failures: int
    seed: int | None
    prepare_logical_one: bool
    circuit_sha256: str
    code: str = "repetition-3-bitflip"
    protects_against: str = "bit-flip only"

    @property
    def rate(self) -> float:
        return self.failures / self.shots

    @property
    def standard_error(self) -> float:
        return math.sqrt(max(self.rate * (1.0 - self.rate), 1e-12) / self.shots)

    @property
    def analytic(self) -> float:
        return analytic_failure_probability(self.probability)

    @property
    def uncorrected(self) -> float:
        """The same physical error rate with no encoding at all."""
        return self.probability

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "distance": DISTANCE,
            "prepared": "1" if self.prepare_logical_one else "0",
            "physical_error_rate": self.probability,
            "shots": self.shots,
            "failures": self.failures,
            "logical_error_rate": self.rate,
            "standard_error": self.standard_error,
            "analytic_failure_probability": self.analytic,
            "uncorrected_physical_rate": self.uncorrected,
            "circuit_sha256": self.circuit_sha256,
            "seed": self.seed,
            "syndrome_extraction": "perfect-ancilla",
            "protects_against": self.protects_against,
        }


def _run(circuit: Circuit, *, shots: int, seed: int | None) -> RunResult:
    return StatevectorBackend().run(circuit, shots=shots, seed=seed)


def logical_error_rate(probability: float, *, shots: int = 4000, seed: int | None = 0,
                       prepare_logical_one: bool = False, code: str = "bit_flip",
                       backend: Any=None) -> RepetitionCodeRun:
    """Measure the logical failure rate of a distance-3 repetition code.

    `code` selects which error the code is built for: `"bit_flip"` protects `|0>`,
    `|1>` against `X`, `"phase_flip"` protects `|+>`, `|->` against `Z`. Both are
    the same repetition code in conjugated bases, so both are compared against the
    same exact prediction, `3p^2 - 2p^3`.
    """
    if shots < 1:
        raise ValueError(f"shots must be positive, got {shots}")
    if code == "bit_flip":
        circuit = bit_flip_code_circuit(prepare_logical_one=prepare_logical_one,
                                        error_probability=probability)
        protects_against = "bit-flip only"
    elif code == "phase_flip":
        circuit = phase_flip_code_circuit(prepare_logical_one=prepare_logical_one,
                                          error_probability=probability)
        protects_against = "phase-flip only"
    else:
        raise ValueError(f"unknown code {code!r}; expected 'bit_flip' or 'phase_flip'")
    backend = backend or StatevectorBackend()
    result = backend.run(circuit, shots=shots, seed=seed)
    failures = 0
    for sample in result.samples:
        readout = sample[DISTANCE - 1:]
        if decode(readout) != prepare_logical_one:
            failures += 1
    return RepetitionCodeRun(probability=float(probability), shots=shots, failures=failures,
                             seed=seed, prepare_logical_one=prepare_logical_one,
                             circuit_sha256=circuit.sha256(), code=circuit.metadata["code"],
                             protects_against=protects_against)


# ------------------------------------------------- repetition codes at any distance
def _repetition_decoder(distance: int, code: str, decoder: Any) -> Any:
    """The decoder object for a run: an instance if given, else the matching decoder."""
    from apedp_dyn.vqc.decoders import repetition_mwpm_decoder

    if decoder is None or decoder == "mwpm":
        return repetition_mwpm_decoder(distance, code=code)
    if decoder == "lookup":
        from apedp_dyn.vqc.decoders import repetition_lookup_decoder

        return repetition_lookup_decoder(distance, code=code)
    if isinstance(decoder, str):
        raise ValueError(f"unknown decoder {decoder!r}; use 'mwpm', 'lookup' or an instance")
    return decoder


def repetition_code_circuit(distance: int, *, code: str = "bit_flip",
                            prepare_logical_one: bool = False,
                            injected_errors: Sequence[int] = (),
                            error_probability: float = 0.0,
                            decoder: Any=None,
                            metadata: dict[str, str] | None = None) -> Circuit:
    """A distance-`distance` repetition code whose correction gates come from `decoder`.

    The protocol is built once: encode, corrupt, extract `distance - 1` parity
    syndromes into ancillas, apply the decoder's correction under a classical
    condition, and read the data out. The phase-flip code is the same circuit seen
    through `H` on every data qubit, which is why it shares the arithmetic exactly.

    The correction is emitted **per syndrome pattern**, so the decoder is a real
    input to the circuit rather than a comment on it: a lookup table and a matching
    decoder give two different circuits for the same code, and the tests compare
    them on the same noise.
    """
    if distance < 3 or distance % 2 == 0:
        raise ValueError(f"distance must be odd and at least 3, got {distance}")
    if code not in ("bit_flip", "phase_flip"):
        raise ValueError(f"code must be 'bit_flip' or 'phase_flip', got {code!r}")
    checks = distance - 1
    resolved = _repetition_decoder(distance, code, decoder)

    data_gate = "x" if code == "bit_flip" else "z"

    ops: list[Any] = []
    if prepare_logical_one:
        ops.append(Gate("x", (0,)))
    ops += [Gate("cx", (0, qubit)) for qubit in range(1, distance)]
    if code == "phase_flip":
        # `|0_L>` is `|+...+>` and `|1_L>` is `|-...->`: the code is the bit-flip code
        # in the X basis, so the data are prepared with Hadamards and the physical
        # error is a `Z`.
        ops += [Gate("h", (qubit,)) for qubit in range(distance)]
    ops += [Gate(data_gate, (qubit,)) for qubit in injected_errors]
    if error_probability > 0.0:
        kind = "bit_flip" if code == "bit_flip" else "phase_flip"
        ops += [Noise(qubit, kind, error_probability) for qubit in range(distance)]

    for check in range(checks):
        ancilla = distance + check
        if code == "phase_flip":
            # Measure `X_i X_(i+1)` by rotating the pair into the Z basis first, then
            # rotate back so the next check sees the same frame.
            ops += [Gate("h", (check,)), Gate("h", (check + 1,))]
        ops += [Gate("cx", (check, ancilla)), Gate("cx", (check + 1, ancilla))]
        ops += [Measure(ancilla, check)]
        if code == "phase_flip":
            ops += [Gate("h", (check,)), Gate("h", (check + 1,))]

    for pattern in range(1 << checks):
        correction = resolved.decode(tuple(bool(pattern >> check & 1)
                                           for check in range(checks))).pauli
        condition = tuple((check, bool(pattern >> check & 1)) for check in range(checks))
        ops += [Gate(data_gate, (qubit,), condition=condition)
                for qubit, letter in enumerate(correction) if letter != "I"]

    if code == "phase_flip":
        # Read `X_q` per data qubit: the logical value is the majority of the readout,
        # exactly as in the bit-flip code.
        ops += [Gate("h", (qubit,)) for qubit in range(distance)]
    ops += [Measure(qubit, checks + qubit) for qubit in range(distance)]

    return Circuit(ops=tuple(ops), num_qubits=2 * distance - 1,
                   num_bits=2 * distance - 1,
                   ancillas=tuple(range(distance, 2 * distance - 1)),
                   metadata={"generator": "vqc.logical.repetition_code_circuit",
                             "code": f"repetition-{distance}-{code}",
                             "distance": str(distance),
                             "decoder": resolved.name,
                             "syndrome_extraction": "perfect-ancilla",
                             **(metadata or {})})


def repetition_exact_failure_probability(distance: int, probability: float, *,
                                         code: str = "bit_flip",
                                         decoder: Any=None) -> float:
    """Exact `P(logical flip)` by enumerating all `2**distance` error patterns.

    This never runs the circuit: it applies the decoder to each pattern of
    independent single-qubit errors, counts the patterns whose residual is a logical
    flip, and sums their weights. It is therefore an independent check on the
    simulated run, and it is the prediction the run is compared against.
    """
    p = float(probability)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability {p} is not in [0, 1]")
    if distance < 3 or distance % 2 == 0:
        raise ValueError(f"distance must be odd and at least 3, got {distance}")
    resolved = _repetition_decoder(distance, code, decoder)
    letter = "X" if code == "bit_flip" else "Z"
    checks = distance - 1
    total = 0.0
    for pattern in range(1 << distance):
        weight = 0.0
        errors: list[str] = []
        for qubit in range(distance):
            flipped = bool(pattern >> qubit & 1)
            errors.append(letter if flipped else "I")
            weight += 1.0 if flipped else 0.0
        probability_of_pattern = p ** weight * (1.0 - p) ** (distance - weight)
        if probability_of_pattern == 0.0:
            continue
        # `s_check = q_check xor q_(check+1)`: XOR, not OR. An OR turns a weight-2
        # error into the wrong syndrome and the whole enumeration into a fiction.
        syndrome = tuple(errors[check] != errors[check + 1] for check in range(checks))
        correction = resolved.decode(syndrome).pauli
        residual = [errors[qubit] != correction[qubit] for qubit in range(distance)]
        # The decoded logical value is the majority of the readout, which is what
        # `decode()` does and what `3p^2 - 2p^3` counts. Parity would be a different
        # (and weaker) observable, so it is not used here.
        if sum(residual) > distance // 2:
            total += probability_of_pattern
    return total


@dataclass(frozen=True)
class RepetitionRun:
    """One measured logical error rate for a repetition code of any odd distance."""

    distance: int
    code: str
    decoder_name: str
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
        return repetition_exact_failure_probability(self.distance, self.probability,
                                                    code=self.code,
                                                    decoder=self.decoder_name)

    @property
    def uncorrected(self) -> float:
        return self.probability

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": f"repetition-{self.distance}-{self.code}",
            "distance": self.distance,
            "decoder": self.decoder_name,
            "prepared": "1" if self.prepare_logical_one else "0",
            "physical_error_rate": self.probability,
            "shots": self.shots,
            "failures": self.failures,
            "logical_error_rate": self.rate,
            "standard_error": self.standard_error,
            "exact_failure_probability": self.exact,
            "uncorrected_physical_rate": self.uncorrected,
            "circuit_sha256": self.circuit_sha256,
            "seed": self.seed,
            "syndrome_extraction": "perfect-ancilla",
        }


def repetition_logical_error_rate(distance: int, probability: float, *, shots: int = 2000,
                                  seed: int | None = 0, code: str = "bit_flip",
                                  decoder: Any=None, prepare_logical_one: bool = False,
                                  backend: Any=None) -> RepetitionRun:
    """Run a repetition code of any odd distance and count logical failures."""
    if shots < 1:
        raise ValueError(f"shots must be positive, got {shots}")
    circuit = repetition_code_circuit(distance, code=code,
                                      prepare_logical_one=prepare_logical_one,
                                      error_probability=probability, decoder=decoder)
    result = (backend or StatevectorBackend()).run(circuit, shots=shots, seed=seed)
    checks = distance - 1
    failures = 0
    for sample in result.samples:
        readout = [bool(sample[checks + qubit]) for qubit in range(distance)]
        if (sum(readout) > distance // 2) != prepare_logical_one:
            failures += 1
    resolved = _repetition_decoder(distance, code, decoder)
    return RepetitionRun(distance=distance, code=code, decoder_name=resolved.name,
                         probability=float(probability), shots=shots, failures=failures,
                         seed=seed, prepare_logical_one=prepare_logical_one,
                         circuit_sha256=circuit.sha256())
