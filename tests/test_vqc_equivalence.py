"""The equivalence referee: what it accepts, and every wrong candidate it rejects.

The cases are chosen so that a weaker checker would pass them:

* `CX` versus `H CZ H` -- different gates, same unitary, must be VALID;
* `CCX` versus `CCX; Z` -- **identical measurement statistics**, different
  relative phase, must be INVALID;
* a measured "read-then-gate" implementation -- identical on basis inputs, not a
  unitary at all, must be INVALID with that reason;
* a global phase -- must stay VALID, because one global phase is not an error.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from apedp_dyn.vqc import (
    Circuit,
    Gate,
    Measure,
    StatevectorBackend,
    adjudicate_equivalence,
    unitary,
)
from apedp_dyn.vqc.equivalence import (
    NotAUnitaryCircuitError,
    global_phase_distance,
    unitary_distance,
)


def cx_reference() -> Circuit:
    return Circuit(ops=(Gate("cx", (0, 1)),), num_qubits=2)


def cx_via_cz() -> Circuit:
    return Circuit(ops=(Gate("h", (1,)), Gate("cz", (0, 1)), Gate("h", (1,))), num_qubits=2)


def ccx_reference() -> Circuit:
    return Circuit(ops=(Gate("ccx", (0, 1, 2)),), num_qubits=3)


def test_different_gate_sequences_that_compute_the_same_unitary_are_valid():
    report = adjudicate_equivalence(cx_reference(), cx_via_cz())
    assert report.verdict == "VALID", report.failed()
    assert report.distance == pytest.approx(0.0, abs=1e-12)
    assert report.qubits == 2


def test_a_global_phase_is_not_an_error():
    """`RX(2*pi)` is `-I`: the same circuit up to one global phase."""
    identity = Circuit(ops=(), num_qubits=1)
    rotated = Circuit(ops=(Gate("rx", (0,), (2.0 * math.pi,)),), num_qubits=1)
    report = adjudicate_equivalence(identity, rotated)
    assert report.verdict == "VALID"
    assert abs(report.global_phase) == pytest.approx(math.pi, abs=1e-9)
    assert report.distance == pytest.approx(0.0, abs=1e-12)


def test_a_global_phase_is_free_standalone_and_not_free_under_control():
    """`RX(2*pi) = -I`: the same operation, until the operation is controlled.

    Under a control qubit the phase sits on one branch, so it is a relative phase
    between branches -- exactly what a standalone comparison is allowed to discard
    and a subroutine comparison must not.
    """
    identity = Circuit(ops=(), num_qubits=1)
    rotated = Circuit(ops=(Gate("rx", (0,), (2.0 * math.pi,)),), num_qubits=1)

    standalone = adjudicate_equivalence(identity, rotated)
    assert standalone.verdict == "VALID"
    assert standalone.scope == "standalone"

    controlled = adjudicate_equivalence(identity, rotated, as_subroutine=True)
    assert controlled.scope == "controlled"
    assert controlled.verdict == "INVALID"
    assert "under control" in controlled.failed()[0]
    # `diag(I, -I)` has zero overlap with `diag(I, I)`, so the distance is the maximum.
    assert controlled.distance == pytest.approx(math.sqrt(2), abs=1e-12)
    assert controlled.as_dict()["decides"].startswith("diag(I, U_c)")


@pytest.mark.parametrize("as_subroutine", (False, True))
def test_an_identity_that_is_actually_an_identity_survives_either_scope(as_subroutine):
    report = adjudicate_equivalence(cx_reference(), cx_via_cz(), as_subroutine=as_subroutine)
    assert report.verdict == "VALID", report.failed()
    assert report.distance == pytest.approx(0.0, abs=1e-12)


def test_a_relative_phase_error_is_rejected_even_when_every_probability_matches():
    """`CCX` and `CCX; Z` are indistinguishable in the Z basis, and not equivalent."""
    reference = ccx_reference()
    candidate = Circuit(ops=(Gate("ccx", (0, 1, 2)), Gate("z", (2,))), num_qubits=3)

    # Same Z-basis statistics on a uniform superposition: the weak check passes.
    setup = [Gate("h", (0,)), Gate("h", (1,)), Gate("h", (2,))]
    states = []
    for circuit in (reference, candidate):
        prepared = Circuit(ops=tuple(setup) + circuit.ops, num_qubits=3)
        state = StatevectorBackend().run(prepared, shots=1, seed=0,
                                         keep_states=True).final_states[0]
        states.append(np.abs(state) ** 2)
    assert np.allclose(states[0], states[1], atol=1e-12), (
        "the two circuits must agree on every Z-basis probability for this test to mean "
        "anything")

    # The referee rejects it, because the unitaries differ by a relative phase.
    report = adjudicate_equivalence(reference, candidate)
    assert report.verdict == "INVALID"
    assert "unitaries agree up to one global phase" in report.failed()
    assert report.distance > 1.0


def test_a_fewer_toffoli_candidate_that_implements_the_wrong_operation_is_rejected():
    """Two CNOTs instead of a Toffoli: cheaper, and not the same map."""
    candidate = Circuit(ops=(Gate("cx", (0, 2)), Gate("cx", (1, 2))), num_qubits=3)
    report = adjudicate_equivalence(ccx_reference(), candidate)
    assert report.verdict == "INVALID"
    assert report.distance > 0.5


def test_a_measure_and_feedforward_candidate_is_a_channel_not_a_unitary():
    candidate = Circuit(ops=(Measure(0, 0), Gate("x", (1,), condition=((0, True),))),
                        num_qubits=2, num_bits=1)
    with pytest.raises(NotAUnitaryCircuitError):
        unitary(candidate)
    report = adjudicate_equivalence(cx_reference(), candidate)
    assert report.verdict == "INVALID"
    assert "candidate defines a unitary" in report.failed()
    reason = next(check for check in report.checks
                  if check.name.startswith("candidate")).measured["reason"]
    assert "channel" in reason


def test_a_candidate_on_a_different_number_of_qubits_is_rejected():
    candidate = Circuit(ops=(Gate("ccx", (0, 1, 2)),), num_qubits=4)
    report = adjudicate_equivalence(ccx_reference(), candidate)
    assert report.verdict == "INVALID"
    assert "both circuits act on the same number of qubits" in report.failed()


def test_a_resource_claim_that_is_not_met_is_reported_apart_from_the_verdict():
    """Equivalent, but larger than the improvement it is offered under.

    The semantics and the economics are two statements: this candidate computes the
    right unitary, so its verdict is VALID, and it is not an improvement, so it is
    not promotable. Reporting it as INVALID would say the circuit is wrong.
    """
    equivalent = cx_via_cz()  # 1 two-qubit gate, 2 single-qubit gates
    met = adjudicate_equivalence(cx_reference(), equivalent,
                                 claims={"two_qubit_gates": 1})
    assert met.verdict == "VALID"
    assert met.claim_verdict == "MET"
    assert met.promotable is True

    unmet = adjudicate_equivalence(cx_reference(), equivalent,
                                   claims={"operations": 1})
    assert unmet.verdict == "VALID", "the circuit is still the right operation"
    assert unmet.failed() == ()
    assert unmet.claim_verdict == "UNMET"
    assert unmet.promotable is False
    assert unmet.claim is not None
    assert unmet.claim.measured["unmet"]["operations"]["actual"] == 3
    assert unmet.as_dict()["claim_verdict"] == "UNMET"


def test_the_reported_distance_is_the_minimised_metric_and_not_a_stale_number():
    """`I` versus `Z`: 1.414214, not 2.0.

    The two matrices agree on every computational-basis probability and are
    orthogonal as operators. The reported distance is
    ``min_phi ||U_c - e^(i phi) U_r||_F / sqrt(2**n)``, minimised in closed form, so
    for `I` and `Z` (zero overlap, so every phase is a minimizer) it is
    ``sqrt(2 - 0) = sqrt(2)``. The max-entry deviation *at* that phase is 2.0, is
    reported separately, and is explicitly not a minimised quantity -- the min-max
    over phi is also sqrt(2), and an earlier version of the module reported the 2.0
    as if it were that min-max.
    """
    identity = np.eye(2, dtype=np.complex128)
    z_gate = np.diag([1.0, -1.0]).astype(np.complex128)
    distance, _phase, entry = unitary_distance(identity, z_gate)
    assert distance == pytest.approx(math.sqrt(2), abs=1e-12)
    assert entry == pytest.approx(2.0)
    assert distance != entry

    # The reviewer's check, at the referee level: same verdict, corrected number.
    report = adjudicate_equivalence(Circuit(ops=(), num_qubits=1),
                                   Circuit(ops=(Gate("z", (0,)),), num_qubits=1))
    assert report.verdict == "INVALID"
    assert report.distance == pytest.approx(math.sqrt(2), abs=1e-12)
    assert report.entry_deviation == pytest.approx(2.0)
    assert report.as_dict()["entry_deviation_is_minimised"] is False


def test_the_metric_is_a_norm_so_zero_means_one_global_phase_and_two_means_orthogonal():
    identity = np.eye(4, dtype=np.complex128)
    phased = -identity
    flipped = np.diag([1.0, 1.0, 1.0, -1.0]).astype(np.complex128)
    assert unitary_distance(identity, phased)[0] == pytest.approx(0.0, abs=1e-12)
    # One flipped amplitude: overlap 2 of 4, so `sqrt(2 - 1) = 1`.
    assert unitary_distance(identity, flipped)[0] == pytest.approx(1.0, abs=1e-12)
    # A permutation with no fixed points has zero Hilbert-Schmidt overlap with the
    # identity, which is the largest this metric gets: sqrt(2).
    cycle = np.zeros((4, 4), dtype=np.complex128)
    for index in range(4):
        cycle[(index + 1) % 4, index] = 1.0
    assert unitary_distance(identity, cycle)[0] == pytest.approx(math.sqrt(2), abs=1e-12)
    # And the swap, which fixes two basis states, is half-way.
    swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                    dtype=np.complex128)
    assert unitary_distance(identity, swap)[0] == pytest.approx(1.0, abs=1e-12)


def test_the_report_is_machine_readable_and_reproducible():
    first = adjudicate_equivalence(cx_reference(), cx_via_cz())
    second = adjudicate_equivalence(cx_reference(), cx_via_cz())
    assert first.as_dict() == second.as_dict()

    payload = json.loads(first.to_json())
    assert payload["verdict"] == "VALID"
    assert payload["reference_sha256"] == cx_reference().sha256()
    assert payload["candidate_sha256"] == cx_via_cz().sha256()
    assert payload["resources"]["candidate"]["two_qubit_gates"] == 1
    assert payload["distance"] <= payload["tolerance"]


def test_the_distance_is_zero_only_up_to_the_phase_and_not_for_a_permutation():
    permutation = Circuit(ops=(Gate("swap", (0, 1)),), num_qubits=2)
    distance, _phase = global_phase_distance(unitary(cx_reference()), unitary(permutation))
    assert distance > 0.5


# ----------------------------------------------- an oracle the referee did not write
#
# `unitary()` and the referee both go through the backend's gate application, so a bug
# in that one implementation would be invisible to both. These matrices are built a
# second way -- bit arithmetic on basis indices, no tensor slicing, no simulator -- and
# every path above is required to agree with them.
_PAULI = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def oracle_gate_matrix(name: str, targets: tuple[int, ...], num_qubits: int) -> np.ndarray:
    """One gate on basis indices, by definition rather than by application."""
    dim = 1 << num_qubits
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    root_half = 1.0 / math.sqrt(2.0)
    for column in range(dim):
        if name == "cx":
            control, target = targets
            row = column ^ (((column >> control) & 1) << target)
            matrix[row, column] = 1.0
        elif name == "cz":
            first, second = targets
            sign = -1.0 if ((column >> first) & 1) and ((column >> second) & 1) else 1.0
            matrix[column, column] = sign
        elif name == "swap":
            first, second = targets
            a, b = (column >> first) & 1, (column >> second) & 1
            row = column ^ ((a ^ b) << first) ^ ((a ^ b) << second)
            matrix[row, column] = 1.0
        elif name == "ccx":
            first, second, target = targets
            fire = ((column >> first) & 1) & ((column >> second) & 1)
            matrix[column ^ (fire << target), column] = 1.0
        else:
            qubit = targets[0]
            bit = (column >> qubit) & 1
            high = column | (1 << qubit)
            low = column & ~(1 << qubit)
            if name == "x":
                matrix[high if bit == 0 else low, column] = 1.0
            elif name == "y":
                matrix[high if bit == 0 else low, column] = 1j if bit == 0 else -1j
            elif name == "z":
                matrix[column, column] = -1.0 if bit else 1.0
            elif name == "s":
                matrix[column, column] = 1j if bit else 1.0
            elif name == "sdg":
                matrix[column, column] = -1j if bit else 1.0
            elif name == "h":
                matrix[low, column] = root_half
                matrix[high, column] = -root_half if bit else root_half
            else:  # pragma: no cover - the test only builds Clifford gates
                raise AssertionError(f"no oracle for {name!r}")
    return matrix


def oracle_unitary(circuit: Circuit) -> np.ndarray:
    dim = 1 << circuit.num_qubits
    matrix = np.eye(dim, dtype=np.complex128)
    for op in circuit.ops:
        matrix = oracle_gate_matrix(op.name, tuple(op.targets), circuit.num_qubits) @ matrix
    return matrix


RANDOM_CLIFFORD_GATES = ("h", "x", "y", "z", "s", "sdg", "cx", "cz", "swap")


@pytest.mark.parametrize("seed", (0, 1, 2, 3))
def test_the_execution_path_agrees_with_an_independent_oracle(seed):
    """Same unitary, computed two ways that share no code."""
    rng = np.random.default_rng(seed)
    num_qubits = int(rng.integers(1, 5))
    ops = []
    for _ in range(12):
        name = RANDOM_CLIFFORD_GATES[int(rng.integers(len(RANDOM_CLIFFORD_GATES)))]
        arity = 2 if name in ("cx", "cz", "swap") else 1
        ops.append(Gate(name, tuple(int(q) for q in rng.choice(num_qubits, arity,
                                                               replace=False))))
    circuit = Circuit(ops=tuple(ops), num_qubits=num_qubits)
    assert np.allclose(unitary(circuit), oracle_unitary(circuit), atol=1e-12)


@pytest.mark.parametrize("seed", (0, 1))
def test_the_tableau_path_agrees_with_the_same_oracle(seed):
    """The Clifford tier's images are `U P U^dagger` for the oracle's `U` too."""
    from apedp_dyn.vqc.stabilizer import clifford_pauli_images

    rng = np.random.default_rng(seed)
    num_qubits = 3
    ops = []
    for _ in range(10):
        name = RANDOM_CLIFFORD_GATES[int(rng.integers(len(RANDOM_CLIFFORD_GATES)))]
        arity = 2 if name in ("cx", "cz", "swap") else 1
        ops.append(Gate(name, tuple(int(q) for q in rng.choice(num_qubits, arity,
                                                               replace=False))))
    circuit = Circuit(ops=tuple(ops), num_qubits=num_qubits)
    u = oracle_unitary(circuit)
    for index, (labels, phase) in enumerate(clifford_pauli_images(circuit)):
        qubit, axis = divmod(index, 2)
        basis = ["I"] * num_qubits
        basis[qubit] = "XZ"[axis]
        # Qubit 0 is the least significant bit of the amplitude index, so the kron
        # factors run from the highest qubit down.
        operator = np.array([[1.0 + 0j]])
        for label in reversed(basis):
            operator = np.kron(operator, _PAULI[label])
        expected = np.array([[1.0 + 0j]])
        for label in reversed(["IXYZ"[value] for value in labels]):
            expected = np.kron(expected, _PAULI[label])
        expected = (1j ** phase) * expected
        assert np.allclose(u @ operator @ u.conj().T, expected, atol=1e-12), index
