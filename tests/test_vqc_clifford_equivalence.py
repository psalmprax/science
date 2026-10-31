"""The Clifford branch of the referee: exact, and above the dense bound.

Three things are checked here, in increasing strength:

* the images a Clifford circuit produces really are ``U P U†`` -- compared against
  the *dense* unitary for circuits small enough to have one;
* the verdicts of the two branches agree on every pair the dense branch can decide;
* a 100-qubit pair is decided at all, which the dense branch cannot do, and the two
  scopes (`unitary` and `state`) disagree exactly where they should.
"""
from __future__ import annotations

import json
import random

import numpy as np
import pytest

from apedp_dyn.vqc import Circuit, Gate, Measure
from apedp_dyn.vqc.equivalence import (
    adjudicate_clifford_equivalence,
    adjudicate_equivalence,
    unitary,
)
from apedp_dyn.vqc.gates import GATE_SPECS
from apedp_dyn.vqc.stabilizer import (
    CLIFFORD_GATES,
    StabilizerState,
    StabilizerUnsupportedError,
    clifford_pauli_images,
    ghz_generators,
    pauli_string,
)

LABELS = "IXYZ"
_PAULI_MATRICES = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


# --------------------------------------------------------------------- circuits
def ghz_ladder(num_qubits: int) -> Circuit:
    ops = [Gate("h", (0,))]
    ops.extend(Gate("cx", (qubit - 1, qubit)) for qubit in range(1, num_qubits))
    return Circuit(ops=tuple(ops), num_qubits=num_qubits)


def ghz_star(num_qubits: int) -> Circuit:
    ops = [Gate("h", (0,))]
    ops.extend(Gate("cx", (0, qubit)) for qubit in range(1, num_qubits))
    return Circuit(ops=tuple(ops), num_qubits=num_qubits)


def global_minus_one(num_qubits: int, qubit: int = 5) -> Circuit:
    """`X Z X Z` on one qubit is `-I`: a Clifford circuit with a global phase."""
    return Circuit(ops=(Gate("x", (qubit,)), Gate("z", (qubit,)),
                        Gate("x", (qubit,)), Gate("z", (qubit,))), num_qubits=num_qubits)


def random_clifford(num_qubits: int, *, seed: int, depth: int = 8) -> Circuit:
    rng = random.Random(seed)
    ops = [Gate("h", (qubit,)) for qubit in range(num_qubits)]
    for _ in range(depth):
        name = rng.choice(sorted(CLIFFORD_GATES))
        if GATE_SPECS[name].arity == 2:
            first, second = rng.sample(range(num_qubits), 2)
            ops.append(Gate(name, (first, second)))
        else:
            ops.append(Gate(name, (rng.randrange(num_qubits),)))
    return Circuit(ops=tuple(ops), num_qubits=num_qubits)


def with_append(circuit: Circuit, *extra: Gate) -> Circuit:
    return Circuit(ops=circuit.ops + extra, num_qubits=circuit.num_qubits)


def pauli_matrix(labels: str) -> np.ndarray:
    matrix = np.ones((1, 1), dtype=np.complex128)
    for label in reversed(labels):  # qubit 0 is the least significant tensor factor
        matrix = np.kron(matrix, _PAULI_MATRICES[label])
    return matrix


# --------------------------------------------------------- the images are the map
@pytest.mark.parametrize("seed", (0, 1, 2))
def test_the_images_are_the_dense_conjugation(seed: int):
    """`U P U†` computed two ways, for every generator: strings and matrices."""
    circuit = random_clifford(3, seed=seed)
    dense = unitary(circuit)
    for index, (labels, phase) in enumerate(clifford_pauli_images(circuit)):
        qubit, axis = divmod(index, 2)
        basis = ["I"] * 3
        basis[qubit] = "XZ"[axis]
        expected = dense @ pauli_matrix("".join(basis)) @ dense.conj().T
        actual = (1j ** phase) * pauli_matrix("".join(LABELS[label] for label in labels))
        assert np.allclose(expected, actual, atol=1e-12), (basis, pauli_string(labels, phase))


# ------------------------------------------------- agreement with the dense branch
@pytest.mark.parametrize("seed", (0, 1, 2))
def test_the_two_branches_agree_on_every_pair_both_can_decide(seed: int):
    base = random_clifford(4, seed=seed)
    identity_tail = (Gate("h", (0,)), Gate("h", (0,)), Gate("cz", (1, 2)), Gate("cz", (1, 2)))
    pairs = [
        with_append(base, *identity_tail),                       # same unitary
        with_append(base, *identity_tail, Gate("z", (3,))),      # different unitary
        with_append(base),                                        # identical
        with_append(base, Gate("s", (2,)), Gate("sdg", (2,))),    # `S S† = I`
    ]
    for candidate in pairs:
        dense = adjudicate_equivalence(base, candidate)
        clifford = adjudicate_clifford_equivalence(base, candidate)
        assert dense.verdict == clifford.verdict, (
            f"dense {dense.verdict} ({dense.distance}) vs clifford "
            f"{clifford.verdict} ({clifford.mismatched}/{clifford.compared})")


def test_a_clifford_relative_phase_is_rejected_by_the_action():
    """`S` versus `Z S = S†`: same state on `|0>`, different unitary."""
    reference = Circuit(ops=(Gate("s", (0,)),), num_qubits=1)
    candidate = Circuit(ops=(Gate("s", (0,)), Gate("z", (0,))), num_qubits=1)

    unitary_scope = adjudicate_clifford_equivalence(reference, candidate)
    assert unitary_scope.verdict == "INVALID"
    assert unitary_scope.mismatched == 1
    assert unitary_scope.mismatches[0]["basis"] == "X_0"

    # The prepared state is identical -- `S|0> = S†|0> = |0>` -- so the weaker scope
    # passes. That is the difference the report's `scope` field is for.
    assert adjudicate_clifford_equivalence(reference, candidate, on="state").verdict == "VALID"
    assert adjudicate_equivalence(reference, candidate).verdict == "INVALID"


def test_a_non_clifford_candidate_is_left_to_the_dense_tier():
    """`CCX` is not Clifford: the dense tier decides it, the stabilizer tier says so."""
    reference = Circuit(ops=(Gate("ccx", (0, 1, 2)),), num_qubits=3)
    candidate = with_append(reference, Gate("z", (2,)))
    assert adjudicate_equivalence(reference, candidate).verdict == "INVALID"

    report = adjudicate_clifford_equivalence(reference, candidate)
    assert report.verdict == "INVALID"
    assert report.compared == 0
    reason = next(check for check in report.checks
                  if check.name.startswith("reference")).measured["reason"]
    assert "not Clifford" in reason


# ------------------------------------------------------- above the dense bound
def test_a_hundred_qubit_pair_is_decided_and_the_dense_tier_refuses():
    reference = ghz_ladder(100)
    candidate = with_append(reference, *global_minus_one(100).ops)
    # The dense branch refuses to decide rather than calling a 100-qubit circuit wrong.
    dense = adjudicate_equivalence(reference, candidate)
    assert dense.verdict == "UNRESOLVED", dense.failed()
    assert dense.as_dict()["checks"][0]["passed"] is None

    report = adjudicate_clifford_equivalence(reference, candidate)
    assert report.verdict == "VALID"
    assert report.compared == 200
    assert report.mismatched == 0


def test_the_same_state_from_two_different_unitaries():
    """Chain and star both prepare GHZ, and are **not** the same unitary."""
    chain, star = ghz_ladder(100), ghz_star(100)
    unitary_scope = adjudicate_clifford_equivalence(chain, star)
    state_scope = adjudicate_clifford_equivalence(chain, star, on="state")

    assert unitary_scope.verdict == "INVALID"
    assert unitary_scope.mismatched > 0
    assert state_scope.verdict == "VALID"

    for circuit in (chain, star):
        prepared = StabilizerState(100)
        for op in circuit.ops:
            prepared.apply_gate(op.name, op.targets)
        # The generator *basis* differs between the two circuits, so the check is
        # group membership of the closed-form stabilizers, not list equality.
        for labels in ghz_generators(100):
            assert prepared.phase_of(labels) == 0
        assert prepared.probability_of_bitstring(tuple([False] * 100)) == 0.5
        assert prepared.probability_of_bitstring((True, *tuple([False] * 99))) == 0.0
    assert state_scope.as_dict()["decides"] == "the prepared state, up to one global phase"
    assert unitary_scope.as_dict()["decides"] == "the unitary, up to one global phase"


def test_a_single_changed_gate_at_a_hundred_qubits_is_named():
    reference = ghz_ladder(100)
    broken = Circuit(
        ops=tuple(Gate("cz", (0, 1)) if (op.name, op.targets) == ("cx", (0, 1)) else op
                  for op in reference.ops),
        num_qubits=100)
    report = adjudicate_clifford_equivalence(reference, broken)
    assert report.verdict == "INVALID"
    assert report.mismatched >= 1
    assert report.mismatches, "the report must carry the mismatch, not only its count"
    basis = report.mismatches[0]["basis"]
    assert basis in {"X_0", "Z_0", "X_1", "Z_1"}


def test_a_h_cz_h_ladder_equals_a_cx_ladder_at_forty_qubits():
    reference = ghz_ladder(40)
    # `H CZ H` on the target *is* `CX`, so this is the same unitary on 40 qubits.
    ops = [Gate("h", (0,))]
    for qubit in range(1, 40):
        ops.extend((Gate("h", (qubit,)), Gate("cz", (qubit - 1, qubit)), Gate("h", (qubit,))))
    candidate = Circuit(ops=tuple(ops), num_qubits=40)
    assert adjudicate_clifford_equivalence(reference, candidate).verdict == "VALID"
    # ... and it is genuinely the same state as well as the same map.
    assert adjudicate_clifford_equivalence(reference, candidate, on="state").verdict == "VALID"


def test_a_hundred_qubit_state_that_differs_is_rejected_by_both_scopes():
    reference = ghz_ladder(100)
    broken = with_append(reference, Gate("z", (99,)))
    assert adjudicate_clifford_equivalence(reference, broken).verdict == "INVALID"
    assert adjudicate_clifford_equivalence(reference, broken, on="state").verdict == "INVALID"


# ------------------------------------------------------------------ domain edges
def test_a_non_clifford_gate_is_refused_by_name():
    reference = ghz_ladder(3)
    candidate = with_append(reference, Gate("t", (1,)))
    report = adjudicate_clifford_equivalence(reference, candidate)
    assert report.verdict == "INVALID"
    reason = next(check for check in report.checks
                  if check.name.startswith("candidate")).measured["reason"]
    assert "not Clifford" in reason
    with pytest.raises(StabilizerUnsupportedError):
        clifford_pauli_images(candidate)


def test_a_measured_candidate_is_refused_as_a_mixture_not_as_a_channel():
    reference = ghz_ladder(2)
    candidate = Circuit(ops=(*reference.ops, Measure(0, 0)), num_qubits=2, num_bits=1)
    report = adjudicate_clifford_equivalence(reference, candidate)
    assert report.verdict == "INVALID"
    reason = next(check for check in report.checks
                  if check.name.startswith("candidate")).measured["reason"]
    assert "mixture" in reason


def test_the_scope_argument_is_validated():
    with pytest.raises(ValueError, match="'unitary' or 'state'"):
        adjudicate_clifford_equivalence(ghz_ladder(3), ghz_ladder(3), on="channel")


def test_a_qubit_count_mismatch_is_rejected():
    report = adjudicate_clifford_equivalence(ghz_ladder(3), ghz_ladder(4))
    assert report.verdict == "INVALID"
    assert report.qubits is None
    assert "both circuits act on the same number of qubits" in report.failed()


def test_claims_are_reported_apart_from_the_clifford_verdict():
    reference = ghz_ladder(5)
    candidate = with_append(reference, Gate("cx", (3, 4)), Gate("cx", (3, 4)))
    assert adjudicate_clifford_equivalence(reference, candidate).verdict == "VALID"
    assert adjudicate_clifford_equivalence(reference, candidate).claim_verdict == \
        "NOT_ASSERTED"

    report = adjudicate_clifford_equivalence(reference, candidate,
                                             claims={"two_qubit_gates": 3})
    # The images agree, so the circuit is right; it just saves nothing.
    assert report.verdict == "VALID"
    assert report.failed() == ()
    assert report.claim_verdict == "UNMET"
    assert report.promotable is False
    assert report.claim is not None
    assert report.claim.measured["unmet"]["two_qubit_gates"]["actual"] == 6


def test_the_report_is_reproducible_and_machine_readable():
    reference, candidate = ghz_ladder(6), with_append(ghz_ladder(6), Gate("z", (2,)))
    first = adjudicate_clifford_equivalence(reference, candidate)
    second = adjudicate_clifford_equivalence(reference, candidate)
    assert first.as_dict() == second.as_dict()

    payload = json.loads(first.to_json())
    assert payload["tier"] == "stabilizer"
    assert payload["scope"] == "unitary"
    assert payload["verdict"] == "INVALID"
    assert payload["reference_sha256"] == reference.sha256()
    assert payload["candidate_sha256"] == candidate.sha256()
    assert payload["compared"] == 12
