"""The third-party referee: a second opinion that can abstain, and is not ours.

The record's own limitations said the comparison was adjudicated by this package's
referee alone. These pins are the second opinion:

* **it decides the same question** -- the same signed stabilizers every engine was
  handed, evaluated on the state the circuit prepares, so a `VALID` here means the
  circuit prepares the requested state rather than something a looser check accepted;
* **it fails when the circuit is wrong** -- a dropped gate, a circuit built for a
  different target, an inverted phase and a mis-wired qubit must all be `INVALID`, so a
  verdict cannot be a constant;
* **it abstains rather than guesses** -- a non-Clifford gate, a measurement, a width
  mismatch and an absent Stim are `REFUSED`/`UNAVAILABLE` with a reason, and none of
  them counts as a verdict about the circuit.
"""
from __future__ import annotations

import pytest

from apedp_dyn.vqc.circuit import Circuit, Gate, Measure
from apedp_dyn.vqc.corpus import CORPUS
from apedp_dyn.vqc.foreign_referee import (
    FOREIGN_REFEREES,
    STIM_GATES,
    ForeignVerdict,
    stim_verdict,
)
from apedp_dyn.vqc.synthesis import (
    I,
    TargetSpecification,
    X,
    Z,
    build,
    ghz_target,
    ring_graph_target,
)

pytest.importorskip("stim")


def _graph_state_target(num_qubits: int, edges: tuple[tuple[int, int], ...]) -> TargetSpecification:
    """A graph state specification written out directly, without a generator."""
    stabilizers = []
    for vertex in range(num_qubits):
        labels = [I] * num_qubits
        labels[vertex] = X
        for parent, child in edges:
            if parent == vertex:
                labels[child] = Z
            elif child == vertex:
                labels[parent] = Z
        stabilizers.append((tuple(labels), 0))
    return TargetSpecification(name="path_graph", num_qubits=num_qubits,
                               stabilizers=tuple(stabilizers),
                               decides="the path graph state, up to one global phase")


def _path_edges(num_qubits: int) -> tuple[tuple[int, int], ...]:
    return tuple((index, index + 1) for index in range(num_qubits - 1))


def _path_circuit(num_qubits: int) -> Circuit:
    ops = [Gate("h", (qubit,)) for qubit in range(num_qubits)]
    ops += [Gate("cz", edge) for edge in _path_edges(num_qubits)]
    return Circuit(tuple(ops), num_qubits)


def test_the_third_party_referee_decides_the_same_specification():
    target = ghz_target(6)
    verdict = stim_verdict(build("chain", target).circuit, target)
    assert isinstance(verdict, ForeignVerdict)
    assert verdict.verdict == "VALID"
    assert verdict.referee == "stim"
    assert verdict.version
    assert verdict.qubits == 6
    assert verdict.generators == verdict.satisfied == 6
    assert verdict.failing == ()
    assert verdict.comparative is True
    assert verdict.seconds >= 0.0


def test_a_correct_state_with_a_negative_phase_is_accepted_and_the_sign_checked():
    """The referee reads the sign, not just the Pauli string."""
    zero_one = TargetSpecification(
        name="zero_one", num_qubits=2,
        stabilizers=(((Z, I), 0), ((I, Z), 2)),          # |01>
        decides="the |01> state, up to one global phase")
    zero_zero = TargetSpecification(
        name="zero_zero", num_qubits=2,
        stabilizers=(((Z, I), 0), ((I, Z), 0)),          # |00>
        decides="the |00> state, up to one global phase")
    flipped = Circuit((Gate("x", (1,)),), 2)
    assert stim_verdict(flipped, zero_one).verdict == "VALID"
    assert stim_verdict(flipped, zero_zero).verdict == "INVALID"
    assert stim_verdict(Circuit((), 2), zero_zero).verdict == "VALID"
    inverted = stim_verdict(Circuit((), 2), zero_one)
    assert inverted.verdict == "INVALID"
    assert inverted.satisfied == 1
    assert inverted.failing[0]["generator"] == 1
    assert inverted.failing[0]["expected_expectation"] == -1
    assert inverted.failing[0]["observed_expectation"] == 1


def test_a_missing_gate_is_invalid_and_named():
    target = _graph_state_target(5, _path_edges(5))
    circuit = _path_circuit(5)
    verdict = stim_verdict(Circuit(circuit.ops[:-1], 5), target)
    assert verdict.verdict == "INVALID"
    # The last entangler touches two vertices, so dropping it breaks two generators.
    assert verdict.satisfied < verdict.generators
    assert verdict.failing
    assert "generators do not stabilize" in verdict.reason


def test_a_circuit_for_a_different_target_is_invalid():
    """A correct circuit for the wrong state must not slip through."""
    other = ring_graph_target(6)
    verdict = stim_verdict(_path_circuit(6), other)
    assert verdict.verdict == "INVALID"


def test_a_mis_wired_qubit_is_invalid():
    """The bridge carries qubit indices, so a wrong wire cannot wash out."""
    target = _graph_state_target(4, _path_edges(4))
    correct = _path_circuit(4)
    wrong = Circuit(tuple([Gate("h", (qubit,)) for qubit in range(4)]
                          + [Gate("cz", (0, 2)), Gate("cz", (1, 3))]), 4)
    assert stim_verdict(correct, target).verdict == "VALID"
    verdict = stim_verdict(wrong, target)
    assert verdict.verdict == "INVALID"
    assert verdict.satisfied < verdict.generators


def test_a_non_clifford_circuit_is_refused_with_a_reason():
    target = ghz_target(3)
    ops = [Gate("t", (0,)), *build("chain", target).circuit.ops]
    verdict = stim_verdict(Circuit(tuple(ops), 3), target)
    assert verdict.verdict == "REFUSED"
    assert "not a Clifford gate" in verdict.reason
    assert "t" in verdict.reason
    assert verdict.comparative is False


def test_a_measured_circuit_is_refused():
    """A preparation circuit only: a measurement is outside a tableau referee's remit."""
    target = ghz_target(3)
    ops = (*build("chain", target).circuit.ops, Measure(qubit=0, bit=0))
    verdict = stim_verdict(Circuit(ops, 3, num_bits=1), target)
    assert verdict.verdict == "REFUSED"
    assert "measure" in verdict.reason
    assert verdict.comparative is False


def test_a_width_mismatch_is_refused_rather_than_compared():
    verdict = stim_verdict(build("chain", ghz_target(4)).circuit, ghz_target(5))
    assert verdict.verdict == "REFUSED"
    assert "qubits" in verdict.reason
    assert verdict.comparative is False


def test_an_absent_stim_is_an_abstention_not_a_loss(monkeypatch):
    from apedp_dyn.vqc import foreign_referee

    def unavailable():
        from apedp_dyn.vqc.foreign_referee import ForeignUnavailable
        raise ForeignUnavailable("stim is not importable (ImportError: test)")

    monkeypatch.setattr(foreign_referee, "_stim_module", unavailable)
    verdict = foreign_referee.stim_verdict(build("chain", ghz_target(4)).circuit,
                                           ghz_target(4))
    assert verdict.verdict == "UNAVAILABLE"
    assert verdict.version is None
    assert "not importable" in verdict.reason
    assert verdict.comparative is False
    assert verdict.as_dict()["decides_the_circuit"] is False


def test_the_referee_scales_to_a_routed_device_circuit():
    """A 127-wire routed circuit is decided, and by the padded specification."""
    from apedp_dyn.vqc.placement import line, place

    target = CORPUS[0].target
    circuit = build("graph_layers", target).circuit
    routed = place(circuit, line(127))
    padded = TargetSpecification(
        name="padded", num_qubits=127,
        stabilizers=tuple(
            (tuple(labels) + (I,) * (127 - len(labels)), phase)
            for labels, phase in target.stabilizers)
        + tuple((tuple(Z if index == qubit else I for index in range(127)), 0)
                for qubit in range(target.num_qubits, 127)))
    verdict = stim_verdict(routed.circuit, padded)
    assert verdict.verdict == "VALID"
    assert verdict.generators == 127


def test_the_referee_table_says_what_it_decides_and_what_it_does_not():
    entry = FOREIGN_REFEREES["stim"]
    assert entry["package"] == "stim"
    assert entry.get("does_not_decide")
    assert set(STIM_GATES) <= {"h", "x", "y", "z", "s", "sdg", "cx", "cz", "swap",
                               "ccx", "cswap"}
