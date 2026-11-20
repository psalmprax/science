"""OpenQASM 2 interchange: what round-trips, and what is refused on purpose.

The interesting property is not "we can print QASM". It is that a circuit which
survives the round trip is the *same circuit* -- same operations, same metadata,
same ancilla contract, same hash -- and that everything else raises instead of
being silently approximated.
"""
from __future__ import annotations

import math

import pytest

from apedp_dyn.vqc import (
    Circuit,
    Gate,
    Measure,
    Noise,
    QasmUnsupportedError,
    Reset,
    StatevectorBackend,
    from_qasm,
    to_qasm,
)

SEED = 20261030


def bell() -> Circuit:
    return Circuit(ops=(Gate("h", (0,)), Gate("cx", (0, 1)),
                        Measure(0, 0), Measure(1, 1)),
                   num_qubits=2, num_bits=2, metadata={"generator": "tests.qasm.bell"})


def ghz() -> Circuit:
    return Circuit(ops=(Gate("h", (0,)), Gate("cx", (0, 1)), Gate("cx", (1, 2)),
                        Measure(0, 0), Measure(1, 1), Measure(2, 2)),
                   num_qubits=3, num_bits=3)


def conditioned() -> Circuit:
    """A condition that tests every declared bit, which is what QASM 2 can carry."""
    return Circuit(ops=(Gate("h", (0,)), Measure(0, 0), Measure(1, 1),
                        Gate("x", (1,), condition=((0, True), (1, False)))),
                   num_qubits=2, num_bits=2, ancillas=(1,),
                   metadata={"code": "toffoli-half"})


@pytest.mark.parametrize("build", [bell, ghz, conditioned])
def test_circuits_round_trip_through_qasm_unchanged(build):
    circuit = build()
    text = to_qasm(circuit)
    assert text.startswith("OPENQASM 2.0;")
    assert 'include "qelib1.inc";' in text

    rebuilt = from_qasm(text)
    assert rebuilt == circuit
    assert rebuilt.sha256() == circuit.sha256()
    assert rebuilt.ancillas == circuit.ancillas
    assert rebuilt.metadata == circuit.metadata

    first = StatevectorBackend().run(circuit, shots=512, seed=SEED)
    second = StatevectorBackend().run(rebuilt, shots=512, seed=SEED)
    assert first.counts == second.counts
    assert first.samples == second.samples


def test_the_text_is_a_faithful_qasm_program_not_a_dump_of_the_ir():
    text = to_qasm(bell())
    assert "qreg q[2];" in text
    assert "creg c[2];" in text
    assert "h q[0];" in text
    assert "cx q[0], q[1];" in text
    assert "measure q[0] -> c[0];" in text
    # No IR-specific syntax leaks into the program body.
    assert "condition" not in text
    assert "schema_version" not in text.split("qreg", 1)[1]


def test_a_hand_written_qasm_program_imports_with_its_parameters_and_resets():
    source = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[1];
// a comment
ry(pi/4) q[0];
h q[1];
cx q[1], q[0];
measure q[0] -> c[0];
reset q[1];
if(c==1) x q[1];
"""
    circuit = from_qasm(source)
    assert circuit.num_qubits == 2
    assert circuit.num_bits == 1
    ops = circuit.ops
    assert ops[0] == Gate("ry", (0,), (math.pi / 4.0,))
    assert ops[1] == Gate("h", (1,))
    assert ops[2] == Gate("cx", (1, 0))
    assert ops[3] == Measure(qubit=0, bit=0)
    assert ops[4] == Reset(qubit=1)
    # `if(c==1)` becomes the full condition over the one declared bit.
    assert ops[5] == Gate("x", (1,), condition=((0, True),))


def test_parameter_parser_evaluates_supported_arithmetic_without_code_execution():
    source = """OPENQASM 2.0;
qreg q[1];
rx(-(pi / 4) + 1.5 * 2) q[0];
"""
    circuit = from_qasm(source)
    assert circuit.ops[0] == Gate("rx", (0,), (-math.pi / 4 + 3.0,))


def test_parameter_parser_rejects_non_finite_results():
    source = "OPENQASM 2.0;\nqreg q[1];\nrx(1e999) q[0];\n"
    with pytest.raises(QasmUnsupportedError, match="non-finite"):
        from_qasm(source)


def test_export_refuses_a_condition_qasm2_cannot_express():
    teleport = Circuit(
        ops=(Gate("ry", (0,), (math.pi / 4.0,)), Gate("h", (1,)), Gate("cx", (2, 1)),
             Gate("cx", (0, 1)), Gate("h", (0,)), Measure(0, 0), Measure(1, 1),
             Gate("x", (2,), condition=((1, True),))),
        num_qubits=3, num_bits=2)
    with pytest.raises(QasmUnsupportedError, match="QASM 3"):
        to_qasm(teleport)


def test_export_refuses_noise_because_openqasm2_has_no_noise_construct():
    noisy = Circuit(ops=(Gate("h", (0,)), Noise(0, "bit_flip", 0.1)), num_qubits=1)
    with pytest.raises(QasmUnsupportedError, match="noise operation"):
        to_qasm(noisy)


@pytest.mark.parametrize("source,reason", [
    ("h q[0];\n", "missing `OPENQASM 2.0;` header"),
    ("OPENQASM 2.0;\nqreg q[1];\ncu1(0.3) q[0], q[1];\n", "not in the supported QASM 2 subset"),
    ("OPENQASM 2.0;\nqreg q[1];\nrx q[0];\n", "takes 1 parameter"),
    ("OPENQASM 2.0;\ninclude \"qelib1.inc\";\n", "no quantum register declared"),
    ("OPENQASM 2.0;\nqreg q[1];\ngate mine a { h a; }\n", "user-defined"),
    ("OPENQASM 2.0;\nqreg q[1];\ncreg c[1];\nif(c==5) x q[0];\n", "does not fit"),
    ("OPENQASM 2.0;\nqreg q[1];\nrx(sqrt(2)) q[0];\n", "unsupported parameter"),
])
def test_import_refuses_what_it_would_have_to_guess(source, reason):
    with pytest.raises(QasmUnsupportedError) as error:
        from_qasm(source)
    assert reason in str(error.value)


def test_the_exported_program_re_imports_to_the_same_hash_after_a_reset_circuit():
    circuit = Circuit(ops=(Gate("x", (0,)), Reset(0), Gate("h", (1,)),
                           Measure(0, 0), Measure(1, 1)),
                      num_qubits=2, num_bits=2, ancillas=(0,))
    rebuilt = from_qasm(to_qasm(circuit))
    assert rebuilt == circuit
    assert rebuilt.sha256() == circuit.sha256()
