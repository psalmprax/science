"""The fault-tolerance referee and the search that uses it, on the Steane code."""
from __future__ import annotations

import itertools

import pytest

from apedp_dyn.vqc import fault_tolerant_prep as ft
from apedp_dyn.vqc.stabilizer import StabilizerBackend

CODE = ft.steane()


def test_code_tables_are_the_steane_code():
    assert len(CODE.x_space) == 8 and len(CODE.zero_z_group) == 16
    assert int(CODE.x_reduced_weight.max()) == 3 and int(CODE.z_reduced_weight.max()) == 1
    # a perfect code for X errors: weight one is corrected, weight two is not
    assert not CODE.decodes_wrongly[[1 << q for q in range(7)]].any()
    assert all(CODE.decodes_wrongly[(1 << a) | (1 << b)] for a, b in itertools.combinations(range(7), 2))
    assert not CODE.decodes_wrongly[list(CODE.x_space)].any()
    with pytest.raises(ValueError, match="commute"):
        ft.CssCode("broken", 3, (0b011,), (0b001,))


def test_the_shortest_encoder_has_eight_cnots_and_is_not_fault_tolerant():
    encoders = ft.shortest_encoders(CODE, limit=50)
    assert 10 <= len(encoders) <= 50 and all(len(e.cnots) == 8 and e.n_anc == 0 for e in encoders)
    assert len({(e.plus, e.cnots) for e in encoders}) == len(encoders)
    for encoder in encoders[:10]:
        certificate = ft.certify(encoder, CODE)
        assert certificate.prepares_state and not certificate.fault_tolerant
        assert certificate.c1 > 1.0 and certificate.worst_x >= 2 and certificate.dangerous


def test_the_prepared_state_is_checked_by_the_package_s_own_simulator():
    encoder = ft.shortest_encoders(CODE, limit=1)[0]
    circuit = encoder.to_circuit()
    run = StabilizerBackend().run(circuit, shots=1, seed=0)
    assert run is not None
    # measuring the data of the logical zero gives a code word of the X space, in every shot
    from apedp_dyn.vqc.circuit import Circuit, Measure
    measured = Circuit(ops=(*circuit.ops, *(Measure(q, q) for q in range(7))), num_qubits=7, num_bits=7)
    counts = StabilizerBackend().run(measured, shots=400, seed=1).counts
    seen = {sum(int(bit) << q for q, bit in enumerate(key)) if not isinstance(key, str)
            else sum(int(ch) << q for q, ch in enumerate(key)) for key in counts}
    assert seen <= set(CODE.x_space) and len(seen) == 8
    broken = ft.PrepCircuit(7, 0, encoder.plus, encoder.cnots[:-1])
    assert not ft.prepares_logical_zero(broken, CODE) and not ft.certify(broken, CODE).prepares_state


def test_fast_fault_propagation_equals_gate_by_gate_propagation():
    encoder = ft.shortest_encoders(CODE, limit=30)[-1]
    circuit = ft.PrepCircuit(7, 1, encoder.plus, (*encoder.cnots, (0, 7), (1, 7), (2, 7)))
    table = ft.single_faults(circuit)

    def run(x: int, z: int, start: int) -> tuple[int, int]:
        for c, t in circuit.cnots[start:]:
            x ^= ((x >> c) & 1) << t
            z ^= ((z >> t) & 1) << c
        return x, z

    expected = [run(0, 1 << q, 0) if q in circuit.plus else run(1 << q, 0, 0) for q in range(8)]
    for i, (c, t) in enumerate(circuit.cnots):
        expected += [run((xc << c) | (xt << t), (zc << c) | (zt << t), i + 1) for xc, zc, xt, zt in ft._TWO_QUBIT_PAULIS]
    for k, (x, z) in enumerate(expected):
        assert int(table["x"][k]) == x & 127 and int(table["z"][k]) == z & 127 and int(table["outcome"][k]) == x >> 7
    assert len(table["x"]) == len(expected) + 1 and int(table["outcome"][-1]) == 1   # the measurement flip
    assert table["weight"].sum() == pytest.approx(8 + 11 + 1)                           # 8 preparations, 11 CNOTs, 1 measurement


def test_one_weight_three_check_makes_the_encoder_fault_tolerant():
    found = ft.discover(CODE, encoder_limit=40)
    assert found, "no fault-tolerant circuit found"
    circuit, certificate = found[0]
    assert len(circuit.cnots) == 11 and circuit.n_anc == 1
    assert certificate.prepares_state and certificate.fault_tolerant and certificate.c1 == 0.0 and certificate.c2 > 0.0
    assert certificate.worst_x <= 1 and certificate.worst_z <= 1
    assert [c.c2 for _, c in found] == sorted(c.c2 for _, c in found)   # ranked by the second-order coefficient
    assert found[-1][1].c2 > 1.1 * certificate.c2                        # gate-optimal circuits are not equally good
    # a check that is not a stabiliser of the state breaks the preparation, and the referee says so
    wrong = ft.PrepCircuit(7, 1, circuit.plus, (*circuit.cnots[:8], (0, 7), (1, 7)))
    assert not ft.certify(wrong, CODE).fault_tolerant or not ft.certify(wrong, CODE).prepares_state


def test_coupling_limits_are_respected():
    line = [(q, q + 1) for q in range(6)]
    encoders = ft.shortest_encoders(CODE, line, limit=5, reduce_symmetry=False)
    assert encoders and all(abs(c - t) == 1 for e in encoders for c, t in e.cnots) and len(encoders[0].cnots) > 8
    # an ancilla that reaches only two data qubits cannot measure any check of this code
    assert ft.verified(encoders[0], CODE, max_ancillas=1, allowed_support=[0b0000011]) == []


def test_exact_coefficients_agree_with_an_outside_simulator():
    stim = pytest.importorskip("stim")
    del stim
    circuit, certificate = ft.discover(CODE, encoder_limit=5)[0]
    p = 0.02
    sampled = ft.stim_logical_error(circuit, CODE, p, 400_000, seed=3)
    predicted = certificate.logical_error_rate(p)
    assert abs(sampled["rate"] - predicted) < 0.25 * predicted + 5 * sampled["standard_error"]
    bare = ft.PrepCircuit(7, 0, circuit.plus, circuit.cnots[:8])
    sampled_bare = ft.stim_logical_error(bare, CODE, p, 200_000, seed=4)
    assert sampled_bare["acceptance"] == 1.0 and sampled_bare["rate"] > 5 * sampled["rate"]
    assert abs(sampled_bare["rate"] - ft.certify(bare, CODE).c1 * p) < 0.25 * sampled_bare["rate"]


def test_reordered_and_relabelled_encoders_are_listed_once():
    assert ft.normal_form([(0, 1), (2, 3), (1, 2)]) == ft.normal_form([(2, 3), (0, 1), (1, 2)])
    assert ft.normal_form([(0, 1), (1, 2), (2, 3)]) != ft.normal_form([(0, 1), (2, 3), (1, 2)])
    reduced = ft.shortest_encoders(CODE, limit=400)
    plain = ft.shortest_encoders(CODE, limit=400, reduce_symmetry=False)
    assert len(reduced) < len(plain) <= 400
    # a symmetry of the code maps a circuit to one with the same certificate, which is why one of them is enough
    order = next(o for o in ft.symmetries(CODE, [(a, b) for a in range(7) for b in range(7) if a != b]) if o != tuple(range(7)))
    circuit = plain[0]
    moved = ft.PrepCircuit(7, 0, tuple(sorted(order[q] for q in circuit.plus)), tuple((order[c], order[t]) for c, t in circuit.cnots))
    first, second = ft.certify(circuit, CODE), ft.certify(moved, CODE)
    assert second.prepares_state and (first.c1, first.c2, first.worst_x) == pytest.approx((second.c1, second.c2, second.worst_x))
