"""Pins for the amplitude tier and the licence boundary it measures.

`docs/VQC_AMPLITUDE_TIER.md` licenses every gate in the DQI arm on the inputs
being computational-basis states. These tests pin the tier's algebra, the
boundary itself (identical on the basis, a complete dephasing off it), the
`HMR`/`MR` substitution, and the recorded JSON. A future edit that widens the
claim past the basis -- or that quietly changes the gadget -- fails here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from apedp_dyn.quantum_compilation.statevector_sim import (
    StatevectorSimulator,
    dimension_report,
    fidelity,
    global_phase_free_equality,
)

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "benchmarks" / "quantum_compilation" / "vqc_licence_boundary.json"
CONTROL, ANCILLA, LEFT, RIGHT = 0, 1, 2, 3


def index_of(value: int, num_qubits: int = 4) -> int:
    return int(value) & ((1 << num_qubits) - 1)


def reference_ops():
    return [("gate", "cswap", (CONTROL, LEFT, RIGHT))]


def gated_ops():
    return [
        ("gate", "cx", (CONTROL, ANCILLA)),
        ("measure", ANCILLA, 0),
        ("conditioned", 0, [("gate", "swap", (LEFT, RIGHT))]),
    ]


def run(initial: int, ops):
    sim = StatevectorSimulator(4)
    sim.set_basis_state(initial)
    return sim.branches(ops)


# ---------------------------------------------------------------- the algebra
def test_x_flips_and_cx_copies():
    sim = StatevectorSimulator(2)
    sim.x(0)
    assert np.argmax(np.abs(sim.state)) == index_of(0b01, 2)
    sim.cx(0, 1)
    assert np.argmax(np.abs(sim.state)) == index_of(0b11, 2)


def test_ccx_needs_both_controls():
    sim = StatevectorSimulator(3)
    sim.x(0)
    sim.ccx(0, 1, 2)
    assert np.argmax(np.abs(sim.state)) == index_of(0b001, 3)
    sim.x(1)
    sim.ccx(0, 1, 2)
    assert np.argmax(np.abs(sim.state)) == index_of(0b111, 3)


def test_cswap_swaps_only_under_the_control():
    for control in (0, 1):
        sim = StatevectorSimulator(3)
        sim.set_basis_state(index_of((1 << 1), 3))  # left = |1>, right = |0>
        if control:
            sim.x(0)
        sim.cswap(0, 1, 2)
        expected = (0b100 if control else 0b010) | (control)
        assert np.argmax(np.abs(sim.state)) == expected


def test_hadamard_creates_an_even_superposition():
    sim = StatevectorSimulator(1)
    sim.h(0)
    assert np.allclose(np.abs(sim.state), [1 / np.sqrt(2), 1 / np.sqrt(2)])


def test_measure_and_reset_reports_then_clears():
    sim = StatevectorSimulator(1)
    sim.x(0)
    assert sim.measure_and_reset(0) is True
    assert np.argmax(np.abs(sim.state)) == 0  # reset to |0>
    assert sim.measure_and_reset(0) is False


def test_cz_is_a_phase_on_the_all_ones_component():
    sim = StatevectorSimulator(2)
    sim.x(0)
    sim.x(1)
    sim.cz(0, 1)
    assert np.isclose(sim.state[3], -1.0)


def test_branches_weights_sum_to_one():
    sim = StatevectorSimulator(2)
    sim.h(0)
    sim.cx(0, 1)
    branches = sim.branches([("measure", 0, 0)])
    assert len(branches) == 2
    assert np.isclose(sum(branch.probability for branch in branches), 1.0)


# ------------------------------------------------------------ the licence
def test_the_gated_gadget_is_the_controlled_one_on_every_basis_input():
    for control in (0, 1):
        for left in (0, 1):
            for right in (0, 1):
                initial = index_of(control | (left << LEFT) | (right << RIGHT))
                reference = run(initial, reference_ops())
                gated = run(initial, gated_ops())
                assert len(reference) == 1 and len(gated) == 1
                assert global_phase_free_equality(reference[0].state, gated[0].state)
                assert gated[0].bits[0] == bool(control)
                # The ancilla is measured and reset, so it holds no amplitude at 1.
                index = np.arange(gated[0].state.size)
                one = ((index >> ANCILLA) & 1).astype(bool)
                assert np.isclose(
                    np.sum(np.abs(gated[0].state[one]) ** 2), 0.0)


def test_the_gated_gadget_dephases_a_superposed_control():
    sim = StatevectorSimulator(4)
    sim.set_basis_state(1 << LEFT)
    sim.h(CONTROL)
    reference = sim.branches(reference_ops())[0]

    gated_sim = StatevectorSimulator(4)
    gated_sim.set_basis_state(1 << LEFT)
    gated_sim.h(CONTROL)
    gated = gated_sim.branches(gated_ops())

    assert len(gated) == 2
    assert np.isclose(fidelity(reference.state, gated), 0.5)
    assert not global_phase_free_equality(reference.state, gated[0].state)


def test_hmr_is_not_mr_and_h_then_hmr_is():
    hmr = [("gate", "h", (CONTROL,)), ("measure", CONTROL, 0)]
    mr = [("measure", CONTROL, 0)]

    sim = StatevectorSimulator(4)
    sim.set_basis_state(0)
    assert len(sim.branches(hmr)) == 2  # randomises a basis-state qubit

    sim = StatevectorSimulator(4)
    sim.set_basis_state(0)
    substituted = sim.branches([("gate", "h", (CONTROL,)), *hmr])
    genuinely = sim.branches(mr)
    assert len(substituted) == 1 and len(genuinely) == 1
    assert substituted[0].bits == genuinely[0].bits
    assert global_phase_free_equality(substituted[0].state, genuinely[0].state)


def test_the_tier_refuses_a_decoder_sized_register():
    with pytest.raises(ValueError):
        StatevectorSimulator(27)


def test_dimension_report_does_not_pretend_to_have_the_amplitudes():
    report = dimension_report(1913)
    assert report["num_qubits"] == 1913
    assert isinstance(report["amplitudes"], str)
    assert 575 < report["log10_amplitudes"] < 576
    assert dimension_report(10)["amplitudes"] == 1024.0


# ------------------------------------------------------------- the record
def test_the_record_pins_the_boundary():
    record = json.loads(RECORD.read_text())
    assert record["basis_states"]["all_equal"] is True
    assert len(record["basis_states"]["rows"]) == 8
    superposition = record["superposition"]
    assert superposition["gated_branches"] == 2
    assert np.isclose(
        superposition["fidelity_of_gated_mixture_against_reference"], 0.5)
    assert np.isclose(superposition["reference_control_offdiagonal"], 0.5)
    assert np.isclose(superposition["gated_control_offdiagonal"], 0.0)
    assert superposition["states_equal_up_to_phase"] is False
    assert record["readout_identity"]["hmr_on_zero"]["outcome_is_random"] is True
    assert record["readout_identity"]["hmr_on_zero"]["matches_mr"] is False
    assert record["readout_identity"]["h_then_hmr"]["matches_mr_on_basis"] is True
    assert record["dimension"]["arm"]["num_qubits"] == 1913
    assert record["failed_checks"] == []
    assert record["verdict"].startswith("the licence is exactly the basis")
