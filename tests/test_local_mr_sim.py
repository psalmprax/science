"""Pins for the local MR-capable simulator and the gated DQI arms.

``kickmix`` has no Z-basis measure-and-reset, so the gated DQI levers could only be
measured with oracle predicate registers. These tests pin the local engine that
makes the readouts real, and the recorded arm results.

Two kinds of pin:

1. **Primitive semantics**, matched against ``kickmix`` itself (the expectations were
   read off the extension, not assumed). If one drifts, every arm result is suspect.
2. **The MR contract**, which is the whole point: ``mr`` measures in Z, clears the
   qubit and leaves the tracked phase alone; ``r`` clears the qubit and randomizes
   the phase; ``hmr`` measures in X and is random on ``|1>``.

The arm results live in ``benchmarks/quantum_compilation/local_mr_arms.json`` and were
produced under the kickmix interpreter (CPython 3.13); the engine here is pure Python
and runs anywhere.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from apedp_dyn.quantum_compilation import local_mr_sim as L

ROOT = Path(__file__).resolve().parents[1]
ARMS = ROOT / "benchmarks" / "quantum_compilation" / "local_mr_arms.json"


class _conditional:
    """``with _conditional(builder, bit):`` for the tests that gate a block."""

    def __init__(self, builder, bit):
        self.builder = builder
        self.bit = bit

    def __enter__(self):
        self.builder.push_condition(self.bit)
        return self

    def __exit__(self, *_exc):
        self.builder.pop_condition()
        return False


def run(build, widths, init, shots=1, seed=7):
    builder = L.Builder()
    registers = [builder.alloc_qubits(w) for w in widths]
    build(builder, registers)
    circuit = builder.finish_circuit()
    sim = L.Simulator(batch_size=shots, seed=seed)
    sim.use_same_registers_as(circuit)
    for register, value in zip(registers, init, strict=False):
        bits = np.array([[(value >> i) & 1] for i in range(len(register))], dtype=bool)
        sim.write_across_shots(register, bits)
    sim.do(circuit)
    values = [int(sum(int(bit) << k for k, bit in enumerate(sim.read_across_shots(reg)[:, 0])))
              for reg in registers]
    return values, sim.read_phase_flipped_across_shots()


# -- primitives, pinned against kickmix -------------------------------------

def test_field_matches_kickmix_arithmetic():
    gf = L.Field(4)
    assert gf.mul(2, 9) == 1          # x * (x^3+1) reduces to 1
    assert gf.div(3, 5) == 14
    assert gf.div(9, 2) == 13
    assert gf.mul(7, 7) == 6
    assert gf.mul(3, 5) == 15
    assert gf.primitive_element == 2


def test_field_reduction_polynomials_are_the_kickmix_ones():
    # Recovered from kickmix.GF2Field: x^4+x+1, x^8+x^4+x^3+x^2+1, x^12+x^6+x^4+x+1.
    assert L._PRIMITIVE_POLYNOMIALS[4] == 0b10011
    assert L._PRIMITIVE_POLYNOMIALS[8] == 0b100011101
    assert L._PRIMITIVE_POLYNOMIALS[12] == 0b1000001010011
    # x^(degree) reduces as kickmix reports it.
    assert L.Field(8).mul(128, 2) == 29
    assert L.Field(12).mul(2048, 2) == 83


def test_rotations_match_kickmix_direction():
    assert run(lambda b, r: b.left_rotate(r[0], 1), [4], [0b1101])[0] == [0b1011]
    assert run(lambda b, r: b.right_rotate(r[0], 1), [4], [0b1101])[0] == [0b1110]
    assert run(lambda b, r: b.left_rotate(r[0], 2), [4], [0b1101])[0] == [0b0111]


def test_comparisons_and_arithmetic_match_kickmix():
    assert run(lambda b, r: b.flip_if_equal(r[0], 5, target=r[1][0]), [4, 1], [5, 0])[0][1] == 1
    assert run(lambda b, r: b.flip_if_equal(r[0], 5, target=r[1][0]), [4, 1], [6, 0])[0][1] == 0
    assert run(lambda b, r: b.flip_if_less_than(r[0], r[1], target=r[2][0]),
               [4, 4, 1], [3, 5, 0])[0][2] == 1
    assert run(lambda b, r: b.flip_if_less_than(r[0], r[1], target=r[2][0]),
               [4, 4, 1], [5, 3, 0])[0][2] == 0
    assert run(lambda b, r: b.iadd(3, target=r[0]), [4], [6])[0] == [9]
    assert run(lambda b, r: b.isub(1, target=r[0]), [4], [0])[0] == [15]
    assert run(lambda b, r: b.iadd(r[1], target=r[0]), [4, 4], [8, 5])[0][0] == 13


def test_and_gadget_broadcasts_the_scalar_control():
    values, _ = run(
        lambda b, r: b.init_and(r[0][0], L.array([r[1][0], r[2][0]]),
                                L.array([r[3][0], r[4][0]])),
        [1, 1, 1, 1, 1], [1, 1, 0, 0, 0])
    assert values[3:] == [1, 0]


def test_unary_iteration_match_is_the_address_comparison():
    def build(b, r):
        with b.unary_iteration(r[0]) as cursor:
            cursor.move_to(2)
            b.cx(cursor.match_qubit, r[1][0])

    assert run(build, [2, 1], [2, 0])[0][1] == 1
    assert run(build, [2, 1], [1, 0])[0][1] == 0


def test_match_address_is_snapshotted_not_shared():
    """Every emitted match keeps the address it was emitted at."""
    def build(b, r):
        with b.unary_iteration(r[0]) as cursor:
            cursor.move_to(0)
            b.cx(cursor.match_qubit, r[1][0])
            cursor.move_to(1)
            b.cx(cursor.match_qubit, r[2][0])

    values, _ = run(build, [2, 1, 1], [1, 0, 0])
    assert values[1:] == [0, 1]


def test_phase_by_product_is_the_masked_parity():
    for a, b, mask, expected in [(3, 5, 1, 1), (3, 5, 4, 1), (3, 5, 15, 0),
                                 (0, 9, 15, 0), (7, 7, 7, 0)]:
        builder = L.Builder()
        qa, qb, mb = builder.alloc_qubits(4), builder.alloc_qubits(4), builder.alloc_bits(4)
        builder.gf2_phase_by_product(qa, qb, mask=mb, field=L.Field(4))
        circuit = builder.finish_circuit()
        sim = L.Simulator(batch_size=1, seed=1)
        sim.use_same_registers_as(circuit)
        for register, value in ((qa, a), (qb, b), (mb, mask)):
            sim.write_across_shots(register,
                                   np.array([[(value >> i) & 1] for i in range(4)], dtype=bool))
        sim.do(circuit)
        assert int(sim.read_phase_flipped_across_shots()[0]) == expected, (a, b, mask)


# -- the MR contract --------------------------------------------------------

def test_mr_measures_z_clears_and_leaves_the_phase_alone():
    def build(b, r):
        bit = b.alloc_bits(1)
        b.x(r[0])
        b.mr(r[0], bit[0])

    for seed in range(8):
        values, phase = run(build, [1], [0], shots=16, seed=seed)
        assert values[0] == 0, "mr must leave the qubit in |0>"
        assert not phase.any(), "mr must not disturb the tracked phase"


def test_r_clears_but_randomizes_the_phase():
    """The kickmix defect the MR specification exists for."""

    def build(b, r):
        b.x(r[0])
        b.r(r[0])

    values, phase = run(build, [1], [0], shots=64, seed=3)
    assert values[0] == 0
    assert 0 < int(phase.sum()) < 64, "r is expected to randomize the tracked phase"


def test_no_reset_reports_no_phase_flip():
    def build(b, r):
        b.x(r[0])

    _, phase = run(build, [1], [0], shots=64, seed=3)
    assert not phase.any()


def test_mr_writes_the_measured_value_and_honours_a_condition():
    def build(b, r):
        out = b.alloc_bits(1)
        gate = b.alloc_bits(1)
        b.x(r[0])
        b.bit_store(gate[0], True, control=r[1][0])
        with _conditional(b, gate[0]):
            b.mr(r[0], out[0])

    # Gate closed: mr never runs, so the qubit stays |1>.
    values, _ = run(build, [1, 1], [0, 0])
    assert values[0] == 1
    # Gate open: mr measures 1 and clears.
    values, _ = run(build, [1, 1], [0, 1])
    assert values[0] == 0


def _hmr_outcomes(start_in_one: bool, shots: int = 64) -> int:
    builder = L.Builder()
    qubit = builder.alloc_qubits(1)[0]
    bit = builder.alloc_bits(1)[0]
    if start_in_one:
        builder.x(qubit)
    builder.hmr(qubit, bit)
    circuit = builder.finish_circuit()
    sim = L.Simulator(batch_size=shots, seed=5)
    sim.use_same_registers_as(circuit)
    sim.do(circuit)
    return int(np.asarray(sim.read_across_shots([bit])).sum())


def test_hmr_is_random_on_one_and_deterministic_on_zero():
    assert _hmr_outcomes(start_in_one=False) == 0
    ones = _hmr_outcomes(start_in_one=True)
    assert 0 < ones < 64, f"X-basis measurement of |1> should be a mixture, got {ones}/64"


# -- recorded arm results ---------------------------------------------------

def _arms(instance: str = "m15") -> dict:
    assert ARMS.is_file(), f"missing arms record: {ARMS}"
    record = json.loads(ARMS.read_text(encoding="utf-8"))
    prefix = f"{instance}_"
    return {key[len(prefix):]: value for key, value in record["results"].items()
            if key.startswith(prefix)}


@pytest.mark.parametrize("instance", ["m15", "m255", "m4095"])
@pytest.mark.parametrize(
    "arm", ["none", "swap", "sweep", "clear", "forney", "shift", "quotient", "all",
            "full"])
def test_every_mr_gated_arm_reproduces_the_reference(instance: str, arm: str):
    result = _arms(instance)[arm]
    assert result["verdict"] == "PASS", result
    assert result["syndromes_match_reference"] is True
    assert result["phase_matches_reference"] is True
    assert result["syndromes_restored"] is True
    assert result["junk_clean"] is True


def test_the_clear_arm_removes_the_reverse_eea_allocations():
    """Only the clear arm shrinks the circuit; the reads cost one shared ancilla.

    The Forney block reads ``sigma`` with ``mr`` and restores it in place, so it adds no
    qubit ids at all; the arms that read a control (``swap``, ``sweep``, ``shift``,
    ``quotient``) share one readout ancilla across every read in the circuit, so each of
    them allocates exactly one id more than the reference — not one per read. These are
    local id counts, not ``Q``: freed ids are never reused here, so they bound
    allocation rather than the register count.
    """
    results = _arms()
    reference = results["none"]["reference_qubits"]
    assert results["none"]["arm_qubits"] == reference
    assert results["clear"]["arm_qubits"] < reference
    assert results["all"]["arm_qubits"] < reference
    assert results["full"]["arm_qubits"] < reference
    assert results["forney"]["arm_qubits"] == reference
    for arm in ("swap", "sweep", "shift"):
        assert results[arm]["arm_qubits"] == reference + 1, arm


def test_reading_with_r_fails_the_same_arm():
    """The falsifier: same circuit, r instead of mr, and the phase no longer matches."""
    result = _arms()["swap_r-readout"]
    assert result["verdict"] == "FAIL"
    assert result["phase_matches_reference"] is False
    assert result["syndromes_match_reference"] is True, "value correctness is not what fails"


def test_the_under_gated_shift_arm_fails():
    """Drop ``¬flag ∧ (c0 ∨ c3)`` and the gate skips shifts the decoder needs."""
    result = _arms()["shift_under-gated"]
    assert result["verdict"] == "FAIL"
    assert result["syndromes_match_reference"] is False
