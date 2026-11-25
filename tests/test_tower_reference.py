"""The 42-product multiplication in GF(2^12) and its circuit: every claim, checked exactly.

The bound is the literature's, not a discovery (``apedp_dyn/quantum/tower_reference.py``). What
these tests fix is that the repository's copy is right: the products reproduce the field's tensor,
the circuit computes ``z + x*y`` on the three registers alone, and the checks themselves refuse a
broken circuit.
"""
from __future__ import annotations

import json
import pathlib
import random
import subprocess
import sys

import numpy as np
import pytest

from apedp_dyn.quantum import tower_reference as tower
from apedp_dyn.quantum.field_tensor_search import (
    DEGREE,
    MODULUS_INT,
    build_54_reference,
    build_karatsuba_reference,
    build_multiplication_tensor,
    gf_mul,
    solve_gf2_bilinear_weights,
    verify_tensor_decomposition,
)
from apedp_dyn.quantum.gf2_rank_search import solve_weights

ROOT = pathlib.Path(__file__).resolve().parents[1]
RECORD = ROOT / "benchmarks" / "gf2_multiplication" / "gf2_mul_tower_42.json"


@pytest.fixture(scope="module")
def tensor() -> np.ndarray:
    return build_multiplication_tensor()


@pytest.fixture(scope="module")
def decomposition():
    return tower.build_tower_42_reference()


@pytest.fixture(scope="module")
def circuit():
    return tower.build_tower_42_circuit()


def test_the_subfield_is_a_field_of_eight_elements():
    theta = tower.subfield_generator()
    elements = {0} | {tower.gf_pow(theta, k, MODULUS_INT) for k in range(7)}
    assert len(elements) == 8
    assert all(a ^ b in elements and gf_mul(a, b, MODULUS_INT) in elements for a in elements for b in elements)


def test_the_tower_basis_is_a_basis(tensor):
    basis = tower.tower_basis(tower.subfield_generator())
    assert tower._solve(basis, np.eye(DEGREE, dtype=np.uint8)).shape == (DEGREE, DEGREE)    # raises if singular


def test_forty_two_products_reproduce_every_entry_of_the_tensor(tensor, decomposition):
    assert decomposition.rank == tower.RANK == 42 and decomposition.U.shape == (42, DEGREE)
    assert np.array_equal(decomposition.U, decomposition.V)                                 # symmetric
    assert verify_tensor_decomposition(tensor, decomposition) == (True, [])


def test_the_weights_are_what_the_repository_s_other_solver_finds(tensor, decomposition):
    solved, weights = solve_weights(tensor, decomposition.U, decomposition.V)
    assert solved and np.array_equal(weights, decomposition.W)


def test_no_product_can_simply_be_dropped(tensor, decomposition):
    for index in range(decomposition.rank):
        fewer = np.delete(decomposition.U, index, axis=0)
        assert solve_gf2_bilinear_weights(fewer, fewer, tensor)[0] is False


def test_every_non_zero_factor_keeps_at_least_twelve_products_alive(decomposition):
    """Necessary in any correct multiplication: ``y -> x*y`` is onto, so twelve products must survive."""
    forms = [sum(int(bit) << k for k, bit in enumerate(row)) for row in decomposition.U]
    assert min(sum(bin(form & x).count("1") & 1 for form in forms) for x in range(1, 1 << DEGREE)) == 12


def test_the_circuit_has_42_toffolis_and_uses_only_the_three_registers(tensor, circuit):
    gates, account = circuit
    assert account == {"toffolis": 42, "cnots": 474, "qubits": 36, "registers_restored": True}
    assert {qubit for _, control, second, target in gates for qubit in (control, second, target) if qubit >= 0} <= set(range(36))
    assert np.array_equal(tower.replay(gates)["tensor"], tensor)


def test_the_circuit_run_bit_by_bit_multiplies_and_gives_x_and_y_back(circuit):
    gates, _ = circuit
    rng = random.Random(7)
    for _ in range(3000):
        x, y, z = (rng.randrange(1 << DEGREE) for _ in range(3))
        assert tower.run(gates, x, y, z) == (x, y, z ^ gf_mul(x, y, MODULUS_INT))


def test_the_same_builder_is_right_on_the_repository_s_older_decompositions(tensor):
    for build, toffolis in ((build_54_reference, 54), (build_karatsuba_reference, 63)):
        account = tower.replay(tower.build_netlist(build()))
        assert account["toffolis"] == toffolis and account["registers_restored"]
        assert np.array_equal(account["tensor"], tensor)


def test_the_check_notices_a_broken_circuit(tensor, circuit):
    gates, _ = circuit
    toffoli = next(index for index, gate in enumerate(gates) if gate[0] == tower.TOFFOLI)
    without_one_product = gates[:toffoli] + gates[toffoli + 1:]
    assert not np.array_equal(tower.replay(without_one_product)["tensor"], tensor)
    cnot = next(index for index, gate in enumerate(gates) if gate[0] == tower.CNOT)
    without_one_cnot = gates[:cnot] + gates[cnot + 1:]
    broken = tower.replay(without_one_cnot)
    assert not (np.array_equal(broken["tensor"], tensor) and broken["registers_restored"])


def test_the_check_refuses_gates_it_cannot_account_for():
    with pytest.raises(ValueError, match="leaves its register"):
        tower.replay([(tower.CNOT, 0, -1, 12)])
    with pytest.raises(ValueError, match="not from x and y into z"):
        tower.replay([(tower.TOFFOLI, 0, 1, 24)])
    with pytest.raises(ValueError, match="unknown gate"):
        tower.replay([("SWAP", 0, -1, 1)])


def test_the_record_is_what_the_code_builds_now():
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "record_gf2_mul_tower_42.py"), "--check"],
                          capture_output=True, text=True, cwd=ROOT, check=False)
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_record_says_what_it_is_and_what_it_is_not():
    record = json.loads(RECORD.read_text())
    assert record["development"] is True and "not a discovery" in record["what"]
    assert record["checks"]["tensor_entries_that_differ_of_1728"] == 0
    assert record["checks"]["circuit_run_bit_by_bit"]["wrong"] == 0
    assert record["circuit"]["qubits_beyond_the_three_registers"] == 0 and record["circuit"]["cnots_minimised"] is False
    assert record["beside_it"]["published_circuit"]["toffolis"] == 51
    assert "NOT a measurement" in record["projection_onto_the_decoder_benchmark"]["what"]
    assert "not known" in record["least_possible"]
