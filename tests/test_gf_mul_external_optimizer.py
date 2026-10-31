"""Pins for the external-optimizer comparison of the 54-AND multiply.

`scripts/compare_gf_mul_external_optimizer.py` records the multiply's forward AND netlist
from the emitter itself and hands it to PyZX 0.10.7 as the same function in Clifford+T.
These tests pin what that measured: 54 ANDs and 1,004 CNOTs on our side, 378 T on the
optimizer's, no T reduction at all from its Clifford pass, its T-reducing pipeline not
returning within the budget, and the decoder not expressible in its gate set.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORD = (ROOT / "benchmarks" / "quantum_compilation"
          / "gf_mul_external_optimizer.json")
SCRIPT = ROOT / "scripts" / "compare_gf_mul_external_optimizer.py"


def _record() -> dict:
    return json.loads(RECORD.read_text())


def test_our_netlist_is_the_emitters_own():
    record = _record()
    assert record["ours"]["ands"] == 54
    assert record["ours"]["cnots"] == 1004
    assert record["ours"]["measured_clears"] == 54
    # The clears are the point: they cost nothing because they are measurements.
    assert record["ours"]["clear_cost_toffoli"] == 0
    assert record["ours"]["controlled_call_toffoli_like"] == 66


def test_the_toffoli_decomposition_was_verified_before_it_was_used():
    check = _record()["decomposition_check"]
    assert check["basis_states_checked"] == 8
    assert check["exact"] is True
    assert check["max_abs_error"] < 1e-12


def test_the_optimizer_does_not_reduce_the_t_count():
    record = _record()
    pyzx = record["pyzx"]
    assert pyzx["pyzx_version"].startswith("0.10")
    assert pyzx["before"]["t"] == 378  # 54 Toffolis x 7 T
    assert pyzx["after"]["t"] == 378
    assert pyzx["t_reduction"] == 0
    # It does compress the Clifford skeleton, which is not the metric.
    assert pyzx["after"]["cnot"] < pyzx["before"]["cnot"]
    assert pyzx["after"]["cnot"] == 824


def test_the_t_reducing_pipeline_is_reported_honestly():
    record = _record()
    full = record["pyzx_full_optimize"]
    assert full["completed"] is False
    assert full["timeout_s"] > 0
    assert "did not return" in full["note"]
    # The verdict must not claim a comparison it did not make.
    assert "did not return" in record["verdict"]


def test_the_unitary_accounting_pays_for_uncomputation():
    accounting = _record()["unitary_accounting"]
    assert accounting["ands_forward"] == 54
    assert accounting["ands_with_adjoint"] == 108
    assert accounting["t_forward_before_optimization"] == 378
    assert accounting["t_with_adjoint_before_optimization"] == 756


def test_the_decoder_is_shown_not_to_be_expressible():
    decoder = _record()["decoder_not_compared"]
    assert decoder["expressible_as_clifford_t"] is False
    ops = decoder["non_unitary_or_non_gate_ops"]
    assert "HMR" in ops
    assert "INFO_TAG" in ops


def test_the_record_pins_the_script_and_its_scope():
    record = _record()
    assert record["reproducibility"]["script_sha256"] == hashlib.sha256(
        SCRIPT.read_bytes()).hexdigest()
    assert "not_measured" in record["scope"]
    assert "Clifford+T optimizers do not model" in record["scope"]["not_measured"]
