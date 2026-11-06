"""Pins for the stabilizer scaling record, and a re-run of its cheap rows.

The record is a measurement, so the pin is not a re-run of 1,000 qubits (five
minutes): it re-derives the two cheapest numbers from the script's own functions,
checks the record's hashes against the code on disk, and refuses a record that
claims a check it did not pass.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "measure_vqc_stabilizer_scaling.py"
RECORD = ROOT / "benchmarks" / "quantum_compilation" / "vqc_stabilizer_scaling.json"


def _load_script():
    spec = importlib.util.spec_from_file_location("vqc_stabilizer_scaling", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(RECORD.read_text())


@pytest.fixture(scope="module")
def script():
    return _load_script()


def test_the_record_covers_every_requested_size_and_family(record):
    assert record["schema"] == "apedp.vqc.stabilizer_scaling.v1"
    assert record["sizes"] == [100, 200, 500, 1000]
    families = {(row["family"], row["qubits"]) for row in record["rows"]}
    for size in record["sizes"]:
        for family in ("ghz_chain", "ghz_star", "ring_graph_state",
                       "random_clifford_layers"):
            assert (family, size) in families, (family, size)
    assert len(record["rows"]) == 4 * len(record["sizes"])


def test_every_check_in_the_record_passed_and_the_verdict_says_so(record):
    assert record["checks"], "a record with no checks is not evidence"
    for check in record["checks"]:
        assert check["passed"] is True, check
    assert record["verdict"].startswith("the stabilizer tier")
    assert "inconsistent" not in record["verdict"]
    for row in record["rows"]:
        assert row["checks"], row["family"]
        for name, passed in row["checks"].items():
            assert passed is True, (row["family"], row["qubits"], name)


def test_the_recorded_cost_matches_the_representation(record):
    """`n**2` labels against `2**n` amplitudes, and both stated as what they are."""
    for row in record["rows"]:
        assert row["generators"] == row["qubits"]
        assert row["tableau_label_bytes"] == row["qubits"] ** 2
        # Recorded to three decimals, so the pin is to three decimals.
        assert row["dense_amplitudes_log10"] == pytest.approx(
            row["qubits"] * 0.30102999566398120, abs=1e-3)
        assert row["build_seconds"] > 0.0
        assert row["validate_seconds"] > 0.0
        assert row["sample_seconds_per_shot"] > 0.0
        assert row["shots"] >= 1


def test_the_equivalence_rows_decide_two_different_circuits_at_scale(record):
    assert len(record["equivalence_at_scale"]) == 2
    smallest, largest = record["equivalence_at_scale"]
    assert smallest["qubits"] == 100 and largest["qubits"] == 1000
    for item in record["equivalence_at_scale"]:
        assert item["rewritten"]["verdict"] == "VALID"
        assert item["rewritten"]["mismatched"] == 0
        assert item["mutated"]["verdict"] == "INVALID"
        assert item["mutated"]["mismatched"] > 0
        assert item["mutated"]["first_mismatch"]["basis"]
        assert item["compared_images"] == 2 * item["qubits"]
        assert item["rewritten"]["seconds"] > 0.0
    for item in record["state_scope_at_scale"]:
        assert item["state_scope"]["verdict"] == "VALID"
        assert item["unitary_scope"]["verdict"] == "INVALID"


def test_the_limits_are_stated_in_the_record_itself(record):
    text = " ".join(record["limitations"]).lower()
    for topic in ("clifford", "pure stabilizer", "noise", "1,000", "z-basis"):
        assert topic in text, topic
    assert "asymptotic" in text


def test_the_record_pins_the_code_that_produced_it(record):
    reproducibility = record["reproducibility"]
    assert reproducibility["script_sha256"] == hashlib.sha256(
        SCRIPT.read_bytes()).hexdigest()
    for key, relative in (("module_sha256", "apedp_dyn/vqc/stabilizer.py"),
                          ("equivalence_sha256", "apedp_dyn/vqc/equivalence.py")):
        assert reproducibility[key] == hashlib.sha256(
            (ROOT / relative).read_bytes()).hexdigest(), key


def test_the_cheapest_rows_still_reproduce(script):
    """The 100-qubit rows re-derived from the script, so the record is not alone."""
    chain = script.ghz_row(100, family="ghz_chain", shots=2, seed=20261030)
    assert all(chain["checks"].values()), chain["checks"]
    star = script.ghz_row(100, family="ghz_star", shots=2, seed=20261030)
    assert all(star["checks"].values()), star["checks"]
    graph = script.graph_row(100, shots=2, seed=20261030)
    assert all(graph["checks"].values()), graph["checks"]
    assert graph["stabilised_generators"] == 100
    random_row = script.random_row(100, shots=2, seed=20261030)
    assert all(random_row["checks"].values()), random_row["checks"]
    # Two circuits, one unitary, decided at 100 qubits in well under a second.
    decided = script.equivalence_at_scale(100, seed=20261030)
    assert decided["rewritten"]["verdict"] == "VALID"
    assert decided["mutated"]["verdict"] == "INVALID"
