"""Pin the Benchmark III precision gate: cheaper families stay inadmissible.

`benchmarks/quantum_compilation/dfthcsos_precision_gate.json` records what
`dfthcsos/verify.py` returned for every precision the 128-qubit cap admits. The
finding it rests on is easy to lose: the cheapest family (8 circuits, 63,584
Toffolis) is also the one the accuracy gate rejects, so any change that makes it
look admissible has to move either the arithmetic or the gate itself.

This module checks both halves. The structural half always runs: the rows must
reconcile with the builder's own identities (`2 * K * 3,974` Toffolis,
`K = ceil(75/J)`, `Q = 12 + J*brot + 11`), the verifier's thresholds must still be
the ones the rows were judged against, and exactly one row may be admissible. The
numerical half re-derives the rows with the verifier's math when the pinned
FCIDUMP is present, and skips otherwise — the file is competition data and is not
in this checkout (see `docs/DFTHCSOS_HANDOVER.md`).
"""

from __future__ import annotations

import json
import math
import pathlib
import re
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
GATE_JSON = REPO_ROOT / "benchmarks" / "quantum_compilation" / "dfthcsos_precision_gate.json"
FCIDUMP = pathlib.Path("/tmp/qcc/verifiers/dfthcsos/FeMoCo76o113e.fcidump")
VERIFIERS = (REPO_ROOT / "apedp_dyn" / "sandbox"
             / "google-verifiable-quantum-circuit-compilation" / "verifiers")
VERIFY_SOURCE = VERIFIERS / "dfthcsos" / "verify.py"


def verifier_constant(name: str) -> float:
    """Read one numeric constant out of `dfthcsos/verify.py` without importing it.

    Importing the module pulls in `scoreinfra.circuits`, which needs the compiled
    `kickmix` extension; reading the assignment keeps this pin runnable everywhere,
    and it still fails if the gate itself is changed.
    """
    match = re.search(rf"^{name}\s*(?::\s*[^=\n]+)?=\s*([0-9.]+)\s*(?:#.*)?$",
                      VERIFY_SOURCE.read_text(), flags=re.MULTILINE)
    assert match is not None, f"{name} is no longer a literal in verify.py"
    return float(match.group(1))

TOFFOLI_PER_CIRCUIT = 3974  # the 3,975-node address tree, minus one early-break
ANGLES = 75
SELECTION_QUBITS = 12
TREE_ANCILLAS = 11


def gate() -> dict:
    return json.loads(GATE_JSON.read_text())


def test_the_gate_file_records_three_precisions():
    rows = gate()["rows"]
    assert [row["brot"] for row in rows] == [12, 11, 10]


def test_every_row_reconciles_with_the_builder_identities():
    for row in gate()["rows"]:
        brot, batch = row["brot"], row["batch_size"]
        assert row["circuits"] == math.ceil(ANGLES / batch)
        assert row["angles_qubits"] == SELECTION_QUBITS + batch * brot + TREE_ANCILLAS
        assert row["angles_qubits"] <= gate()["qubit_cap"]
        assert row["angles_toffoli"] == 2 * row["circuits"] * TOFFOLI_PER_CIRCUIT
        # Each batch is the largest that still fits the cap at that precision.
        assert row["angles_qubits"] + brot > gate()["qubit_cap"]


def test_only_the_twelve_bit_family_is_admitted():
    rows = gate()["rows"]
    limit = gate()["error_limit"]
    admissible = [row["brot"] for row in rows if row["error_1"] <= limit and row["error_2"] <= limit]
    assert admissible == [12]
    for row in rows:
        assert row["admitted"] is (row["brot"] == 12)


def test_the_cheapest_family_is_the_rejected_one():
    """The 19.98% saving and the rejection are the same row, which is the whole finding."""
    rows = {row["brot"]: row for row in gate()["rows"]}
    cheapest = min(rows.values(), key=lambda row: row["angles_toffoli"])
    assert cheapest["brot"] == 10
    assert cheapest["admitted"] is False
    assert cheapest["angles_toffoli"] < rows[12]["angles_toffoli"]
    assert cheapest["error_1"] > 10 * gate()["error_limit"]


def test_both_residuals_grow_as_the_grid_coarsens():
    rows = gate()["rows"]
    for row, coarser in zip(rows, rows[1:]):
        assert coarser["error_1"] > row["error_1"]
        assert coarser["error_2"] > row["error_2"]


def test_the_verifier_still_enforces_the_gate_these_rows_were_judged_against():
    """A relaxed threshold would make the cheap family admissible without moving a circuit."""
    assert verifier_constant("ERROR_1_THRESHOLD") == gate()["error_limit"]
    assert verifier_constant("ERROR_2_THRESHOLD") == gate()["error_limit"]
    assert verifier_constant("QUBIT_CAP") == gate()["qubit_cap"]


@pytest.mark.skipif(not FCIDUMP.is_file(), reason="pinned FCIDUMP is competition data, not in this checkout")
def test_the_recorded_residuals_reproduce_from_the_verifier_math():
    """Re-derive every row without building a circuit; takes about a minute."""

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    sys.path.insert(0, str(
        REPO_ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation" / "verifiers"))

    import refit_dfthcsos_words as refit

    inputs = refit.load_inputs(FCIDUMP)
    for row in gate()["rows"]:
        words = refit.requantise(inputs["words"], 12, row["brot"])
        metrics = refit.evaluate(
            inputs, refit.vectors_from_words(words, row["brot"], inputs["tree"]))
        assert metrics["error_1"] == pytest.approx(row["error_1"], abs=5e-6)
        assert metrics["error_2"] == pytest.approx(row["error_2"], abs=5e-6)
