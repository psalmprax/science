"""Pins for the representation-aware uncomputation checklist.

`docs/REPRESENTATION_AWARE_UNCOMPUTATION.md` is a method note, but every row in it is a
measurement: `del_and` and `del_gf2_mul` are phase-safe clears on their own
preconditions, `hmr` is not a clear at all, and the reduced-window multiply that made
`del_gf2_mul` applicable is the only correct cheap candidate the pinned binary can
build. These tests hold the note to those records and to its own nine gates.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "REPRESENTATION_AWARE_UNCOMPUTATION.md"
CANDIDATE = ROOT / "benchmarks" / "quantum_compilation" / "dqi_official_candidate.json"
SUBMISSION = ROOT / "benchmarks" / "quantum_compilation" / "dqi_submission_score.json"


@pytest.fixture(scope="module")
def candidate() -> dict:
    return json.loads(CANDIDATE.read_text())


@pytest.fixture(scope="module")
def submission() -> dict:
    return json.loads(SUBMISSION.read_text())


def test_del_and_is_phase_safe_on_its_own_precondition(candidate):
    safety = candidate["clear_phase_safety"]["del_and_on_one"]
    assert safety["phase_flips"] == 0
    assert safety["phase_safe"] is True


def test_hmr_is_not_a_clear_at_all(candidate, submission):
    safety = candidate["clear_phase_safety"]["hmr_on_one"]
    assert safety["phase_safe"] is False
    assert safety["phase_flips"] == safety["measured_ones"] > 0
    # The probe in Experiment E is the same statement on |1> over 64 shots: the bit it
    # writes is a measurement, not the qubit's value.
    probe = submission["lever_primitives"]["probes"]["hmr"]
    assert probe["shots"] == 64
    assert probe["bit_ones"] == 24
    assert probe["reads_qubit"] is False
    assert probe["phase_flips"] == probe["bit_ones"]


def test_the_reduced_window_is_what_made_del_gf2_mul_applicable(candidate):
    row = candidate["arms"]["phase_safe_multiply"]
    reference = candidate["arms"]["reference"]
    assert row["value_errors"] == 0 and row["phase_errors"] == 0
    assert row["junk_clean"] is True
    # `del_gf2_mul` costs 0 Toffolis and no workspace: the saving reaches `Q` = 1,893
    # unchanged, which is the checklist's qubit-ledger gate.
    assert row["qubits"] == reference["qubits"] == 1893
    assert candidate["score_ratios"]["phase_safe_multiply"] == pytest.approx(0.949054,
                                                                            abs=1e-5)
    # And the two paths are the multiply's 54-AND base case plus the 12-Toffoli add.
    assert row["mean_toffoli_like"] == pytest.approx(5_433_399.609375, abs=1e-6)


def test_the_note_carries_the_nine_gates_and_the_three_correctness_ones():
    text = DOC.read_text()
    for number in range(1, 10):
        assert f"{number}. **" in text, number
    # Gates 3, 4 and 7 are the correctness deciders; the note says which they are.
    assert "Gates 3, 4 and 7 are the ones that decide correctness" in text
    assert "13 of 21 records" in text


def test_the_note_quotes_the_measured_rows_and_not_more():
    text = DOC.read_text()
    assert "66 Toffolis per controlled call against 75" in text
    assert "24/64" in text and "37/64" in text
    assert "0.949054" in text
    # The note must keep its boundaries: no minimality claim, and the external
    # comparison scoped to what a Clifford+T optimizer can express.
    assert "not a minimality proof" in text
    assert "378 → 378" in text
    assert "108 ANDs" in text
    assert "gf_mul_external_optimizer.json" in text


def test_the_note_names_the_hooks_it_plugs_into():
    text = DOC.read_text()
    for hook in ("CircuitCandidate", "CandidateStatus", "run_dqi_verifier",
                 "dqi_verification_scope", "contracts.py", "gf54_mul_reduced.py",
                 "test_gf54_mul_reduced.py"):
        assert hook in text, hook
