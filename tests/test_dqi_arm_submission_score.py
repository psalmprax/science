"""Pins for Experiment F: the MR-gated arm's score, measured by the verifier.

Experiment E scored what the pinned binary can compile (5.0946%, 2.319438e13). The
six-lever arm's 1,221,466.8 had only ever been multiplied out by hand, and only on a
build the competition does not ship. These tests pin the measured row: the arm passes
`verify()` at 102,400 shots with zero value and zero phase errors, scores
5.324728e12 at `Q` = 1,913, is 4.5898x cheaper than the pinned reference at the same
budget, and is still 3.137x the derived Benchmark II target -- and the record says it
is not a receipt.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "benchmarks" / "quantum_compilation" / "dqi_arm_submission_score.json"
SCRIPT = ROOT / "scripts" / "measure_dqi_arm_submission_score.py"
REFERENCE_RECORD = (ROOT / "benchmarks" / "quantum_compilation"
                    / "dqi_submission_score.json")
DOC = ROOT / "docs" / "DQI_OFFICIAL_COMPATIBILITY.md"

#: The row `gf54_native_opcode.json` and the schedule campaign's control carry.
ARM_T_MEAN = 1_221_466.8
ARM_QUBITS = 1913
#: Benchmark II's derived target.
TARGET_SCORE = 1.6975e12


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(RECORD.read_text())


def repetition_overhead(eps: float) -> float:
    return math.log(2.0) / -math.log(0.5 + 2.0 * math.sqrt(eps))


def test_the_arm_passes_the_competitions_verifier(record):
    verified = record["verified"]
    assert verified["status"] == "PASS"
    assert verified["value_errors"] == 0
    assert verified["phase_errors"] == 0
    assert verified["verified_shots"] == 102_400
    assert verified["effective_error_rate"] < 1 / 16
    assert record["instance"]["shots"] == record["budget"]["official_num_shots"]


def test_the_arm_reproduces_the_row_it_is_compared_against(record):
    verified = record["verified"]
    assert verified["num_qubits"] == ARM_QUBITS
    # `T` is a mean over sampled syndromes; the tolerance is 0.15%.
    assert verified["toffoli_count"] == pytest.approx(ARM_T_MEAN, rel=1.5e-3)
    # The static count is exact, and the executed mean sits far below it because the
    # arm's classical-condition reads skip work per shot.
    assert verified["max_toffoli_count"] == 3_132_510
    assert verified["toffoli_count"] < 0.5 * verified["max_toffoli_count"]


def test_the_score_is_R_times_q_squared_times_t(record):
    verified = record["verified"]
    r = repetition_overhead(verified["effective_error_rate"])
    assert verified["repetition_factor"] == pytest.approx(r, rel=1e-12)
    assert verified["score"] == pytest.approx(
        r * verified["num_qubits"] ** 2 * verified["toffoli_count"], rel=1e-12)


def test_the_ratio_to_the_pinned_reference_is_like_for_like(record):
    reference = json.loads(REFERENCE_RECORD.read_text())["verified"]["reference"]
    comparisons = record["comparisons"]
    assert comparisons["ratio_is_like_for_like"] is True
    assert comparisons["pinned_reference"]["verified_shots"] == 102_400
    # Both rows share a zero-error `eps_plus` and therefore `R`, so the ratio is the
    # `Q^2 T` ratio: 1,221,408.0 at 1,913 against 5,725,108.4 at 1,893.
    assert comparisons["ratio_to_pinned_reference"] == pytest.approx(
        record["verified"]["score"] / reference["score"], rel=1e-12)
    assert comparisons["ratio_to_pinned_reference"] == pytest.approx(0.217874, abs=1e-6)
    assert 1 / comparisons["ratio_to_pinned_reference"] == pytest.approx(4.5898, abs=1e-3)


def test_the_arm_is_still_above_the_derived_target(record):
    comparisons = record["comparisons"]
    assert comparisons["target"]["score"] == TARGET_SCORE
    assert comparisons["ratio_to_target"] == pytest.approx(3.137, abs=1e-3)
    # The gap is the decoder, not `Q`: at 1,913 the target's Toffoli budget is 509,088.
    assert comparisons["toffolis_needed_at_this_Q"] == pytest.approx(509_088, abs=1)
    assert comparisons["toffoli_ratio_to_target"] == pytest.approx(2.7433, abs=1e-3)
    assert comparisons["qubit_ratio_squared_to_target"] == pytest.approx(1.1434, abs=1e-3)
    assert any(entry["check"].startswith("the arm is still above")
               and entry["passed"] for entry in record["checks"])


def test_the_record_says_it_is_not_a_receipt(record):
    assert "NOT a receipt" in record["status"]
    assert "mr" in record["kickmix_role"] and "pinned" in record["kickmix_role"]
    assert record["instance"]["clear_with"] == "mr"
    assert record["instance"]["product_policy"] == "and54"
    assert "not a submission" in record["scope"]["not_measured"]
    assert record["failed_checks"] == []


def test_the_record_pins_the_script_and_the_verifier_patch(record):
    assert record["reproducibility"]["script_sha256"] == hashlib.sha256(
        SCRIPT.read_bytes()).hexdigest()
    assert record["verifier_patch"]["replaced"] == "'reaction_depth': circuit.reaction_depth(),"


def test_the_doc_states_the_measured_arm_row():
    text = DOC.read_text()
    assert "Experiment F" in text
    assert "5.324728e12" in text
    assert "0.2179" in text and "3.137" in text
    assert "509,088" in text
    assert "not a receipt" in text
