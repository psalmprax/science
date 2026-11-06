"""Pins for Experiment G: the campaign arm's score, measured by the verifier.

The best circuit this repository has built is the six-lever arm with the five
schedule/control gates stacked on it. Its 686,014.8 per shot had only ever been a
`Q^2 T` product. These tests pin the verifier's own row: PASS with zero value and zero
phase errors at 102,400 shots, 685,936.2 per shot at `Q` = 1,913, score 2.990339e12,
0.12236 of the pinned reference, and still 1.762x the derived Benchmark II target.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RECORD = (ROOT / "benchmarks" / "quantum_compilation"
          / "dqi_campaign_arm_submission_score.json")
SCRIPT = ROOT / "scripts" / "measure_dqi_campaign_arm_submission_score.py"
REFERENCE_RECORD = (ROOT / "benchmarks" / "quantum_compilation"
                    / "dqi_submission_score.json")
ARM_RECORD = (ROOT / "benchmarks" / "quantum_compilation"
              / "dqi_arm_submission_score.json")

STACKED_T_MEAN = 686_014.8
STACKED_QUBITS = 1913
TARGET_SCORE = 1.6975e12


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(RECORD.read_text())


def repetition_overhead(eps: float) -> float:
    return math.log(2.0) / -math.log(0.5 + 2.0 * math.sqrt(eps))


def test_the_campaign_arm_passes_the_verifier(record):
    verified = record["verified"]
    assert verified["status"] == "PASS"
    assert verified["value_errors"] == 0
    assert verified["phase_errors"] == 0
    assert verified["verified_shots"] == 102_400
    assert verified["effective_error_rate"] < 1 / 16


def test_it_reproduces_the_campaigns_stacked_row(record):
    verified = record["verified"]
    assert verified["num_qubits"] == STACKED_QUBITS
    assert verified["toffoli_count"] == pytest.approx(STACKED_T_MEAN, rel=1.5e-3)
    assert verified["max_toffoli_count"] == 3_130_734


def test_the_score_is_R_times_q_squared_times_t(record):
    verified = record["verified"]
    r = repetition_overhead(verified["effective_error_rate"])
    assert verified["repetition_factor"] == pytest.approx(r, rel=1e-12)
    assert verified["score"] == pytest.approx(
        r * verified["num_qubits"] ** 2 * verified["toffoli_count"], rel=1e-12)


def test_it_is_the_frontier_and_the_ratios_say_so(record):
    reference = json.loads(REFERENCE_RECORD.read_text())["verified"]["reference"]
    six_lever = json.loads(ARM_RECORD.read_text())["verified"]
    comparisons = record["comparisons"]
    assert comparisons["ratio_is_like_for_like"] is True
    assert comparisons["ratio_to_pinned_reference"] == pytest.approx(
        record["verified"]["score"] / reference["score"], rel=1e-12)
    assert comparisons["ratio_to_pinned_reference"] == pytest.approx(0.122357, abs=1e-6)
    # 1.78x cheaper than the six-lever arm at the same Q: the five gates are 0.5616.
    expected = record["verified"]["score"] / six_lever["score"]
    assert comparisons["ratio_to_six_lever_arm"] == pytest.approx(expected, rel=1e-12)
    assert comparisons["ratio_to_six_lever_arm"] == pytest.approx(0.561595, abs=1e-6)
    assert comparisons["ratio_to_target"] == pytest.approx(1.762, abs=1e-3)
    assert comparisons["toffolis_needed_at_this_Q"] == pytest.approx(509_088, abs=1)


def test_the_record_says_it_is_not_a_receipt(record):
    assert "NOT a receipt" in record["status"]
    assert "pinned" in record["kickmix_role"] and "mr" in record["kickmix_role"]
    assert record["instance"]["clear_with"] == "mr"
    assert record["instance"]["product_policy"] == "and54"
    # The six levers and the two promoted mixins are named, not implied.
    assert record["instance"]["gates"] == {
        "gate_forney": True, "clear_eea": True, "gate_sweep": True,
        "gate_swap": True, "gate_shift": True, "gate_quotient": True}
    assert record["instance"]["mixins"] == ["HaltGatedDecoder",
                                            "ReusedFlagPredicateDecoder"]
    assert record["failed_checks"] == []


def test_the_record_pins_the_script(record):
    assert record["reproducibility"]["script_sha256"] == hashlib.sha256(
        SCRIPT.read_bytes()).hexdigest()
    assert record["verifier_patch"]["replaced"] == \
        "'reaction_depth': circuit.reaction_depth(),"
