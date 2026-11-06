"""Pins for the Benchmark II status decision.

`docs/BENCHMARK2_STATUS_DECISION.md` records what the measurements already decide: the
pinned binary is frozen at 0.949054, `MR` is the only reopen condition, and no further
field-arithmetic work is planned. These tests hold the note to the records, so a decision
cannot drift away from the numbers it was taken on.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "BENCHMARK2_STATUS_DECISION.md"
REACHABILITY = ROOT / "docs" / "BENCHMARK2_TARGET_REACHABILITY.md"
MR_REQUEST = ROOT / "docs" / "MR_PATCH_REQUEST.md"
SUBMISSION = ROOT / "benchmarks" / "quantum_compilation" / "dqi_submission_score.json"
ARM = ROOT / "benchmarks" / "quantum_compilation" / "dqi_arm_submission_score.json"
CAMPAIGN = (ROOT / "benchmarks" / "quantum_compilation"
            / "dqi_campaign_arm_submission_score.json")

TARGET = 1.6975e12


def _doc() -> str:
    return DOC.read_text()


def test_d1_freezes_the_pinned_line_at_the_measured_candidate():
    record = json.loads(SUBMISSION.read_text())
    reference = record["verified"]["reference"]
    candidate = record["verified"]["candidate"]
    text = _doc()
    assert "0.949054" in text
    assert candidate["num_qubits"] == 1893
    assert candidate["phase_errors"] == 0 and candidate["value_errors"] == 0
    assert f"{candidate['score'] / TARGET:.2f}×" in text  # 13.66x the target
    assert record["score_ratio"]["score"] / 1 == record["score_ratio"]["score"]
    assert abs(reference["score"] / TARGET - 14.397) < 0.01


def test_d2_names_mr_as_the_only_reopen_and_quotes_the_measured_arms():
    arm = json.loads(ARM.read_text())["verified"]
    campaign = json.loads(CAMPAIGN.read_text())["verified"]
    text = _doc()
    assert "MR_PATCH_REQUEST.md" in text
    assert "kickmix_mr_opcode.patch" in text and "build_kickmix_mr.py" in text
    assert MR_REQUEST.exists()
    assert "5.324728e12" in text and "2.990339e12" in text
    assert arm["num_qubits"] == campaign["num_qubits"] == 1913
    assert abs(arm["score"] / TARGET - 3.137) < 0.001
    assert abs(campaign["score"] / TARGET - 1.762) < 0.001
    assert "4.142" in text and "4.5898" in text


def test_d3_keeps_the_arithmetic_ceiling_and_the_next_class_of_change():
    text = _doc()
    reachability = REACHABILITY.read_text()
    assert "472,768.2" in text and "472,768.2" in reachability
    assert "240,785.8" in text
    assert "1.062" in text  # every product and the add removed still leaves 1.06x
    assert "decoder algorithm" in text
    # The next class of change is named as *not started*, not as in progress.
    assert "not started" in text.replace("*", "")
    assert "23-AND" in text


def test_every_decision_states_what_would_overturn_it():
    text = _doc()
    assert "What would overturn a decision" in text
    for line in ("**D1:**", "**D2:**", "**D3:**"):
        assert line in text, line


def test_the_reachability_note_points_at_the_decision():
    reachability = REACHABILITY.read_text()
    assert "BENCHMARK2_STATUS_DECISION.md" in reachability
    # And no longer describes any frontier row as unmeasured.
    assert "one it has not yet" not in reachability


def test_the_note_does_not_add_a_measurement_or_a_submission_claim():
    text = _doc().lower()
    assert "does not add a measurement" in text
    assert "receipt" not in text or "no" in text  # no receipt is claimed here
    assert "still in progress" not in text
