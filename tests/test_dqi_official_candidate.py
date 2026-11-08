"""Pins for Experiment D2: is the officially compilable candidate *correct*?

`docs/DQI_OFFICIAL_COMPATIBILITY.md` §5 reported a candidate the pinned binary can
compile -- the reference with the 54-AND multiply and `hmr` clears -- at a 0.952
score ratio, having checked its syndromes and not its phase. Experiment D2 measures
the phase, and these tests pin what it found: `hmr` is not phase-safe, `del_and` is,
the reference makes no phase errors and both cheap candidates make about one in two.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "benchmarks" / "quantum_compilation" / "dqi_official_candidate.json"
SCRIPT = ROOT / "scripts" / "measure_dqi_official_candidate.py"
DOC = ROOT / "docs" / "DQI_OFFICIAL_COMPATIBILITY.md"


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(RECORD.read_text())


def test_the_reference_is_phase_correct(record):
    reference = record["arms"]["reference"]
    assert reference["qubits"] == 1893
    assert reference["mean_toffoli_like"] == pytest.approx(5_725_071.609375, abs=1e-6)
    assert reference["value_errors"] == 0
    assert reference["phase_errors"] == 0


def test_hmr_is_not_phase_safe_and_del_and_is(record):
    safety = record["clear_phase_safety"]
    assert safety["hmr_on_zero"]["phase_flips"] == 0
    assert safety["hmr_on_one"]["phase_flips"] > 0
    # The flips are the measurement outcomes: an X-basis measurement of |1> puts the
    # -1 on the branch that measures 1, and the counter tracks it.
    assert safety["hmr_on_one"]["phase_flips"] == safety["hmr_on_one"]["measured_ones"]
    assert safety["del_and_on_zero"]["phase_flips"] == 0
    assert safety["del_and_on_one"]["phase_flips"] == 0
    assert safety["hmr_on_one"]["phase_safe"] is False
    assert safety["del_and_on_one"]["phase_safe"] is True


def test_the_54_and_candidate_reproduces_its_count_and_fails_on_phase(record):
    candidate = record["arms"]["hmr_candidate"]
    assert candidate["qubits"] == 1896
    assert candidate["mean_toffoli_like"] == pytest.approx(5_433_399.609375, abs=1e-6)
    assert candidate["value_errors"] == 0
    # A coin flip per shot, not a small defect: this is why D's 0.952 is withdrawn.
    assert candidate["phase_errors"] > 0.4 * candidate["shots"]
    assert candidate["phase_errors"] < 0.75 * candidate["shots"]


def test_the_stream_of_the_working_candidate_has_no_z_basis_read(record):
    """The two clears it rests on are `del_and` and `del_gf2_mul`, not `mr`/`hmr`."""
    assert record["build"]["has_mr"] is False
    assert record["clear_phase_safety"]["del_and_on_one"]["phase_safe"] is True
    assert record["clear_phase_safety"]["hmr_on_one"]["phase_safe"] is False
    assert "del_gf2_mul" in record["finding"]


def test_the_copied_eea_candidate_removes_the_reverse_pass_and_still_fails(record):
    copied = record["arms"]["copied_eea"]
    assert copied["mean_toffoli_like"] < 0.7 * record["arms"]["reference"]["mean_toffoli_like"]
    assert copied["value_errors"] == 0
    assert copied["phase_errors"] > 0.4 * copied["shots"]
    # Its clears are `hmr` too, and holding the whole scratch until the end costs
    # more qubits than the removed pass saves.
    assert copied["qubits"] > record["arms"]["reference"]["qubits"]


def test_no_score_ratio_is_quoted_for_a_candidate_with_phase_errors(record):
    # The ratios are recorded — they are what the Toffoli count and Q are worth — but
    # the record says which of them the phase invalidates, and the finding names the
    # 0.952 row as unattainable.
    for ratio in record["score_ratios"].values():
        assert ratio > 0
    assert record["score_ratios"]["hmr_candidate"] == pytest.approx(0.952064, abs=1e-5)
    assert "0.952 score ratio D recorded is not" in record["finding"]
    assert record["scope"]["corrects"].startswith("the D record's candidate is not "
                                                  "phase-correct")


def test_the_phase_safe_multiply_is_the_first_correct_candidate(record):
    row = record["arms"]["phase_safe_multiply"]
    reference = record["arms"]["reference"]
    assert row["value_errors"] == 0 and row["phase_errors"] == 0
    assert row["junk_clean"] is True and row["syndromes_restored"] is True
    # The reduced-window design costs *no* extra qubits: peak Q is the reference's.
    assert row["qubits"] == reference["qubits"] == 1893
    assert row["mean_toffoli_like"] == pytest.approx(5_433_399.609375, abs=1e-6)
    assert record["score_ratios"]["phase_safe_multiply"] == pytest.approx(0.949054,
                                                                         abs=1e-5)
    assert reference["phase_errors"] == 0


def test_only_the_phase_safe_arm_is_correct_and_cheaper(record):
    reference = record["arms"]["reference"]["mean_toffoli_like"]
    correct = {name for name, row in record["arms"].items()
               if row["phase_errors"] == 0 and row["value_errors"] == 0
               and row["mean_toffoli_like"] < reference}
    assert correct == {"phase_safe_multiply"}
    assert any(entry["check"].startswith("the phase-safe multiply")
               and entry["passed"] for entry in record["checks"])


def test_the_record_pins_the_script_and_passes_its_checks(record):
    assert record["reproducibility"]["script_sha256"] == hashlib.sha256(
        SCRIPT.read_bytes()).hexdigest()
    assert record["failed_checks"] == []
    assert record["build"]["has_mr"] is False


def test_the_doc_states_the_correction_and_the_replacement():
    text = DOC.read_text()
    assert "Correction (Experiment D2)" in text
    assert "0.952" in text and "withdrawn" in text
    assert "37/64" in text and "35/64" in text
    # And the candidate that replaced it: phase-safe clears, no extra qubits.
    assert "del_gf2_mul" in text and "0.9491" in text
    assert "reduced-window" in text
    assert "no lever can be compiled" in text.lower()
