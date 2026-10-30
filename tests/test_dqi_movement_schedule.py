"""Pins for Experiment C: the data-movement schedule of the A/B exchange.

`docs/DQI_MOVEMENT_SCHEDULE.md` asks whether the exchange can be resolved at
build time -- which would make it a *static* saving, the only kind the pinned
competition binary can compile -- and measures the answer. These tests pin the
control, the rebased arm (region removed, statically cheaper, incorrect), the
per-shot predicate pattern that explains it, and the audit of the two
zero-saving arms the schedule campaign shipped.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "benchmarks" / "quantum_compilation" / "dqi_movement_schedule.json"
SCRIPT = ROOT / "scripts" / "measure_dqi_movement_schedule.py"


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(RECORD.read_text())


def test_the_control_is_the_five_gate_incumbent(record):
    assert record["control"]["mean_toffoli_like"] == pytest.approx(686_014.84, abs=0.01)
    assert record["control"]["qubits"] == 1913
    assert record["control"]["swap_region"] == pytest.approx(64_224.0, abs=0.01)


def test_the_prediction_was_written_before_the_arms(record):
    prediction = record["prediction"]
    assert prediction["written_before_the_arms"] is True
    assert prediction["C1_rebased"]["predicted_saving_per_shot"] == 64_224.0
    assert prediction["C1_rebased"]["predicted_correct_if_and_only_if"] == \
        "the c3-round set is the same for every shot"


def test_resolving_the_exchange_at_build_time_removes_the_region(record):
    arms = record["arms"]["rebased"]
    assert arms["costs"].get("swap_a_and_b", 0.0) == 0.0
    assert record["observation"]["rebased_swap_region"] == 0.0
    # The static count falls by the region's whole static cost: this is the saving
    # an official-compatible schedule would have delivered.
    assert record["observation"]["rebased_static_saving"] == 198_024


def test_the_rebased_arm_is_incorrect(record):
    arms = record["arms"]["rebased"]
    assert arms["syndromes"] is True  # the input registers come back ...
    assert arms["junk"] is False or arms["phase"] is False  # ... the state does not
    assert record["observation"]["rebased_correct"] is False


def test_the_schedule_is_data_dependent(record):
    pattern = record["observation"]["c3_pattern"]
    assert pattern["rounds"] == 222
    assert pattern["shots"] == 64
    assert pattern["distinct_patterns"] > 1
    assert pattern["identical_to_period_four"] is False
    # The *number* of exchange rounds is invariant; the order is not. That is why
    # a schedule can be counted but not resolved at build time.
    assert set(pattern["firings_per_shot"]) == {72}


def test_the_flag_operand_of_the_exchange_cannot_be_dropped(record):
    observation = record["observation"]
    assert observation["c3_only_prediction"]["predicted_correct"] is True
    assert observation["c3_only_prediction"]["outcome"].startswith("falsified")
    assert observation["c3_only_correct"] is False


def test_the_campaigns_zero_saving_arms_are_audited(record):
    audit = record["observation"]["campaign_arm_audit"]
    assert audit["composed_is_a_copy"] is True
    assert audit["drop_shift_c3_is_a_copy"] is True
    assert audit["arms"]["composed"] == audit["arms"]["none"]


def test_every_arm_keeps_the_rounds_and_the_qubit_count(record):
    for name, arm in record["arms"].items():
        # The probe carries 222 one-bit registers of its own; every other arm must sit
        # at the incumbent's qubit count, because a read costs no qubit.
        expected = record["control"]["qubits"] if name == "probe" else 1913
        if name == "probe":
            expected = 1913 + 222
        assert arm["qubits"] == expected, name
        assert arm["eea_steps"] == 222, name


def test_the_reads_are_counted_beside_every_toffoli_figure(record):
    for name, arm in record["arms"].items():
        assert "mid_circuit_reads_per_shot" in arm, name
        assert "instructions_per_shot" in arm, name


def test_the_record_pins_the_script(record):
    assert record["reproducibility"]["script_sha256"] == hashlib.sha256(
        SCRIPT.read_bytes()).hexdigest()
    assert "not the pinned competition binary" in record["kickmix_role"]
    assert record["failed_checks"] == []
    assert record["verdict"].startswith("the schedule is data-dependent")
