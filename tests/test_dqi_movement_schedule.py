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


def test_the_exchange_predicate_explains_the_falsification(record):
    # `c3` is a count and `c3 and flag` is a condition: the post-update flag is clear
    # in more than half the c3 rounds, so exchanging on `c3` alone over-swaps.
    predicate = record["observation"]["exchange_predicate"]
    assert predicate["rounds"] == 222
    assert predicate["c3_rounds_total"] == 72 * 64
    assert predicate["exchanges_total"] < predicate["c3_rounds_total"]
    assert predicate["c3_rounds_with_flag_clear_total"] > 0
    assert predicate["flag_is_redundant_in_c3_rounds"] is False
    assert predicate["c3_rounds_total"] - (predicate["exchanges_total"]
                                          + predicate["c3_rounds_with_flag_clear_total"]) == 0
    assert predicate["distinct_exchange_sets"] > 1
    assert set(predicate["c3_firings_per_shot"]) == {72}
    assert max(predicate["exchanges_per_shot"]) < 72


def test_the_campaigns_zero_saving_arms_were_copies_and_now_are_not(record):
    observation = record["observation"]
    audit = observation["campaign_arm_audit"]
    # What the campaign's run measured: with the gate in a method nothing called, the
    # two arms *were* the control circuit. That is the finding, kept.
    assert "never called" in audit["why"]
    # Now that the method is named for what it overrides, the gate is in the stream:
    # the composed arm saves exactly 0.0 and stays correct, the c3-dropping arm finds
    # 45,377.7 of work the decoder needs and is wrong.
    assert observation["campaign_composed_saving"] == 0.0
    assert observation["campaign_composed_correct"] is True
    assert observation["campaign_composed_shift_region"] == \
        observation["campaign_none_shift_region"]
    assert observation["campaign_drop_c3_saving"] > 40_000.0
    assert observation["campaign_drop_c3_correct"] is False
    assert observation["campaign_drop_c3_shift_region"] < \
        observation["campaign_none_shift_region"]


def test_every_arm_keeps_the_rounds_and_the_qubit_count(record):
    for name, arm in record["arms"].items():
        # The probe carries 222 one-bit registers of its own; every other arm must sit
        # at the incumbent's qubit count, because a read costs no qubit.
        expected = record["control"]["qubits"] if name == "probe" else 1913
        if name == "probe":
            # Two probes of one qubit per round: the exchange's `c3` and its flag.
            expected = 1913 + 2 * 222
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
