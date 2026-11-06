"""Pins for the shift/sweep/swap schedule campaign record.

`scripts/measure_dqi_schedule_campaign.py` reads each movement and product predicate
into a classical bit and gates the work on it; `docs/EEA_SCHEDULE_CAMPAIGN.md` is the
campaign it discharges. The record is a build artifact of the local `MR` rebuild, so
re-running it needs that extension, but the *numbers* it produced are pinned here:
a record that drifts back to the six-lever total, invents a saving its region rows do
not support, promotes a gate whose falsifier passes, or loses the Q/iteration invariants
fails without the extension being built.

Everything here is arithmetic over the committed record plus the documentation links, so
the pin runs on the project interpreter (3.12) as well as the extension's.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORD = REPO_ROOT / "benchmarks" / "quantum_compilation" / "dqi_schedule_campaign.json"
SCRIPT = REPO_ROOT / "scripts" / "measure_dqi_schedule_campaign.py"
CAMPAIGN_DOC = REPO_ROOT / "docs" / "EEA_SCHEDULE_CAMPAIGN.md"
PLAN_DOC = REPO_ROOT / "docs" / "EEA_AND_PREDICATE_PLAN.md"
REACHABILITY_DOC = REPO_ROOT / "docs" / "BENCHMARK2_TARGET_REACHABILITY.md"

CONTROL_TOTAL = 1_221_466.8
CONTROL_QUBITS = 1_913
ITERATIONS = 222
TARGET = 445_229.0
#: The four measured arms, to the recorded 0.1 Toffoli.
SAVINGS = {"sweep": 103_968.0, "shift": 65_895.0, "products": 362_637.0,
           "everything": 532_500.0}
EVERYTHING_TOTAL = 688_966.8


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(RECORD.read_text())


def test_the_campaign_is_declared_before_it_is_measured(record):
    """The prediction must be computable from the control arm alone, and say so."""
    prediction = record["prediction"]
    assert prediction["written_before_variants"] is True
    control = prediction["control"]
    # Every predicted quantity is derived from the run's own control arm.
    assert control["selection_chain_per_shot"] == pytest.approx(105_696.0, abs=0.1)
    assert control["shift_chain_per_shot"] == pytest.approx(131_220.0, abs=0.1)
    s1 = prediction["S1_sweep"]
    assert s1["burn_per_shot"] == 5_328.0  # 12 CCX x 2 selections x 222 rounds
    assert s1["headroom_per_shot"] == pytest.approx(
        s1["chain_per_shot"] - s1["burn_per_shot"])
    assert s1["predicted_saving_at_least"] == pytest.approx(100_368.0)
    # S2 and S4 were declared as "no saving predicted", not as a guess.
    assert prediction["S2_shift"]["predicted_saving_at_least"] == 0.0
    assert prediction["S4_products"]["predicted_saving_at_least"] == 0.0
    assert prediction["S4_products"]["product_stream_per_shot"] == pytest.approx(
        693_660.0, abs=0.1)
    # The composition was declared to be worth exactly nothing.
    assert prediction["S3_swap"]["predicted_saving_per_shot"] == 0.0
    assert prediction["S3b_shift_c3_branch"]["predicted_saving_per_shot"] == 0.0


def test_the_five_gate_stack_was_declared_before_it_was_built(record):
    """The 686,014.8 is a prediction that matched, not arithmetic done afterwards."""
    declared = record["prediction"]["S5_stack"]
    assert declared["written_before_the_stacked_arm"] is True
    assert declared["predicted_saving_per_shot"] == pytest.approx(535_452.0, abs=0.1)
    assert declared["predicted_total_per_shot"] == pytest.approx(686_014.8, abs=0.1)
    # Its inputs are named: this run's `everything` and the prior record's 2,952.
    assert declared["inputs"]["everything_measured_in_this_run"] == pytest.approx(
        532_500.0, abs=0.1)
    assert declared["inputs"]["prior_combined_arm_from_dqi_combined_arm_json"] == 2_952.0
    observation = record["observation"]
    assert observation["stacked_predicted_saving"] == pytest.approx(535_452.0, abs=0.1)
    assert observation["stacked_error_per_shot"] == pytest.approx(0.0, abs=0.1)
    checks = {c["check"]: c for c in record["promotion_checks"]}
    s5 = next(c for k, c in checks.items() if k.startswith("S5"))
    assert s5["passed"] is True
    assert s5["measured"]["increment"] == pytest.approx(2_952.0, abs=0.1)


def test_the_control_copy_reproduces_the_six_lever_arm(record):
    """The variants are only trustworthy because the control *copy* is the real arm."""
    control = record["control"]
    assert control["mean_toffoli_like"] == pytest.approx(CONTROL_TOTAL, abs=0.1)
    assert control["qubits"] == CONTROL_QUBITS
    assert control["correct"] is True
    assert control["chains"]["sweep"] == pytest.approx(105_696.0, abs=0.1)
    assert control["chains"]["shift"] == pytest.approx(131_220.0, abs=0.1)
    rows = {row["region"]: row for row in control["regions"]}
    # The three movement rows of `dqi_region_ledger.json`, non-product and products.
    assert (rows["shift_b_array_right"]["executed_toffoli_per_shot"]
            - rows["shift_b_array_right"]["product_toffoli_per_shot"]) == pytest.approx(
        162_479.6, abs=0.1)
    assert (rows["sweep_positions"]["executed_toffoli_per_shot"]
            - rows["sweep_positions"]["product_toffoli_per_shot"]) == pytest.approx(
        143_038.0, abs=0.1)
    assert rows["swap_a_and_b"]["executed_toffoli_per_shot"] == pytest.approx(
        64_224.0, abs=0.1)
    assert rows["sweep_positions"]["product_toffoli_per_shot"] == pytest.approx(
        693_660.0, abs=0.1)


def test_every_measured_arm_saves_what_the_record_claims(record):
    """The four savings, and that each is the chain it says it moved."""
    control = record["control"]["mean_toffoli_like"]
    arms = record["arms"]
    for name, saving in SAVINGS.items():
        arm = arms[name]
        assert arm["correct"] is True, name
        assert control - arm["mean_toffoli_like"] == pytest.approx(saving, abs=0.1), name

    # The chains themselves: the sweep's collapses to the 144 selections that fire,
    # the shift's halves, and the product stream loses what the window excludes.
    assert arms["sweep"]["selection_chain_per_shot"] == pytest.approx(1_728.0, abs=0.1)
    assert arms["shift"]["selection_chain_per_shot"] == pytest.approx(105_696.0, abs=0.1)
    assert arms["products"]["selection_chain_per_shot"] == pytest.approx(105_696.0, abs=0.1)
    assert arms["shift"]["shift_chain_per_shot"] == pytest.approx(65_325.0, abs=0.1)
    assert arms["products"]["shift_chain_per_shot"] == pytest.approx(131_220.0, abs=0.1)
    assert arms["products"]["product_toffoli_per_shot"] == pytest.approx(331_023.0, abs=0.1)
    assert arms["sweep"]["product_toffoli_per_shot"] == pytest.approx(693_660.0, abs=0.1)
    assert arms["everything"]["selection_chain_per_shot"] == pytest.approx(1_728.0, abs=0.1)
    assert arms["everything"]["shift_chain_per_shot"] == pytest.approx(65_325.0, abs=0.1)
    assert arms["everything"]["product_toffoli_per_shot"] == pytest.approx(
        331_023.0, abs=0.1)


def test_the_prior_combined_arm_is_reproduced_in_this_record(record):
    """The previous session's 2,952.0 is re-measured here, not cited across a report."""
    arm = record["arms"]["control_gates"]
    control = record["control"]["mean_toffoli_like"]
    assert arm["correct"] is True
    assert arm["mean_toffoli_like"] == pytest.approx(1_218_514.8, abs=0.1)
    assert control - arm["mean_toffoli_like"] == pytest.approx(2_952.0, abs=0.1)
    # That arm gates the flag/metadata updates on the control it already has: no reads.
    assert arm["mid_circuit_reads_per_shot"] == 0
    assert arm["qubits"] == CONTROL_QUBITS
    observation = record["observation"]
    assert observation["control_gates_total"] == pytest.approx(1_218_514.8, abs=0.1)
    assert observation["control_gates_saving"] == pytest.approx(2_952.0, abs=0.1)


def test_the_stacked_arm_is_all_five_gates_and_adds_up(record):
    """The headline: five gates in one circuit, and two sessions' savings compose."""
    arms = record["arms"]
    control = record["control"]["mean_toffoli_like"]
    stacked = arms["stacked"]
    assert stacked["correct"] is True
    assert stacked["mean_toffoli_like"] == pytest.approx(686_014.8, abs=0.1)
    assert control - stacked["mean_toffoli_like"] == pytest.approx(535_452.0, abs=0.1)
    # Exactly everything + control_gates: the two gates do not interact.
    assert (control - stacked["mean_toffoli_like"]) == pytest.approx(
        (control - arms["everything"]["mean_toffoli_like"])
        + (control - arms["control_gates"]["mean_toffoli_like"]), abs=0.1)
    assert stacked["qubits"] == CONTROL_QUBITS
    assert stacked["mean_toffoli_like"] / TARGET == pytest.approx(1.5408, abs=0.001)
    assert (control - stacked["mean_toffoli_like"]) / control == pytest.approx(
        0.438368, abs=1e-5)


def test_the_saving_removes_work_rather_than_relocating_it(record):
    """The reviewer's question, answered on the instruction stream as well as the T count.

    A schedule change that is cheaper in Toffolis but longer in instructions has moved
    work, not removed it. Here both fall, and the added reads are counted rather than
    left implicit.
    """
    control = record["control"]
    observation = record["observation"]
    stacked = record["arms"]["stacked"]
    toffoli_saving = control["mean_toffoli_like"] - stacked["mean_toffoli_like"]
    assert observation["control_instructions_per_shot"] == pytest.approx(37_808_019.7,
                                                                       abs=0.5)
    assert stacked["instructions_per_shot"] == pytest.approx(32_451_795.1, abs=0.5)
    instruction_saving = (observation["control_instructions_per_shot"]
                          - stacked["instructions_per_shot"])
    assert toffoli_saving == pytest.approx(535_452.0, abs=0.1)
    assert instruction_saving == pytest.approx(5_356_224.6, abs=1.0)
    assert instruction_saving > 9 * toffoli_saving  # the removed CCXes' expansion
    reads = stacked["mid_circuit_reads_per_shot"]
    assert reads == pytest.approx(30_253.0, abs=1.0)
    # Each read is one cx and one mr: the added cost is 2 instructions per read.
    assert 2 * reads < 0.02 * instruction_saving
    # And the reads are not hidden: the arm that gates on predicates reports them.
    assert record["prediction"]["relocation"]["question"].startswith(
        "does the Toffoli saving relocate work")


def test_the_three_gates_are_exactly_additive(record):
    """`everything` is the sum of its parts, and the arm is 1.547x the target."""
    arms = record["arms"]
    total = control = record["control"]["mean_toffoli_like"]
    assert arms["both"]["correct"] is True
    assert arms["both"]["mean_toffoli_like"] == pytest.approx(
        total - SAVINGS["sweep"] - SAVINGS["shift"], abs=0.1)
    assert arms["everything"]["mean_toffoli_like"] == pytest.approx(
        total - SAVINGS["sweep"] - SAVINGS["shift"] - SAVINGS["products"], abs=0.1)
    assert arms["everything"]["mean_toffoli_like"] == pytest.approx(EVERYTHING_TOTAL,
                                                                   abs=0.1)
    assert EVERYTHING_TOTAL / TARGET == pytest.approx(1.5474, abs=0.001)
    # The pieces still add to the arm: products + non-product = total.
    rows = arms["everything"]["regions"]
    products = sum(row["product_toffoli_per_shot"] for row in rows)
    non_product = sum(row["executed_toffoli_per_shot"]
                      - row["product_toffoli_per_shot"] for row in rows)
    assert products == pytest.approx(386_061.7, abs=0.2)
    assert non_product == pytest.approx(302_905.2, abs=0.2)
    assert products + non_product == pytest.approx(EVERYTHING_TOTAL, abs=0.1)
    assert 100 * SAVINGS["everything"] / control == pytest.approx(43.60, abs=0.01)


def test_the_reads_cost_no_qubit_and_no_iteration(record):
    """A classical condition is classical: peak qubits and the EEA schedule are fixed."""
    control = record["control"]
    for name, arm in record["arms"].items():
        assert arm["qubits"] == control["qubits"] == CONTROL_QUBITS, name
    assert record["instance"]["iterations"] == ITERATIONS
    assert record["instance"]["shots"] == 64


def test_the_composition_is_worth_nothing_and_says_so(record):
    """S3 as a *measurement*, and S3b as a falsified prediction.

    These two arms were the control circuit when this record was first written -- the
    method implementing both was called `_shift_a_and_b` and nothing calls that -- so
    their 0.0 was vacuous. Renamed to `_shift_b_array_right`, the gate is in the
    stream, and the re-measured arms are what these assertions pin: composing the
    shift with the exchange still saves exactly 0.0 and stays correct, while dropping
    the shift from every `c3` round finds real work and breaks the decoder.
    """
    arms = record["arms"]
    assert arms["composed"]["correct"] is True
    assert arms["composed"]["saving_per_shot"] == pytest.approx(0.0, abs=0.05)
    assert arms["drop_shift_c3"]["correct"] is False
    assert arms["drop_shift_c3"]["saving_per_shot"] > 40_000.0
    s3b = next(entry for entry in record["promotion_checks"]
               if entry["check"].startswith("S3b"))
    assert s3b["measured"]["outcome"].startswith("falsified")
    assert s3b["measured"]["declared_saving"] == 0.0
    # The per-round table is an aggregate over the shot batch, and the record says so
    # rather than letting it read as per-shot evidence.
    assert "aggregate" not in record["round_overlap"]["note"].lower()
    assert "summed over the shot batch" in record["round_overlap"]["note"]
    assert record["round_overlap"]["rounds"] == ITERATIONS


def test_every_falsifier_breaks_the_arm_and_was_informative(record):
    """A wrong condition must fail, and it must have skipped work to count at all."""
    falsifiers = record["falsifiers"]
    assert set(falsifiers) == {"falsify_sweep_stale", "falsify_shift_complement",
                               "falsify_shift_stale", "falsify_zero_q"}
    for name, row in falsifiers.items():
        assert row["correct"] is False, name
        assert row["informative"] is True, name
        assert row["skipped_per_shot"] > 0.5, name
    # The stale-selection falsifier is the trap: it skips the most work of any arm and
    # still costs more than the control, because a wrong state runs a new control flow.
    stale = falsifiers["falsify_sweep_stale"]
    assert stale["skipped_per_shot"] == pytest.approx(104_768.1, abs=0.1)
    assert stale["arm_total_change"] < -200_000
    # And the falsifiers that save more than the honest gate are the ones that are wrong.
    honest_shift = SAVINGS["shift"]
    for name in ("falsify_shift_complement", "falsify_shift_stale"):
        assert falsifiers[name]["arm_total_change"] > honest_shift, name


def test_experiment_b_is_measured_and_closed(record):
    """The product stream's own operand predicate: sound, correct, and a measured loss.

    This is the one arm in the campaign whose prediction is that nothing is gained. It has
    to stay recorded as a negative result: a correct arm that costs more, with the reason
    (the read price) in the record rather than in a conversation.
    """
    declared = record["prediction"]["S6_zero_q"]
    assert declared["written_before_the_arm"] is True
    assert declared["declared_saving_per_shot"] == 0.0
    arms = record["arms"]
    arm = arms["zero_q"]
    assert arm["correct"] is True  # the predicate is admissible, not wrong
    assert arm["mean_toffoli_like"] == pytest.approx(997_795.6, abs=0.1)
    # It loses against the incumbent it was built on, not just against the control.
    assert arm["mean_toffoli_like"] - arms["stacked"]["mean_toffoli_like"] == pytest.approx(
        311_780.7, abs=0.1)
    assert arm["mid_circuit_reads_per_shot"] - arms["stacked"][
        "mid_circuit_reads_per_shot"] == pytest.approx(327_094.0, abs=1.0)
    # It is not promoted, and the verdict still names only the six that earned it.
    assert "zero_q" not in record["observation"]["promoted_arms"]
    assert record["verdict"] == ("PROMOTE sweep, shift, products, both, everything, "
                                 "stacked")
    # Its falsifier is wrong and informative: the wrong condition skips real products.
    falsifier = record["falsifiers"]["falsify_zero_q"]
    assert falsifier["correct"] is False
    assert falsifier["informative"] is True
    assert falsifier["skipped_per_shot"] > 100_000


def test_the_campaign_and_the_plan_agree_about_what_it_changed():
    """Docs: the campaign note exists, and both other notes carry its numbers."""
    text = CAMPAIGN_DOC.read_text(encoding="utf-8")
    assert "1,221,466.8 → 686,014.8" in text
    assert "1.541×" in text
    assert "535,452.0" in text and "686,014.8" in text
    assert "written_before_variants" in text
    assert "adaptive execution" in text
    # The relocation question and the preconditions are in the note, not only in the data.
    assert "remove work, or move it" in text
    assert "30,253 reads" in text
    assert "P1" in text and "P3" in text
    assert "5,356,224.6 instructions" in text
    # Experiment B's negative result is documented, with its price.
    assert "Experiment B" in text
    assert "311,780.7" in text and "327,094" in text

    # S3b is quoted on this record's own base, and the note never swaps in the
    # movement script's figure (45,377.7, a base 2,952 lower) as if it were this
    # record's: the two differ, and only the record decides which is which.
    record = json.loads(RECORD.read_text())
    s3b = record["arms"]["drop_shift_c3"]
    assert s3b["mean_toffoli_like"] == 1_176_070.703125
    assert s3b["saving_per_shot"] == 45_396.140625
    assert f"{s3b['saving_per_shot']:,.1f}" in text  # 45,396.1
    assert f"{record['control']['mean_toffoli_like'] - s3b['mean_toffoli_like']:,.1f}" \
        in text  # 45,396.1 again, from the two T values

    plan = PLAN_DOC.read_text(encoding="utf-8")
    assert "EEA_SCHEDULE_CAMPAIGN.md" in plan
    assert "532,500" in plan and "688,966.8" in plan
    # The earlier static-count claim is still there, and is scoped rather than deleted.
    assert "No amount of EEA/predicate" in plan

    reachability = REACHABILITY_DOC.read_text(encoding="utf-8")
    assert "1.541×" in reachability
    assert "686,014.8" in reachability
    assert "240,785.8" in reachability

    # The record cites the note that declares it, and the script is in the registry.
    assert json.loads(RECORD.read_text())["declared_in"] == "docs/EEA_SCHEDULE_CAMPAIGN.md"
    registry = (REPO_ROOT / "docs" / "DOCUMENTATION.md").read_text(encoding="utf-8")
    assert "scripts/measure_dqi_schedule_campaign.py" in registry
    assert "dqi_schedule_campaign.json" in registry


def test_the_record_was_produced_by_the_script_on_disk(record):
    """A stale record is worse than none: the script hash must match the file.

    Editing the harness without re-running it would leave a record that no longer
    describes the code it cites. The campaign's own promotion gate records the hash;
    this pins that the recorded one is still the script's.
    """
    import hashlib

    recorded = record["reproducibility"]["script_sha256"]
    assert recorded == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()


def test_the_zone_of_claimed_scope_is_stated(record):
    """No receipt, and the reason is in the record rather than in a reader's memory."""
    assert record["scope"]["is_official_result"] is False
    assert "mr" in record["scope"]["reason"].lower()
    assert record["verdict"].startswith("PROMOTE")
    assert record["all_checks_passed"] is True
    assert record["failed_checks"] == []
    # The script's own promotion checks are all green in the committed record.
    assert all(check["passed"] for check in record["promotion_checks"])
    assert re.search(r"sha256", json.dumps(record["reproducibility"]))
