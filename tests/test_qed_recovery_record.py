"""Pins on the QED-001 record: its claims are re-derivable from its own rows.

A record is only evidence if it cannot drift away from the code that produced
it. These tests re-run the parts that are cheap to re-run (the exhaustive
fairness sweep, the hashes), check the arithmetic the record states about its own
budgets, and refuse a row whose status disagrees with its own numbers.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from apedp_dyn.qed.records import RECORD, SCHEMA, oracle_sweep
from apedp_dyn.qed.referee import (
    ABSTAIN,
    ALTERNATIVE,
    HOLDOUT_R2_FLOOR,
    INVALID,
    VALID,
)
from apedp_dyn.qed.tasks import damped_oscillator_task

ROOT = Path(__file__).resolve().parent.parent
RECORD_PATH = ROOT / "benchmarks" / "equation_discovery" / "qed_001_recovery.json"
STATUSES = {VALID, ALTERNATIVE, INVALID, ABSTAIN}


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(RECORD_PATH.read_text(encoding="utf-8"))


def test_the_record_identifies_itself_and_the_question_it_answers(record):
    assert record["record"] == RECORD
    assert record["schema"] == SCHEMA
    assert "same number of objective fits" in record["purpose"]
    # The record is a measurement, so its date moves when it is re-run; what is
    # pinned is that it identifies the day it was produced, not one particular day.
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", record["generated_on"]), record["generated_on"]
    assert record["generated_on"] >= "2026-10-30"
    assert len(record["conditions"]) == 2


def test_both_conditions_use_the_same_task_and_differ_only_in_budget(record):
    first, second = record["conditions"]
    assert {first["budget"], second["budget"]} == {256, 64}
    assert {
        "seed": first["task"]["seed"],
        "noise": first["task"]["noise_level"],
        "penalty": first["task"]["penalty"],
        "library": first["task"]["library"],
    } == {
        "seed": second["task"]["seed"],
        "noise": second["task"]["noise_level"],
        "penalty": second["task"]["penalty"],
        "library": second["task"]["library"],
    }
    for condition in (first, second):
        assert condition["task"]["budget_per_arm"] == condition["budget"]


def test_every_check_the_record_states_passed(record):
    failed = [check["name"] for check in record["checks"] if not check["passed"]]
    assert not failed, failed
    names = [check["name"] for check in record["checks"]]
    assert len(names) == len(set(names)), "check names must be unique across conditions"


def test_the_task_is_fair_in_every_condition_and_once_was_the_truth(record):
    for condition in record["conditions"]:
        oracle = condition["oracle"]
        assert oracle["agrees_with_truth"] is True
        assert oracle["margin"] > 0
        assert oracle["best"]["support"] == condition["task"]["truth_support"]
        assert oracle["truth_objective"] == pytest.approx(oracle["best"]["objective"])
        assert oracle["space"] == 2 ** len(condition["task"]["library"])


def test_the_recorded_oracle_is_reproducible_from_the_recorded_task(record):
    """The sweep is recomputed from the task spec in the record, not trusted."""
    for condition in record["conditions"]:
        task = condition["task"]
        rebuilt = damped_oscillator_task(
            seed=task["seed"],
            noise_level=task["noise_level"],
            penalty=task["penalty"],
            budget=condition["budget"],
        )
        sweep = oracle_sweep(rebuilt)
        assert sweep["best"]["support"] == condition["oracle"]["best"]["support"]
        assert sweep["margin"] == pytest.approx(condition["oracle"]["margin"], rel=1e-12)
        assert sweep["runner_up"]["support"] == condition["oracle"]["runner_up"]["support"]


def test_no_arm_spent_more_than_its_budget_and_an_abstention_carries_no_candidate(record):
    for condition in record["conditions"]:
        for row in condition["arms"]:
            for run in row["runs"]:
                result = run["result"]
                assert result["evaluations"] <= result["budget"] == condition["budget"]
                if result["status"] == "abstained":
                    assert result["best"] is None
                    assert result["reason"], "an abstention must say why"
                else:
                    assert result["best"] is not None


def test_each_status_agrees_with_its_own_numbers(record):
    for condition in record["conditions"]:
        for row in condition["arms"]:
            for run in row["runs"]:
                verdict = run["adjudication"]
                assert verdict["status"] in STATUSES
                if verdict["status"] == ABSTAIN:
                    assert verdict["holdout_r2"] is None
                elif verdict["status"] in (VALID, ALTERNATIVE):
                    assert verdict["holdout_r2"] >= HOLDOUT_R2_FLOOR
                elif verdict["status"] == INVALID:
                    assert verdict["holdout_r2"] < HOLDOUT_R2_FLOOR
                if verdict["status"] == VALID:
                    assert verdict["recovered"] is True
                if verdict["status"] == ALTERNATIVE:
                    assert verdict["recovered"] is False


def test_the_classical_arms_recover_the_generating_support_at_the_generous_budget(record):
    generous = next(c for c in record["conditions"] if c["budget"] == 256)
    recoveries = {row["arm"]: row["recovered_runs"] for row in generous["arms"]}
    for arm in ("random_restart", "simulated_annealing", "genetic_algorithm", "boolean_gp"):
        assert recoveries[arm] == 3, recoveries
    assert sum(1 for count in recoveries.values() if count == 3) >= 4, recoveries


def test_the_quantum_arm_is_recorded_as_worse_and_claims_no_advantage(record):
    """The measured outcome: a deficit, recorded with a reason, never a claim."""
    for condition in record["conditions"]:
        advantage = condition["comparison"]["advantage"]
        assert advantage["claimed"] is False
        assert "search quality at a fixed budget" in advantage["reason"]
        assert advantage["quantum_minus_classical"] == pytest.approx(
            advantage["quantum_best_objective"] - advantage["best_classical_objective"]
        )
        assert advantage["quantum_minus_classical"] >= 0, (
            "the quantum arm never beat the best classical arm on this task"
        )
    generous = next(c for c in record["conditions"] if c["budget"] == 256)
    lean = next(c for c in record["conditions"] if c["budget"] == 64)
    # Best-of-three hides the difference; the per-run recovery counts do not.
    for condition, expected in ((generous, 1), (lean, 0)):
        qaoa = next(r for r in condition["arms"] if r["arm"] == "qaoa_surrogate")
        advantage = condition["comparison"]["advantage"]
        assert qaoa["recovered_runs"] == expected
        assert advantage["quantum_recovered_runs"] == expected
        assert advantage["quantum_recovered_runs"] < advantage["best_classical_recovered_runs"]


def test_the_quantum_arm_reports_only_candidates_it_scored(record):
    """A surrogate fit is model estimation, not a proposed equation.

    The expansion evaluates every singleton and every pair, so if those counted
    as proposals the arm's answer would be an exhaustive pair search with a
    circuit wrapped around it. The recorded rows show the difference: on the
    lean budget the arm returned multi-term selections, not the two-term answer
    its own expansion fitted earlier.
    """
    lean = next(c for c in record["conditions"] if c["budget"] == 64)
    truth = sorted(lean["task"]["truth_support"])
    qaoa = next(row for row in lean["arms"] if row["arm"] == "qaoa_surrogate")
    for run in qaoa["runs"]:
        assert sorted(run["result"]["best"]["support"]) != truth
        assert run["result"]["detail"]["scored_new_fits"] >= 1
        assert run["result"]["detail"]["candidates_offered"] >= run["result"]["detail"]["scored_new_fits"]
    table = record["cross_condition"]["per_arm"]
    assert set(table) >= {"qaoa_surrogate", "boolean_gp", "rime_relaxed"}
    for rows in table.values():
        assert [row["budget"] for row in rows] == [condition["budget"] for condition in record["conditions"]]


def test_the_surrogate_is_paid_for_out_of_the_same_budget_and_is_reported_apart(record):
    for condition in record["conditions"]:
        block = condition["surrogate"]
        truth = sorted(condition["task"]["truth_support"])
        assert block["arm"] == "qaoa_surrogate"
        assert "reported separately" in block["note"]
        assert "circuit probe" in block["note"]
        for run in block["runs"]:
            assert run["surrogate_fits"] == 1 + 10 + 45
            assert run["surrogate_fits"] + run["scored_new_fits"] <= condition["budget"]
            assert run["scored_new_fits"] <= run["candidates_offered"]
            assert run["surrogate_optimum_matches_truth"] == (
                sorted(run["surrogate_optimum"]) == truth
            )
        assert block["surrogate_optimum_matched_truth"] == sum(
            1 for run in block["runs"] if run["surrogate_optimum_matches_truth"]
        )


def test_the_surrogate_minimiser_is_not_the_generating_support(record):
    """The measured defect: its minimiser is not the generating support here.

    If this ever starts matching, the surrogate has become exact on this task and
    the record's prose about it is stale -- so the number is pinned, not assumed.
    """
    for condition in record["conditions"]:
        block = condition["surrogate"]
        assert block["surrogate_optimum_matched_truth"] == 0
        assert all(
            "surrogate_optimum" in run and run["surrogate_optimum"] for run in block["runs"]
        )
        for run in block["runs"]:
            assert len(run["surrogate_optimum"]) > len(condition["task"]["truth_support"])


def test_the_circuit_itself_put_almost_no_probability_on_the_answer(record):
    """Why the arm's recovery came from sampling and not from the circuit.

    ``effective_selections`` is ``1 / sum(p**2)``: the number of selections the
    circuit behaves as if it were choosing between. On this task it is close to
    the full 1024, and the generating support sits a few hundred ranks down. The
    number is pinned because it is the entire argument that the circuit did not
    identify the answer.
    """
    for condition in record["conditions"]:
        probes = condition["surrogate"]["circuit"]
        assert len(probes) == len(condition["arms"][0]["runs"])
        for probe in probes:
            assert probe["space"] == condition["space"]
            assert probe["effective_selections"] > 0.5 * probe["space"]
            assert probe["truth_probability"] <= 2.0 * probe["uniform_probability"]
            assert 100 < probe["truth_rank"] <= probe["space"]
        assert condition["surrogate"]["truth_probability_below_uniform_runs"] == len(probes)


def test_the_circuit_probe_reproduces_the_arm_it_probes(record):
    """A probe is only evidence if it is the same circuit, not a lookalike."""
    for condition in record["conditions"]:
        runs = next(
            row for row in condition["arms"] if row["arm"] == "qaoa_surrogate"
        )["runs"]
        probes = condition["surrogate"]["circuit"]
        for run, probe in zip(runs, probes, strict=True):
            assert run["seed"] == probe["seed"]
            assert run["result"]["detail"]["surrogate_fits"] == probe["surrogate_fits"]
            assert run["result"]["detail"]["surrogate_minimum"] == pytest.approx(
                probe["surrogate_minimum"], rel=1e-12
            )
            assert run["result"]["detail"]["shots"] == probe["shots"]
            assert run["result"]["detail"]["depth"] == probe["depth"]


def test_the_holdout_threshold_in_the_record_is_the_one_the_referee_used(record):
    assert record["thresholds"]["holdout_r2_floor"] == HOLDOUT_R2_FLOOR
    assert "not part of the canonical preregistered specification" in record["thresholds"]["note"]


def test_an_alternative_equation_is_recorded_as_an_alternative(record):
    generous = next(c for c in record["conditions"] if c["budget"] == 256)
    rime = next(row for row in generous["arms"] if row["arm"] == "rime_relaxed")
    assert rime["recovered_runs"] == 0
    verdict = rime["runs"][0]["adjudication"]
    assert verdict["status"] == ALTERNATIVE
    assert verdict["holdout_r2"] >= HOLDOUT_R2_FLOOR
    checks = {check["name"]: check["passed"] for check in verdict["checks"]}
    assert checks["structural_recovery"] is False
    assert checks["forward_prediction"] is True
    assert checks["ablation_separability"] is False


def test_the_record_pins_the_script_and_modules_that_produced_it(record):
    reproducibility = record["reproducibility"]
    assert reproducibility["script"] == "scripts/measure_qed_recovery.py"
    script = ROOT / reproducibility["script"]
    assert hashlib.sha256(script.read_bytes()).hexdigest() == reproducibility["script_sha256"]
    assert reproducibility["modules"], "the record must pin every module it relies on"
    for path, digest in reproducibility["modules"].items():
        target = ROOT / path
        assert target.exists(), f"pinned module vanished: {path}"
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest, path


def test_the_limitations_are_in_the_record_rather_than_implied(record):
    text = " ".join(record["limitations"]).lower()
    for phrase in (
        "no quantum-advantage claim",
        "simulated classically",
        "one synthetic system",
        "finite difference",
        "kinematic identity",
        "three seeds",
    ):
        assert phrase in text, phrase


def test_the_record_contains_no_non_finite_numbers(record):
    text = RECORD_PATH.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
