"""Pins for the order score: the place of the true law in a proposer's list, comparisons, and what can be seen."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from apedp_dyn.recursion import order_score
from apedp_dyn.recursion.curriculum import Problem

ROOT = Path(__file__).resolve().parents[1]
PROBLEM = Problem("order-score|test|0", "1.3*x0/(x0 + 0.7)", 1, 0, 0.01)


def _script():
    spec = importlib.util.spec_from_file_location("measure_order_score", ROOT / "scripts" / "measure_order_score.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a_candidate_is_judged_as_the_judge_judges_a_reported_law() -> None:
    truth = order_score.judge_candidate(PROBLEM, PROBLEM.text)
    assert truth.accurate and truth.solved and truth.same_form and truth.r2 == 1.0
    # the same law with a constant that does nothing: accurate, but longer than the true law, so not solved
    padded = order_score.judge_candidate(PROBLEM, "1.3*x0/(x0 + 0.7) + 0.00000001*x0**2")
    assert padded.accurate and not padded.solved and padded.bits > truth.bits
    wrong = order_score.judge_candidate(PROBLEM, "0.4*x0")
    assert not wrong.accurate and not wrong.solved and not wrong.same_form


def test_what_cannot_be_read_is_not_the_law_and_never_raises() -> None:
    for text in ("x0 +* 2", "1.3*y/(y + 0.7)", "", "1/(x0 - x0)"):
        judged = order_score.judge_candidate(PROBLEM, text)
        assert not judged.solved and not judged.same_form


def test_the_places_are_read_off_the_list_in_its_order() -> None:
    row = order_score.measure(PROBLEM, ["0.4*x0", "1.3*x0/(x0 + 0.7) + 0.00000001*x0**2", PROBLEM.text])
    assert (row["solved_place"], row["form_place"], row["accurate_place"]) == (3, 3, 2)
    assert row["proposals"] == 3 and [c["solved"] for c in row["candidates"]] == [False, False, True]
    nothing = order_score.measure(PROBLEM, ["0.4*x0"])
    assert nothing["solved_place"] is None and order_score.reciprocal(None) == 0.0 and order_score.reciprocal(4) == 0.25


def _rows(places: list[int | None], level: int = 1) -> list[dict]:
    return [{"key": f"k{i}", "level": level, "solved_place": place} for i, place in enumerate(places)]


def test_a_set_is_summarised_by_share_found_and_mean_reciprocal_place() -> None:
    summary = order_score.summarise(_rows([1, 2, None, 4]))
    assert (summary["problems"], summary["found"], summary["first"], summary["within_3"]) == (4, 3, 1, 2)
    assert summary["share_found"] == 0.75 and summary["mean_reciprocal_place"] == pytest.approx((1 + 0.5 + 0 + 0.25) / 4, abs=1e-4)
    assert summary["by_level"]["1"]["problems"] == 4


def test_two_proposers_are_compared_problem_by_problem() -> None:
    first, second = _rows([None, 5, 2, 1, None, None]), _rows([3, 1, 2, 1, 4, None])
    got = order_score.compare(first, second)
    assert (got["problems"], got["earlier"], got["later"], got["same"], got["gained"], got["lost"]) == (6, 3, 0, 3, 2, 0)
    assert got["sign_test_p"] == 0.25 and got["mean_reciprocal_difference"] > 0
    assert got["interval_95"][0] <= got["mean_reciprocal_difference"] <= got["interval_95"][1]
    back = order_score.compare(second, first)
    assert (back["earlier"], back["later"]) == (0, 3) and back["mean_reciprocal_difference"] < 0
    assert order_score.compare(first, first)["sign_test_p"] == 1.0


def test_ten_problems_see_almost_nothing_and_three_hundred_see_a_modest_gain() -> None:
    measured = _rows([1] * 30 + [None] * 70)                     # the law found on 30 % of problems
    few = order_score.power(measured, 10, 0.5, trials=400, seed=1)
    many = order_score.power(measured, 300, 0.5, trials=400, seed=1)
    assert few < 0.2 < 0.9 < many                                # a gain of half: invisible on 10, plain on 300
    assert order_score.power(measured, 300, 0.0, trials=400, seed=1) < 0.1          # no gain is rarely called a gain
    with_churn = order_score.power(measured, 300, 0.2, churn=0.25, trials=400, seed=1)
    without = order_score.power(measured, 300, 0.2, churn=0.0, trials=400, seed=1)
    assert with_churn < without                                   # problems lost on the way make a gain harder to see
    table = order_score.smallest_visible_gain(measured, 300, trials=300)
    assert table["smallest_visible_gain"] is not None and table["power_by_gain"]["2"] >= table["power_by_gain"]["0.05"]
    assert order_score.power(_rows([None] * 20), 100, 0.5) == 0.0                   # nothing found: nothing to plant a gain on


def test_the_runner_draws_teaching_laws_apart_from_the_test_problems() -> None:
    script = _script()
    import sympy as sp

    from apedp_dyn.recursion.curriculum import structure

    tests = script.test_problems(20)
    laws = script.teaching_laws(12, tests)
    assert len(laws) == 12 and not {structure(sp.sympify(law)) for law in laws} & {structure(p.truth) for p in tests}
    assert script.teaching_laws(0, tests) == [] and script.memory_of("taught-0", tests).laws == 0
    assert script.memory_of("taught-5", tests).laws == 5


def test_the_runner_s_pools_keep_numbers_only_and_its_summary_reads_them() -> None:
    script = _script()
    candidate = {"bic": -10.0, "law_bits": 40.0, "constants": 2, "small_constants": 0, "large_constants": 0, "operations": 3,
                 "steps": 2, "train_r2": 0.99991, "held_r2": 0.9999, "accurate": True, "solved": True, "same_form": False,
                 "text": "1.3*x0/(x0 + 0.7)"}
    rows = [{"arm": arm, "key": f"k{i}", "level": 1, "inputs": 1, "noise": 0.0, "truth_bits": 44.9, "motifs": 3, "seconds": 2.0,
             "solved_place": place, "form_place": None, "accurate_place": 1, "candidates": [candidate]}
            for arm, places in (("taught-5", [None, None, 2, None]), ("taught-20", [1, None, 1, 3]))
            for i, place in enumerate(places)]
    packed = script.pools(rows)
    assert packed["fields"] == list(script.FIELDS) and len(packed["rows"]) == 8
    assert packed["rows"][0][6][0] == [-10.0, 40.0, 2, 0, 0, 3, 2, 0.99991, 0.9999, 1, 1, 0]
    read = script.summary(rows)
    assert read["arms"]["taught-20"]["solved_place"]["found"] == 3 and read["arms"]["taught-5"]["solved_among_the_first_3"] == 1
    step = read["steps"]["taught-20 against taught-5"]
    assert (step["earlier"], step["later"], step["gained"], step["lost"]) == (3, 0, 2, 0)
