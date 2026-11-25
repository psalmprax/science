"""Pins for the learned order of candidate laws and the rule that stands beside it."""
from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.proposers import learned_order
from apedp_dyn.recursion import order_score


def _pool(rng: np.random.Generator, size: int = 12, solved_at: int | None = 7) -> list[dict]:
    """A list as the memory writes it: best fit first, over-fitted laws ahead, and (at ``solved_at``) one short law that is the answer."""
    pool = []
    for index in range(size):
        answer = index == solved_at
        fit = 0.999999 - 1e-7 * index
        pool.append({"train_r2": 0.99993 if answer else fit, "held_r2": 0.99992 if answer else fit - 2e-6 * rng.random(),
                     "law_bits": 60.0 if answer else float(150 + 10 * rng.integers(0, 8)), "constants": 2 if answer else int(5 + rng.integers(0, 4)),
                     "small_constants": 0 if answer else int(rng.integers(0, 2)), "large_constants": 0, "operations": 4 if answer else 11,
                     "steps": 2, "bic": -9000.0 + index, "solved": answer, "accurate": True, "same_form": answer})
    return pool


def _rows(count: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [{"key": f"p{seed}-{i}", "level": 1 + i % 3, "candidates": _pool(rng, solved_at=None if i % 5 == 0 else int(rng.integers(3, 12))),
             "solved_place": None} for i in range(count)]


def _measured(rows: list[dict]) -> list[dict]:
    return [{**row, "solved_place": learned_order.place_under(row["candidates"], range(len(row["candidates"])))} for row in rows]


def test_features_are_read_beside_the_other_candidates_of_the_list() -> None:
    pool = _pool(np.random.default_rng(0))
    table = learned_order.features(pool)
    assert table.shape == (12, len(learned_order.FEATURES))
    names = dict(zip(learned_order.FEATURES, table[7], strict=True))
    assert names["bits_above_shortest"] == 0.0 and names["fit_gap"] > 0.0          # the answer: the shortest, not the best fit
    assert learned_order.features([]).shape == (0, len(learned_order.FEATURES))
    assert learned_order._digits(None) == 0.0 and learned_order._digits(0.999) == pytest.approx(3.0)


def test_the_rule_puts_the_shortest_within_tolerance_first_and_keeps_the_rest_in_place() -> None:
    pool = _pool(np.random.default_rng(1))
    assert learned_order.simplest_within(pool, tolerance=5.0)[0] == 7             # a tolerance wide enough to reach the answer
    tight = learned_order.simplest_within(pool, tolerance=0.0)
    assert sorted(tight) == list(range(12)) and tight[0] != 7                     # the answer fits worse: outside a zero tolerance
    assert learned_order.simplest_within([]) == []


def test_an_untrained_order_is_the_lists_own_order() -> None:
    pool = _pool(np.random.default_rng(2))
    model = learned_order.LearnedOrder()
    assert model.order(pool) == list(range(12))
    assert learned_order.LearnedOrder().fit([]).order(pool) == list(range(12))
    all_wrong = [[{**item, "solved": False} for item in pool]]
    assert learned_order.LearnedOrder().fit(all_wrong).order(pool) == list(range(12))   # nothing solved: nothing to learn


def test_what_it_learns_on_some_problems_brings_the_answer_forward_on_others() -> None:
    model = learned_order.LearnedOrder().fit([row["candidates"] for row in _rows(60, seed=3)])
    assert model.trained_on["problems"] == 60 and model.trained_on["solved"] == 48
    fresh = _measured(_rows(40, seed=4))
    learned = learned_order.reorder(fresh, model)
    before, after = order_score.summarise(fresh), order_score.summarise(learned)
    assert before["first"] == 0 and after["first"] == after["found"] == before["found"] == 32
    compared = order_score.compare(fresh, learned)
    assert compared["earlier"] == 32 and compared["later"] == 0 and compared["sign_test_p"] < 1e-6
    assert compared["gained"] == compared["lost"] == 0                            # an order moves candidates; it adds none


def test_its_state_is_a_few_numbers_and_comes_back_the_same() -> None:
    model = learned_order.LearnedOrder().fit([row["candidates"] for row in _rows(30, seed=5)])
    again = learned_order.LearnedOrder.from_dict(model.as_dict())
    pool = _pool(np.random.default_rng(6))
    assert again.order(pool) == model.order(pool) and np.allclose(again.scores(pool), model.scores(pool))
    with pytest.raises(ValueError, match="other features"):
        learned_order.LearnedOrder.from_dict({**model.as_dict(), "features": ["place"]})


def test_out_of_fold_scores_every_problem_once_with_an_order_learned_elsewhere() -> None:
    rows = _measured(_rows(50, seed=7))
    scored = learned_order.out_of_fold(rows, folds=5, seed=1)
    assert [row["key"] for row in scored] == sorted(row["key"] for row in rows)
    assert order_score.summarise(scored)["first"] == 40
