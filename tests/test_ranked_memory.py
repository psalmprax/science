"""Pins for the memory's list handed over in a learned order: the rows it sets aside, its order, and that nothing builds it."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.proposers import learned_order, ranked_memory
from apedp_dyn.proposers.structure_memory import Memory
from apedp_dyn.recursion.curriculum import Problem

ROOT = Path(__file__).resolve().parents[1]
FITS = {"grow_fits": 60, "whole_fits": 40}           # a small share of the product's work: enough for a one-input law
TAUGHT = ("0.7*x0/(x0 + 2.3)", "1.7*exp(-0.45*x0)", "2.9*sqrt(x0 + 1.3)", "0.45*x0**2", "1.3*log(1.7*x0 + 1)")


@pytest.fixture(scope="module")
def memory() -> Memory:
    out = Memory()
    for law in TAUGHT:
        out.learn(law)
    return out


@pytest.fixture(scope="module")
def table() -> dict[str, np.ndarray]:
    return Problem("ranked-memory|0", "1.3*x0/(x0 + 0.7)", 1, 1, 0.01).rows()


def test_the_rows_set_aside_are_a_fifth_chosen_by_the_table_itself(table) -> None:
    X, y = table["X"], table["y"]
    fit, held = ranked_memory.split(X, y)
    assert len(held) == round(ranked_memory.HELD_SHARE * len(y)) and len(fit) + len(held) == len(y)
    assert not set(fit) & set(held) and sorted([*fit, *held]) == list(range(len(y)))
    again = ranked_memory.split(X.copy(), y.copy())
    assert np.array_equal(again[0], fit) and np.array_equal(again[1], held)            # the same table, the same rows
    other = ranked_memory.split(X, y + 1e-9)
    assert not np.array_equal(other[1], held)                                          # another table, other rows


def test_the_list_is_fitted_on_the_rows_kept_and_read_on_the_rows_set_aside(memory, table) -> None:
    pool, report = ranked_memory.split_pool(memory, ["x0"], table["X"], table["y"], **FITS)
    assert report["rows_fitted"] == 560 and report["rows_set_aside"] == 140 and len(pool) >= 5
    assert [item["bic"] for item in pool] == sorted(item["bic"] for item in pool)      # as the memory hands it over
    needed = {"text", "bic", "steps", "constants", "small_constants", "large_constants", "operations", "train_r2", "held_r2",
              "law_bits"}
    assert all(set(item) == needed for item in pool)
    assert learned_order.features(pool).shape == (len(pool), len(learned_order.FEATURES))


def test_an_order_decides_which_three_are_handed_over_and_adds_no_candidate(memory, table) -> None:
    frame = pd.DataFrame({"x0": table["X"][:, 0], "target": table["y"]})
    by_rule = ranked_memory.RankedMemoryProposer(memory, **FITS)
    listed = by_rule.ordered(["x0"], table["X"], table["y"])
    proposals = by_rule.fit_and_propose(frame)
    assert len(proposals) == 3 and by_rule.last_run["order"] == "shortest within tolerance" and by_rule.last_run["proposed"] == 3
    assert by_rule.proposer_id == "structure_memory_ranked"

    longest_first = learned_order.LearnedOrder()                 # an order that wants the longest law: the reverse of the rule
    longest_first.weights[learned_order.FEATURES.index("law_bits")] = 1.0
    reverse = ranked_memory.RankedMemoryProposer(memory, longest_first, **FITS)
    other = reverse.ordered(["x0"], table["X"], table["y"])
    assert sorted(other) == sorted(listed) and other != listed   # the same candidates, another order
    assert reverse.last_run["order"] == "learned" and len(reverse.last_run["moved_to_the_front"]) == 3
    pool, _ = ranked_memory.split_pool(memory, ["x0"], table["X"], table["y"], **FITS)
    bits = {item["text"]: item["law_bits"] for item in pool}
    assert bits[other[0]] == max(bits.values())


def test_an_order_is_loaded_from_its_record_or_from_a_record_that_holds_it(tmp_path) -> None:
    order = learned_order.LearnedOrder()
    order.weights[0] = -2.0
    bare, held = tmp_path / "bare.json", tmp_path / "held.json"
    bare.write_text(json.dumps(order.as_dict()))
    held.write_text(json.dumps({"tag": "x", "model": order.as_dict()}))
    assert np.allclose(ranked_memory.load_order(bare).weights, order.weights)
    assert np.allclose(ranked_memory.load_order(held).weights, order.weights)


def test_nothing_in_force_builds_it() -> None:
    """One module of the package builds on it, the stage of the referee test (``ranked_stage.py``), which nothing builds."""
    sources = {path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8") for path in (ROOT / "apedp_dyn").rglob("*.py")}
    users = sorted(name for name, text in sources.items() if "ranked_memory" in text and not name.endswith("/ranked_memory.py"))
    assert users == ["apedp_dyn/discovery_l2/ranked_stage.py"]
    assert [name for name, text in sources.items() if "ranked_stage" in text and not name.endswith("/ranked_stage.py")] == []
