"""The fallbacks behind the family reading: what each hands over, and the order they are tried in."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import sympy as sp

from apedp_dyn.discovery_l2 import fallbacks
from apedp_dyn.discovery_l2.across_tables import Table, shape_of
from apedp_dyn.proposers import experience


def _table(names: list[str], law: Any, rows: int = 300, seed: int = 0, noise: float = 0.01) -> Table:
    rng = np.random.default_rng(seed)
    X = rng.uniform(1.0, 5.0, (rows, len(names)))
    clean = law(*X.T)
    y = clean + noise * float(np.sqrt(np.mean(clean ** 2))) * rng.standard_normal(rows)
    return Table("t", names, X, y, X[:50], lambda asked: y[:50][asked])


def _store() -> experience.Experience:
    """Solved tables in x0, x1: a saturating law times a power, and two that are nothing like the test tables."""
    store = experience.Experience()
    rng = np.random.default_rng(9)
    X = rng.uniform(1.0, 5.0, (300, 2))
    for key, law, target in (("saturating", "1.3*x0/(0.7 + x0)*x1**2", 1.3 * X[:, 0] / (0.7 + X[:, 0]) * X[:, 1] ** 2),
                             ("sum", "0.45*x0 + 2.9*sin(1.7*x1)", 0.45 * X[:, 0] + 2.9 * np.sin(1.7 * X[:, 1])),
                             ("decay", "3.7*exp(-0.3*x0)/x1", 3.7 * np.exp(-0.3 * X[:, 0]) / X[:, 1])):
        store.add(key, law, X, target)
    return store


def test_solved_cases_hand_over_their_laws_refitted_here_in_the_table_s_own_names() -> None:
    table = _table(["flow", "width"], lambda a, b: 2.2 * a / (1.9 + a) * b ** 2)
    laws = fallbacks.SolvedCases(_store()).laws(table)
    assert laws and all(shape_of(law, table.names, 60) is not None for law in laws)
    best = sp.sympify(laws[0], locals={n: sp.Symbol(n) for n in table.names})
    predicted = np.asarray(sp.lambdify([sp.Symbol(n) for n in table.names], best, "numpy")(*table.X.T), dtype=float)
    clean = 2.2 * table.X[:, 0] / (1.9 + table.X[:, 0]) * table.X[:, 1] ** 2
    assert float(np.sqrt(np.mean((predicted - clean) ** 2)) / np.sqrt(np.mean(clean ** 2))) < 0.01
    assert len(fallbacks.SolvedCases(_store(), handed=1).laws(table)) == 1


def test_a_case_s_law_is_tried_with_its_inputs_in_every_order() -> None:
    # the same law with the two inputs the other way round: only the swapped reading of the case fits
    table = _table(["width", "flow"], lambda b, a: 2.2 * a / (1.9 + a) * b ** 2)
    best = sp.sympify(fallbacks.SolvedCases(_store()).laws(table)[0], locals={n: sp.Symbol(n) for n in table.names})
    predicted = np.asarray(sp.lambdify([sp.Symbol(n) for n in table.names], best, "numpy")(*table.X.T), dtype=float)
    clean = 2.2 * table.X[:, 1] / (1.9 + table.X[:, 1]) * table.X[:, 0] ** 2
    assert float(np.sqrt(np.mean((predicted - clean) ** 2)) / np.sqrt(np.mean(clean ** 2))) < 0.01


def test_each_fallback_says_why_it_cannot_be_tried() -> None:
    one = _table(["x"], lambda a: 2.0 * a ** 2)
    two = _table(["x", "z"], lambda a, b: 2.0 * a ** 2 * b)
    assert "no solved table of 1 inputs" in fallbacks.SolvedCases(_store()).applies(one)
    assert fallbacks.SolvedCases(_store()).applies(two) is None
    assert "one input" in fallbacks.TensorTrain().applies(one) and fallbacks.TensorTrain().applies(two) is None
    assert "no client" in fallbacks.LanguageModel().applies(two)
    assert fallbacks.LanguageModel(client=lambda prompt: "").applies(two) is None


def test_the_language_model_s_shapes_are_fitted_here_and_cost_two_calls() -> None:
    table = _table(["height", "time"], lambda a, b: 4.9 * b ** 2 + 0.3 * a)
    prompts: list[str] = []

    def client(prompt: str) -> str:
        prompts.append(prompt)
        return "" if "No data is given" in prompt else "c0*x1**2 + c1*x0\nc0*x0*x1\nimport os"

    laws = fallbacks.LanguageModel(client=client).laws(table)
    assert len(prompts) == 2 and "height" not in prompts[1]                 # the model sees x0, x1 and twelve rows
    assert len(laws) == 2
    best = sp.sympify(laws[0], locals={n: sp.Symbol(n) for n in table.names})
    assert best.free_symbols == {sp.Symbol("height"), sp.Symbol("time")}
    assert sorted(round(float(f), 1) for f in best.atoms(sp.Float)) == [0.3, 4.9]


class _Stub:
    def __init__(self, name: str, laws: list[str] | None = None, why_not: str | None = None, fails: bool = False) -> None:
        self.name, self._laws, self._why_not, self._fails, self.asked = name, laws or [], why_not, fails, 0

    def applies(self, table: Table) -> str | None:
        return self._why_not

    def laws(self, table: Table) -> list[str]:
        self.asked += 1
        if self._fails:
            raise RuntimeError("the service is down")
        return list(self._laws)


def test_fallbacks_are_tried_in_order_and_the_chain_stops_at_the_first_reported_law() -> None:
    table = _table(["x", "z"], lambda a, b: 2.0 * a ** 2 * b)
    handed: list[list[str]] = []

    def read(_table: Table, laws: list[str]) -> dict[str, Any]:
        handed.append(list(laws))
        reported = any("x**2*z" in law for law in laws)
        return {"reported": reported, "law": "2.0*x**2*z" if reported else None}

    chain = [_Stub("absent", why_not="not for this table"), _Stub("broken", fails=True), _Stub("empty"),
             _Stub("wrong", ["0.5*x + 1.5*z", "not a law of these inputs: q"]), _Stub("right", ["1.9*x**2*z"]),
             _Stub("never reached", ["3.0*x*z"])]
    reading, trail = fallbacks.try_in_order(table, ["0.1*x - 0.2*z"], chain, read)
    assert reading == {"reported": True, "law": "2.0*x**2*z"}
    assert [step["fallback"] for step in trail] == ["absent", "broken", "empty", "wrong", "right"]
    assert [step["tried"] for step in trail] == [False, True, True, True, True]
    assert trail[0]["why_not"] == "not for this table" and "the service is down" in trail[1]["error"]
    assert trail[2]["proposed"] == 0 and trail[3]["proposed"] == 1 and trail[3]["reported"] is False
    assert trail[4]["reported"] is True and trail[4]["the_reported_law_is_one_of_its_own"] is True
    # what is handed to the route: the fallback's own laws first, then what the table had already been given
    assert handed == [["0.5*x + 1.5*z", "0.1*x - 0.2*z"], ["1.9*x**2*z", "0.1*x - 0.2*z"]]
    assert chain[0].asked == 0 and chain[5].asked == 0


def test_a_chain_that_reports_nothing_says_what_each_did() -> None:
    table = _table(["x", "z"], lambda a, b: 2.0 * a ** 2 * b)
    reading, trail = fallbacks.try_in_order(table, [], [_Stub("one", ["0.5*x + 1.5*z"])], lambda t, laws: {"reported": False, "law": None})
    assert reading is None and trail[0]["reported"] is False and trail[0]["law"] is None
    assert fallbacks.try_in_order(table, [], [], lambda t, laws: pytest.fail("nothing to read")) == (None, [])
