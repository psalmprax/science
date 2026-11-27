"""A family read together: who is read again and with what, where each law came from, and one run through the real route."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.discovery_l2 import across_tables as at
from apedp_dyn.discovery_l2 import family_route
from apedp_dyn.ode.extended_route import replayed

RIGHT, WRONG = "1.0*x0**2/(x1 + 2.0)", "0.5*x0**2 - 0.3*x1"
RIGHT_KEY = at.shape_of(RIGHT, ["x0", "x1"]).key


def _member(c: float, rng: np.random.Generator, noise: float = 0.02, rows: int = 300, pool: int = 100) -> at.Table:
    X = rng.uniform(1.0, 5.0, (rows + pool, 2))
    clean = (0.7 + 0.5 * c) * X[:, 0] ** 2 / (X[:, 1] + 2.3)
    y = clean + noise * float(np.sqrt(np.mean(clean ** 2))) * rng.standard_normal(len(X))
    return at.Table(f"c={c:g}", ["x0", "x1"], X[:rows], y[:rows], X[rows:], lambda asked, y=y, rows=rows: y[rows:][asked],
                    {"T": float(c)})


def _family(members: int = 4, seed: int = 0) -> at.Family:
    rng = np.random.default_rng(seed)
    return at.Family("demo", [_member(float(c), rng) for c in range(1, members + 1)])


class _Referee:
    """In place of the route: reports a law exactly when a law of the right shape is among those handed over."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def __call__(self, table: at.Table, proposer: Any, *, noise: str = "absolute", seed: int = 0) -> dict[str, Any]:
        formulas: list[str] = []
        if proposer is not None:
            frame = pd.DataFrame({**table.columns(), "__target__": table.y})
            formulas = [str(p.formula) for p in proposer.fit_and_propose(frame, target_col="__target__")]
        self.calls.append((table.key, formulas))
        wanted = RIGHT_KEY if "T" not in table.names else None
        right = [f for f in formulas if (at.shape_of(f, table.names).key == wanted if wanted else "T" in f and "x0**2" in f)]
        law = right[0] if right else None
        return {"record": {"vouching": {"in_force": True, "applied": bool(law), "version": "second"},
                           "measured_range": {"state": "inside"}},
                "reported": law is not None, "law": law, "tentative_law": None, "action": "confirmed" if law else "widen_search",
                "formulas": formulas, "listed": list(formulas), "rows_read": 30, "seconds": 0.0}


@pytest.fixture
def referee(monkeypatch: pytest.MonkeyPatch) -> _Referee:
    stub = _Referee()
    monkeypatch.setattr(family_route, "read", stub)
    return stub


def _stage(found_by: set[str]) -> family_route.Stage:
    return lambda table: replayed([RIGHT if table.key in found_by else WRONG])


def test_a_member_whose_search_missed_the_shape_gets_it_from_a_sibling_and_is_read_again(referee: _Referee) -> None:
    out = family_route.discover_family(_family(), _stage({"c=1"}), wider=False)
    assert out["reported_first"] == 1 and out["reported_final"] == 4
    assert [m["source"] for m in out["members"]] == ["route", "siblings", "siblings", "siblings"]
    assert [key for key, _ in referee.calls] == ["c=1", "c=2", "c=3", "c=4", "c=2", "c=3", "c=4"]      # c=1 is not read again
    again = referee.calls[4][1]
    assert again[0] == WRONG and at.shape_of(again[1], ["x0", "x1"]).key == RIGHT_KEY   # its own first, a sibling's beside it
    assert out["members"][1]["handed_from_siblings"] == again[1:] and out["members"][0]["handed_from_siblings"] == []
    assert all(m["passed_over"] == [] for m in out["members"])
    assert out["readings"] == 7 and out["wider"] is None
    assert out["shapes"][0]["shape"] == RIGHT_KEY and out["shapes"][0]["reported_on"] == [0, 1, 2, 3]
    assert out["shapes"][0]["from_members"] == [0] and out["shapes"][0]["seen_on"] == [0]


def test_with_siblings_switched_off_it_is_the_route_table_by_table(referee: _Referee) -> None:
    out = family_route.discover_family(_family(), _stage({"c=1"}), siblings=False, wider=False)
    assert out["reported_final"] == 1 and out["readings"] == 4 and len(referee.calls) == 4
    assert [m["source"] for m in out["members"]] == ["route", None, None, None]


def test_a_family_in_which_no_search_found_the_shape_gets_nothing_from_its_siblings(referee: _Referee) -> None:
    out = family_route.discover_family(_family(), _stage(set()), wider=False)
    assert out["reported_final"] == 0 and all(m["source"] is None for m in out["members"])
    assert out["readings"] == 4            # every member holds the one wrong shape already: nothing comes from a sibling


def test_the_wider_law_is_read_on_the_pooled_table_and_then_written_out_for_a_member_without_a_law(
        referee: _Referee, monkeypatch: pytest.MonkeyPatch) -> None:
    family = _family()
    # one member's sibling reading fails here, so that the wider law has someone left to help
    real = referee.__call__

    def stubborn(table: at.Table, proposer: Any, **kwargs: Any) -> dict[str, Any]:
        reading = real(table, proposer, **kwargs)
        if table.key == "c=3" and len(referee.calls) <= 7:
            reading.update({"reported": False, "law": None, "action": "widen_search"})
        return reading

    monkeypatch.setattr(family_route, "read", stubborn)
    out = family_route.discover_family(family, _stage({"c=1"}))
    wide = out["wider"]
    assert wide["condition"] == "T" and wide["reported"] and "T" in wide["law"] and len(wide["handed"]) >= 1
    pooled_call = next(formulas for key, formulas in referee.calls if key.endswith("pooled over T"))
    assert pooled_call == wide["handed"]
    assert [m["source"] for m in out["members"]] == ["route", "siblings", "wider law", "siblings"]
    last_key, last_laws = referee.calls[-1]
    assert last_key == "c=3" and last_laws[0] == WRONG                      # its own law first, the wider law beside it
    assert "T" not in last_laws[1] and at.shape_of(last_laws[1], ["x0", "x1"]).key == RIGHT_KEY and len(last_laws) == 2
    assert out["reported_final"] == 4 and out["readings"] == 4 + 3 + 1 + 1
    assert wide["source"] == "family" and wide["alone"]["reported"] and wide["searched"] is None and wide["beside"] is None


def test_a_re_reading_that_reports_one_of_the_member_s_own_laws_is_not_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    """What the first development run showed: read again before other rivals, the referee may report a law it refused
    the first time. That is not something the family brought, and the member stays as its first reading left it."""
    calls: list[tuple[str, list[str]]] = []

    def lenient(table: at.Table, proposer: Any, *, noise: str = "absolute", seed: int = 0) -> dict[str, Any]:
        frame = pd.DataFrame({**table.columns(), "__target__": table.y})
        formulas = [str(p.formula) for p in proposer.fit_and_propose(frame, target_col="__target__")]
        first_reading = sum(key == table.key for key, _ in calls) == 0
        calls.append((table.key, formulas))
        # c=1 has the law. Every other member is refused at first; read again it gets its own law back, not the handed one
        law = RIGHT if table.key == "c=1" else None if first_reading else WRONG
        return {"record": {"vouching": {}, "measured_range": {}}, "reported": law is not None, "law": law, "confirmed_as": law,
                "tentative_law": None, "action": "confirmed" if law else "widen_search", "formulas": formulas,
                "listed": list(formulas), "rows_read": 30, "seconds": 0.0}

    monkeypatch.setattr(family_route, "read", lenient)
    out = family_route.discover_family(_family(), _stage({"c=1"}), wider=False)
    assert out["reported_first"] == 1 and out["reported_final"] == 1
    assert [m["source"] for m in out["members"]] == ["route", None, None, None]
    assert [m["final"]["reported"] for m in out["members"]] == [True, False, False, False]
    assert all(m["passed_over"] == [{"reading": "siblings", "law": WRONG}] for m in out["members"][1:])
    # and the same law, had it been handed in, would have counted
    assert family_route.brought_in({"reported": True, "law": WRONG, "confirmed_as": WRONG}, [WRONG], ["x0", "x1"])
    assert not family_route.brought_in({"reported": True, "law": WRONG, "confirmed_as": WRONG}, [RIGHT], ["x0", "x1"])
    assert not family_route.brought_in({"reported": False, "law": None}, [WRONG], ["x0", "x1"])
    # what the referee confirmed decides, not the form the route rewrote it into
    assert family_route.brought_in({"reported": True, "law": WRONG, "confirmed_as": RIGHT}, [RIGHT], ["x0", "x1"])


def test_the_pooled_table_can_be_searched_too_and_the_family_s_law_then_stands_beside_that_search(referee: _Referee) -> None:
    searched_and_missed: family_route.Stage = lambda table: replayed(["0.4*x0 + 0.1*T"])
    out = family_route.discover_family(_family(), _stage({"c=1"}), pooled_stage=searched_and_missed)
    wide = out["wider"]
    assert wide["searched"]["reported"] is False and wide["alone"]["reported"] and wide["beside"]["reported"]
    assert wide["reported"] and wide["source"] == "family" and "T" in wide["law"]
    beside_call = [formulas for key, formulas in referee.calls if key.endswith("pooled over T")][-1]
    assert beside_call[0] == "0.4*x0 + 0.1*T" and beside_call[1] == wide["handed"][0]     # the search's law first
    found: family_route.Stage = lambda table: replayed(["0.9*T*x0**2/(x1 + 1.0)"])
    other = family_route.discover_family(_family(), _stage({"c=1"}), pooled_stage=found)
    assert other["wider"]["source"] == "pooled search" and other["wider"]["beside"] is None
    assert other["wider"]["law"] == "0.9*T*x0**2/(x1 + 1.0)"
    assert other["readings"] == out["readings"] - 1                 # no reading of the two side by side was needed


class _Fallback:
    def __init__(self, name: str, laws: list[str]) -> None:
        self.name, self._laws, self.asked = name, laws, []

    def applies(self, table: at.Table) -> str | None:
        return None

    def laws(self, table: at.Table) -> list[str]:
        self.asked.append(table.key)
        return list(self._laws)


def test_fallbacks_are_only_for_members_still_without_a_law_and_say_which_one_led_to_it(referee: _Referee) -> None:
    empty, right = _Fallback("solved cases", ["0.2*x0 + 0.1*x1"]), _Fallback("tensor train", ["1.3*x0**2/(x1 + 1.0)"])
    out = family_route.discover_family(_family(), _stage(set()), wider=False, fallbacks=[empty, right])
    assert out["reported_first"] == 0 and out["reported_final"] == 4
    assert all(m["source"] == "tensor train" for m in out["members"])
    assert [step["fallback"] for step in out["members"][0]["fallbacks"]] == ["solved cases", "tensor train"]
    assert empty.asked == right.asked == ["c=1", "c=2", "c=3", "c=4"]
    again = family_route.discover_family(_family(), _stage({"c=1", "c=2", "c=3", "c=4"}), wider=False, fallbacks=[empty, right])
    assert again["reported_first"] == 4 and all(m["fallbacks"] == [] for m in again["members"])


def test_one_table_is_the_route_then_its_levels_as_a_family_then_the_fallbacks(referee: _Referee) -> None:
    rng = np.random.default_rng(4)
    rows, pool = 600, 400
    X = np.column_stack([rng.uniform(1.0, 5.0, (rows + pool, 2)), rng.choice([1.0, 2.0, 3.0, 4.0], rows + pool)])
    y = (0.7 + 0.5 * X[:, 2]) * X[:, 0] ** 2 / (X[:, 1] + 2.3)
    table = at.Table("grid", ["x0", "x1", "T"], X[:rows], y[:rows], X[rows:], lambda asked: y[rows:][asked])
    stage: family_route.Stage = lambda t: replayed([WRONG if "T" in t.names else RIGHT])
    out = family_route.discover_table(table, stage)
    assert out["first"]["reported"] is False                       # the route on the whole table: nothing
    assert out["reported"] and out["source"] == "family" and "T" in out["law"]
    assert out["family"]["reported_first"] == 4 and out["family"]["wider"]["reported"]
    smooth = at.Table("smooth", ["x0", "x1"], X[:rows, :2], y[:rows], X[rows:, :2], lambda asked: y[rows:][asked])
    plain = family_route.discover_table(smooth, lambda t: replayed([WRONG]), fallbacks=[_Fallback("solved cases", [RIGHT])])
    assert plain["family"] is None and plain["reported"] and plain["source"] == "solved cases"
    nothing = family_route.discover_table(smooth, lambda t: replayed([WRONG]))
    assert not nothing["reported"] and nothing["source"] is None and nothing["fallbacks"] == []


def test_a_family_through_the_route_in_force() -> None:
    # No stub: the real route and its referee. One member's "search" proposes the shape with rough constants, the
    # others' a wrong law. Each of the others must be reported from its sibling's shape, on its own rows, and the law
    # over the inputs and the condition must be reported on the pooled table.
    family = _family(members=4, seed=1)
    out = family_route.discover_family(family, _stage({"c=1"}))
    assert out["reported_first"] == 1 and out["reported_final"] == 4
    assert [m["source"] for m in out["members"]] == ["route", "siblings", "siblings", "siblings"]
    for member, c in zip(out["members"], (1.0, 2.0, 3.0, 4.0), strict=True):
        shape = at.shape_of(member["final"]["law"], ["x0", "x1"])
        assert shape.key == RIGHT_KEY and shape.start == pytest.approx((0.7 + 0.5 * c, 2.3), rel=0.05)
        assert member["final"]["last_checks"]["applied"] is True
    wide = out["wider"]
    assert wide["reported"] and wide["reading"]["action"] in {"confirmed", "equivalent_class"}
    X = np.random.default_rng(8).uniform(1.0, 5.0, (200, 3))
    import sympy as sp

    law = sp.lambdify([sp.Symbol(n) for n in ("x0", "x1", "T")], sp.sympify(wide["law"]), "numpy")
    truth = (0.7 + 0.5 * X[:, 2]) * X[:, 0] ** 2 / (X[:, 1] + 2.3)
    assert float(np.sqrt(np.mean((np.asarray(law(*X.T), dtype=float) - truth) ** 2)) / np.sqrt(np.mean(truth ** 2))) < 0.01


def test_a_reading_the_route_cannot_make_is_written_down_and_the_family_goes_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """The route in force can end in a fitting error on a law a search proposed. That table then has no law, as in
    the product, and its siblings are still read."""
    real = family_route.vouched_route_v2.discover_law
    broken: list[str] = []

    def discover_law(X: Any, Y: Any, pool_X: Any, measure: Any, **kwargs: Any) -> dict[str, Any]:
        if len(broken) < 1:                          # the first reading of the first member cannot be made
            broken.append("c=1")
            kwargs["proposer"].fit_and_propose(pd.DataFrame({**X, "__target__": Y}), target_col="__target__")
            raise np.linalg.LinAlgError("SVD did not converge in Linear Least Squares")
        return real(X, Y, pool_X, measure, **kwargs)

    monkeypatch.setattr(family_route.vouched_route_v2, "discover_law", discover_law)
    family = _family(members=4, seed=1)
    reading = family_route.read(family.tables[0], replayed([WRONG]))
    assert reading["reported"] is False and reading["action"] == "not_run" and reading["law"] is None
    assert reading["error"] == "LinAlgError: SVD did not converge in Linear Least Squares"
    assert reading["formulas"] == [WRONG] and reading["listed"] == []      # what its search proposed is kept
    assert family_route.summary(reading)["error"] == reading["error"]
    broken.clear()
    out = family_route.discover_family(family, _stage({"c=2"}), wider=False)
    first = out["members"][0]["first"]
    assert first["action"] == "not_run" and "SVD did not converge" in first["error"] and out["reported_first"] == 1
    assert out["members"][1]["source"] == "route" and out["reported_final"] == 4       # the others, and then this one too
    assert out["members"][0]["source"] == "siblings" and out["members"][0]["final"]["error"] is None
