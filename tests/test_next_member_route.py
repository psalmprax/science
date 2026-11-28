"""The next law of a class when the one the route took fails the residual check (experimental, not in force)."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from apedp_dyn.discovery_l2 import next_member_route as nm
from apedp_dyn.discovery_l2 import vouching_v2

RNG = np.random.default_rng(0)
X = {"x": RNG.uniform(1.0, 5.0, 400)}
Y = 2.0 * X["x"] + 1.0 + 0.01 * RNG.standard_normal(400)
POOL = RNG.uniform(1.0, 5.0, (60, 1))
WRONG, CURVED, RIGHT, LONGER = "0.55*x**2", "1.9*x**1.1", "2.0*x + 1.0", "2.0*x + 1.0 + 1.0e-9*x**3"


def _record(*, action: str = "widen_search", rejected: str | None = WRONG, verdict: str = "equivalent_class", route: str = "refined",
            survivor: str = WRONG, equivalent: tuple[str, ...] = (LONGER, CURVED, RIGHT)) -> dict[str, Any]:
    contrast = SimpleNamespace(verdict=verdict, survivor=survivor, equivalent=list(equivalent))
    path = SimpleNamespace(action=action, rejected_law=rejected, contrast=contrast,
                           candidates=[{"formula": law} for law in (WRONG, LONGER, CURVED, RIGHT)])
    return {"route": route, "main_search": path, "law": None, "reported": False, "action": action, "tentative_law": None,
            "vouching": {"in_force": True, "version": "second", "applied": False, "why_not": "the route reported no law"}}


@pytest.fixture
def vouched(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list[str]]]:
    """The last checks vouch for whatever they are handed, and say what that was."""
    seen: list[tuple[str, list[str]]] = []

    def vouch(law: str, others: Any, names: Any, X_: Any, y: Any, more: Any = None, **_: Any) -> Any:
        seen.append((law, list(others)))
        return vouching_v2.Vouched(law, "")

    monkeypatch.setattr(nm.vouching_v2, "vouch", vouch)
    return seen


@pytest.mark.parametrize("change, why", [
    ({"route": "main"}, "refined path only"), ({"action": "confirmed"}, "did not end at the residual check"),
    ({"rejected": None}, "did not end at the residual check"), ({"verdict": "unique"}, "not taken from a class"),
])
def test_a_reading_that_ended_any_other_way_is_returned_as_it_is(change: dict[str, Any], why: str, vouched: list[Any]) -> None:
    record = _record(**change)
    out = nm.next_member(record, X, Y, POOL, searched=True)
    assert out["next_member"]["applied"] is False and why in out["next_member"]["why_not"] and not vouched
    assert {k: v for k, v in out.items() if k != "next_member"} == record
    unsearched = nm.next_member(_record(), X, Y, POOL, searched=False)
    assert unsearched["next_member"]["applied"] is False and "no search stage" in unsearched["next_member"]["why_not"]
    for asked in ("first", "off"):                                              # the route was asked for other last checks, or none
        other = {**_record(), "vouching": {"in_force": asked != "off", "version": asked, "applied": False}}
        assert "second version" in nm.next_member(other, X, Y, POOL, searched=True)["next_member"]["why_not"] and not vouched
    reported = {**_record(), "reported": True, "law": WRONG}                    # the step adds a report and never takes one away
    kept = nm.next_member(reported, X, Y, POOL, searched=True)
    assert kept["law"] == WRONG and kept["reported"] is True and kept["next_member"]["why_not"] == "the route reported a law"


def test_the_class_s_other_laws_are_tried_shortest_first_and_the_first_with_clean_residuals_goes_to_the_last_checks(
        vouched: list[tuple[str, list[str]]]) -> None:
    out = nm.next_member(_record(), X, Y, POOL, searched=True)
    state = out["next_member"]
    assert state["applied"] and state["class"] == 4 and state["rejected_first"] == WRONG
    # the power law is the shortest and fails the residual check; the line is next and passes; the longer line is not reached
    assert [t["residual_check"] for t in state["tried"]] == [False, True] and state["took"] == 2
    assert out["reported"] is True and out["law"] == RIGHT and out["action"] == "equivalent_class" and out["tentative_law"] is None
    law, rivals = vouched[0]
    assert law == RIGHT and set(rivals) == {WRONG, LONGER, CURVED}        # every other member is its rival, the failed one too


def test_a_law_the_last_checks_do_not_vouch_for_is_held_back_and_nothing_further_is_tried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nm.vouching_v2, "vouch", lambda *a, **k: vouching_v2.Vouched(None, "2 candidates passed every test"))
    out = nm.next_member(_record(), X, Y, POOL, searched=True)
    assert out["reported"] is False and out["law"] is None and out["tentative_law"] == RIGHT
    assert out["tentative_reason"].startswith(nm.vouched_route_v2.NOT_VOUCHED) and len(out["next_member"]["tried"]) == 2

    def crash(*_: Any, **__: Any) -> Any:
        raise FloatingPointError("singular")

    monkeypatch.setattr(nm.vouching_v2, "vouch", crash)
    crashed = nm.next_member(_record(), X, Y, POOL, searched=True)
    assert crashed["reported"] is False and "the checks failed: FloatingPointError" in crashed["tentative_reason"]


def test_a_member_the_last_checks_choose_in_its_place_has_to_pass_the_residual_check_too(monkeypatch: pytest.MonkeyPatch) -> None:
    def to(other: str) -> Any:
        return lambda *a, **k: vouching_v2.Vouched(other, "another member of the class is reported: the first failed a test", True)

    monkeypatch.setattr(nm.vouching_v2, "vouch", to(WRONG))                     # the member the main search had rejected
    held = nm.next_member(_record(), X, Y, POOL, searched=True)
    assert held["reported"] is False and held["tentative_law"] == RIGHT and "fails the residual check" in held["tentative_reason"]
    assert held["next_member"]["tried"][-1] == {"residual_check": True, "vouched": False, "switched": True,
                                                "reason": "the member the checks chose in its place fails the residual check"}
    monkeypatch.setattr(nm.vouching_v2, "vouch", to(LONGER))                    # a member whose residuals are clean
    moved = nm.next_member(_record(), X, Y, POOL, searched=True)
    assert moved["reported"] is True and moved["law"] == LONGER and moved["next_member"]["tried"][-1]["switched"] is True


def test_a_class_with_no_law_of_clean_residuals_stays_without_a_law(vouched: list[Any]) -> None:
    out = nm.next_member(_record(equivalent=(CURVED, "0.5*x**3")), X, Y, POOL, searched=True)
    assert out["reported"] is False and out["law"] is None and out["action"] == "widen_search" and not vouched
    assert [t["residual_check"] for t in out["next_member"]["tried"]] == [False, False]
    unreadable = nm.next_member(_record(equivalent=("log(x - 9)",)), X, Y, POOL, searched=True)       # not finite on these rows
    assert unreadable["next_member"]["tried"] == [{"residual_check": None}] and unreadable["reported"] is False


def test_the_real_last_checks_decide_when_they_are_not_stood_in_for() -> None:
    out = nm.next_member(_record(equivalent=(RIGHT,)), X, Y, POOL, searched=True)
    tried = out["next_member"]["tried"]
    assert tried[0]["residual_check"] is True and isinstance(tried[0]["vouched"], bool)
    assert out["reported"] is tried[0]["vouched"]


def test_the_route_is_called_once_and_then_the_step(monkeypatch: pytest.MonkeyPatch, vouched: list[Any]) -> None:
    calls: list[dict[str, Any]] = []

    def route(X_: Any, Y_: Any, pool: Any, measure: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return _record()

    monkeypatch.setattr(nm.vouched_route_v2, "discover_law", route)
    out = nm.discover_law(X, Y, POOL, lambda rows: Y[:len(rows)], proposer=object(), noise="absolute", seed=3)  # type: ignore[arg-type]
    assert len(calls) == 1 and calls[0]["noise"] == "absolute" and calls[0]["seed"] == 3 and out["law"] == RIGHT
    assert nm.discover_law(X, Y, POOL, lambda rows: Y[:len(rows)])["next_member"]["why_not"].startswith("no search stage")
    assert nm.NEIGHBOURS == 8
