"""The check of whether the route's report on a table depends on the laws that stand beside its own."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import study_list_sensitivity as probe

from apedp_dyn.discovery_l2 import across_tables as at
from apedp_dyn.discovery_l2 import family_route


@pytest.mark.parametrize("inputs", [1, 2])
def test_the_three_lists_hold_the_table_s_ten_laws_and_put_the_six_at_the_head_or_in_turn(inputs: int) -> None:
    names = [f"x{i}" for i in range(inputs)]
    own, other = list(probe.OWN[inputs]), list(probe.UNRELATED[inputs])
    assert len(own) == 10 and len(other) == 6
    shapes = [at.shape_of(law, names) for law in (*own, *other)]
    assert all(shape is not None for shape in shapes) and len({shape.key for shape in shapes}) == 16
    lists = probe.lists(inputs)
    assert set(lists) == set(probe.WAYS) and lists["as_it_is"] == own
    assert lists["unrelated_at_the_head"] == [*other, *own]
    turn = lists["unrelated_in_turn"]
    assert turn[:12:2] == own[:6] and turn[1:12:2] == other and turn[12:] == own[6:]
    # the route keeps the first ten: at the head the six leave room for the table's first four, in turn for its first five
    assert [law in own for law in lists["unrelated_at_the_head"][:10]].count(True) == 4
    assert [law in own for law in turn[:10]].count(True) == 5


def _row(as_it_is: bool, head: bool, turn: bool, accurate: bool = True, unrelated: bool = False) -> dict[str, Any]:
    reading = lambda reported: {"reported": reported, "accurate": reported and accurate, "an_unrelated_law": reported and unrelated}
    return {"noise": 0.01, "as_it_is": reading(as_it_is), "unrelated_at_the_head": reading(head), "unrelated_in_turn": reading(turn)}


def test_gains_and_losses_are_counted_against_the_list_as_it_is() -> None:
    rows = [_row(True, True, True), _row(True, False, True), _row(True, False, False), _row(False, True, False),
            _row(False, False, False), _row(False, True, True, accurate=False, unrelated=True)]
    summary = probe.summarise(rows)
    assert summary["tables"] == 6 and summary["as_it_is"] == {"reported": 3, "reported_and_not_accurate": 0, "reported_an_unrelated_law": 0}
    head = summary["unrelated_at_the_head"]
    assert head["reported"] == 3 and head["gained_a_report"] == 2 and head["lost_a_report"] == 2 and head["verdicts_changed"] == 4
    assert head["of_those_accurate"] == 1 and head["reported_and_not_accurate"] == 1 and head["reported_an_unrelated_law"] == 1
    assert head["of_tables_without_a_report_gained"] == "2 of 3" and head["of_tables_with_a_report_lost"] == "2 of 3"
    turn = summary["unrelated_in_turn"]
    assert turn["gained_a_report"] == 1 and turn["lost_a_report"] == 1 and turn["reported"] == 3


def test_every_table_of_a_family_is_read_the_three_ways(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, list[str]]] = []

    def read(table: at.Table, proposer: Any, *, noise: str = "absolute", seed: int = 0) -> dict[str, Any]:
        frame = pd.DataFrame({**table.columns(), "__target__": table.y})
        formulas = [str(p.formula) for p in proposer.fit_and_propose(frame, target_col="__target__")]
        seen.append((table.key, formulas))
        law = formulas[0] if formulas[0] in probe.UNRELATED[len(table.names)] else None     # reports whatever leads, if unrelated
        return {"reported": law is not None, "law": law, "confirmed_as": law, "action": "confirmed" if law else "widen_search"}

    monkeypatch.setattr(family_route, "read", read)
    rows = probe.one((0, 0.01))
    assert len(rows) == 5 and len(seen) == 15 and len({key for key, _ in seen}) == 5
    assert [len(formulas) for _, formulas in seen[:3]] == [10, 16, 16]
    assert all(set(probe.WAYS) <= set(row) and row["family"] == 0 and row["inputs"] == 1 for row in rows)
    assert all(not row["as_it_is"]["reported"] and not row["unrelated_in_turn"]["reported"] for row in rows)
    assert all(row["unrelated_at_the_head"]["reported"] and row["unrelated_at_the_head"]["an_unrelated_law"]
               and not row["unrelated_at_the_head"]["accurate"] for row in rows)
