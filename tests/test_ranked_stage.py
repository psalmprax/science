"""Pins for the stage with the memory's list in a learned order beside it: what is added, what is displaced, and that nothing builds it."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from apedp_dyn.discovery_l2 import ranked_stage
from apedp_dyn.discovery_l2.search_stage import MAX_PROPOSALS
from apedp_dyn.proposers.portfolio import FixedFormulasProposer
from apedp_dyn.proposers.structure_memory import Memory
from apedp_dyn.recursion.curriculum import Problem

ROOT = Path(__file__).resolve().parents[1]
STAGE = [f"{1.5 + k}*x0**{k + 1} + 0.5*x1" for k in range(MAX_PROPOSALS)]          # ten shapes, as the stage in force hands over


def _frame() -> pd.DataFrame:
    rows = Problem("ranked-stage|0", "1.3*x0/(x0 + 0.7) + 0.9*x1", 2, 2, 0.0).rows()
    return pd.DataFrame({"x0": rows["X"][:, 0], "x1": rows["X"][:, 1], "__target__": rows["y"]})


def test_two_laws_of_one_shape_differ_only_in_their_constants() -> None:
    names = ["x0", "x1"]
    assert ranked_stage.shape("1.3*x0/(x0 + 0.7)", names) == ranked_stage.shape("2.9*x0/(x0 + 1.31)", names)
    assert ranked_stage.shape("1.3*x0/(x0 + 0.7)", names) != ranked_stage.shape("1.3*x1/(x1 + 0.7)", names)
    assert ranked_stage.shape("x0**2", names) != ranked_stage.shape("x0**3", names)       # a whole exponent is part of the shape
    assert ranked_stage.shape("not ( a law", names) == "not ( a law"


def test_the_first_new_shapes_are_added_and_take_the_place_of_the_stage_s_last() -> None:
    offered = ["7.7*x0**2 + 0.1*x1",                       # the stage's second shape with other constants: not new
               "1.1*sin(0.3*x0)", "2.2*exp(-0.4*x1)", "3.3*log(x0 + 1.5)", "4.4*sqrt(x1 + 0.2)"]
    stage = ranked_stage.WithRanked(FixedFormulasProposer(STAGE), FixedFormulasProposer(offered))
    out = [str(p.formula) for p in stage.fit_and_propose(_frame(), target_col="__target__")]
    assert out == [*STAGE[:7], *offered[1:4]] and len(out) == MAX_PROPOSALS
    report = stage.last_run["ranked"]
    assert report["added"] == offered[1:4] and report["displaced"] == STAGE[7:] and report["offered"] == 5
    assert report["handed_by_the_stage"] == MAX_PROPOSALS and stage.failures == []


def test_nothing_is_displaced_while_the_stage_s_list_has_room() -> None:
    stage = ranked_stage.WithRanked(FixedFormulasProposer(STAGE[:6]), FixedFormulasProposer(["1.1*sin(0.3*x0)", "2.2*exp(-0.4*x1)"]))
    out = [str(p.formula) for p in stage.fit_and_propose(_frame(), target_col="__target__")]
    assert out == [*STAGE[:6], "1.1*sin(0.3*x0)", "2.2*exp(-0.4*x1)"] and stage.last_run["ranked"]["displaced"] == []
    nothing_new = ranked_stage.WithRanked(FixedFormulasProposer(STAGE), FixedFormulasProposer(["9.9*x0**3 + 0.2*x1"]))
    assert [str(p.formula) for p in nothing_new.fit_and_propose(_frame(), target_col="__target__")] == STAGE


def test_a_ranked_proposer_that_fails_costs_the_stage_nothing() -> None:
    stage = ranked_stage.WithRanked(FixedFormulasProposer(STAGE), FixedFormulasProposer([], fail="no list today"))
    out = [str(p.formula) for p in stage.fit_and_propose(_frame(), target_col="__target__")]
    assert out == STAGE and len(stage.failures) == 1 and "no list today" in stage.failures[0]
    assert stage.last_run["ranked"]["added"] == [] and stage.last_run["ranked"]["displaced"] == []


def test_the_ranked_proposer_alone_stands_behind_the_fit_guard_and_proposes_from_the_memory() -> None:
    memory = Memory()
    for law in ("0.7*x0/(x0 + 2.3)", "1.7*exp(-0.45*x0)", "2.9*sqrt(x0 + 1.3)", "0.45*x0**2", "1.3*log(1.7*x0 + 1)"):
        memory.learn(law)
    rows = Problem("ranked-stage|1", "1.3*x0/(x0 + 0.7)", 1, 1, 0.01).rows()
    frame = pd.DataFrame({"x0": rows["X"][:, 0], "__target__": rows["y"]})
    alone = ranked_stage.with_ranked(None, memory, None, offered=4)
    proposals = alone.fit_and_propose(frame, target_col="__target__")
    assert 1 <= len(proposals) <= 4 and type(alone).__name__ == "FitGuardProposer"
    assert ranked_stage.shape("1.3*x0/(x0 + 0.7)", ["x0"]) in {ranked_stage.shape(str(p.formula), ["x0"]) for p in proposals}


def test_nothing_in_force_builds_the_ranked_stage() -> None:
    users = sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / "apedp_dyn").rglob("*.py")
                   if path.name != "ranked_stage.py" and "ranked_stage" in path.read_text(encoding="utf-8"))
    assert users == []                                    # its runner script and these tests are its only callers
