"""The routes put in force by the runs of 2026-11-22, held to what those runs ran.

Status: audit-only

``ode/route_in_force.py`` and ``discovery_l2/stage_in_force.py`` are thin layers around modules a
preregistration froze. These tests hold each to the composition the frozen runner used, and to
the rules the result records state: no reinstatement by default, nothing reported without a
search, an earlier route handed on as it was measured, the memory beside the searches and never
in their place.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from apedp_dyn.discovery_l2 import memory_stage, stage_in_force
from apedp_dyn.discovery_l2.evolved_stage import SideBySide
from apedp_dyn.ode import default_route, experiment_pruning, extended_route, route_in_force
from apedp_dyn.ode.window_rows import WindowRows

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def toy():
    system = {**_load("run_odebench_main_search").toy_systems()[1], "part": "toy"}     # dx0 = 1 - x0*x1, dx1 = x0 - 0.5*x1
    return system, _load("odebench_worlds").world(system, 0.0, tag="test")


def _rate(states: np.ndarray) -> np.ndarray:
    states = np.atleast_2d(states)
    return np.column_stack([1.0 - states[:, 0] * states[:, 1], states[:, 0] - 0.5 * states[:, 1]])


FORMULAS = ["1 - x0*x1", "x0 + x1"]


# ------------------------------------------------------------------ trajectories
def test_the_ode_route_in_force_is_the_second_engine_arm_as_the_frozen_runner_composed_it(toy):
    _, world = toy
    found, done = route_in_force.discover_component(world["dev"], world["pool"], 0, proposer=extended_route.replayed(FORMULAS))
    arm, arm_done = extended_route.discover_component(world["dev"], world["pool"], 0, proposer=extended_route.replayed(FORMULAS),
                                                      seed=0, second_engine=True, at_precision=False)
    assert (found.law, found.action, found.unpruned_law) == (arm.law, arm.action, arm.unpruned_law) and found.law
    assert done == {**arm_done, "route": "second_engine"} and done["second_engine"] is True and done["at_precision"] is False
    assert "experiment" not in done and json.dumps(done)
    assert route_in_force.DEFAULT_ROUTE == "second_engine" and route_in_force.route_in_force() == "second_engine"


def test_with_a_system_that_can_be_started_again_the_reported_law_goes_through_the_experiment(toy):
    _, world = toy
    dev = world["dev"]
    rates = _rate(dev.states)
    rates[:, 0] += 2e-3 * dev.states[:, 0] ** 3 * dev.states[:, 1] ** 2      # an error of the rates that a term can fit
    padded_rows = WindowRows(dev.states, rates, dev.times, dev.degree, dev.fit_rms, dev.rate_sd)
    restart = route_in_force.Restart(_rate, 0.0, world["spread"])
    stage = extended_route.replayed(["1.0 - 1.0*x0*x1 + 2e-3*x0**3*x1**2", "x0 + x1"])
    found, done = route_in_force.discover_component(padded_rows, world["pool"], 0, proposer=stage, restart=restart, seed=1)
    plain, _ = route_in_force.discover_component(padded_rows, world["pool"], 0, proposer=stage, seed=1)
    by_hand, record = experiment_pruning.decided(plain, padded_rows, 0, _rate, level=0.0, spread=world["spread"], seed=1,
                                                 chosen=True)
    assert found.law == by_hand.law and done["experiment"]["dropped"] == record["experiment"]["dropped"]
    assert done["experiment"]["chosen"] is True
    if plain.law and "x0**3" in plain.law:                               # where the padded law was reported, the term goes
        assert done["experiment"]["dropped"] and "x0**3" not in (found.law or "")


def test_the_reinstatement_is_available_and_off(toy):
    _, world = toy
    stage = extended_route.replayed(FORMULAS)
    _, off = route_in_force.discover_component(world["dev"], world["pool"], 0, proposer=stage)
    _, on = route_in_force.discover_component(world["dev"], world["pool"], 0, proposer=stage, at_precision=True)
    assert off["at_precision"] is False and "precision" not in off
    assert on["at_precision"] is True and "precision" in on


def test_without_a_search_nothing_is_reported_and_the_second_engine_does_not_stand_in(toy):
    _, world = toy
    found, done = route_in_force.discover_component(world["dev"], world["pool"], 0, proposer=None)
    earlier = default_route.discover_component(world["dev"], world["pool"], 0, proposer=None)
    assert found.law is None and found.law == earlier.law and found.tentative_law == earlier.tentative_law
    assert done == {"route": "second_engine", "second_engine": False}
    none_by_budget, _ = route_in_force.discover_component(world["dev"], world["pool"], 0, budget_seconds=0.0)
    assert none_by_budget.law is None


def test_an_earlier_route_is_handed_on_as_it_was_measured(toy, monkeypatch):
    _, world = toy
    stage = extended_route.replayed(FORMULAS)
    restart = route_in_force.Restart(_rate, 0.0, world["spread"])
    found, done = route_in_force.discover_component(world["dev"], world["pool"], 0, proposer=stage, route_name="repaired",
                                                    restart=restart)
    earlier = default_route.discover_component(world["dev"], world["pool"], 0, proposer=stage, route_name="repaired")
    assert (found.route, found.law, found.action) == (earlier.route, earlier.law, earlier.action)
    assert done == {"route": "repaired", "second_engine": False}
    monkeypatch.setenv(route_in_force.ENVIRONMENT_VARIABLE, "repaired")
    assert route_in_force.route_in_force() == "repaired"
    monkeypatch.setenv(route_in_force.ENVIRONMENT_VARIABLE, "something else")
    with pytest.raises(ValueError, match="must be one of"):
        route_in_force.route_in_force()
    with pytest.raises(ValueError, match="route must be one of"):
        route_in_force.discover_component(world["dev"], world["pool"], 0, proposer=stage, route_name="nothing")


def test_the_route_states_its_evidence_and_the_record_exists():
    for evidence in (route_in_force.EVIDENCE, route_in_force.EARLIER_EVIDENCE, stage_in_force.EVIDENCE):
        assert (ROOT / evidence).is_file(), evidence
    text = (ROOT / route_in_force.EVIDENCE).read_text(encoding="utf-8")
    assert "Deviation, on the cautious side" in text and "one of them is the law" in text
    for needed in ("120 recovered, 165 reported, 3 wrong", "at_precision=True", "not claimed that"):
        assert needed in (route_in_force.__doc__ or ""), needed


# ------------------------------------------------------------------ tables
def test_the_table_stage_in_force_is_the_memory_stage_the_run_measured(monkeypatch):
    calls: list[tuple[str, float, dict]] = []
    monkeypatch.setattr(memory_stage, "memory_search_stage",
                        lambda budget, **kw: calls.append(("memory", budget, kw)) or ("with memory", "described"))
    monkeypatch.setattr(stage_in_force, "default_search_stage",
                        lambda budget, **kw: calls.append(("default", budget, kw)) or ("searches alone", "plain"))
    monkeypatch.delenv(stage_in_force.ENVIRONMENT_VARIABLE, raising=False)
    assert stage_in_force.memory_in_force() and stage_in_force.search_stage(30.0, cores=4, seed=2) == ("with memory", "described")
    monkeypatch.setenv(stage_in_force.ENVIRONMENT_VARIABLE, "OFF")
    assert not stage_in_force.memory_in_force() and stage_in_force.search_stage(30.0) == ("searches alone", "plain")
    assert calls == [("memory", 30.0, {"cores": 4, "seed": 2}), ("default", 30.0, {"cores": None, "seed": 0})]


def test_the_memory_stands_beside_the_searches_and_the_stage_says_so(monkeypatch):
    class Searches:
        proposer_id = "searches"

    monkeypatch.setattr(memory_stage, "default_search_stage", lambda budget, **kw: (Searches(), "four searches"))
    monkeypatch.delenv(stage_in_force.ENVIRONMENT_VARIABLE, raising=False)
    stage, described = stage_in_force.search_stage(30.0)
    assert isinstance(stage, SideBySide) and described.startswith("four searches; and laws grown from ")
    assert "remembered motifs" in described
    for needed in ("23", "17", "one grammar", "APEDP_TABLE_MEMORY=off"):
        assert needed in (stage_in_force.__doc__ or ""), needed


def test_the_endpoint_builds_the_stage_in_force(monkeypatch):
    from apedp_dyn.proposers import hard_exit
    from apedp_dyn.ui import route_endpoint

    monkeypatch.setattr(stage_in_force, "search_stage", lambda budget, **kw: ("the stage in force", f"in force, {budget:g} s"))
    monkeypatch.setitem(sys.modules, "pysr", type(sys)("pysr"))          # the endpoint only asks whether it can be loaded
    # since 2026-11-25 the endpoint builds the same stage with search processes that cannot stop a reading for ever
    # (tests/test_hard_exit.py); the stage as frozen is built where the environment asks for it
    monkeypatch.setenv(hard_exit.ENVIRONMENT_VARIABLE, hard_exit.WAIT)
    assert route_endpoint._proposer(12.0) == ("the stage in force", "in force, 12 s")
    assert route_endpoint._proposer(0.0)[0] is None
    block = route_endpoint.status_block()
    assert "RESULT-MEMORY-2026-11.md" in block["search_stage_measured"]
    ode = next(entry for entry in block["routes"] if entry["name"] == "ode")
    assert ode["evidence"] == route_in_force.EVIDENCE and "3 of 165" in ode["measured"]


# ------------------------------------------------------------------ the filed forecasts
def test_the_filed_forecasts_are_scored_as_recorded():
    forecasts = _load("forecast_runs")
    recorded = json.loads((ROOT / "benchmarks" / "outcome_forecast_scored.json").read_text(encoding="utf-8"))
    assert recorded["forecasts"] == [forecasts.score(*entry) for entry in forecasts.FILED]
    assert recorded["totals_inside_their_intervals"] == [5, 6]
    memory = recorded["forecasts"][1]["outcomes"]["recovered"]
    assert (memory["seen"], memory["inside"], memory["interval"]) == (17, False, [37, 62])
