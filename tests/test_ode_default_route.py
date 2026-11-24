"""The ODE route in force: that it is the route the blind test measured, composed the same way.

Nothing here runs a search engine or a system of the blind set. The route is exercised on the
frozen runner's toy systems, with a fixed list of candidates in place of the searches.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from apedp_dyn.ode import default_route
from apedp_dyn.proposers.fit_guard import FitGuardProposer
from apedp_dyn.proposers.portfolio import FixedFormulasProposer

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ["1 - x0*x1", "x0 + x1", "log(x0 - 50)"]


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
def world():
    system = _load("run_odebench_main_search").toy_systems()[1]          # dx0 = 1 - x0*x1, dx1 = x0 - 0.5*x1
    return _load("odebench_worlds").world(system, 0.0, tag="test")


def _stage() -> FitGuardProposer:
    return FitGuardProposer(FixedFormulasProposer(CANDIDATES))


def test_the_default_is_the_repaired_route_and_the_environment_can_restore_the_recorded_one(monkeypatch):
    monkeypatch.delenv(default_route.ENVIRONMENT_VARIABLE, raising=False)
    assert default_route.DEFAULT_ROUTE == default_route.REPAIRED == default_route.default_route()
    monkeypatch.setenv(default_route.ENVIRONMENT_VARIABLE, "main_search")
    assert default_route.default_route() == default_route.RECORDED
    monkeypatch.setenv(default_route.ENVIRONMENT_VARIABLE, "something_else")
    with pytest.raises(ValueError, match="must be one of"):
        default_route.default_route()


def test_the_route_gives_what_the_frozen_runner_of_the_blind_test_gave(world, monkeypatch):
    runner = _load("run_ode_blind")
    monkeypatch.setattr(runner, "STAGE", lambda budget: _stage())
    measured = runner.run_repaired(world, 0, budget=1.0)
    monkeypatch.delenv(default_route.ENVIRONMENT_VARIABLE, raising=False)
    here = default_route.discover_component(world["dev"], world["pool"], 0, proposer=_stage())
    assert here.route == default_route.REPAIRED and here.law and here.law == measured["formula"]
    assert here.unpruned_law == measured["extra"]["unpruned_law"] and here.action == measured["extra"]["action"]
    assert here.precision_dropped == measured["extra"]["precision_dropped"] and here.prune_error is None
    assert here.fit_guard_dropped == measured["extra"]["fit_guard_dropped"]
    assert [formula for formula, _reason in here.fit_guard_dropped] == ["log(x0 - 50)"]     # undefined on these rows
    assert here.tentative_law is None and here.result.path is not None


def test_the_settings_are_the_blind_tests(monkeypatch):
    runner = _load("run_ode_blind")
    ode_run = _load("run_odebench_main_search")
    assert default_route.MEMBERS == runner.MEMBERS == 4 and default_route.SEARCH_BUDGET_S == ode_run.SEARCH_BUDGET_S
    assert default_route.search_stage(0.0) is None
    stage = default_route.search_stage(240.0, seed=7)
    assert isinstance(stage, FitGuardProposer) and len(stage._inner._members) == 4
    assert stage._inner._max == default_route.MAX_PROPOSALS == 10
    assert stage._inner._timeout == 240.0 + default_route.MEMBER_GRACE_S == 540.0


def test_without_a_search_stage_nothing_is_reported(world, monkeypatch):
    monkeypatch.delenv(default_route.ENVIRONMENT_VARIABLE, raising=False)
    found = default_route.discover_component(world["dev"], world["pool"], 0, proposer=None)
    assert found.law is None and found.unpruned_law is None and found.precision_dropped == []
    again = default_route.discover_component(world["dev"], world["pool"], 0, budget_seconds=0.0)
    assert again.law is None and again.tentative_law == found.tentative_law


def test_the_recorded_route_is_handed_on_as_it_was_measured(world, monkeypatch):
    monkeypatch.setenv(default_route.ENVIRONMENT_VARIABLE, default_route.RECORDED)
    found = default_route.discover_component(world["dev"], world["pool"], 0, proposer=FixedFormulasProposer(CANDIDATES[:2]))
    assert found.route == default_route.RECORDED and found.unpruned_law is None and found.precision_dropped == []
    assert found.law and "x0*x1" in found.law and found.result.action == found.action
    with pytest.raises(ValueError, match="route must be one of"):
        default_route.discover_component(world["dev"], world["pool"], 0, proposer=None, route_name="nope")


def test_the_evidence_is_on_file_and_the_module_states_both_runs():
    for path in (default_route.EVIDENCE, default_route.EARLIER_EVIDENCE):
        assert (ROOT / path).exists(), path
    doc = default_route.__doc__ or ""
    for phrase in ("PREREG-ODE-BLIND-2026-11", "3 were", "199 and 210", "132 and 201", "PREREG-ODE-REFINED-2026-11",
                   "10 of the 195", "5.1 %", "not adopted", "four cores", "M-26"):
        assert phrase in doc, phrase
    summary = json.loads((ROOT / "preregistration/RESULT-ODE-BLIND-2026-11/results.json").read_text())["summary"]
    expectations = summary["expectations"]
    assert all(expectations[f"{k}_holds"] for k in (1, 2, 3, 4))
    assert expectations["2_recovered_without_noise"] == [199, 132] and expectations["3_recovered_with_noise"] == [210, 201]
    assert summary["repaired|all"]["reported"] == 484 and summary["repaired|all"]["reported_wrong"] == 3
    assert summary["repaired|all"]["recoveries_gained_by_pruning"] == 24
    assert summary["repaired|all"]["recoveries_lost_by_pruning"] == 0
