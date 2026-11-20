"""The blind test of the repaired ODE route: its systems, its runner, its summary.

Nothing here runs a discovery on a system of the blind set. The route is exercised on the frozen
runner's toy systems, with a fixed list of candidates in place of the search engine.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import sympy as sp

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
def runner():
    return _load("run_ode_blind")


@pytest.fixture(scope="module")
def builder():
    return _load("build_dysts_polynomial_systems")


# --- the systems ----------------------------------------------------------------------------------

def test_every_system_is_a_polynomial_field_of_three_or_four_variables(runner, builder):
    record = json.loads(runner.SYSTEMS_FILE.read_text(encoding="utf-8"))
    systems = runner.systems()
    # 100 pass the rule; the frozen world builder gives no rows in time for two, which are not run
    assert len(record["systems"]) == 100 and sorted(record["not_buildable"]) == ["Malasoma", "SprottJerk"]
    assert len(systems) == 98 and sum(s["dim"] for s in systems) == 310
    assert [s["id"] for s in record["systems"]] == list(range(1000, 1100))
    assert not {s["name"] for s in systems} & set(record["not_buildable"])
    assert len({s["name"] for s in systems}) == 98 and [s["name"] for s in systems] == sorted(s["name"] for s in systems)
    for system in systems:
        symbols = [sp.Symbol(f"x{i}") for i in range(system["dim"])]
        assert system["dim"] in (3, 4) and len(system["components"]) == system["dim"]
        assert len(system["inits"]) == 2 and all(len(init) == system["dim"] for init in system["inits"])
        assert np.all(np.isfinite(system["inits"])) and system["period"] > 0
        for component in system["components"]:
            poly = sp.Poly(component, *symbols)                    # raises if not a polynomial in the state
            assert poly.total_degree() <= builder.MAX_DEGREE and len(poly.terms()) <= builder.MAX_TERMS
            assert all(power == int(power) for monomial in poly.monoms() for power in monomial)


def test_no_system_of_the_blind_set_is_one_the_routes_were_built_on(runner, builder):
    record = json.loads(runner.SYSTEMS_FILE.read_text(encoding="utf-8"))
    names = {s["name"] for s in record["systems"]}
    assert not names & set(builder.IN_ODEBENCH) and not names & set(builder.SAME_TERMS_AS_ODEBENCH)
    assert all(record["excluded"][name] == "in ODEBench" for name in builder.IN_ODEBENCH if name in record["excluded"])
    assert all(record["excluded"][name] == "same terms as an ODEBench system" for name in builder.SAME_TERMS_AS_ODEBENCH)
    # the toy systems the smoke job runs are not in the set either: they have fewer variables
    toy = _load("run_odebench_main_search").toy_systems()
    assert all(s["dim"] < 3 for s in toy)


def test_powers_are_written_as_whole_numbers_and_coefficients_in_full(builder):
    x0, x1 = sp.symbols("x0 x1")
    text = builder.text_of(sp.expand(0.1 * x0 ** 3 - (1.0 / 3.0) * x0 * x1 + 2), [x0, x1])
    assert "x0**3" in text and "**3.0" not in text
    assert sp.expand(sp.sympify(text) - (0.1 * x0 ** 3 - (1.0 / 3.0) * x0 * x1 + 2)).xreplace({x0: 1.3, x1: -0.7}) == pytest.approx(0)
    assert repr(1.0 / 3.0) in text.replace("-", "")


def test_the_rule_refuses_what_it_says_it_refuses(builder):
    class Flow:
        nonautonomous = False
        ic = np.array([1.0, 2.0, 3.0])
        period = 2.0
        params: ClassVar[dict] = {"a": 0.5}

        @staticmethod
        def _rhs(x, y, z, t, a):
            return a * y, x * z - y, x * y - z

    field, reason = builder.eligible("Made", Flow())
    assert reason == "eligible" and field is not None and str(field[0]) == "0.5*x1"
    assert builder.eligible("Lorenz", Flow())[1] == "in ODEBench"
    assert builder.eligible("Chen", Flow())[1] == "same terms as an ODEBench system"

    forced = type("Forced", (Flow,), {"nonautonomous": True})
    assert builder.eligible("Forced", forced())[1] == "not autonomous"
    planar = type("Planar", (Flow,), {"ic": np.array([1.0, 2.0])})
    assert builder.eligible("Planar", planar())[1] == "not three or four variables"
    rational = type("Rational", (Flow,), {"_rhs": staticmethod(lambda x, y, z, t, a: (a / (1 + x), y, z))})
    assert builder.eligible("Rational", rational())[1] == "not a polynomial"
    numeric = type("Numeric", (Flow,), {"_rhs": staticmethod(lambda x, y, z, t, a: (np.sin(x), y, z))})
    assert builder.eligible("Numeric", numeric())[1] == "not plain arithmetic"
    quintic = type("Quintic", (Flow,), {"_rhs": staticmethod(lambda x, y, z, t, a: (x ** 5, y, z))})
    assert builder.eligible("Quintic", quintic())[1] == f"degree above {builder.MAX_DEGREE}"
    timed = type("Timed", (Flow,), {"_rhs": staticmethod(lambda x, y, z, t, a: (x + t, y, z))})
    assert builder.eligible("Timed", timed())[1] == "depends on something other than the state"


def test_the_committed_systems_are_what_the_package_gives(builder):
    pytest.importorskip("dysts")
    assert builder.main(["--check"]) == 0


# --- the runner -----------------------------------------------------------------------------------

def test_the_jobs_cover_every_component_noise_level_and_engine_once(runner):
    jobs = runner.all_jobs()
    assert len(jobs) == 310 * len(runner.NOISE_LEVELS) * len(runner.ENGINES) == 1860
    assert len({(j["system"], j["component"], j["noise"], j["engine"]) for j in jobs}) == len(jobs)
    ode_run = _load("run_odebench_main_search")
    parts = [ode_run.shard(jobs, k, 48) for k in range(48)]
    assert sum(len(p) for p in parts) == len(jobs)
    # a component's six runs stay in one shard, so its world is built once
    assert all(len({(j["system"], j["component"]) for j in part}) * 6 == len(part) for part in parts)


def test_the_repaired_route_runs_end_to_end_with_candidates_in_place_of_a_search(runner, monkeypatch):
    from apedp_dyn.proposers.fit_guard import FitGuardProposer
    from apedp_dyn.proposers.portfolio import FixedFormulasProposer

    worlds = _load("odebench_worlds")
    system = _load("run_odebench_main_search").toy_systems()[1]          # dx0 = 1 - x0*x1, dx1 = x0 - 0.5*x1
    world = worlds.world(system, 0.0, tag="test")
    monkeypatch.setattr(runner, "STAGE", lambda budget: FitGuardProposer(
        FixedFormulasProposer(["1 - x0*x1", "x0 + x1", "log(x0 - 50)"])))
    row = runner.run_one({"system": system["id"], "component": 0, "noise": 0.0, "engine": runner.REPAIRED},
                         system, world, budget=1.0, time_limit=120.0)
    assert row["status"] == "ok", row.get("trace")
    assert row["symbolic"] and row["accuracy_solution"] and row["search_stage"] is True
    assert row["unpruned_law"] and row["unpruned_symbolic"] and row["prune_error"] is None
    assert [formula for formula, _ in row["fit_guard_dropped"]] == ["log(x0 - 50)"]      # undefined on these rows
    assert row["portfolio"]["fit_guard_dropped"] == 1


def test_without_a_search_stage_nothing_is_reported_as_a_law(runner):
    worlds = _load("odebench_worlds")
    system = _load("run_odebench_main_search").toy_systems()[2]
    world = worlds.world(system, 0.0, tag="test")
    row = runner.run_one({"system": system["id"], "component": 0, "noise": 0.0, "engine": runner.REPAIRED},
                         system, world, budget=0.0, time_limit=120.0)
    assert row["status"] == "ok", row.get("trace")
    assert row["model"] in (None, "None") and row["search_stage"] is False and row.get("unpruned_law") is None


def _row(engine, noise, system, component, *, symbolic, accurate, reported=True, **extra):
    return {"engine": engine, "noise": noise, "system": system, "component": component, "status": "ok", "seconds": 1.0,
            "model": "x0" if reported else None, "symbolic": symbolic, "accuracy_solution": accurate, **extra}


def test_the_summary_judges_the_four_expectations_as_written(runner):
    rows = []
    for noise in runner.NOISE_LEVELS:
        for component in range(20):
            # the recorded route recovers 10 of 20; the repaired one 12, reports 20, and one of them is wrong
            rows.append(_row("main_search", noise, 1, component, symbolic=component < 10, accurate=component < 12))
            rows.append(_row("sparse_poly3", noise, 1, component, symbolic=component < 5, accurate=component < 5))
            good = component < 12
            rows.append(_row(runner.REPAIRED, noise, 1, component, symbolic=good, accurate=component < 19,
                             unpruned_law="x0", unpruned_symbolic=good and component != 3, unpruned_accurate=component < 18,
                             precision_dropped=["x1"] if component in (3, 18) else [], fit_guard_dropped=[]))
    summary = runner.summarise(rows)
    cell = summary[f"{runner.REPAIRED}|all"]
    assert cell["reported"] == 40 and cell["reported_wrong"] == 2 and cell["unpruned_wrong"] == 4
    assert cell["recoveries_gained_by_pruning"] == 2 and cell["recoveries_lost_by_pruning"] == 0 and cell["pruning_changed"] == 4
    verdict = summary["expectations"]
    assert verdict["1_share_of_reported_laws_wrong"] == pytest.approx(0.05) and verdict["1_holds"] is True
    assert verdict["2_recovered_without_noise"] == [12, 10] and verdict["2_holds"] and verdict["3_holds"]
    assert verdict["4_holds"] is True

    worse = [dict(r) for r in rows]
    for r in worse:
        if r["engine"] == runner.REPAIRED and r["component"] in (17, 18):
            r["accuracy_solution"] = False                      # three wrong laws of twenty at each noise level
        if r["engine"] == runner.REPAIRED and r["component"] == 0:
            r["symbolic"], r["accuracy_solution"] = False, True  # a recovery that pruning lost
    verdict = runner.summarise(worse)["expectations"]
    assert verdict["1_holds"] is False and verdict["1_share_of_reported_laws_wrong"] == pytest.approx(6 / 40)
    assert verdict["4_recoveries_lost_by_pruning"] == 2 and verdict["4_holds"] is False


def test_a_law_that_fits_or_is_the_law_is_not_counted_as_wrong(runner):
    assert runner._wrong({"accuracy_solution": False, "symbolic": False}) is True
    assert runner._wrong({"accuracy_solution": True, "symbolic": False}) is False
    assert runner._wrong({"accuracy_solution": False, "symbolic": True}) is False


def test_the_run_refuses_to_start_if_a_frozen_file_has_changed(runner, tmp_path, monkeypatch):
    recorded = dict(runner.frozen_digests())
    recorded[runner.FROZEN[2]] = "0" * 64
    digest = tmp_path / "DIGEST.json"
    digest.write_text(json.dumps({"sha256": recorded}), encoding="utf-8")
    monkeypatch.setattr(runner, "DIGEST", digest)
    with pytest.raises(SystemExit, match="frozen files differ"):
        runner.main(["run", "--out", str(tmp_path / "out"), "--shard", "0/48"])
    monkeypatch.setattr(runner, "DIGEST", tmp_path / "missing.json")
    with pytest.raises(SystemExit, match="not frozen"):
        runner.check_frozen()


def test_the_frozen_files_match_the_recorded_digest(runner):
    recorded = json.loads(runner.DIGEST.read_text(encoding="utf-8"))
    assert recorded["preregistration"] == runner.PREREGISTRATION_ID
    assert recorded["sha256"] == runner.frozen_digests()
    runner.check_frozen()
