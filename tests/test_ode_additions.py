"""Three additions to the ODE route (a second engine, adequacy at the rows' precision, a chosen experiment).

Nothing here runs a search engine or a system of either blind set. The pieces are exercised on
made-up rows and on the frozen runner's toy systems, with fixed candidates in the searches' place.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import sympy as sp

from apedp_dyn.discovery_l2 import contrast_confirmation as cc
from apedp_dyn.discovery_l2 import deciding_measurement as dm
from apedp_dyn.discovery_l2 import refined_search
from apedp_dyn.ode import chosen_experiment as ce
from apedp_dyn.ode import default_route, extended_route, precision_adequacy, second_engine
from apedp_dyn.ode.window_rows import WindowRows
from apedp_dyn.proposers.portfolio import FixedFormulasProposer
from apedp_dyn.proposers.sparse_library import SparseLibraryProposer, supports

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


# ------------------------------------------------------------------ the second engine
def _frame(noise: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    X = rng.uniform(-2.0, 2.0, (300, 3))
    y = 1.5 * X[:, 0] * X[:, 2] - 0.7 * X[:, 1] ** 3 + 2.0 + noise * rng.standard_normal(300)
    return pd.DataFrame({"a": X[:, 0], "b": X[:, 1], "c": X[:, 2], "rate": y})


def test_the_sparse_engine_proposes_the_law_in_the_frames_own_names():
    proposer = SparseLibraryProposer(target_col="rate")
    proposals = proposer.fit_and_propose(_frame(), target_col="rate")
    law = sp.sympify(proposals[0].formula)
    assert sp.expand(law - sp.sympify("1.5*a*c - 0.7*b**3 + 2.0")).xreplace(
        {n: sp.Float(round(float(n), 8)) for n in sp.expand(law - sp.sympify("1.5*a*c - 0.7*b**3 + 2.0")).atoms(sp.Float)}) == 0
    assert proposals[0].proposer_id == "sparse_library" and proposals[0].metadata["terms"] == 3
    frame = _frame()
    assert np.allclose(proposals[0].compute_score(frame), frame["rate"])
    assert 1 <= len(proposals) <= 2 and proposer.last_run["library_terms"] == 20


def test_the_supports_come_best_first_and_each_once():
    frame = _frame(noise=0.05)
    X, y = frame[["a", "b", "c"]].to_numpy(), frame["rate"].to_numpy()
    library = np.column_stack([np.ones(len(y)), X[:, 0] * X[:, 2], X[:, 1] ** 3, X[:, 0], X[:, 1], X[:, 2]])
    fits = supports(library, y)
    assert fits and [bic for bic, _ in fits] == sorted(bic for bic, _ in fits)
    kept = [tuple(np.flatnonzero(coef)) for _, coef in fits]
    assert len(set(kept)) == len(kept) and set(kept[0]) >= {0, 1, 2}


def test_the_sparse_engine_gives_what_the_frozen_reference_gives(toy):
    _, world = toy
    reference = _load("run_odebench_main_search").run_sparse_poly3(world, 0)
    frame = pd.DataFrame(world["dev"].states, columns=["x0", "x1"])
    frame["__target__"] = world["dev"].rates[:, 0]
    ours = SparseLibraryProposer(target_col="__target__").fit_and_propose(frame, target_col="__target__")
    difference = sp.expand(sp.sympify(ours[0].formula) - reference["model"])
    assert all(abs(float(c)) < 1e-9 for c in difference.as_coefficients_dict().values())


def test_the_second_engine_stands_beside_the_searches_or_alone():
    alone = second_engine.with_sparse(None)
    assert alone.proposer_id == "sparse_library"
    both = second_engine.with_sparse(FixedFormulasProposer(["a + b", "a*c"]))
    proposals = both.fit_and_propose(_frame().rename(columns={"rate": "__target__"}), target_col="__target__")
    assert [p.proposer_id for p in proposals[:4]] == ["fixed_formulas", "sparse_library", "fixed_formulas", "sparse_library"]
    assert both.dropped == [] and second_engine.search_stage(0.0).proposer_id == "sparse_library"


# ------------------------------------------------------------------ the deciding measurement
def test_rivals_are_told_apart_where_they_differ_most():
    rivals = ["2*x0", "2*x0 + 0.5*x0**2"]                   # equal at 0, far apart at the ends
    gap, pairs = dm.disagreement(rivals, ["x0"], np.array([[0.0], [1.0], [3.0]]), sd=0.1)
    assert np.allclose(gap, [0.0, 5.0, 45.0]) and pairs[2].tolist() == [0, 1]
    points = dm.deciding_points(rivals, ["x0"], [-1.0], [3.0], sd=0.1, count=3, seed=1)
    assert len(points) == 3 and points[0].point[0] > 2.8 and points[0].spread > 39
    assert all(abs(a.point[0] - b.point[0]) >= dm.SEPARATION * 4.0 for a in points for b in points if a is not b)
    assert "Measure at x0 =" in dm.describe(points, ["x0"], rivals) and "`2*x0`" in dm.describe(points, ["x0"], rivals)


def test_rivals_that_agree_or_a_single_law_give_nothing_to_measure():
    assert dm.deciding_points(["x0 + x1"], ["x0", "x1"], [0, 0], [1, 1]) == []
    assert dm.deciding_points(["x0 + x1", "x1 + x0"], ["x0", "x1"], [0, 0], [1, 1]) == []
    assert "no measurement here separates them" in dm.describe([], ["x0"], ["a", "b"])
    assert dm.deciding_rows(["x0"], ["x0"], np.ones((3, 1))) == []


def test_a_rival_that_predicts_nothing_somewhere_does_not_decide_there():
    rivals = ["log(x0)", "x0 - 1", "x0**2 - 1"]
    gap, _ = dm.disagreement(rivals, ["x0"], np.array([[-1.0], [2.0]]))
    assert gap[0] == pytest.approx(0.0 - (-2.0)) and gap[1] == pytest.approx(3.0 - np.log(2.0))   # at -1 only two predict
    assert dm.predictions(["not a law ((", "x0"], ["x0"], np.array([[1.0]]))[0, 0] != dm.predictions(["x0"], ["x0"], np.array([[1.0]]))[0, 0]


def test_where_inputs_cannot_be_set_the_rows_to_watch_are_chosen_from_a_pool():
    pool = np.array([[0.0], [0.5], [2.0], [1.0]])
    assert dm.deciding_rows(["2*x0", "2*x0 + x0**2"], ["x0"], pool, count=2) == [2, 3]
    assert len(dm.random_points([0, 0], [1, 2], 5, seed=3)) == 5


# ------------------------------------------------------------------ the follow-up experiment
def _rate(states: np.ndarray) -> np.ndarray:
    states = np.atleast_2d(states)
    return np.column_stack([1.0 - states[:, 0] * states[:, 1], states[:, 0] - 0.5 * states[:, 1]])


def test_an_experiment_gives_rows_like_any_other_and_they_follow_the_law():
    rows, failed = ce.measured_rows(_rate, [(1.0, 0.5), (0.2, 1.5)], 0.0, np.ones(2), np.random.default_rng(0))
    assert failed == 0 and rows is not None and rows.states.shape == (20, 2) and rows.rates.shape == (20, 2)
    assert np.allclose(rows.rates, _rate(rows.states), atol=1e-5)
    nothing, failed = ce.measured_rows(lambda s: np.full((1, 2), 1e9), [(1.0, 1.0)], 0.0, np.ones(2), np.random.default_rng(0))
    assert nothing is None and failed == 1


def test_the_follow_up_starts_where_the_rivals_part_and_a_random_one_does_not_choose(toy):
    _, world = toy
    candidates = ["1 - x0*x1", "1 - x0*x1 + 0.01*x0**3", "x0 + x1"]
    larger, done = ce.follow_up(world["dev"], world["pool"], 0, candidates, _rate, level=0.0, spread=world["spread"], seed=2)
    assert done.chosen and done.rows > 0 and len(larger.states) == len(world["pool"].states) + done.rows
    assert done.spreads == sorted(done.spreads, reverse=True) and len(done.rivals) == 3
    low, high = world["dev"].states.min(axis=0), world["dev"].states.max(axis=0)
    assert all(np.all(np.asarray(s) >= low) and np.all(np.asarray(s) <= high) for s in done.starts)
    by_chance, random_done = ce.follow_up(world["dev"], world["pool"], 0, candidates, _rate, level=0.0,
                                          spread=world["spread"], seed=2, chosen=False)
    assert not random_done.chosen and random_done.spreads == [] and len(random_done.starts) == ce.STARTS
    assert len(by_chance.states) == len(world["pool"].states) + random_done.rows
    same, none = ce.follow_up(world["dev"], world["pool"], 0, ["1 - x0*x1"], _rate, level=0.0, spread=world["spread"])
    assert same is world["pool"] and "agree" in none.note
    assert json.dumps(ce.as_record(done))


def test_the_rivals_are_the_candidates_that_fit_best_each_once(toy):
    _, world = toy
    kept = ce.rivals_of(["x0 + x1", "1 - x0*x1", "1 - x1*x0", "log(x0 - 50)"], ["x0", "x1"], world["dev"].states,
                        world["dev"].rates[:, 0])
    assert kept[0] == "1 - x0*x1" and "1 - x1*x0" not in kept and "log(x0 - 50)" not in kept and len(kept) == 2


# ------------------------------------------------------------------ adequacy at the rows' precision
def _rows(law: str, error: float, precision: float) -> WindowRows:
    rng = np.random.default_rng(1)
    states = rng.uniform(0.5, 2.0, (200, 2))
    rate = np.asarray(sp.lambdify(sp.symbols("x0 x1"), sp.sympify(law), "numpy")(*states.T), dtype=float)
    wobble = error * np.sin(3.0 * states[:, 0])               # a smooth error: structure, not noise
    return WindowRows(states, np.column_stack([rate + wobble, rate]), np.arange(200.0), (3, 3), np.zeros(2),
                      np.array([precision, precision]))


def _refused(law: str, verdict: str = cc.UNIQUE) -> refined_search.RefinedResult:
    path = SimpleNamespace(action="widen_search", law=None, rejected_law=law, adequacy={"passed": False},
                           contrast=SimpleNamespace(verdict=verdict))
    return refined_search.RefinedResult("widen_search", None, None, path=path)


def test_a_law_refused_for_structure_beneath_the_rows_precision_is_reinstated():
    dev = _rows("2.0*x0*x1 - 1.0*x1", error=0.001, precision=0.001)
    result, info = precision_adequacy.judged_at_precision(_refused("2.0*x0*x1 - 1.0*x1"), dev, 0)
    assert info["reinstated"] and info["residual_in_precisions"] < precision_adequacy.MULTIPLES
    assert result.action == "confirmed" and result.law and "x0*x1" in result.law and result.tentative_law is None
    assert precision_adequacy.MULTIPLES == 2.0


def test_a_law_whose_residual_exceeds_the_rows_precision_stays_refused():
    dev = _rows("2.0*x0*x1 - 1.0*x1", error=0.05, precision=0.001)
    refused = _refused("2.0*x0*x1 - 1.0*x1")
    result, info = precision_adequacy.judged_at_precision(refused, dev, 0)
    assert result is refused and not info["reinstated"] and info["residual_in_precisions"] > 10


def test_what_was_not_refused_at_that_check_is_left_alone():
    dev = _rows("2.0*x0*x1", error=0.0, precision=0.001)
    reported = refined_search.RefinedResult("confirmed", "2.0*x0*x1", None, path=SimpleNamespace(
        action="confirmed", law="2.0*x0*x1", rejected_law=None, adequacy={}, contrast=None))
    no_survivor = refined_search.RefinedResult("widen_search", None, None, path=SimpleNamespace(
        action="widen_search", law=None, rejected_law=None, adequacy={}, contrast=None))
    for result in (reported, no_survivor, refined_search.RefinedResult("widen_search", None, None)):
        same, info = precision_adequacy.judged_at_precision(result, dev, 0)
        assert same is result and info == {"reinstated": False}
    unchallenged, info = precision_adequacy.judged_at_precision(_refused("2.0*x0*x1", verdict=cc.UNCHALLENGED), dev, 0)
    assert info["reinstated"] and unchallenged.law is None and unchallenged.tentative_law     # weaker evidence, kept apart


# ------------------------------------------------------------------ the composed route and the runner
def test_with_nothing_switched_on_the_composed_route_is_the_route_in_force(toy):
    _, world = toy
    formulas = ["1 - x0*x1", "x0 + x1"]
    plain = default_route.discover_component(world["dev"], world["pool"], 0, proposer=extended_route.replayed(formulas))
    composed, done = extended_route.discover_component(world["dev"], world["pool"], 0,
                                                       proposer=extended_route.replayed(formulas))
    assert composed.law == plain.law and composed.action == plain.action and done == {"second_engine": False, "at_precision": False}
    assert "1 - x0*x1" in " ".join(extended_route.candidates_of(composed)).replace("1.0", "1").replace("- 1*", "- ") or composed.law
    assert extended_route.replayed([]) is None


def test_a_recording_stage_keeps_what_it_proposed():
    stage = extended_route.Recording(FixedFormulasProposer(["a + b", "a*c"]))
    stage.fit_and_propose(_frame(), target_col="rate")
    assert stage.formulas == ["a + b", "a*c"] and stage.proposer_id == "fixed_formulas" and stage.dropped == []


def test_every_arm_gets_the_same_candidates_and_a_row_each(toy):
    runner = _load("run_ode_additions")
    system, world = toy
    rows = runner.run_unit(system, world, 0, 0.0, ["1 - x0*x1", "x0 + x1"], time_limit=600.0)
    assert [r["arm"] for r in rows] == [*runner.ARMS, runner.REFERENCE] and all(r["status"] == "ok" for r in rows), rows[0]
    assert rows[0]["searched"] == ["1 - x0*x1", "x0 + x1"] and rows[0]["symbolic"] and rows[3]["second_engine"] is True
    assert rows[4]["experiment"]["chosen"] is True and rows[5]["experiment"]["chosen"] is False
    assert all(r["part"] == "toy" and r["truth"] and r["n_pool"] == 100 for r in rows)


def _row(arm, noise, component, *, symbolic, accurate=True, reported=True, part="invented"):
    return {"arm": arm, "noise": noise, "system": 1, "component": component, "part": part, "status": "ok",
            "symbolic": symbolic, "accuracy_solution": accurate, "model": "x0" if reported else None}


def test_the_summary_judges_the_expectations_as_written():
    runner = _load("run_ode_additions")
    rows = []
    for noise in runner.NOISE_LEVELS:
        for component in range(10):
            rows.append(_row("default", noise, component, symbolic=component < 4))
            rows.append(_row("both", noise, component, symbolic=component < 6))
            rows.append(_row("both_chosen", noise, component, symbolic=component < 8))
            rows.append(_row("both_random", noise, component, symbolic=component < 7))
    summary = runner.summarise(rows)
    e = summary["expectations"]
    assert e["1_holds"] and e["2_holds"] and e["3_holds"]
    assert e["2_recovered_both_and_default"] == {"0.0": [6, 4], "0.01": [6, 4]}
    assert summary["run_for_run"]["both against default|0.0"] == {"gained": 2, "lost": 0}
    assert summary["both|all|invented"]["recovered"] == 12
    # one wrong law among twenty reports of the chosen arm is 5 %: at the limit, it holds; two is over it
    worse = [dict(r) for r in rows]
    spoiled = [r for r in worse if r["arm"] == "both_chosen" and not r["symbolic"]][:2]
    spoiled[0]["accuracy_solution"] = False
    assert runner.summarise(worse)["expectations"]["1_holds"]
    spoiled[1]["accuracy_solution"] = False
    assert not runner.summarise(worse)["expectations"]["1_holds"]
    # a tie between a chosen and a random experiment is not a pass
    tied = [dict(r, symbolic=(r["component"] < 7)) if r["arm"] == "both_chosen" else r for r in rows]
    assert not runner.summarise(tied)["expectations"]["3_holds"]


def test_the_second_blind_set_is_what_the_rules_give_and_shares_no_system_with_the_first():
    builder = _load("build_ode_second_blind_systems")
    record = json.loads(builder.OUT.read_text(encoding="utf-8"))
    names = [s["name"] for s in record["systems"]]
    assert sum(s["part"] == "invented" for s in record["systems"]) == 30 and len(names) == len(set(names))
    assert record["not_buildable"] == {} and record["seed"] == builder.SEED
    first = json.loads((ROOT / "benchmarks" / "dysts_polynomial_systems.json").read_text(encoding="utf-8"))
    assert not set(names) & {s["name"] for s in first["systems"]}
    runner = _load("run_ode_additions")
    systems = runner.systems()
    assert len(systems) == len(names) and len(runner.units()) == 2 * sum(s["dim"] for s in systems)
    for system in systems:
        symbols = [sp.Symbol(f"x{i}") for i in range(system["dim"])]
        polynomial = all(c.is_polynomial(*symbols) for c in system["components"])
        assert polynomial == (system["part"] == "invented")
        assert all(c.is_rational_function(*symbols) for c in system["components"])
        if system["part"] == "invented":                 # every component is held in by its own cubic term
            assert all(sp.Poly(c, *symbols).coeff_monomial(symbols[i] ** 3) < 0 for i, c in enumerate(system["components"]))
