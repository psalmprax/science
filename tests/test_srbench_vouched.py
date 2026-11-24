"""PREREG-VOUCHED-SRBENCH: one search, two reports that differ only by the checks, and how the run is judged."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from apedp_dyn.proposers.portfolio import FixedFormulasProposer

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
def run():
    return _load("run_srbench_vouched")


def _problem(noise: float = 0.002, seed: int = 0):
    """A toy problem in the benchmark's own shapes: a manifest entry and a split."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(1.0, 5.0, (1500, 2))
    y = -4.0 * X[:, 0] - 0.2 * X[:, 1]
    train = y[:1000] + noise * float(np.sqrt(np.mean(y[:1000] ** 2))) * rng.standard_normal(1000)
    entry = {"name": "toy_line", "family": "strogatz", "features": ["a", "b"], "units": {}, "target_units": {},
             "formula": "-4.0*a - 0.2*b"}
    split = {"names": ["x0", "x1"], "X_train": X[:1000], "y_train": train, "X_test": X[1000:], "y_test": y[1000:]}
    return entry, split


def test_one_search_gives_the_routes_report_and_the_same_report_after_the_checks(run, monkeypatch) -> None:
    portfolio_run = _load("run_srbench_portfolio")
    monkeypatch.setattr(portfolio_run, "members", lambda budget: [lambda: FixedFormulasProposer(["x0 + x1", "x0*x1", "x0 - x1**2"])])
    entry, split = _problem()
    out = run.run_vouched(split, entry, budget=1.0)
    extra = out["extra"]
    route, state = extra["route"], extra["vouching"]
    assert route["formula"] and route["symbolic"] and route["accuracy_solution"]       # the route finds the line
    assert state["in_force"] and state["applied"] and state["route_law"] == route["formula"]
    assert isinstance(extra["equivalent"], list) and extra["action"] in ("confirmed", "equivalent_class")
    # the law handed to the scorer is the one after the checks: here the same, since nothing was wrong with it
    assert out["formula"] == route["formula"] and state["reason"] == "" and out["pred"] is not None
    # checks that withhold leave the route's report on record and hand the scorer nothing
    from apedp_dyn.discovery_l2 import vouching

    monkeypatch.setattr(vouching, "vouch", lambda law, *a, **k: vouching.Vouched(None, "every candidate left structure in its residuals"))
    out = run.run_vouched(split, entry, budget=1.0)
    assert out["model"] is None and out["formula"] is None and out["pred"] is None
    assert out["extra"]["route"]["symbolic"] and out["extra"]["vouching"]["reason"].startswith("every candidate")


def _row(route: str, after: str, noise: float = 0.01, problem: str = "feynman_I_6_2a", **state) -> dict:
    """A scored run: each arm ``right``, ``close`` (accurate, not the law), ``wrong`` or ``silent``."""
    scored = {"right": {"model": "x0", "formula": "x0", "symbolic": True, "accuracy_solution": True, "symbolic_rel_rmse": 0.0},
              "close": {"model": "x0 + 0.01", "formula": "x0 + 0.01", "symbolic": False, "accuracy_solution": True, "symbolic_rel_rmse": 1e-3},
              "wrong": {"model": "x0**2", "formula": "x0**2", "symbolic": False, "accuracy_solution": False, "symbolic_rel_rmse": 0.3},
              "silent": {"model": None, "formula": None, "symbolic": False, "accuracy_solution": False, "symbolic_rel_rmse": None}}
    applied = route != "silent"
    return {"status": "ok", "problem": problem, "noise": noise, "family": "feynman", "seconds": 300.0, **scored[after],
            "route": dict(scored[route]),
            "vouching": {"applied": applied, "reason": "" if after != "silent" else "an error large enough to make it wrong would have been seen 40% of the time, not 80%",
                         "switched": False, **state}}


def test_the_run_is_judged_as_written(run) -> None:
    assert run.MAX_WRONG == 0.05 and run.MIN_KEPT == 0.75 and run.KEPT_WITHOUT_COST == pytest.approx(0.95)
    rows = [_row("right", "right") for _ in range(19)] + [_row("right", "silent")] + [_row("close", "right", switched=True)]
    rows += [_row("wrong", "silent"), _row("silent", "silent"), {"status": "timeout", "problem": "p", "noise": 0.0, "family": "strogatz", "seconds": 780.0}]
    s = run.summarise(rows)
    route, vouched = s["all"]["route"], s["all"]["vouched"]
    assert (route["reported"], route["recovered"], route["not_accurate"], route["accurate_not_the_law"]) == (22, 20, 1, 1)
    assert (vouched["reported"], vouched["recovered"], vouched["not_accurate"], vouched["timeouts"]) == (20, 20, 0, 1)
    e = s["expectations"]
    assert e["1_holds"] and e["2_recovered_laws_of_the_route_kept"] == [19, 20] and e["2_holds"] and e["3_holds"]
    assert e["route_not_accurate_share"] == pytest.approx(1 / 22)
    checks = s["checks"]
    assert checks["withheld"] == 2 and checks["passed_to_another_member"] == 1 and checks["not_accurate_before_and_withheld"] == 1
    assert len(checks["recovered_laws_lost"]) == 1 and len(checks["gained"]) == 1
    assert sum(checks["withheld_by_reason"].values()) == 2
    # nineteen of twenty kept is the line of the third expectation; eighteen is under it and still over the second's
    rows[0] = _row("right", "silent")
    e = run.summarise(rows)["expectations"]
    assert e["2_recovered_laws_of_the_route_kept"] == [18, 20] and e["2_holds"] and not e["3_holds"]
    # fourteen of twenty is under the canonical share a referee must retain
    for k in range(1, 5):
        rows[k] = _row("right", "silent")
    e = run.summarise(rows)["expectations"]
    assert e["2_recovered_laws_of_the_route_kept"] == [14, 20] and not e["2_holds"]
    # a checked route whose reports are inaccurate more than one time in twenty fails the first
    bad = [_row("right", "right") for _ in range(9)] + [_row("wrong", "wrong")]
    assert not run.summarise(bad)["expectations"]["1_holds"]


def test_the_protocol_is_the_recorded_runs(run) -> None:
    portfolio_run = _load("run_srbench_portfolio")
    assert run.NOISE_LEVELS == portfolio_run.NOISE_LEVELS and run.SEARCH_BUDGET_S == portfolio_run.SEARCH_BUDGET_S
    assert [(job["problem"], job["noise"]) for job in run.all_jobs()] == [(job["problem"], job["noise"]) for job in portfolio_run.all_jobs()]
    assert len(run.all_jobs()) == 266 and all(job["engine"] == run.ENGINE for job in run.all_jobs())
    for rel in run.FROZEN:
        assert (ROOT / rel).is_file() or rel.endswith("PREREG-VOUCHED-SRBENCH-2026-11.md"), rel
