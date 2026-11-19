"""The series route: a law has to forecast the later period better than the development mean."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import market_planted_worlds as worlds

from apedp_dyn.discovery_l2 import series_route
from apedp_referee.canonical_spec import DIEBOLD_MARIANO_P_VALUE_THRESHOLD


def _run(law: str | None, share: float, seed: int) -> tuple[series_route.SeriesResult, dict]:
    world = worlds.world(law, share, seed, tag="test")
    X, y = world["X"], world["y"]
    result = series_route.discover_predictive_law(
        {n: X[world["dev"], i] for i, n in enumerate(worlds.NAMES)}, y[world["dev"]], X[world["pool"]], y[world["pool"]],
        always_search=False)
    return result, world


def test_worlds_look_like_returns():
    world = worlds.world(None, 0.0, 0, tag="test")
    y = world["y"]
    kurtosis = float(np.mean((y - y.mean()) ** 4) / y.var() ** 2)
    assert kurtosis > 5.0                                                # heavy tails
    assert np.corrcoef(np.abs(y), world["X"][:, 3])[0, 1] > 0.2           # the noise scale follows an input
    assert np.corrcoef(y[1:] ** 2, y[:-1] ** 2)[0, 1] > 0.05              # volatility clusters
    assert world["truth"] is None and not world["signal"].any()
    planted = worlds.world("trend", 0.05, 0, tag="test")
    assert abs(planted["signal"].var() / planted["y"].var() - 0.05) < 0.03


def test_the_forecast_test_removes_laws_found_in_pure_noise():
    results = [_run(None, 0.0, seed)[0] for seed in range(6)]
    assert sum(r.ungated_law is not None for r in results) >= 3   # the path alone reports laws where there are none
    assert sum(r.law is not None for r in results) == 0
    assert all(r.action == series_route.NO_PREDICTIVE_LAW for r in results)
    assert all(r.forecast_test["alpha"] == DIEBOLD_MARIANO_P_VALUE_THRESHOLD for r in results if r.forecast_test)
    # a candidate that does not beat the mean inside the development period never reaches the later period
    assert all(bool(r.forecast_test) == (r.selected["chosen"] is not None) for r in results)


def test_a_law_that_forecasts_is_kept_and_reported_as_tested():
    result, world = _run("trend", 0.2, 0)
    assert result.action == series_route.PREDICTIVE_LAW and result.law is not None and result.forecast_test["passed"]
    assert result.selected["chosen"] is not None and result.selected["fit_rows"] == 2000 and result.selected["validation_rows"] == 1000
    assert result.forecast_test["rows"] == world["pool"].stop - world["pool"].start
    model = sp.sympify(result.law)
    prediction = series_route.predict(model, list(worlds.NAMES), world["X"][world["test"]])
    signal = world["signal"][world["test"]]
    assert np.corrcoef(prediction, signal)[0, 1] > 0.9


def test_the_choice_is_made_inside_the_development_period_and_prefers_what_predicts():
    world = worlds.world("trend_with_flow", 0.2, 1, tag="test")
    names = list(worlds.NAMES)
    X, y = world["X"][world["dev"]], world["y"][world["dev"]]
    chosen = series_route.select_on_development(["x1", "x0*x2", "exp(x3)", "x0*x2 + x1"], names, X, y)
    scores = {c["formula"]: c["validation_mse"] for c in chosen["candidates"]}
    assert chosen["chosen"] in ("x0*x2", "x0*x2 + x1") and scores["x0*x2"] < chosen["no_law_validation_mse"] < scores["exp(x3)"] * 1.5
    null = worlds.world(None, 0.0, 3, tag="test")
    nothing = series_route.select_on_development([], names, null["X"][null["dev"]], null["y"][null["dev"]])
    assert nothing["chosen"] is None and nothing["candidates"] == []
