"""The system's forecast of its own runs: the features, the forecaster, the scoring, the record.

A forecast here is a statement about the system, to be scored against what happens. The backtest
pinned below is not a test of it: every outcome was known when the forecaster was written.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from apedp_dyn.discovery_l2 import outcome_forecast as of

ROOT = Path(__file__).resolve().parents[1]


def _script():
    name = "forecast_runs"
    if name not in sys.modules:
        sys.path.insert(0, str(ROOT / "scripts"))
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def test_the_features_say_what_a_law_looks_like():
    f = of.law_features("2.5*x0**2*x1 - 0.3*x1", 3, 0.0)
    assert f["polynomial"] == 1 and f["degree"] == 3 and f["rational"] == 1 and f["terms"] == 2 and f["inputs"] == 3
    # the numbers in a law other than 0 and 1: two coefficients and one power
    assert f["noise_is_zero"] == 1 and f["constants"] == 3 and not any(f[k] for k in ("exp", "log", "trig", "tanh", "root"))
    g = of.law_features("1.3*exp(-0.7*x0)/(x1 + 2) + sqrt(x0)*sin(x1)", 2, 0.01)
    assert g["polynomial"] == 0 and g["degree"] == 0 and g["rational"] == 0 and g["exp"] == g["trig"] == g["root"] == 1
    assert g["divides_by_inputs"] == 1 and g["noise_is_zero"] == 0 and g["log10_noise"] == pytest.approx(-2.0)
    assert set(f) == set(of.FEATURES) == set(g)


def _runs(n: int = 400, seed: int = 0) -> list[of.Run]:
    """Polynomials are recovered nine times in ten, laws with a sine one time in ten; nothing reported is wrong."""
    rng = np.random.default_rng(seed)
    runs = []
    for i in range(n):
        easy = bool(i % 2)
        law = "2.0*x0*x1 + 1.5*x0" if easy else "2.0*sin(1.3*x0) + x1"
        recovered = bool(rng.random() < (0.9 if easy else 0.1))
        runs.append(of.Run(law, 2, 0.01, "route_a", recovered, True, False, source="made up", name=f"p{i}"))
    return runs


def test_the_forecaster_learns_which_laws_are_recovered_and_beats_the_average():
    runs = _runs()
    model = of.OutcomeForecaster().fit(runs[:300])
    held = runs[300:]
    forecast = model.forecast(held)
    easy = [c for c, r in zip(forecast.chances["recovered"], held, strict=True) if "sin" not in r.law]
    hard = [c for c, r in zip(forecast.chances["recovered"], held, strict=True) if "sin" in r.law]
    assert min(easy) > 0.8 and max(hard) < 0.2 and model.trained_on == 300
    scored, flat = of.score(forecast, held), of.score(of.baseline(runs[:300], len(held)), held)
    assert scored["recovered"]["brier"] < 0.12 < 0.2 < flat["recovered"]["brier"]
    assert abs(scored["recovered"]["expected"] - scored["recovered"]["seen"]) < 12
    low, high = scored["recovered"]["interval"]
    assert low < scored["recovered"]["expected"] < high and scored["recovered"]["inside"] == (low <= scored["recovered"]["seen"] <= high)
    # reported always, wrong never: forecast at the smoothed frequency, never 0 or 1
    assert forecast.chances["reported"][0] == pytest.approx(301 / 302) and forecast.chances["wrong"][0] == pytest.approx(1 / 302)


def test_a_route_never_seen_gets_the_average_of_the_routes_on_record():
    a = _runs(400)
    rng = np.random.default_rng(5)
    # a second route that recovers the same laws less often
    b = [of.Run(r.law, r.inputs, r.noise, "route_b", bool(rng.random() < (0.5 if "sin" not in r.law else 0.02)), True, False,
                name=r.name) for r in _runs(400, seed=1)]
    model = of.OutcomeForecaster().fit(a + b)
    new = [of.Run("2.0*x0*x1 + 1.5*x0", 2, 0.01, "route_new", name="q")]
    known_a = model.forecast(new, route="route_a").chances["recovered"][0]
    known_b = model.forecast(new, route="route_b").chances["recovered"][0]
    unseen = model.forecast(new).chances["recovered"][0]
    assert known_a > 0.8 and known_b < 0.65 and known_b < unseen < known_a


def test_the_interval_of_a_total_widens_with_the_spread_between_whole_runs():
    forecast = of.Forecast("r", {"recovered": [0.5] * 200}, ["x"] * 200)
    low, high = forecast.interval("recovered")
    wide_low, wide_high = forecast.interval("recovered", between_runs=0.1)
    assert low <= 100 <= high and 14 <= high - low <= 24
    assert wide_low < low and wide_high > high and 45 <= wide_high - wide_low <= 60
    assert forecast.as_dict(between_runs=0.1)["totals"]["recovered"]["interval"] == [wide_low, wide_high]
    assert forecast.expected("recovered") == 100.0


def test_brier_and_hardest():
    assert of.brier([1.0, 0.0], [True, False]) == 0.0 and of.brier([0.5, 0.5], [True, False]) == 0.25
    forecast = of.Forecast("r", {"recovered": [0.9, 0.1, 0.5, 0.2]}, list("abcd"))
    assert of.hardest(forecast, 2) == [1, 3]


def test_a_forecaster_refuses_to_forecast_before_it_has_learned_and_to_learn_from_nothing():
    with pytest.raises(ValueError, match="fit"):
        of.OutcomeForecaster().forecast([of.Run("x0", 1, 0.0, "r")])
    with pytest.raises(ValueError, match="no runs"):
        of.OutcomeForecaster().fit([])
    with pytest.raises(ValueError, match="no recorded outcome"):
        of.Run("x0", 1, 0.0, "r").outcome("recovered")


def test_the_runs_on_record_are_read_with_their_laws_and_outcomes():
    script = _script()
    sparse = script.recorded("sparse")
    assert len(sparse) == 234 + 620 and {r.route for r in sparse} == {"sparse_poly3"}
    assert sum(r.recovered for r in sparse) == 131 + 444 and all(r.reported for r in sparse)
    blind = [r for r in script.recorded("trajectories") if r.source == "RESULT-ODE-BLIND-2026-11" and r.route == "ode_repaired"]
    assert len(blind) == 620 and sum(r.recovered for r in blind) == 409 and sum(r.wrong for r in blind) == 3
    assert sum(r.reported for r in blind) == 484 and {r.noise for r in blind} == {0.0, 0.01}


def test_the_backtest_on_file_is_what_the_script_gives_and_says_what_the_documents_say():
    script = _script()
    record = json.loads(script.BACKTEST.read_text(encoding="utf-8"))
    assert "not a test" in record["what"] and record["forecast_beats_averages"] == [7, 10]
    assert record["total_inside_unwidened_interval"] == [4, 10]
    assert record["between_runs_spread"]["recovered"] == pytest.approx(0.1057, abs=1e-4)
    fresh = script.backtest("sparse")
    on_file = [row for row in record["held_out_runs"] if row["group"] == "sparse"]
    assert fresh == on_file and script.between_runs_spread(record["held_out_runs"]) == pytest.approx(0.1057, abs=1e-4)
    tables = [row for row in record["held_out_runs"] if row["group"] == "tables"]
    assert all(row["forecast"]["recovered"]["brier"] < row["averages_only"]["recovered"]["brier"] for row in tables)
