"""Window rows and the ODE wrapper around the main search path."""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp
from scipy.integrate import solve_ivp

from apedp_dyn.ode import main_search_ode
from apedp_dyn.ode.window_rows import window_rows

T = np.linspace(0.0, 10.0, 10_000)


def _duffing(_t: float, x: np.ndarray) -> list[float]:
    return [x[1], -0.3 * x[1] - 1.2 * x[0] - 0.8 * x[0] ** 3]


def _trajectory(init: list[float]) -> np.ndarray:
    return solve_ivp(_duffing, (0.0, 10.0), init, t_eval=T, method="DOP853", rtol=1e-12, atol=1e-14).y.T


def _true_rates(states: np.ndarray) -> np.ndarray:
    return np.array([_duffing(0.0, row) for row in states])


def test_clean_data_gives_rates_to_many_digits():
    rows = window_rows(T, _trajectory([1.5, 0.0]))
    assert rows.states.shape == rows.rates.shape == (100, 2)
    truth = _true_rates(rows.states)
    assert np.sqrt(np.mean((rows.rates - truth) ** 2)) < 1e-8 * np.sqrt(np.mean(truth ** 2))
    assert min(rows.degree) >= 7


def test_noisy_data_takes_a_low_degree_and_reports_its_own_error_honestly():
    clean = _trajectory([1.5, 0.0])
    rng = np.random.default_rng(0)
    rows = window_rows(T, clean + 0.01 * clean.std(axis=0) * rng.standard_normal(clean.shape))
    assert max(rows.degree) <= 3
    error = rows.rates - _true_rates(rows.states)
    assert np.all(rows.rate_sd / error.std(axis=0) > 0.7) and np.all(rows.rate_sd / error.std(axis=0) < 1.4)
    assert np.all(rows.fit_rms / (0.01 * clean.std(axis=0)) > 0.9) and np.all(rows.fit_rms / (0.01 * clean.std(axis=0)) < 1.1)


def test_errors_of_neighbouring_windows_are_not_shared():
    # the reason for disjoint windows: the path's adequacy check assumes independent row errors
    clean = _trajectory([1.5, 0.0])
    rng = np.random.default_rng(1)
    rows = window_rows(T, clean + 0.01 * clean.std(axis=0) * rng.standard_normal(clean.shape))
    error = rows.rates - _true_rates(rows.states)
    for j in range(2):
        assert abs(np.corrcoef(error[:-1, j], error[1:, j])[0, 1]) < 0.3


def test_bad_input_is_refused():
    with pytest.raises(ValueError, match="uniformly"):
        window_rows(np.sort(np.random.default_rng(0).uniform(0, 1, 500)), np.zeros((500, 1)))
    with pytest.raises(ValueError, match="same number"):
        window_rows(T, np.zeros((50, 1)))
    with pytest.raises(ValueError, match="max_degree"):
        window_rows(T, np.zeros((len(T), 1)), window=8, max_degree=7)


def test_components_go_through_the_path_and_are_reported_as_confirmed():
    dev, pool = window_rows(T, _trajectory([1.5, 0.0])), window_rows(T, _trajectory([-0.7, 1.1]))
    first, second = main_search_ode.discover_system(dev, pool)
    assert first.action in ("confirmed", "equivalent_class") and second.action in ("confirmed", "equivalent_class")
    names = main_search_ode.state_names(2)
    law = main_search_ode.confirmed_form(second.law, names, dev.states, dev.rates[:, 1])
    x0, x1 = sp.symbols("x0 x1")
    coefficients = sp.Poly(law, x0, x1).as_dict()
    assert coefficients[(3, 0)] == pytest.approx(-0.8, abs=1e-6)
    assert coefficients[(1, 0)] == pytest.approx(-1.2, abs=1e-6)
    assert coefficients[(0, 1)] == pytest.approx(-0.3, abs=1e-6)


def test_pool_rows_can_be_restricted():
    dev, pool = window_rows(T, _trajectory([1.5, 0.0])), window_rows(T, _trajectory([-0.7, 1.1]))
    result = main_search_ode.discover_component(dev, pool, 0, pool_rows=np.arange(0, 100, 2))
    assert result.law is not None and result.rows_read <= 50
