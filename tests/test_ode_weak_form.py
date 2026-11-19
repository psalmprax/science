"""Weak-form ODE identification (apedp_dyn/ode/weak_form.py)."""
from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.ode import weak_form


def _oscillator(noise: float, seed: int = 0):
    """x' = y, y' = -4x - 0.3y, sampled at 0.01 for 30 time units."""
    t = np.arange(0.0, 30.0, 0.01)
    lam = np.roots([1.0, 0.3, 4.0])[0]
    x = np.real(np.exp(lam * t))
    y = np.real(lam * np.exp(lam * t))
    S = np.column_stack([x, y])
    return t, S + np.random.default_rng(seed).normal(0, 1, S.shape) * noise * S.std(axis=0)


def _library(S):
    x, y = S.T
    return np.column_stack([np.ones(len(S)), x, y, x * y, x * x, y * y])


TRUTH = np.array([[0, 0], [0, -4.0], [1.0, -0.3], [0, 0], [0, 0], [0, 0]])


def test_clean_data_gives_the_system_to_high_precision():
    t, S = _oscillator(0.0)
    coef = weak_form.identify(t, S, _library, threshold=0.05)
    assert np.array_equal(coef != 0, TRUTH != 0)
    assert np.max(np.abs(coef - TRUTH)) < 1e-6


def test_at_ten_percent_noise_the_coefficients_stay_close():
    """No claim against the derivative route here: on this easy linear system both
    do well. The comparison that matters is scripts/study_weak_form_lorenz.py."""
    errors = []
    for seed in range(8):
        t, S = _oscillator(0.10, seed)
        weak = weak_form.identify(t, S, _library, threshold=0.05)
        on = TRUTH != 0
        errors.append(np.max(np.abs(weak[on] - TRUTH[on])))
    assert np.median(errors) < 0.05


def test_the_test_function_derivative_integrates_a_known_case():
    """For x = t the weak form must return x' = 1 exactly: -int(phi' t) = int(phi)."""
    t = np.linspace(0, 1, 501)
    G, b = weak_form.weak_system(t, t[:, None], lambda S: np.ones((len(S), 1)), window=101, stride=50)
    assert np.allclose(b[:, 0], G[:, 0], rtol=1e-4)


@pytest.mark.parametrize("kwargs, message", [
    ({"window": 100}, "odd"), ({"window": 100001}, "odd"),
])
def test_bad_windows_are_refused(kwargs, message):
    t, S = _oscillator(0.0)
    with pytest.raises(ValueError, match=message):
        weak_form.weak_system(t, S, _library, **kwargs)


def test_nonuniform_times_are_refused():
    t, S = _oscillator(0.0)
    t = t.copy()
    t[50:] += 0.004
    with pytest.raises(ValueError, match="uniformly"):
        weak_form.weak_system(t, S, _library)


def test_the_proposer_can_refit_its_selected_terms_with_the_weak_form():
    """Same terms as the kernel chose, better coefficients under noise; opt-in, and the default is unchanged."""
    pd = pytest.importorskip("pandas")
    from apedp.rust_bridge import is_rust_kernel_available
    from apedp_dyn.ode.proposer import CoupledOdeProposer

    if not is_rust_kernel_available():
        pytest.skip("native kernel is not built")
    errors = {"kernel": [], "weak_form": []}
    for seed in range(6):
        _t, S = _oscillator(0.10, seed)
        frame = pd.DataFrame({"x0": S[:, 0], "x1": S[:, 1]})
        chosen = {}
        for estimator in errors:
            proposals = CoupledOdeProposer(dt=0.01, coefficient_estimator=estimator).fit_and_propose(frame)
            eq = next(p for p in proposals if p.metadata["state_variable"] == "x1")
            assert eq.metadata["coefficient_estimator"] == estimator
            chosen[estimator] = set(eq.parameters)
            errors[estimator].append(max(abs(eq.parameters.get("x0", 0.0) + 4.0), abs(eq.parameters.get("x1", 0.0) + 0.3)))
            if estimator == "weak_form":
                assert "dx1/dt = " in eq.formula and f"{eq.parameters['x0']:.4f}" in eq.formula
        assert chosen["kernel"] == chosen["weak_form"]  # selection is untouched
    assert np.median(errors["weak_form"]) < 0.5 * np.median(errors["kernel"])
    assert CoupledOdeProposer()._coefficient_estimator == "kernel"
    with pytest.raises(ValueError, match="coefficient_estimator"):
        CoupledOdeProposer(coefficient_estimator="spline")


def test_refit_returns_none_when_there_is_too_little_data():
    series = {"x0": np.arange(4.0), "x1": np.arange(4.0)}
    assert weak_form.refit_coefficients(series, "x0", ["x1"], 0.1) is None
    assert weak_form.refit_coefficients({"x0": np.arange(50.0), "x1": np.arange(50.0)}, "x0", [], 0.1) is None
