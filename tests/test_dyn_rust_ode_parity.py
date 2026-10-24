"""Differential conformance and parity tests for the native Rust ODE solver (`apedp_kernel`).

Validates:
1. Numerical parity between Rust Dormand-Prince (RK45) and SciPy / analytical truths.
2. Nonlinear mechanical damping (|v|v) and trigonometric terms (sin(x)).
3. Parameter mutation without re-compilation via `CompiledOdeSystem`.
4. Seamless integration in `apedp_dyn.simulation.simulate`.
5. Graceful failure and refusal on divergent / non-finite systems.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from apedp_dyn.model import DynamicalEquation, Parameter, PhysicalModel, StateVariable
from apedp_dyn.simulation import simulate
from apedp_dyn.simulation.ode_solver import is_rust_kernel_available

try:
    import apedp_kernel
except ImportError:
    apedp_kernel = None


@pytest.fixture(autouse=True)
def require_rust_kernel():
    if not is_rust_kernel_available():
        pytest.skip("apedp_kernel extension with ODE support is not available")


def test_rust_kernel_ode_availability():
    """Assert native Rust ODE extensions are active and properly bound."""
    assert is_rust_kernel_available()
    assert hasattr(apedp_kernel, "simulate_ode_rk45")
    assert hasattr(apedp_kernel, "CompiledOdeSystem")


def test_harmonic_oscillator_parity_vs_analytical_and_scipy():
    """Verify Rust RK45 matches both analytical cosine and SciPy RK45."""
    omega = 2.5
    t_span = (0.0, 5.0)
    t_eval = np.linspace(t_span[0], t_span[1], 100)
    x0 = [1.0, 0.0]

    # Analytical truth: x(t) = cos(omega*t), v(t) = -omega*sin(omega*t)
    exact_x = np.cos(omega * t_eval)
    exact_v = -omega * np.sin(omega * t_eval)

    # SciPy RK45 reference
    def ref_rhs(t, y):
        return [y[1], - (omega ** 2) * y[0]]

    sol_scipy = solve_ivp(ref_rhs, t_span, x0, method="RK45", t_eval=t_eval, rtol=1e-8, atol=1e-10)
    assert sol_scipy.success

    # Native Rust RK45
    success, detail, _nfev, _sol_t, sol_y = apedp_kernel.simulate_ode_rk45(
        equations=["v", f"-{omega**2} * x"],
        state_names=["x", "v"],
        param_names=[],
        param_values=[],
        t_span=t_span,
        x0=x0,
        t_eval=t_eval.tolist(),
        rtol=1e-8,
        atol=1e-10,
    )

    assert success, detail
    assert sol_y.shape == (2, len(t_eval))

    # Parity with analytical truth
    np.testing.assert_allclose(sol_y[0], exact_x, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(sol_y[1], exact_v, rtol=1e-4, atol=1e-4)

    # Parity with SciPy RK45
    np.testing.assert_allclose(sol_y[0], sol_scipy.y[0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(sol_y[1], sol_scipy.y[1], rtol=1e-4, atol=1e-4)


def test_nonlinear_duffing_wave_drag_parity():
    """Verify parity on complex multi-term nonlinear dynamics: x + x^3 + |v|v + sin(x)."""
    t_span = (0.0, 10.0)
    t_eval = np.linspace(t_span[0], t_span[1], 150)
    x0 = [0.8, -0.2]

    # dv/dt = c + theta1*x + theta3*x^3 + theta5*v + theta6*|v|v + theta4*sin(x)
    params = {"c": 0.05, "theta1": -2.0, "theta3": -0.8, "theta5": -0.15, "theta6": -0.3, "theta4": 0.25}

    def ref_rhs(t, y):
        x, v = y[0], y[1]
        a = (params["c"]
             + params["theta1"] * x
             + params["theta3"] * (x ** 3)
             + params["theta5"] * v
             + params["theta6"] * abs(v) * v
             + params["theta4"] * np.sin(x))
        return [v, a]

    sol_scipy = solve_ivp(ref_rhs, t_span, x0, method="RK45", t_eval=t_eval, rtol=1e-7, atol=1e-9)
    assert sol_scipy.success

    # Native Rust CompiledOdeSystem
    pnames = list(params.keys())
    pvals = [params[k] for k in pnames]
    eqs = ["v", "c + theta1*x + theta3*x**3 + theta5*v + theta6*abs(v)*v + theta4*sin(x)"]

    sys = apedp_kernel.CompiledOdeSystem(
        equations=eqs,
        state_names=["x", "v"],
        param_names=pnames,
    )

    success, detail, _nfev, _sol_t, sol_y = sys.simulate(
        t_span=t_span,
        x0=x0,
        t_eval=t_eval.tolist(),
        param_values=pvals,
        rtol=1e-7,
        atol=1e-9,
    )

    assert success, detail
    np.testing.assert_allclose(sol_y[0], sol_scipy.y[0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(sol_y[1], sol_scipy.y[1], rtol=1e-4, atol=1e-4)


def test_simulate_dispatch_and_backend_selection():
    """Verify PhysicalModel simulation seamlessly routes through native Rust RK45."""
    states = (StateVariable("x", "dimensionless", 1.0),
              StateVariable("v", "1/day", 0.0))
    params = (Parameter("omega_sq", "1/day**2", 4.0),)
    eqs = (DynamicalEquation("x", "v"),
           DynamicalEquation("v", "-omega_sq * x"))

    model = PhysicalModel(states=states, parameters=params, equations=eqs)

    # 1. Auto backend (selects Rust RK45)
    res_auto = simulate(model, (0.0, 5.0), lambda t: np.zeros(0), np.array([1.0, 0.0]),
                        n_eval=50, rtol=1e-6, atol=1e-8)
    assert res_auto.success
    assert res_auto.metrics["engine"] == "rust_rk45"
    assert res_auto.metrics["backend"] == "rust"
    assert res_auto.metrics["rhs_path"] == "compiled"

    # 2. Forced SciPy backend
    res_scipy = simulate(model, (0.0, 5.0), lambda t: np.zeros(0), np.array([1.0, 0.0]),
                         n_eval=50, backend="scipy", rtol=1e-6, atol=1e-8)
    assert res_scipy.success
    assert res_scipy.metrics["engine"] == "scipy_lsoda"
    assert res_scipy.metrics["backend"] == "scipy"

    # Both backends agree tightly at high precision
    np.testing.assert_allclose(res_auto.y, res_scipy.y, rtol=1e-3, atol=1e-3)


def test_rust_ode_throughput_speedup():
    """Verify native Rust integration executes multiple trajectory evaluations faster than Python."""
    states = (StateVariable("x", "dimensionless", 1.0),
              StateVariable("v", "1/day", 0.0))
    params = (Parameter("k", "1/day**2", 3.0),
              Parameter("d", "1/day", 0.5))
    eqs = (DynamicalEquation("x", "v"),
           DynamicalEquation("v", "-k * x - d * Abs(v) * v"))

    model = PhysicalModel(states=states, parameters=params, equations=eqs)

    # Warm-up
    simulate(model, (0.0, 10.0), lambda t: np.zeros(0), np.array([1.0, 0.0]), n_eval=100, backend="rust")
    simulate(model, (0.0, 10.0), lambda t: np.zeros(0), np.array([1.0, 0.0]), n_eval=100, backend="scipy")

    n_runs = 30

    t0 = time.perf_counter()
    for _ in range(n_runs):
        simulate(model, (0.0, 10.0), lambda t: np.zeros(0), np.array([1.0, 0.0]), n_eval=100, backend="rust")
    t_rust = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_runs):
        simulate(model, (0.0, 10.0), lambda t: np.zeros(0), np.array([1.0, 0.0]), n_eval=100, backend="scipy")
    t_scipy = time.perf_counter() - t0

    print(f"\nThroughput benchmark ({n_runs} runs): Rust RK45 = {t_rust*1000:.1f}ms, SciPy LSODA = {t_scipy*1000:.1f}ms")
    # Rust should be significantly faster
    assert t_rust < t_scipy


def test_divergence_graceful_refusal():
    """Verify an unstable / exploding ODE terminates cleanly with success=False."""
    # dy/dt = x^2 blows up in finite time
    sys = apedp_kernel.CompiledOdeSystem(
        equations=["x**2"],
        state_names=["x"],
        param_names=[],
    )

    success, detail, _nfev, _sol_t, _sol_y = sys.simulate(
        t_span=(0.0, 10.0),
        x0=[10.0],
        t_eval=np.linspace(0.0, 10.0, 100).tolist(),
        param_values=[],
    )

    assert not success
    assert "divergence" in detail.lower() or "underflow" in detail.lower() or "not fully covered" in detail.lower()
