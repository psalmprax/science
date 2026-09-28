"""D5 generalized energy validators: the law is derived from the model.

The conserved quantity must follow the structure rather than a hardcoded
mass-spring form, so the same validators cover cubic restoring (Duffing),
multi-DOF coupling and the pendulum — and must REFUSE (never guess) when the
restoring force is not a gradient field.
"""
import dataclasses

import numpy as np
import pytest
import sympy as sp

from apedp_dyn import (
    DynamicalEquation,
    Parameter,
    PhysicalModel,
    StateVariable,
    check_conservative,
    check_dissipative,
    derive_energy_law,
    simulate_trajectory,
)
from apedp_dyn.problems import coupled, duffing, mass_spring_damper, pendulum

_T = np.linspace(0.0, 10.0, 400)


def _model(prob, values=None) -> PhysicalModel:
    model = PhysicalModel(prob.states, prob.inputs, prob.parameters,
                          prob.truth_equations)
    if values:
        model = dataclasses.replace(model, parameters=tuple(
            dataclasses.replace(p, value=values.get(p.name, p.value))
            for p in model.parameters))
    return model


def _free(model, x0) -> np.ndarray:
    """Undamped, unforced trajectory (zero input, matching input arity)."""
    def zero(t: float) -> np.ndarray:
        return np.zeros(len(model.inputs))

    return simulate_trajectory(model, _T, zero, x0)


def test_derived_potential_matches_analytic_forms():
    msd = derive_energy_law(_model(mass_spring_damper.load()))
    assert msd.conservative
    (x,) = msd.position_symbols
    assert sp.simplify(msd.potential - 2 * x ** 2) == 0  # k=4: (1/2) k x^2

    duff = derive_energy_law(_model(duffing.load()))
    assert duff.conservative
    (x,) = duff.position_symbols
    # alpha=4, bnl=2: (1/2) a x^2 + (1/4) b x^4
    assert sp.simplify(duff.potential - (2 * x ** 2 + sp.Rational(1, 2) * x ** 4)) == 0

    comp = derive_energy_law(_model(coupled.load()))
    assert comp.conservative and comp.velocity_indices == (1, 3)
    x1, x2 = comp.position_symbols
    # k1=4, k2=6, kc=1.5: (1/2)k1 x1^2 + (1/2)k2 x2^2 + (1/2)kc (x1-x2)^2
    want = (sp.Rational(11, 4) * x1 ** 2 - sp.Rational(3, 2) * x1 * x2
            + sp.Rational(15, 4) * x2 ** 2)
    assert sp.simplify(comp.potential - want) == 0


def test_duffing_conserves_energy_without_damping_or_drive():
    model = _model(duffing.load(), {"delta": 0.0, "gmu": 0.0})
    ok, drift = check_conservative(model, _free(model, (0.5, 0.0)))
    assert ok, drift
    assert drift < 1e-6


def test_duffing_anti_damped_impostor_is_exposed():
    model = _model(duffing.load(), {"delta": -0.2, "gmu": 0.0})
    ok, drift = check_conservative(model, _free(model, (0.5, 0.0)))
    assert not ok and drift > 0.1, drift


def test_coupled_conserves_energy_without_damping_or_drive():
    model = _model(coupled.load(),
                   {"c1": 0.0, "c2": 0.0, "g1": 0.0, "g2": 0.0})
    ok, drift = check_conservative(model, _free(model, (0.4, 0.0, -0.3, 0.0)))
    assert ok, drift
    assert drift < 1e-6


def _asymmetric_coupling() -> PhysicalModel:
    """Independent k12 != k21: the restoring 1-form is not exact."""
    states = (StateVariable("x1", "meter", 0.5),
              StateVariable("v1", "meter/second", 0.0),
              StateVariable("x2", "meter", -0.3),
              StateVariable("v2", "meter/second", 0.0))
    params = (Parameter("k1", "1/second**2", 4.0),
              Parameter("k2", "1/second**2", 6.0),
              Parameter("k12", "1/second**2", 1.5),
              Parameter("k21", "1/second**2", 0.4))
    eqs = (DynamicalEquation("x1", "v1"),
           DynamicalEquation("v1", "-k1*x1 - k12*(x1 - x2)"),
           DynamicalEquation("x2", "v2"),
           DynamicalEquation("v2", "-k2*x2 - k21*(x2 - x1)"))
    return PhysicalModel(states, (), tuple(params), eqs)


def test_asymmetric_coupling_has_no_energy_law_and_is_not_guessed():
    model = _asymmetric_coupling()
    law = derive_energy_law(model)
    assert not law.conservative
    assert law.potential is None
    assert "not exact" in law.detail, law.detail
    # An underivable balance is not evidence of conservation.
    ok, drift = check_conservative(model, _free(model, (0.5, 0.0, -0.3, 0.0)))
    assert not ok and drift == float("inf")


def test_non_mechanical_model_reports_underivable_law():
    model = PhysicalModel((StateVariable("y", "meter", 1.0),), (),
                          (Parameter("lam", "1/second", 0.5),),
                          (DynamicalEquation("y", "-lam*y"),))
    law = derive_energy_law(model)
    assert not law.conservative and law.potential is None
    assert "no position/velocity state pair" in law.detail, law.detail


def test_dissipative_balance_on_duffing_and_coupled():
    damped_duff = _model(duffing.load(), {"gmu": 0.0})
    ok, info = check_dissipative(damped_duff, _free(damped_duff, (0.5, 0.0)))
    assert ok, info

    anti = _model(duffing.load(), {"delta": -0.2, "gmu": 0.0})
    ok, info = check_dissipative(anti, _free(anti, (0.5, 0.0)))
    assert not ok, info
    # Sustained exponential growth has an almost constant per-step relative
    # change, so the CUMULATIVE measure is what exposes it: negative decay.
    assert info["decay_fraction"] < 0.0, info

    damped_cpl = _model(coupled.load(), {"g1": 0.0, "g2": 0.0})
    ok, info = check_dissipative(damped_cpl, _free(damped_cpl, (0.4, 0.0, -0.3, 0.0)))
    assert ok, info


def _restoring_model(expr: str) -> PhysicalModel:
    """1-DOF model whose acceleration law is the given expression."""
    states = (StateVariable("x", "meter", 0.5),
              StateVariable("v", "meter/second", 0.0))
    return PhysicalModel(states, (), (Parameter("k", "1/second**2", 4.0),),
                         (DynamicalEquation("x", "v"),
                          DynamicalEquation("v", expr)))


@pytest.mark.parametrize(("expr", "needle"), [
    ("-k*x - exp(x)", "outside the derivable class"),
    ("-k*x - erf(x)", "outside the derivable class"),
    ("-k*x - x**9", "exceeds budget"),
    ("-k*x - sin(x)/x", "not polynomial"),
])
def test_bounded_derivation_refuses_instead_of_hanging(expr: str, needle: str):
    """Exact integration has no natural stopping point, so it must be bounded.

    A validator that hangs is worse than one that refuses: the gate would never
    return, and nothing would report why. Anything outside the admitted class
    comes back as ``conservative=False`` with the reason.
    """
    law = derive_energy_law(_restoring_model(expr))
    assert not law.conservative
    assert law.potential is None
    assert needle in law.detail, law.detail


def test_bounded_derivation_still_admits_every_real_problem():
    """The budgets must not refuse the mechanical forms the grammar emits."""
    for prob in (mass_spring_damper.load(), duffing.load(), pendulum.load(),
                 coupled.load()):
        law = derive_energy_law(_model(prob))
        assert law.conservative, (prob.problem_id, law.detail)


def test_pendulum_potential_and_conservation():
    model = _model(pendulum.load(), {"b": 0.0})
    law = derive_energy_law(model)
    assert law.conservative
    (theta,) = law.position_symbols
    # U(theta) - U(0) == (g/L)(1 - cos theta), g/L = 9.81
    assert sp.simplify((law.potential - law.potential.subs(theta, 0))
                       - 9.81 * (1 - sp.cos(theta))) == 0
    ok, drift = check_conservative(model, _free(model, (0.6, 0.0)))
    assert ok, drift
