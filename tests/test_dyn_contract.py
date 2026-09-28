"""D1 contract + gate unit tests (problem-independent)."""
import numpy as np

from apedp_dyn import (
    DynamicalEquation,
    InputVariable,
    Parameter,
    PhysicalModel,
    StateVariable,
    dimensional_validity,
    equation_admissibility,
    run_all_gates,
)
from apedp_dyn.problems import mass_spring_damper


def _truth_model():
    p = mass_spring_damper.spec()
    return PhysicalModel(p.states, p.inputs, p.parameters, p.truth_equations)


def test_truth_model_passes_gates_1_and_2():
    m = _truth_model()
    g1 = dimensional_validity(m)
    g2 = equation_admissibility(m)
    assert g1.passed, g1.detail
    assert g2.passed, g2.detail


def test_dimensional_gate_rejects_bad_units():
    m = _truth_model()
    bad = PhysicalModel(
        m.states, m.inputs, m.parameters,
        (DynamicalEquation("x", "v"),
         DynamicalEquation("v", "u - c*v - k*x")))  # missing /m: m/s^2 vs required
    g = dimensional_validity(bad)
    assert not g.passed, g.detail


def test_admissibility_rejects_unknown_symbol():
    m = _truth_model()
    bad = PhysicalModel(
        m.states, m.inputs, m.parameters,
        (DynamicalEquation("x", "v"),
         DynamicalEquation("v", "(u - c*v - k*x)/m + damping Gremlin".replace(" Gremlin", "_zzz"))))
    g = equation_admissibility(bad)
    assert not g.passed, g.detail


def test_admissibility_rejects_missing_equation():
    m = _truth_model()
    bad = PhysicalModel(m.states, m.inputs, m.parameters,
                        (DynamicalEquation("x", "v"),))
    g = equation_admissibility(bad)
    assert not g.passed, g.detail


def test_admissibility_rejects_unidentified_parameter():
    p = mass_spring_damper.spec()
    params = tuple(Parameter(q.name, q.units, None) if q.name == "k" else q
                   for q in p.parameters)
    m = PhysicalModel(p.states, p.inputs, params, p.truth_equations)
    g = equation_admissibility(m)
    assert not g.passed and "k" in g.detail, g.detail


def test_admissibility_rejects_derivative_in_rhs():
    import sympy as sp
    m = _truth_model()
    t = sp.Symbol("t")
    x = sp.Function("x")
    bad_eq = DynamicalEquation.__new__(DynamicalEquation)
    object.__setattr__(bad_eq, "state", "v")
    object.__setattr__(bad_eq, "expression", "Derivative(x(t), t)")
    object.__setattr__(bad_eq, "sympy_expr", sp.Derivative(x(t), t))
    bad = PhysicalModel(m.states, m.inputs, m.parameters,
                        (m.equations[0], bad_eq))
    g = equation_admissibility(bad)
    assert not g.passed, g.detail


def test_truth_model_passes_all_four_gates_on_mass_spring_damper():
    prob = mass_spring_damper.load()
    m = PhysicalModel(prob.states, prob.inputs, prob.parameters, prob.truth_equations)
    results = run_all_gates(m, prob)
    assert [r.name for r in results] == [
        "dimensional_validity", "equation_admissibility",
        "forward_simulation_validity", "intervention_holdout"]
    for r in results:
        assert r.passed, f"{r.name}: {r.detail}"
