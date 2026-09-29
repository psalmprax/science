"""D1 contract + gate unit tests (problem-independent)."""

from apedp_dyn import (
    DynamicalEquation,
    Parameter,
    PhysicalModel,
    dimensional_validity,
    equation_admissibility,
    run_all_gates,
)
from apedp_dyn.gates import dao_validity
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


def test_truth_model_passes_every_gate_on_mass_spring_damper():
    prob = mass_spring_damper.load()
    m = PhysicalModel(prob.states, prob.inputs, prob.parameters, prob.truth_equations)
    results = run_all_gates(m, prob)
    # The full stack, in order. Pinned explicitly so adding or reordering a gate
    # is a deliberate edit to this test rather than a silent change.
    assert [r.name for r in results] == [
        "dimensional_validity", "equation_admissibility",
        "forward_simulation_validity", "intervention_holdout", "dao_validity"]
    for r in results:
        assert r.passed, f"{r.name}: {r.detail}"


def test_fit_candidate_evaluates_every_gate_run_all_gates_evaluates():
    """The promotion path must not skip a gate the audit path evaluates.

    dao_validity was added to run_all_gates but not to fit_candidate, so a
    candidate could be promoted without ever being checked against it — a gate
    only the audit path runs prevents nothing. The two sets are pinned equal so
    a future gate cannot be added to one path only.
    """
    from apedp_dyn.fitting import fit_candidate
    from apedp_dyn.fitting.parameter_fit import _GATE_PROVENANCE_KEY
    from apedp_dyn.grammar import enumerate_candidates, prune_admissible

    prob = mass_spring_damper.load()
    truth = next(s for s in prune_admissible(prob, enumerate_candidates(
        required=("u",), optional=("x", "v"), max_terms=3))
        if set(s.candidate.term_set()) == {"u", "x", "v"})
    res = fit_candidate(prob, truth.candidate)

    audit = {r.name for r in run_all_gates(_truth_model(), prob)}
    assert set(_GATE_PROVENANCE_KEY) == audit, (set(_GATE_PROVENANCE_KEY), audit)
    assert set(_GATE_PROVENANCE_KEY.values()) <= set(res.provenance), res.provenance
    # A gate that never runs cannot silently read as "passed".
    assert res.provenance["dao_status"]["passed"], res.provenance["dao_status"]


# --- Gate 1.5 (dao_validity) -------------------------------------------------
#
# The DAE branches used to be unreachable: PhysicalModel carried no
# ``dae_mode``, so dao_validity always took the pure-ODE early return, and
# the constrained branch iterated ``eq.terms``, an attribute
# DynamicalEquation does not have. The D3 tests wrapped every call in
# ``try/except: pass``, so a gate that returned garbage still "passed".
# These tests assert the verdicts instead.

def _dae_model(**kwargs):
    p = mass_spring_damper.spec()
    base = {
        "states": p.states, "inputs": p.inputs, "parameters": p.parameters,
        "equations": p.truth_equations,
    }
    base.update(kwargs)
    return PhysicalModel(**base)


def test_dao_gate_passes_pure_ode():
    g = dao_validity(_dae_model())
    assert g.passed, g.detail
    assert "pure ODE" in g.detail


def test_dao_gate_rejects_constrained_model_with_no_constraint_terms():
    """A constrained-mode model must actually carry a constraint parameter.

    Otherwise the DAE hypothesis is a label with nothing behind it, and the
    model is really just an ODE wearing a DAE badge.
    """
    g = dao_validity(_dae_model(dae_mode="constrained"))
    assert not g.passed
    assert "no constraint parameters" in g.detail


def test_dao_gate_passes_constrained_model_with_dimensionless_constraint():
    p = mass_spring_damper.spec()
    m = PhysicalModel(
        p.states, p.inputs, p.parameters, p.truth_equations,
        dae_mode="constrained", dae_constraint_params=("con",),
    )
    g = dao_validity(m)
    assert g.passed, g.detail
    assert "dimensionless" in g.detail


def test_dao_gate_rejects_dimensionful_constraint_parameter():
    """The constraint basis term is X**0, so its coefficient must be unitless."""
    p = mass_spring_damper.spec()
    params = (Parameter("con", "newton", None, (-1e6, 1e6)),)
    m = PhysicalModel(
        p.states, p.inputs, params, p.truth_equations,
        dae_mode="constrained", dae_constraint_params=("con",),
    )
    g = dao_validity(m)
    assert not g.passed
    assert "dimensionally inhomogeneous" in g.detail


def test_dao_gate_rejects_unknown_dae_mode():
    g = dao_validity(_dae_model(dae_mode="not_a_mode"))
    assert not g.passed
    assert "unknown dae_mode" in g.detail


def test_dao_gate_multiscale_accepts_declared_timescales():
    m = _dae_model(dae_mode="multi_timescale", dae_timescales=("fast", "slow"))
    g = dao_validity(m)
    assert g.passed, g.detail
    assert "timescales labeled" in g.detail


def test_dao_gate_multiscale_rejects_unlabeled_model():
    m = _dae_model(dae_mode="multi_timescale")
    g = dao_validity(m)
    assert not g.passed
    assert "cannot verify" in g.detail
