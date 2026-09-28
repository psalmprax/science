"""D1 problem tests: generation integrity + gate behavior on truth vs wrong structure."""
import numpy as np

from apedp_dyn import (
    DynamicalEquation,
    PhysicalModel,
    intervention_holdout,
    run_all_gates,
)
from apedp_dyn.problems import mass_spring_damper, pendulum


def _model_of(prob):
    return PhysicalModel(prob.states, prob.inputs, prob.parameters, prob.truth_equations)


def test_problem_trajectories_finite_and_shaped():
    for prob in (mass_spring_damper.load(), pendulum.load()):
        for regime in ("train", "holdout"):
            reg = getattr(prob, regime)
            assert reg["truth_t"].shape == (400,)
            assert reg["truth_y"].shape == (400, 2)
            assert np.all(np.isfinite(reg["truth_y"]))


def test_truth_passes_all_gates_both_problems():
    for prob in (mass_spring_damper.load(), pendulum.load()):
        results = run_all_gates(_model_of(prob), prob)
        for r in results:
            assert r.passed, f"{prob.problem_id}/{r.name}: {r.detail}"


def test_holdout_discriminates_structure_mass_spring_damper():
    prob = mass_spring_damper.load()
    truth_rmse = intervention_holdout(_model_of(prob), prob).metrics["rmse"]
    undamped = PhysicalModel(
        prob.states, prob.inputs, prob.parameters,
        (DynamicalEquation("x", "v"),
         DynamicalEquation("v", "(u - k*x)/m")))  # drops damping: passes gates 1-3
    res = intervention_holdout(undamped, prob)
    assert res.metrics["rmse"] > 10 * truth_rmse, res.metrics


def test_holdout_discriminates_structure_pendulum():
    prob = pendulum.load()
    truth_rmse = intervention_holdout(_model_of(prob), prob).metrics["rmse"]
    linear = PhysicalModel(
        prob.states, prob.inputs, prob.parameters,
        (DynamicalEquation("theta", "omega"),
         DynamicalEquation("omega", "-(g/L)*theta - b*omega + u")))
    res = intervention_holdout(linear, prob)
    assert res.metrics["rmse"] > 10 * truth_rmse, res.metrics
