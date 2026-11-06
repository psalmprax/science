"""Two-DOF saturation, fitted end to end.

`_bind_scale` in `apedp_dyn/grammar/coupled.py` now registers a saturation law's
reference level as its own fitted parameter, so `xddot = a*x/(L+|x|)` is
buildable in the two-DOF grammar. Bindable is not the same as fitted, and fitted
is not the same as identified: these pins walk the whole chain on a planted law

    enumerate -> prune (gates 1-2) -> materialize -> trajectory fit (gates 3-4)

and check that each mass gets its own reference level, that both are estimated
from the trajectory, and that a saturation law does not displace the linear
coupled truth on the real D2-COUPLED problem.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from measure_two_dof_saturation_fit import PLANT as _PLANT
from measure_two_dof_saturation_fit import planted_two_dof

from apedp_dyn import fit_candidate, prune_admissible
from apedp_dyn.grammar.coupled import enumerate_coupled, materialize_system
from apedp_dyn.problems import coupled

# One definition of the plant, in the script that records the run: both masses
# carry Langmuir (Michaelis-Menten) drag with their own half-saturation level, so
# a shared scale parameter would be visibly wrong.
_C1, _C2 = _PLANT["c"]
_A1, _A2 = _PLANT["a"]
_S1, _S2 = _PLANT["s"]
_planted_two_dof = planted_two_dof


def _pick(systems, per_dof: set[str]):
    """The candidate whose two equations both carry exactly ``per_dof``."""
    for system in systems:
        if all(set(equation.term_set()) == per_dof
               for equation in system.equations):
            return system
    raise AssertionError(f"no candidate with {sorted(per_dof)} in both equations")


def _saturated_system():
    """The slate that matches the plant exactly: u, damping and one saturated drag.

    `x` is deliberately absent. With it in the slate the fitter can trade a
    linear term against the Langmuir knee (measured: q_x = +0.23 to offset
    q_langmuir = -2.43 where the plant has -2.0), which makes parameter recovery
    unreadable even though the trajectory is right.
    """
    return _pick(
        enumerate_coupled(optional=("v", "langmuir_bind"), max_terms=4),
        {"u", "v", "langmuir_bind"})


def test_the_enumerator_gives_each_mass_its_own_reference_level():
    system = _saturated_system()
    scales = {term.scale_parameter
              for equation in system.equations for term in equation.terms
              if term.scale_parameter}
    assert scales == {"s0_langmuir", "s1_langmuir"}


def test_the_saturated_two_dof_model_materialises_with_scale_parameters():
    model = materialize_system(_planted_two_dof(), _saturated_system())
    by_name = {p.name: (p.role, p.units) for p in model.parameters}
    assert by_name["s0_langmuir"] == ("scale", "meter")
    assert by_name["s1_langmuir"] == ("scale", "meter")
    # The generic build symbol must not survive into an emitted expression.
    for equation in model.equations:
        assert "gL" not in equation.expression, equation.expression
    assert "Abs" in model.equations[1].expression


def test_the_saturated_system_passes_gates_1_and_2():
    kept = prune_admissible(_planted_two_dof(), [_saturated_system()])
    assert len(kept) == 1, (
        "the two-DOF saturation system was rejected before fitting: "
        f"{kept}")
    assert kept[0].provenance["gate1_status"]["passed"]
    assert kept[0].provenance["gate2_status"]["passed"]


def test_the_two_dof_saturation_law_is_fitted_end_to_end():
    """The chain runs and the reference levels come back from the data."""
    prob = _planted_two_dof()
    scored = fit_candidate(prob, _saturated_system(), max_nfev=400)
    assert scored.train_loss is not None, scored.provenance
    assert scored.train_loss < 5e-3, scored.train_loss
    fitted = scored.fit.param_values
    assert {"s0_langmuir", "s1_langmuir"} <= set(fitted), fitted

    # Each level is identified to within a factor of two of what was planted,
    # and the two differ: a single shared scale could not produce both.
    assert 0.5 * _S1 < fitted["s0_langmuir"] < 2.0 * _S1, fitted
    assert 0.5 * _S2 < fitted["s1_langmuir"] < 2.0 * _S2, fitted
    assert abs(fitted["s0_langmuir"] - fitted["s1_langmuir"]) > 0.02, fitted

    # Signs: saturation pulls back toward the origin, damping opposes velocity.
    coefficients = [v for k, v in fitted.items() if k.startswith("q")]
    assert min(coefficients) < 0.0, fitted


@pytest.mark.parametrize("law", ["langmuir_bind", "hill2_bind", "hill3_bind"])
def test_every_scaled_law_reaches_a_fittable_two_dof_model(law):
    """All three saturation laws, not just the one the enumeration test uses."""
    system = _pick(
        enumerate_coupled(optional=("x", "v", law), max_terms=4),
        {"u", "x", "v", law})
    model = materialize_system(_planted_two_dof(), system)
    roles = {p.name: p.role for p in model.parameters}
    assert roles[f"s0_{law.removesuffix('_bind')}"] == "scale"
    assert roles[f"s1_{law.removesuffix('_bind')}"] == "scale"


def _coupled_system(*, with_saturation: bool):
    """D2-COUPLED's truth, optionally nested inside a Langmuir rival.

    Built explicitly rather than picked out of the enumeration: the two systems
    under comparison must differ in exactly one term, and the enumeration's
    ordering decides which parameter name a slate receives.
    """
    from apedp_dyn.grammar.candidate_ir import (
        CandidateDynamics,
        SystemCandidate,
        Term,
    )

    bounds = dict.fromkeys(
        ("q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"), (-1e6, 1e6))
    equations = []
    for dof, base in ((0, 0), (1, 4)):
        terms = [Term("u", f"q{base}"), Term("x", f"q{base + 1}"),
                 Term("dxr", f"q{base + 2}"), Term("v", f"q{base + 3}")]
        if with_saturation:
            scale = f"s{dof}_langmuir"
            terms.append(Term("langmuir_bind", f"q{base + 8}",
                              scale_parameter=scale))
            bounds[scale] = (1e-9, 1e9)
            bounds[f"q{base + 8}"] = (-1e6, 1e6)
        equations.append(CandidateDynamics(
            lhs="xddot", terms=tuple(terms),
            candidate_id=f"d2coupled-d{dof}", parameter_bounds=bounds))
    return SystemCandidate(equations=tuple(equations),
                           candidate_id=("d2coupled-sat" if with_saturation
                                         else "d2coupled-truth"))


def test_a_saturation_rival_buys_nothing_on_the_linear_coupled_problem():
    """D2-COUPLED is linear. Extra saturation structure must not pay.

    The rival carries the coupling terms *and* a Langmuir term, so it nests the
    truth: if gate 4 let a richer hypothesis win on confirmation, this is where
    it would show.
    """
    prob = coupled.load()
    scored_truth = fit_candidate(prob, _coupled_system(with_saturation=False),
                                 max_nfev=400)
    scored_rival = fit_candidate(prob, _coupled_system(with_saturation=True),
                                 max_nfev=400)
    assert scored_truth.train_loss is not None, scored_truth.provenance
    assert scored_rival.train_loss is not None, scored_rival.provenance
    assert scored_truth.holdout_rmse is not None, scored_truth.provenance
    assert scored_rival.holdout_rmse is not None, scored_rival.provenance

    # Nesting means the rival can only fit train better; the question is
    # whether that extra freedom transfers.
    assert scored_rival.train_loss <= scored_truth.train_loss * 1.05
    assert scored_rival.holdout_rmse >= scored_truth.holdout_rmse * 0.99, (
        "the saturating rival beat the linear truth on confirmation: "
        f"{scored_rival.holdout_rmse} vs {scored_truth.holdout_rmse}")


# ------------------------------------------------------- the recorded run
#
# `scripts/measure_two_dof_saturation_fit.py` walks the whole chain and writes
# every number it uses. These pins keep the record honest rather than letting it
# drift into a snapshot nothing re-derives.

RECORD = ROOT / "benchmarks" / "two_dof_saturation_fit_results.json"


def test_the_recorded_run_recovers_the_planted_parameters():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["schema"] == "two_dof_saturation_fit/v1"
    assert record["slate"]["kept_after_gates_1_2"] >= 4
    recovery = record["recovery"]
    for role in ("dof0.scale", "dof1.scale",
                 "dof0.saturation", "dof1.saturation"):
        assert role in recovery, sorted(recovery)
        assert recovery[role]["relative_error"] < 0.35, recovery[role]
    assert record["fits"]["langmuir_bind"]["train_loss"] < 5e-3


def test_the_recorded_run_shows_the_warm_start_falling_short_of_the_minimum():
    """Pinned defect: the self-started fit stops two orders of magnitude high.

    A plant-seeded fit reaches every parameter to within 1 % at a loss ~260x
    lower, while the automatic warm start terminates in about two dozen
    evaluations at the higher loss and reports success. So the recorded recovery
    is the warm start's ceiling, not the information in the data. When the warm
    start is fixed, this test should fail and be rewritten with the new numbers.
    """
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    floor = record["objective_floor"]
    assert floor["warm_start_over_floor"] > 50, floor
    assert floor["recovery"], floor
    for role, entry in floor["recovery"].items():
        assert entry["relative_error"] < 0.05, (role, entry)


def test_the_recorded_noise_sweep_covers_three_levels():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    sweep = record["noise_sweep"]
    assert [entry["relative_noise"] for entry in sweep] == [0.0, 0.01, 0.03]
    for entry in sweep:
        assert set(entry["scale_recovery"]) == {"dof0.scale", "dof1.scale"}
