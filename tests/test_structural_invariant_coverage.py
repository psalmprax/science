"""Every declared structural invariant must actually be decided by the model.

The structural axis is the axis most at risk of the failure mode this repository
keeps re-finding: a check that reads the *record* (a variable's name, a declared
bound, the text of a formula) instead of the model's behaviour, so it passes
whether or not the physics is right. Each case below therefore pairs a model that
must violate the invariant with one that must not, and asserts on the probe that
reports the finding -- not on a verdict, which other axes also influence.

Expectations come from the invariant's meaning, not from a recorded run: e.g.
non-negativity is violated by a quantity that evaluates negative *anywhere* in the
declared envelope, so ``x**2 - 0.5`` on ``[-1, 1]`` is a violation even though
both declared bounds are clean. A case that fails here is a finding to fix, not a
baseline to re-pin.
"""
from __future__ import annotations

import pytest

from apedp.falsification import FalsificationAxis, IndependentAdversarialFalsifier

#: (invariant, formula, domains, probe that must fire, why)
VIOLATIONS = [
    ("positivity", "x**2 - 0.5", {"x": (-1.0, 1.0)}, "PositivityViolationProbe",
     "negative in the interior though not at the declared bounds"),
    ("strictly_positive", "-1.0 + 0.0 * x", {"x": (1.0, 2.0)}, "PositivityViolationProbe",
     "constant negative quantity"),
    ("non_negative", "x - 2.0", {"x": (1.0, 3.0)}, "PositivityViolationProbe",
     "crosses zero inside the envelope"),
    ("probability_bound", "0.5 + 0.6 * x", {"x": (0.0, 1.0)}, "ProbabilityCeilingProbe",
     "exceeds the absolute probability ceiling"),
    ("third_law_thermodynamics", "0.0 - x", {"x": (0.0, 1.0)}, "ThirdLawThermodynamicsProbe",
     "temperature below absolute zero"),
    ("second_law_thermodynamics", "-1.0 * x", {"x": (1.0, 2.0)}, "SecondLawThermodynamicsProbe",
     "negative entropy production"),
    ("cosmic_censorship", "-1.0 * x", {"x": (1.0, 2.0)}, "CosmicCensorshipProbe",
     "positive-energy law evaluates negative"),
    ("unitarity_bound", "x**2", {"x": (0.0, 1.0e3)}, "UnitarityBoundProbe",
     "cross-section diverges with s"),
    ("current_density_ceiling", "1.0e5 * x", {"x": (1.0, 2.0)}, "CurrentDensityCeilingProbe",
     "current density far above the mass-transport ceiling"),
    ("velocity_saturation_ceiling", "1.0e6 * x", {"x": (1.0, 2.0)}, "VelocitySaturationCeilingProbe",
     "velocity above the optical-phonon ceiling"),
    ("yield_strength_ceiling", "1.0e10 * x", {"x": (1.0, 2.0)}, "YieldStrengthCeilingProbe",
     "stress above the material yield ceiling"),
    ("stokes_drag_ceiling", "1.0 * x", {"x": (1.0, 2.0)}, "StokesLaminarCeilingProbe",
     "drag force in the non-creeping regime"),
    ("saturation_ceiling", "1.0e10 * x", {"x": (1.0, 2.0)}, "RateSaturationCeilingProbe",
     "reaction rate above the physical saturation ceiling"),
    ("knudsen_continuum", "1.0 * x", {"x": (1.0e-9, 1.0e-6)}, "KnudsenContinuumBreakdownProbe",
     "envelope reaches continuum-invalid scales"),
    ("third_law_heat_capacity", "1.0 + x", {"x": (0.0, 1.0)}, "ThirdLawHeatCapacityProbe",
     "heat capacity does not vanish at absolute zero"),
    ("lorentz_causality", "sqrt(1.0 - x**2)", {"x": (0.0, 2.0)}, "LorentzCausalityProbe",
     "tachyonic: imaginary somewhere in the envelope"),
]

CLEAN = [
    ("positivity", "x**2 + 1.0", {"x": (-1.0, 1.0)}, "PositivityViolationProbe"),
    ("strictly_positive", "1.0 + x**2", {"x": (0.0, 2.0)}, "PositivityViolationProbe"),
    ("non_negative", "abs(x)", {"x": (-2.0, 2.0)}, "PositivityViolationProbe"),
    ("probability_bound", "0.5 + 0.4 * x", {"x": (0.0, 1.0)}, "ProbabilityCeilingProbe"),
    ("third_law_thermodynamics", "273.0 + x", {"x": (0.0, 100.0)}, "ThirdLawThermodynamicsProbe"),
    ("second_law_thermodynamics", "x", {"x": (1.0, 2.0)}, "SecondLawThermodynamicsProbe"),
    ("cosmic_censorship", "x**2 + 1.0", {"x": (0.0, 2.0)}, "CosmicCensorshipProbe"),
    ("unitarity_bound", "x", {"x": (0.0, 1.0e3)}, "UnitarityBoundProbe"),
    ("current_density_ceiling", "1.0 * x", {"x": (1.0, 2.0)}, "CurrentDensityCeilingProbe"),
    ("velocity_saturation_ceiling", "1.0 * x", {"x": (1.0, 2.0)}, "VelocitySaturationCeilingProbe"),
    ("yield_strength_ceiling", "1.0 * x", {"x": (1.0, 2.0)}, "YieldStrengthCeilingProbe"),
    ("stokes_drag_ceiling", "1.0e-6 * x", {"x": (1.0, 2.0)}, "StokesLaminarCeilingProbe"),
    ("saturation_ceiling", "1.0 * x", {"x": (1.0, 2.0)}, "RateSaturationCeilingProbe"),
    ("knudsen_continuum", "1.0 * x", {"x": (1.0e-3, 1.0e-2)}, "KnudsenContinuumBreakdownProbe"),
    ("third_law_heat_capacity", "x", {"x": (0.0, 1.0)}, "ThirdLawHeatCapacityProbe"),
    ("lorentz_causality", "sqrt(1.0 - x**2)", {"x": (0.0, 0.9)}, "LorentzCausalityProbe"),
]


def _structural_probes(formula: str, domains: dict[str, tuple[float, float]],
                       invariant: str) -> set[str]:
    report = IndependentAdversarialFalsifier().falsify(
        formula=formula,
        variable_domains=domains,
        physical_invariants=[invariant],
        critical_boundaries={},
    )
    return {
        c.probe_name for c in report.counterexamples
        if c.axis is FalsificationAxis.STRUCTURAL_INVARIANT
    }


@pytest.mark.parametrize("invariant,formula,domains,probe,why", VIOLATIONS,
                         ids=[f"{v[0]}:{v[1]}" for v in VIOLATIONS])
def test_declared_invariant_is_violated_and_reported(invariant, formula, domains, probe, why):
    fired = _structural_probes(formula, domains, invariant)
    assert probe in fired, (
        f"{invariant}: {formula} on {domains} should be refuted by {probe} ({why}); "
        f"structural probes that fired: {sorted(fired)}"
    )


@pytest.mark.parametrize("invariant,formula,domains,probe", CLEAN,
                         ids=[f"{c[0]}:{c[1]}" for c in CLEAN])
def test_clean_model_is_not_refuted_by_that_probe(invariant, formula, domains, probe):
    fired = _structural_probes(formula, domains, invariant)
    assert probe not in fired, (
        f"{invariant}: {formula} on {domains} satisfies the invariant but was refuted by {probe}"
    )


def test_a_constant_model_is_still_checked():
    """A model with no free symbols must not skip the axis silently.

    Regression: `_probe_structural_invariants` returned early when the expression
    had no free symbols, so a constant law was declared with invariants, evaluated
    by no check, and reported as `invariants_checked = 0` with an *empty*
    `probe_errors` list -- an unchecked invariant that the report never mentioned.
    """
    report = IndependentAdversarialFalsifier().falsify(
        formula="-1.0",
        variable_domains={"x": (1.0, 2.0)},
        physical_invariants=["positivity"],
        critical_boundaries={},
    )
    metrics = report.detailed_metrics["invariants"]
    assert metrics["invariants_declared"] == 1
    assert metrics["invariants_checked"] == 1, "the constant was declared but never checked"
    assert "PositivityViolationProbe" in {c.probe_name for c in report.counterexamples}
    assert report.survived is False


def test_a_satisfied_constant_model_is_checked_and_not_refuted():
    report = IndependentAdversarialFalsifier().falsify(
        formula="2.0 + 0.0 * x",
        variable_domains={"x": (1.0, 2.0)},
        physical_invariants=["positivity"],
        critical_boundaries={},
    )
    metrics = report.detailed_metrics["invariants"]
    assert metrics["invariants_checked"] == 1
    assert "PositivityViolationProbe" not in {c.probe_name for c in report.counterexamples}


def test_a_non_finite_constant_is_reported_as_unevaluable():
    """A constant with no finite real value is an unresolved probe, not a pass."""
    report = IndependentAdversarialFalsifier().falsify(
        formula="1e400",
        variable_domains={"x": (1.0, 2.0)},
        physical_invariants=["positivity"],
        critical_boundaries={},
    )
    metrics = report.detailed_metrics["invariants"]
    assert metrics["invariants_checked"] == 0
    assert [e["error"] for e in metrics["probe_errors"]] == ["NoRealValues"], metrics["probe_errors"]
    assert metrics["probe_errors"][0]["unresolved"] is True


def test_an_undeclared_invariant_is_not_checked():
    """Silence on an invariant nobody declared: no probes, zero checks attributed."""
    report = IndependentAdversarialFalsifier().falsify(
        formula="x**2 - 0.5",
        variable_domains={"x": (-1.0, 1.0)},
        physical_invariants=[],
        critical_boundaries={},
    )
    structural = [c for c in report.counterexamples if c.axis is FalsificationAxis.STRUCTURAL_INVARIANT]
    assert structural == [], [c.probe_name for c in structural]
    metrics = report.detailed_metrics["invariants"]
    assert metrics["invariants_declared"] == 0
    assert metrics["invariants_checked"] == 0


def test_renaming_a_symbol_cannot_disable_a_check():
    """The probes read the model, not the variable's name."""
    fired_named = _structural_probes("24.942 * (1.0 + 0.0001 * T_temp)", {"T_temp": (1.0, 300.0)},
                                     "third_law_heat_capacity")
    fired_renamed = _structural_probes("24.942 * (1.0 + 0.0001 * zzz)", {"zzz": (1.0, 300.0)},
                                       "third_law_heat_capacity")
    assert fired_named == fired_renamed == {"ThirdLawHeatCapacityProbe"}


def test_dimensional_mismatch_is_reported_when_units_are_declared():
    report = IndependentAdversarialFalsifier().falsify(
        formula="m * a",
        variable_domains={"m": (1.0, 2.0), "a": (1.0, 2.0)},
        physical_invariants=[],
        critical_boundaries={},
        variable_units={"m": "kilogram", "a": "meter / second**2"},
        expected_output_unit="newton",
    )
    assert "DimensionalHomogeneityProbe" not in {
        c.probe_name for c in report.counterexamples
    }, "kg*m/s**2 is a newton; the dimensional probe must not fire"
