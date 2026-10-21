"""Level-5 Pillar 2 pins: autonomous dimensional & boundary synthesis.

The Π reduction must be physically verified (products are actually
dimensionless under pint), the boundary synthesis must recover BOTH
Langmuir constraints (origin + saturation) from the training partition
ALONE, and ambiguous curves must fail closed (no guessed constraint).
"""
from __future__ import annotations

import numpy as np
import pint
import pytest

from apedp.dimensional_invariant_engine import (
    infer_boundary_constraints,
    synthesize_invariants,
)


def test_pendulum_pi_group_is_autonomously_discovered_and_truly_dimensionless():
    units = {"m": "kilogram", "L": "meter", "g": "meter / second**2", "T": "second"}
    synthesis = synthesize_invariants(units)
    assert len(synthesis.pi_groups) == 1, synthesis.pi_groups
    group = synthesis.pi_groups[0]
    # Verify the product Π = ∏ x_i^{γ_i} is ACTUALLY dimensionless (pint),
    # not merely claimed by the group's name/definition string.
    ureg = pint.UnitRegistry()
    product = 1.0 * ureg.dimensionless
    for name, exp in group.exponent_map().items():
        product = product * (1.0 * ureg(units[name])) ** float(exp)
    assert product.dimensionality == ureg.dimensionless.dimensionality
    # Pendulum law shape: the group must be g·T²/L up to orientation and
    # the mass must not appear (mass carries exponent 0).
    exp_map = {k: float(v) for k, v in group.exponent_map().items()}
    assert exp_map.get("m", 0.0) == 0.0
    assert exp_map.get("T", 0.0) != 0.0


def test_langmuir_boundary_synthesis_recovers_origin_and_saturation():
    # Langmuir on its natural axis: log-spaced pressure spanning 7 decades.
    # The plateau is asymptotic, so the log-band estimators are the honest
    # ones (the roadmap's p→∞ limit IS a log-axis statement).
    n = 200
    x = np.logspace(-3, 4, n)
    y = 5.0 * x / (2.0 + x)
    synthesis = synthesize_invariants(
        {"p": "dimensionless", "q": "dimensionless"}, x=x, y=y)
    bc = synthesis.boundary
    assert bc.origin_constrained, bc.evidence  # q(0) = 0 in the training data
    assert bc.saturation_limit is not None
    assert abs(bc.saturation_limit - 5.0) < 0.4, bc.evidence
    assert "rational_origin_constrained_b0" in synthesis.grammar_hints
    assert "rational_saturation_wx_over_s_plus_x" in synthesis.grammar_hints
    # Characteristic scale: half-max crossing of q = 5p/(2+p) is at p = 2.
    assert abs(synthesis.characteristic_scale - 2.0) / 2.0 < 0.35, synthesis.characteristic_scale
    lo, mid, hi = synthesis.scale_bank
    assert mid == pytest.approx(synthesis.characteristic_scale)
    assert lo == pytest.approx(mid / 3.0) and hi == pytest.approx(mid * 3.0)


def test_ambiguous_training_curve_fails_closed_no_constraint():
    x = np.linspace(0.0, 10.0, 50)
    y = 3.0 * x + 0.5  # growing, no origin, no plateau
    bc = infer_boundary_constraints(x, y)
    assert not bc.origin_constrained
    assert bc.saturation_limit is None


def test_growing_curve_without_origin_never_gets_origin_constraint():
    x = np.linspace(1.0, 10.0, 50)
    y = 2.0 * x
    assert not infer_boundary_constraints(x, y).origin_constrained


def test_non_integer_si_exponents_are_refused():
    with pytest.raises(ValueError, match="non-integer SI exponents"):
        synthesize_invariants({"k": "meter / second**0.5"})
