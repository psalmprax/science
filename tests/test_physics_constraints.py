"""Unit tests for the Dedicated Physics Constraint Layer and Parameter Uncertainty.

PROVENANCE: re-authored, not recovered, and adapted. This file existed only
as ``tests/__pycache__/test_physics_constraints.cpython-312-pytest-*.pyc``
-- compiled, never committed, and therefore invisible to git, to review, and
to CI. The source was lost; the bytecode was not.

No decompiler exists for CPython 3.12, so the assertions were reconstructed
from the code objects: docstrings, literal constants, keyword names, and
attribute access all survive in a ``.pyc``. Original formatting, comments,
and local variable names did not.

Two honest departures from a pure recovery, both forced:

1. API drift. The orphaned bytecode targeted an older signature. ``evaluate``
   now takes ``x_domain`` as its own argument rather than reading it from the
   spec, ``boundary_conditions`` is a list of ``(point_dict, expected)``
   pairs rather than a single mapping, and ``compute_parameter_uncertainty``
   takes ``eval_fn(x, params)`` plus a ``param_values`` dict rather than a
   model callable and an initial guess. The tests below are written against
   the API that exists. Running the bytecode original against today's source
   raises TypeError on six of its eight tests, so it is not a usable
   regression baseline in its original form -- only its intent is.

2. Two negative directions are added rather than recovered. The original
   asserted only that the even-parity expression certifies and that the
   symmetric expression passes. A check that cannot fail is not a check, and
   the reconstruction is the right place to say so. These are additions and
   are marked as such inline.

Validated by running against today's source. The mutation-equivalence
harness (``tools/equivalence_check.py``) was built for the singularity file,
where both versions run; it does not apply here for reason 1.

This file matters more than its size suggests. ``apedp/physics_constraints.py``
is the module whose five silent exception handlers were fixed in 0f119d7 --
handlers that each contributed a full point to ``compliance_score`` for a
check that never ran, so an expression diverging at the origin could be
certified physically sound. The file covering that engine was in this state.
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp.parameter_diagnostics import compute_parameter_uncertainty
from apedp.physics_constraints import (
    PhysicsConstraintEngine,
    PhysicsSpecification,
)


@pytest.fixture
def physics_engine():
    return PhysicsConstraintEngine()


def test_origin_limit_and_far_field_extinction(physics_engine):
    """Verify compliance for rational-exponential envelope obeying physical extinction."""
    spec = PhysicsSpecification(
        origin_limit=0.0, far_field_limit=0.0, positivity=True
    )
    rep = physics_engine.evaluate(
        "2.45 * x**1.5 / (1.0 + 0.85 * x**2) * exp(-0.3 * x)", spec,
        x_domain=(0.1, 10.0),
    )
    assert rep.is_physically_sound is True
    assert rep.boundary_valid is True
    assert rep.positivity_valid is True
    assert rep.asymptotics_valid is True
    assert not rep.violations, rep.violations


def test_asymptotic_divergence_violation(physics_engine):
    """Verify rejection when a model diverges asymptotically instead of extinguishing."""
    # x**2 grows without bound, so the far-field limit is not the declared
    # extinction. This is the test a failed sp.limit would have broken
    # silently before the _unverified() handlers existed.
    spec = PhysicsSpecification(origin_limit=0.0, far_field_limit=0.0)
    rep = physics_engine.evaluate("2.5 * x**2 + 1.2 * x", spec, x_domain=(0.1, 10.0))
    assert rep.is_physically_sound is False
    assert rep.asymptotics_valid is False
    assert any("asymptotic" in v.lower() for v in rep.violations), rep.violations


def test_positivity_violation(physics_engine):
    """Verify detection when candidate violates physical state positivity."""
    spec = PhysicsSpecification(positivity=True)
    rep = physics_engine.evaluate("x - 4.0", spec, x_domain=(0.1, 5.0))
    assert rep.is_physically_sound is False
    assert rep.positivity_valid is False
    assert any("positiv" in v.lower() for v in rep.violations), rep.violations


def test_parity_symmetry_checks(physics_engine):
    """Verify parity invariance certification for even and odd physical functions."""
    spec_even = PhysicsSpecification(parity="even")
    rep_even = physics_engine.evaluate("x**2 + 2.0", spec_even, x_domain=(-5.0, 5.0))
    assert rep_even.parity_valid is True

    # ADDITION, not recovery: the original bytecode asserted only the even
    # case. The same spec against an expression that is not even must fail,
    # or the parity check is decoration.
    rep_not_even = physics_engine.evaluate(
        "x**2 + 3.0 * x", spec_even, x_domain=(-5.0, 5.0)
    )
    assert rep_not_even.parity_valid is False


def test_multivariable_dimensional_consistency(physics_engine):
    """Verify dimensional consistency check on multi-variable kinetic energy."""
    spec = PhysicsSpecification(
        variable_units={"m": "kilogram", "v": "meter / second"},
        target_unit="joule",
    )
    rep = physics_engine.evaluate("0.5 * m * v**2", spec)
    assert rep.dimensional_valid is True
    assert rep.is_physically_sound is True
    assert not any("dimension" in v.lower() for v in rep.violations), rep.violations

    # ADDITION: the same check must reject dimensionally inhomogeneous input.
    # Without this, a checker that returned True unconditionally would pass.
    rep_bad = physics_engine.evaluate("0.5 * m * v", spec)
    assert rep_bad.dimensional_valid is False


def test_multivariable_boundary_conditions(physics_engine):
    """Verify multi-variable boundary condition evaluations."""
    # No origin_limit here, deliberately. sp.limit on a two-variable
    # expression takes the limit in one symbol and leaves the other free, so
    # .evalf() raises "Cannot convert expression to float". The engine
    # reports that as "Origin limit check could not be evaluated" and scores
    # it as a failure rather than a pass -- which is the 0f119d7 behaviour
    # working, not a bug. Declaring an origin limit here would make this test
    # about that check instead of about boundary conditions.
    spec = PhysicsSpecification(
        # Current shape: a list of (point_dict, expected_value) pairs.
        boundary_conditions=[({"x": 0.0, "t": 0.0}, 10.0)],
    )
    rep = physics_engine.evaluate("10.0 * (1.0 - x) * exp(-t)", spec)
    assert rep.boundary_valid is True
    assert rep.is_physically_sound is True

    # ADDITION: a boundary condition the expression violates must be caught.
    spec_bad = PhysicsSpecification(
        boundary_conditions=[({"x": 0.0, "t": 0.0}, 999.0)],
    )
    rep_bad = physics_engine.evaluate("10.0 * (1.0 - x) * exp(-t)", spec_bad)
    assert rep_bad.boundary_valid is False


def test_an_unevaluable_origin_limit_is_not_a_pass(physics_engine):
    """A check that raised must not certify the report sound.

    This is the exact failure mode commit 0f119d7 fixed: before it, a handler
    that only logged left boundary_valid True and appended nothing, so a
    limit it never computed contributed a full point to compliance_score. The
    multivariable expression above is a routine way to reach it -- sp.limit
    leaves the second variable free and .evalf() then raises.
    """
    spec = PhysicsSpecification(
        origin_limit=0.0,
        boundary_conditions=[({"x": 0.0, "t": 0.0}, 10.0)],
    )
    rep = physics_engine.evaluate("10.0 * (1.0 - x) * exp(-t)", spec)
    assert any(
        "Origin limit check could not be evaluated" in v for v in rep.violations
    ), rep.violations
    # The unevaluable check is recorded, the report is not certified sound,
    # and it does not score a full point. boundary_valid is False too: the
    # engine shares that flag between the boundary and origin-limit checks,
    # so a raised origin limit withdraws it.
    assert rep.is_physically_sound is False
    assert rep.boundary_valid is False
    assert rep.compliance_score < 1.0


def test_multivariable_symmetries_and_conservation(physics_engine):
    """Verify permutation symmetry, scale invariance, and time conservation."""
    spec = PhysicsSpecification(symmetries=["exchange_x_y", "scale_invariance"])
    # Symmetric in x and y, and invariant under a common rescaling.
    rep = physics_engine.evaluate("(x**2 + y**2) / (x * y)", spec)
    assert rep.symmetry_valid is True
    assert rep.is_physically_sound is True
    assert not any("symmetr" in v.lower() for v in rep.violations), rep.violations

    # ADDITION: not symmetric under x <-> y, so the same spec must reject it.
    spec_asym = PhysicsSpecification(symmetries=["exchange_x_y"])
    rep_asym = physics_engine.evaluate("(x**2 + 2.0 * y**2) / (x * y)", spec_asym)
    assert rep_asym.symmetry_valid is False


def test_parameter_uncertainty_and_confidence_intervals():
    """Verify Gauss-Newton parameter covariance and 95% confidence intervals."""
    rng = np.random.default_rng(42)
    x = np.linspace(0.0, 5.0, 50)
    y_true = 1.0 * np.exp(-5.0 * x)
    y_obs = y_true + rng.normal(0.0, 0.05, size=x.shape)

    # param_values is the point the covariance is linearized about, so it is
    # seeded at the generating values. Seeding it elsewhere measures the
    # uncertainty around wherever it was told to look, not around the fit.
    result = compute_parameter_uncertainty(
        eval_fn=lambda xs, p: p["a"] * np.exp(-p["b"] * xs),
        param_values={"a": 1.0, "b": 5.0},
        x_data=x,
        y_data=y_obs,
        confidence_level=0.95,
    )
    assert result.is_well_identified is True
    assert result.residual_variance > 0.0
    assert result.degrees_of_freedom > 0
    assert set(result.intervals) == {"a", "b"}
    for name, truth in (("a", 1.0), ("b", 5.0)):
        ci = result.intervals[name]
        # The interval must bracket the point it was built around, and that
        # point is inside the data's support, so it must also bracket truth.
        assert ci.ci_lower < truth < ci.ci_upper, (
            name, ci.ci_lower, truth, ci.ci_upper)
        assert ci.standard_error > 0.0
        assert ci.confidence_level == 0.95
        assert ci.point_estimate == truth
