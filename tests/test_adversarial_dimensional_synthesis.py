"""Adversarial Dimensional Synthesis Audit Test Suite.

Deliberately attacks the Dimensional Constant Synthesis engine with physically
incompatible expressions (bare dimensioned transcendentals, heterogeneous additions,
and dimensioned exponents) to verify that dimensional synthesis is NOT a loophole
and strictly enforces physical conservation laws.
"""
from __future__ import annotations

import pytest

from apedp.dimensional_layer import audit_dimensional_admissibility


@pytest.fixture
def physics_units() -> dict[str, str]:
    return {
        "L": "meter",
        "M": "kilogram",
        "T": "second",
        "v": "meter / second",
        "a": "astronomical_unit",
        "r": "meter",
    }


# ==============================================================================
# Adversarial Attack Suite: Strictly MUST Be REJECTED (INVALID)
# ==============================================================================

@pytest.mark.parametrize(
    "expression, expected_reason_keyword",
    [
        ("sin(L)", "Transcendental sin applied to bare dimensioned expression"),
        ("exp(L)", "Transcendental exp applied to bare dimensioned expression"),
        ("log(M)", "Transcendental log applied to bare dimensioned expression"),
        ("cos(T)", "Transcendental cos applied to bare dimensioned expression"),
        ("sin(L**2)", "Transcendental sin applied to bare dimensioned expression"),
        ("exp(L / T)", "Transcendental exp applied to bare dimensioned expression"),
        ("log(M**3)", "Transcendental log applied to bare dimensioned expression"),
        ("M + L", "Addition of incompatible bare variable terms"),
        ("L**2 + T**2", "Addition of incompatible bare variable terms"),
        ("M + v", "Addition of incompatible bare variable terms"),
        ("L**3 - M**3", "Addition of incompatible bare variable terms"),
        ("sqrt(L) + sqrt(T)", "Addition of incompatible bare variable terms"),
        ("v**2 + L / T**2", "Addition of incompatible bare variable terms"),
    ],
)
def test_adversarial_violations_strictly_rejected(physics_units, expression, expected_reason_keyword):
    """Verifies that all bare transcendental and incompatible additive violations are REJECTED."""
    res = audit_dimensional_admissibility(
        expression=expression,
        variable_units=physics_units,
        target_unit="meter",
    )
    assert res["admissible"] is False, f"Expression '{expression}' should have been rejected as INVALID!"
    assert res["status"] == "INVALID"
    assert expected_reason_keyword in res["reason"]


# ==============================================================================
# Legitimate Physical Expressions: MUST Be ACCEPTED (VALID)
# ==============================================================================

@pytest.mark.parametrize(
    "expression, target_unit",
    [
        # 1. Monomial power laws with bare or float scaling constants
        ("10.0 / r**2", "newton"),
        ("a**1.5", "year"),
        ("0.5 * M * v**2", "joule"),
        # 2. Homogeneous bare additions (compatible dimensions)
        ("L + v * T", "meter"),
        # 3. Dimensionless transcendental arguments
        ("sin(v * T / L)", "dimensionless"),
        # 4. Parameterized transcendental decay (e.g. Yukawa potential)
        ("5.0 * exp(-0.8 * r) / r", "joule"),
        # 5. Multi-term parametric models with adjustable coefficients
        ("10.0 * L**2 + 5.0 * T**2", "joule"),
    ],
)
def test_legitimate_physical_laws_accepted(physics_units, expression, target_unit):
    """Verifies that physically consistent and parameterized scaling laws are ACCEPTED."""
    res = audit_dimensional_admissibility(
        expression=expression,
        variable_units=physics_units,
        target_unit=target_unit,
    )
    assert res["admissible"] is True, f"Legitimate expression '{expression}' was improperly rejected: {res.get('reason')}"
    assert res["status"] == "VALID"
