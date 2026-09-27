"""Tests for Symbolic Equivalence and Structural Canonicalization Engine.

Verifies:
- Identity simplification (log(exp(x)) -> x).
- Algebraic equivalence testing.
- Redundant term consolidation (a*x^2 + b*x^2 -> C*x^2).
- Structural family classification (linear, quadratic, shomate, exponential).
- Structural fingerprint matching across different coefficient values.
"""
from __future__ import annotations

from apedp.symbolic_equivalence import SymbolicEquivalenceEngine


def test_algebraic_identity_simplification():
    engine = SymbolicEquivalenceEngine()
    canon = engine.canonicalize("log(exp(x))", known_variables=["x"])
    assert canon.simplified_expression == "x"
    assert canon.structural_family == "linear"


def test_algebraic_equivalence_different_syntactic_forms():
    engine = SymbolicEquivalenceEngine()
    expr_a = "(x + 1)**2"
    expr_b = "x**2 + 2*x + 1"
    assert engine.are_algebraically_equivalent(expr_a, expr_b, known_variables=["x"])

    expr_c = "x**2 + 2*x + 2"
    assert not engine.are_algebraically_equivalent(expr_a, expr_c, known_variables=["x"])


def test_like_term_parameter_consolidation():
    engine = SymbolicEquivalenceEngine()
    # 2.5*x**2 + 1.5*x**2 should collapse to a single quadratic term in the structural skeleton
    canon = engine.canonicalize("2.5*x**2 + 1.5*x**2", known_variables=["x"])
    assert "x**2" in canon.canonical_skeleton
    # Should not have multiple x**2 terms in the skeleton
    assert canon.canonical_skeleton.count("x**2") == 1
    assert canon.structural_family == "quadratic"


def test_same_structural_family_across_different_parameter_values():
    engine = SymbolicEquivalenceEngine()
    # Two quadratics with different numeric parameters
    q1 = "3.2*x**2 + 1.5*x + 0.8"
    q2 = "-0.4*x**2 + 12.0*x - 5.5"

    assert engine.same_structural_family(q1, q2, known_variables=["x"])

    # Linear vs quadratic should not be in the same structural family
    lin = "2.0*x + 1.0"
    assert not engine.same_structural_family(q1, lin, known_variables=["x"])


def test_structural_family_classification():
    engine = SymbolicEquivalenceEngine()

    assert engine.canonicalize("2.0*x + 3.0", known_variables=["x"]).structural_family == "linear"
    assert engine.canonicalize("0.5*x**2 + 2.0*x - 1.0", known_variables=["x"]).structural_family == "quadratic"
    assert engine.canonicalize("exp(-0.2*x)", known_variables=["x"]).structural_family == "exponential_relaxation"
    assert engine.canonicalize("sin(2.0*x)", known_variables=["x"]).structural_family == "trigonometric_oscillatory"

    shomate_expr = "A + B*T + C*T**2 + D*T**3 + E/T**2"
    assert engine.canonicalize(shomate_expr, known_variables=["T"]).structural_family == "shomate_thermodynamic"


def test_group_by_structural_family():
    engine = SymbolicEquivalenceEngine()
    candidates = [
        "1.0*x + 2.0",
        "3.5*x - 0.5",
        "0.2*x**2 + 1.0*x",
        "0.8*x**2 - 2.0*x + 4.0",
        "exp(0.5*x)",
    ]

    groups = engine.group_by_structural_family(candidates, known_variables=["x"])
    # Should have 3 distinct structural families: linear, quadratic, exponential
    family_types = {k.split("::")[0] for k in groups}
    assert family_types == {"linear", "quadratic", "exponential_relaxation"}
    assert len(candidates) == sum(len(members) for members in groups.values())


def test_certify_identity_tiers():
    engine = SymbolicEquivalenceEngine()

    # Sympy-provable + dual-backend agreement -> P2.
    assert engine.certify_identity("log(exp(2*x))", "2*x") == "P2"
    assert engine.certify_identity("1.0*x+1.0*x", "2.0*x") == "P2"
    # Distinct structures and unprovable pairs stay P0 (never inflated).
    assert engine.certify_identity("3*x**1.7", "22.1*x**1.25*log(0.144*x**0.5+1)") == "P0"
    assert engine.certify_identity("x**2", "x**3") == "P0"


def test_certify_special_function_paths():
    engine = SymbolicEquivalenceEngine()

    # True trigonometric identity, dual-backend corroborated -> P2.
    assert engine.certify_identity("sin(x)**2+cos(x)**2", "1") == "P2"
    # zeta(s) self-identity on a domain containing the s=1 pole is rejected
    # by domain safety (P0), not certified: pole conflict != proof.
    assert (
        engine.certify_identity("zeta(s)", "zeta(s)", domain={"s": (0.5, 1.5)})
        == "P0"
    )
    # Gamma recurrence is sympy-provable (>= P1); tier depends on backend
    # availability, so assert it is never demoted to P0.
    assert engine.certify_identity("gamma(x+1)/gamma(x)", "x") in {"P1", "P2"}
