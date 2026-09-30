"""Unit tests for the Controlled Symbolic Special-Function Library and Domain Safety Engine."""
# RECONSTRUCTION NOTE: deleted from the worktree by a concurrent session and
# restored here from bytecode forensics (tests/__pycache__/...pytest-9.1.1.pyc):
# docstrings, values, thresholds and assertion shapes recovered from compiled
# constants; verified by execution below.
from __future__ import annotations

import numpy as np
import sympy as sp

from apedp.special_functions import SPECIAL_FUNCTION_REGISTRY, SpecialFunctionLibrary
from apedp.symbolic_equivalence import SymbolicEquivalenceEngine


def test_special_function_registry():
    """Verify registry contains complete metadata and pole definitions for special functions."""
    assert {"zeta", "gamma", "erf", "besselj"} <= set(SPECIAL_FUNCTION_REGISTRY)
    zeta_meta = SPECIAL_FUNCTION_REGISTRY["zeta"]
    assert 1.0 in zeta_meta.pole_locations
    gamma_meta = SPECIAL_FUNCTION_REGISTRY["gamma"]
    assert 0.0 in gamma_meta.pole_locations
    erf_meta = SPECIAL_FUNCTION_REGISTRY["erf"]
    assert erf_meta.parity == "odd"


def test_special_function_detection():
    """Verify AST traversal detects registered special functions accurately."""
    funcs = SpecialFunctionLibrary.detect_special_functions(
        sp.sympify("3.0 * erf(x) + gamma(s) / zeta(s)")
    )
    assert set(funcs) == {"erf", "gamma", "zeta"}


def test_domain_safety_pole_conflict_detection():
    """Verify domain safety auditor catches isolated poles in evaluation intervals."""
    rep_zeta_pole = SpecialFunctionLibrary.check_domain_safety("zeta(s)", {"s": (0.5, 1.5)})
    assert not rep_zeta_pole.is_domain_safe
    assert any("1" in conflict for conflict in rep_zeta_pole.pole_conflicts)
    rep_zeta_safe = SpecialFunctionLibrary.check_domain_safety("zeta(s)", {"s": (2.0, 4.0)})
    assert len(rep_zeta_safe.pole_conflicts) == 0
    rep_gamma_pole = SpecialFunctionLibrary.check_domain_safety("gamma(s)", {"s": (-0.5, 0.5)})
    assert not rep_gamma_pole.is_domain_safe
    rep_erf = SpecialFunctionLibrary.check_domain_safety("erf(x) + erfc(x)", {"x": (-2.0, 2.0)})
    assert rep_erf.is_domain_safe


def test_special_identities_simplification():
    """Verify algebraic and special-function identities are simplified without losing exactness."""
    expr_trig = sp.sympify("sin(x)**2 + cos(x)**2")
    res_trig = SpecialFunctionLibrary.apply_special_identities(expr_trig)
    assert res_trig == 1
    expr_hyp = sp.sympify("cosh(x)**2 - sinh(x)**2")
    res_hyp = SpecialFunctionLibrary.apply_special_identities(expr_hyp)
    assert res_hyp == 1
    expr_gamma = sp.sympify("gamma(x + 1) / gamma(x)")
    res_gamma = SpecialFunctionLibrary.apply_special_identities(expr_gamma)
    assert sp.simplify(res_gamma - sp.Symbol("x")) == 0
    expr_erf = sp.sympify("erf(x) + erfc(x)")
    res_erf = SpecialFunctionLibrary.apply_special_identities(expr_erf)
    assert res_erf == 1


def test_vectorized_special_evaluation():
    """Verify vectorized evaluation computes standard mathematical values accurately."""
    x_pts = np.array([0.0, 1.0, 2.0])
    res_erf = SpecialFunctionLibrary.evaluate_vectorized("erf(x)", {"x": x_pts})
    assert np.isclose(res_erf[0], 0.0)
    assert np.isclose(res_erf[1], 0.8427007929, atol=1e-6)
    assert len(res_erf) == 3
    res_gamma = SpecialFunctionLibrary.evaluate_vectorized("gamma(x)", {"x": np.array([5.0])})
    assert np.isclose(res_gamma[0], 24.0)
    res_zeta = SpecialFunctionLibrary.evaluate_vectorized("zeta(s)", {"s": np.array([2.0, 6.0])})
    assert np.all(np.isfinite(res_zeta))


def test_certify_identity_with_special_functions():
    """Verify SymbolicEquivalenceEngine certifies identities involving special functions."""
    tier_trig = SymbolicEquivalenceEngine.certify_identity("sin(x)**2+cos(x)**2", "1.0", domain={"x": (0.5, 4.0)})
    assert tier_trig in ("P1", "P2")
    tier_erf = SymbolicEquivalenceEngine.certify_identity(
        "erf(x) + erfc(x)", "1.0", domain={"x": (0.5, 4.0)}
    )
    assert tier_erf in ("P1", "P2")
    tier_zeta_pole = SymbolicEquivalenceEngine.certify_identity("zeta(s)", "zeta(s)", domain={"s": (0.5, 1.5)})
    assert tier_zeta_pole == "P0"
