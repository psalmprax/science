"""Tests for in-loop epistemic guard analytical vetoes and prior art labeling."""
from __future__ import annotations

import sympy as sp

from apedp.epistemic_pruning import EpistemicVetoResult, InLoopEpistemicGuard


def test_interior_singularity_vetoed():
    guard = InLoopEpistemicGuard()
    x = sp.Symbol("x")
    bounds = {"x": (1.0, 5.0)}

    # Pole inside domain at x = 3.0
    bad_expr = 1.0 / (x - 3.0)
    res = guard.evaluate(bad_expr, bounds)
    assert isinstance(res, EpistemicVetoResult)
    assert not res.admitted
    assert "POLE" in res.reason


def test_outside_singularity_admitted():
    guard = InLoopEpistemicGuard()
    x = sp.Symbol("x")
    bounds = {"x": (1.0, 5.0)}

    # Pole outside domain at x = -2.0
    safe_expr = 1.0 / (x + 2.0)
    res = guard.evaluate(safe_expr, bounds)
    assert res.admitted
    assert res.reason == "ADMITTED"


def test_asymptotic_explosion_vetoed_when_constrained():
    guard = InLoopEpistemicGuard()
    x = sp.Symbol("x")
    bounds = {"x": (0.0, 10.0)}

    # Diverges to infinity
    exploding_expr = sp.exp(x)
    res = guard.evaluate(exploding_expr, bounds, bounded_vars={"x"})
    assert not res.admitted
    assert "ASYMPTOTIC_EXPLOSION" in res.reason


def test_asymptotic_divergence_allowed_without_constraint():
    guard = InLoopEpistemicGuard()
    x = sp.Symbol("x")
    bounds = {"x": (0.0, 10.0)}

    # Physical resonance or unconstrained divergence allowed if bounded_vars is None
    exploding_expr = sp.exp(x)
    res = guard.evaluate(exploding_expr, bounds, bounded_vars=None)
    assert res.admitted


def test_prior_art_labeled_and_admitted_by_default():
    guard = InLoopEpistemicGuard(veto_known_prior_art=False)
    m = sp.Symbol("m")
    v = sp.Symbol("v")
    kinetic_energy = 0.5 * m * v**2

    res = guard.evaluate(kinetic_energy, bounds={"m": (0.1, 10.0), "v": (0.1, 10.0)}, check_prior_art=True)
    # Crucial epistemic test: Admitted by default, transparently labeled!
    assert res.admitted
    assert res.details.get("prior_art") == "KINETIC_ENERGY"
