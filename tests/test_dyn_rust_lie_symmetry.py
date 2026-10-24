"""Lie point-symmetry discovery through the PyO3 bindings.

Covers the mathematically checkable cases: the scaling symmetry of y' = y,
the full translation algebra of y' = 0, and y-translation for y' = x.
"""
from __future__ import annotations

import json

import apedp_kernel as ak


def _discover(f_terms, config=None):
    return json.loads(ak.rime_lie_symmetry_json(
        json.dumps(f_terms), json.dumps(config or {})))


def test_lie_scaling_symmetry_of_exponential_ode():
    # y' = y admits at least x-translation (1, 0) and scaling (0, y),
    # both independently verifiable in the determining equation.
    res = _discover([[0, 1, 1.0]])
    assert res["nullity"] >= 2, f"nullity: {res['nullity']}"
    assert res["max_basis_residual"] < 1e-6


def test_lie_trivial_ode_translation_algebra():
    res = _discover([])
    assert res["nullity"] == 9
    assert res["max_basis_residual"] < 1e-9


def test_lie_y_translation_for_x_dependent_ode():
    res = _discover([[1, 0, 1.0]])
    assert res["nullity"] >= 1
    assert res["max_basis_residual"] < 1e-6
