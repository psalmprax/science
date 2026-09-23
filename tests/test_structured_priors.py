# Databricks notebook source
from __future__ import annotations

import sympy as sp

from apedp.structured_priors import (
    bounded_ratio_priors,
    gaussian_envelope_priors,
    reciprocal_structure_priors,
    trig_interaction_priors,
)


def test_reciprocal_priors_cover_generic_three_variable_forms():
    priors = reciprocal_structure_priors(["d1", "d2", "n"])
    target = 1 / (1 / sp.Symbol("d1") + sp.Symbol("n") / sp.Symbol("d2"))
    assert any(sp.simplify(candidate - target) == 0 for candidate in priors)
    assert len(priors) > 1


def test_trig_interaction_priors_cover_generic_four_variable_forms():
    priors = trig_interaction_priors(["q", "v", "B", "theta"])
    target = sp.Symbol("q") * sp.Symbol("v") * sp.Symbol("B") * sp.sin(sp.Symbol("theta"))
    assert any(sp.simplify(candidate - target) == 0 for candidate in priors)


def test_gaussian_priors_cover_normalized_envelope_forms():
    priors = gaussian_envelope_priors(["theta", "sigma"])
    target = sp.exp(-sp.Symbol("theta")**2 / (2 * sp.Symbol("sigma")**2)) / sp.Symbol("sigma")
    assert any(sp.simplify(candidate - target) == 0 for candidate in priors)


def test_bounded_ratio_priors_cover_affine_relativistic_forms():
    # The limit symbol is a separate parameter (default "c_light"); the target
    # below names it "c", so it must be passed explicitly or nothing can match.
    priors = bounded_ratio_priors(["x", "u", "t"], limit_symbol="c")
    target = (sp.Symbol("x") - sp.Symbol("u") * sp.Symbol("t")) / sp.sqrt(1 - (sp.Symbol("u") / sp.Symbol("c"))**2)
    assert any(sp.simplify(candidate - target) == 0 for candidate in priors)
