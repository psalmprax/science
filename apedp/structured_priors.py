# Databricks notebook source
"""Generic structure priors for difficult symbolic-regression families."""
from __future__ import annotations

import itertools

import sympy as sp


def reciprocal_structure_priors(variables: list[str]) -> list[sp.Expr]:
    """Generate generic reciprocal-combination priors, not named equations."""
    symbols = [sp.Symbol(name) for name in variables]
    priors: list[sp.Expr] = []
    for first, second, ratio in itertools.permutations(symbols, 3):
        priors.append(1 / (1 / first + ratio / second))
    for first, second in itertools.combinations(symbols, 2):
        priors.append(1 / (1 / first + 1 / second))
    return _unique(priors)


def trig_interaction_priors(variables: list[str]) -> list[sp.Expr]:
    """Generate generic product-times-trigonometric interaction structures.

    The generator does not name a physical law. It proposes all products of
    two or three variables multiplied by ``sin`` or ``cos`` of another
    variable, which covers force/phase/wave families without hard-coding one.
    """
    symbols = [sp.Symbol(name) for name in variables]
    priors: list[sp.Expr] = []
    for argument in symbols:
        factors = [symbol for symbol in symbols if symbol != argument]
        for width in (1, 2, 3):
            for product in itertools.combinations(factors, width):
                base = sp.prod(product)
                priors.extend((base * sp.sin(argument), base * sp.cos(argument)))
    return _unique(priors)


def gaussian_envelope_priors(variables: list[str]) -> list[sp.Expr]:
    """Generate normalized Gaussian envelope structures.

    The family is generic: a scale-normalized variable enters a squared
    exponential and the scale variable may also normalize the output. This
    covers probability-density, diffusion-kernel, and response-envelope forms
    without naming a particular law.
    """
    symbols = [sp.Symbol(name) for name in variables]
    priors: list[sp.Expr] = []
    for value, scale in itertools.permutations(symbols, 2):
        ratio = value / scale
        envelope = sp.exp(-ratio**2 / 2)
        priors.extend((envelope, envelope / scale))
    return _unique(priors)


def bounded_ratio_priors(variables: list[str], limit_symbol: str = "c_light") -> list[sp.Expr]:
    """Generate generic bounded-ratio denominator structures.

    Forms have an affine numerator divided by ``sqrt(1 - ratio**2)``. This
    covers relativistic, saturation, and bounded-transport families without
    naming a specific physical equation.
    """
    symbols = [sp.Symbol(name) for name in variables]
    limit = sp.Symbol(limit_symbol)
    priors: list[sp.Expr] = []
    for position, velocity, time in itertools.permutations(symbols, 3):
        numerator = position - velocity * time
        denominator = sp.sqrt(1 - (velocity / limit) ** 2)
        priors.append(numerator / denominator)
    return _unique(priors)


def _unique(expressions: list[sp.Expr]) -> list[sp.Expr]:
    unique: list[sp.Expr] = []
    for expression in expressions:
        if all(sp.simplify(expression - existing) != 0 for existing in unique):
            unique.append(expression)
    return unique


__all__ = ["bounded_ratio_priors", "gaussian_envelope_priors", "reciprocal_structure_priors", "trig_interaction_priors"]