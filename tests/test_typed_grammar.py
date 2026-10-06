"""Tests for dimensionally typed context-free production grammar."""
from __future__ import annotations

import random

import pytest
import sympy as sp

from apedp.typed_grammar import DimVector, TypedProductionGrammar


@pytest.fixture
def physics_grammar():
    return TypedProductionGrammar({
        "m": "kilogram",
        "v": "meter / second",
        "r": "meter",
        "t": "second",
        "F": "newton",
    })


def test_dim_vector_algebra():
    # Force: mass * length / time^2 -> [1, 1, -2, 0, 0, 0, 0]
    m_dim = DimVector((1, 0, 0, 0, 0, 0, 0))
    a_dim = DimVector((0, 1, -2, 0, 0, 0, 0))
    f_dim = m_dim + a_dim
    assert f_dim.exponents == (1, 1, -2, 0, 0, 0, 0)
    assert not f_dim.is_dimensionless
    assert (f_dim - f_dim).is_dimensionless


def test_sample_expression_homogeneity(physics_grammar):
    rng = random.Random(123)
    # Target: Energy = [1, 2, -2, 0, 0, 0, 0]
    energy_dim = DimVector((1, 2, -2, 0, 0, 0, 0))
    joule_sym = physics_grammar.engine.register_variable("J", "joule")

    for _ in range(10):
        expr = physics_grammar.sample_expression(energy_dim, depth=2, rng=rng)
        assert physics_grammar.engine.validate_homogeneity(expr, joule_sym)


def test_mutate_expression_preserves_dimension(physics_grammar):
    rng = random.Random(456)
    force_dim = DimVector((1, 1, -2, 0, 0, 0, 0))
    f_sym = sp.Symbol("F")

    expr = physics_grammar.sample_expression(force_dim, depth=2, rng=rng)
    for _ in range(5):
        mut = physics_grammar.mutate_expression(expr, rng=rng)
        assert physics_grammar.engine.validate_homogeneity(mut, f_sym)
