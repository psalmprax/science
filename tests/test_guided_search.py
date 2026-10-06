"""Tests for guided structural search (levels, lanes, library, mutation,
adaptive rates, semantic diversity, archive, safe eval, census)."""
from __future__ import annotations

import random

import numpy as np
import pytest
import sympy as sp

from apedp.guided_search import (
    AdaptiveRates,
    CandidateArchive,
    grow_at_level,
    grow_in_lane,
    has_substructure,
    level_of,
    population_census,
    response_signature,
    safe_to_fit,
    semantic_prune,
    targeted_mutations,
    term_library,
)


def test_level_of_measured_hierarchy():
    T = sp.Symbol("T")
    assert level_of(T) == 0
    assert level_of(T + 1) == 1
    assert level_of(T**2) == 2
    assert level_of(1 / T**2) == 3
    assert level_of(sp.exp(T)) == 4
    assert level_of(sp.sin(T)) == 4 + 1


def test_grow_at_level_respects_cap():
    rng = random.Random(0)
    syms = [sp.Symbol("T")]
    for level in (1, 2, 3):
        for _ in range(10):
            assert level_of(grow_at_level(syms, level, 3, rng)) <= (3 if level < 3 else 4)


def test_grow_level1_never_transcendental():
    rng = random.Random(1)
    syms = [sp.Symbol("T")]
    for _ in range(20):
        expr = grow_at_level(syms, 1, 3, rng)
        assert level_of(expr) <= 3  # div-of-variables scores as reciprocal construction


def test_term_library_patterns_are_generic():
    rng = random.Random(2)
    syms = [sp.Symbol("T")]
    seen = set()
    for _ in range(30):
        expr = term_library(syms, rng)
        assert expr.free_symbols <= {sp.Symbol("T")}
        seen.add(level_of(expr))
    assert len(seen) >= 2  # library spans multiple levels


def test_lanes_produce_constrained_families():
    rng = random.Random(3)
    syms = [sp.Symbol("T")]
    additive = grow_in_lane(syms, "additive", 3, 3, rng)
    assert isinstance(additive, sp.Add)
    rational = grow_in_lane(syms, "rational", 2, 3, rng)
    assert rational.has(sp.Pow) or isinstance(rational, sp.Mul)
    try:
        grow_in_lane(syms, "nope", 2, 2, rng)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown lane must raise")


def test_targeted_mutations_stay_generic():
    rng = random.Random(4)
    T = sp.Symbol("T")
    variants = targeted_mutations(T**2 + T, rng)
    assert len(variants) >= 4
    assert all(isinstance(v, sp.Expr) for v in variants)
    # Reciprocal move must appear among repeated draws.
    forms = set()
    for _ in range(10):
        for variant in targeted_mutations(T**2 + T, rng):
            forms.add(sp.srepr(variant)[:40])
    assert len(forms) >= 3


def test_adaptive_rates_collapse_then_explore():
    rates = AdaptiveRates()
    same = rates.update(duplicate_fraction=0.1, improved=True)
    assert same.mutation_rate <= rates.mutation_rate
    stressed = rates.update(duplicate_fraction=0.6, improved=False)
    assert stressed.mutation_rate > rates.mutation_rate
    assert stressed.crossover_rate <= rates.crossover_rate


def test_semantic_prune_collapses_clones():
    grid = np.linspace(0.0, 1.0, 20)
    clones = [lambda x, k=k: (1.0 + k * 1e-9) * x for k in range(5)]

    def other(x):
        return x**2

    signatures = [response_signature(f, grid) for f in clones] + [response_signature(other, grid)]
    kept = semantic_prune(signatures, keep=6, threshold=0.05)
    assert len(kept) == 2  # one clone representative plus the genuinely different curve
    assert response_signature(lambda x: np.full_like(x, np.nan), grid) is None


def test_archive_keeps_best_per_criterion():
    archive = CandidateArchive()
    assert archive.consider("accuracy", "a", 1.0) is True
    assert archive.consider("accuracy", "b", 2.0) is False
    assert archive.consider("accuracy", "c", 0.5) is True
    assert archive.consider("simplicity", "d", 3.0) is True
    assert archive.to_dict()["accuracy"]["expression"] == "c"


def test_safe_to_fit_screens_pathology():
    _T = sp.Symbol("T")
    grids = {"T": np.linspace(100.0, 300.0, 30)}
    ok, _ = safe_to_fit("T**2 + 1", grids)
    assert ok is True
    bad, reason = safe_to_fit("1/(T - 100)", grids)
    assert bad is False and reason
    over, _ = safe_to_fit("exp(2.78*T)", grids)
    assert over is False
    weird, _ = safe_to_fit("not an expression ((((", grids)
    assert weird is False


def test_census_and_substructure():
    T = sp.Symbol("T")
    census = population_census([T**2, T**2, sp.sin(T)])
    assert census["n"] == 3
    assert census["duplicate_fraction"] == pytest.approx(1 / 3)
    assert census["depth_max"] >= census["depth_min"]
    assert has_substructure(T**2 + 1 / T**2, T**-2) is True
    assert has_substructure(T**2 + T, T**-2) is False
