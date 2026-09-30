"""D5-CONF nullspace regression tests (frozen exact path).

Patch-first order: exact SymPy rational nullspace → these 4 tests →
freeze → noise sweep. No SVD tolerance, no limit_denominator anywhere
in the frozen path.

1. Exact annihilation: D^T γ = 0 exactly for every emitted basis vector.
2. Independence: rank(Γ) = k for k generated groups.
3. Canonical equivalence: commuted / inverted groups collapse deterministically.
4. Dimension-count sanity: dim ker(D^T) = n_vars - rank(D).
"""
from __future__ import annotations

from fractions import Fraction

import sympy as sp

from apedp_dyn.conf.dimensionless import (
    D5_CONF_VERSION,
    DimensionlessGroup,
    build_integer_dimension_matrix,
    discover_pi_groups,
    exact_integer_rank,
    exact_nullspace_basis,
)

CASES: dict[str, dict[str, dict[str, int]]] = {
    "xtv": {
        "x": {"length": 1},
        "t": {"time": 1},
        "v": {"length": 1, "time": -1},
    },
    "bohm_like": {
        "tau_E": {"time": 1},
        "rho_star": {"length": -1},
        "beta": {},
        "nu_star": {},
    },
    "rank_deficient": {
        "a": {"length": 1},
        "b": {"length": 2},
    },
    "all_dimensionless": {
        "p": {},
        "q": {},
    },
}


def _annihilation_holds(
    D_int: tuple[tuple[int, ...], ...],
    gamma: tuple[Fraction, ...],
) -> bool:
    m = len(D_int)
    n = len(D_int[0]) if m else 0
    assert len(gamma) == m
    for j in range(n):
        s = sum(gamma[i] * D_int[i][j] for i in range(m))
        if s != 0:
            return False
    return True


class TestExactAnnihilation:
    def test_d_gamma_zero_exactly_all_cases(self):
        for name, spec in CASES.items():
            D_int, var_order, _ = build_integer_dimension_matrix(spec)
            basis = exact_nullspace_basis(D_int)
            assert basis, f"{name}: empty basis"
            for gamma in basis:
                assert all(isinstance(e, Fraction) for e in gamma), name
                assert _annihilation_holds(D_int, gamma), (
                    f"{name}: D^T γ != 0 for {gamma}"
                )


class TestIndependence:
    def test_rank_gamma_equals_k(self):
        for name, spec in CASES.items():
            D_int, var_order, _ = build_integer_dimension_matrix(spec)
            basis = exact_nullspace_basis(D_int)
            k = len(basis)
            assert k > 0, name
            Gamma = sp.Matrix([[sp.Rational(e) for e in g] for g in basis]).T
            assert int(Gamma.rank()) == k, (
                f"{name}: rank(Γ)={int(Gamma.rank())} != k={k}"
            )


class TestCanonicalEquivalence:
    def test_commuted_groups_same_key(self):
        dg1 = DimensionlessGroup(
            name="a", variables=("x", "y", "z"),
            exponents=(Fraction(1), Fraction(-2), Fraction(1)),
            convention="t", definition="d",
        )
        dg2 = DimensionlessGroup(
            name="b", variables=("z", "y", "x"),
            exponents=(Fraction(1), Fraction(-2), Fraction(1)),
            convention="t", definition="d",
        )
        assert dg1.structural_key() == dg2.structural_key()

    def test_inverse_groups_same_key(self):
        # Deterministic orientation: Π and Π^-1 collapse via first-nonzero->positive.
        dg1 = DimensionlessGroup(
            name="a", variables=("x", "t", "v"),
            exponents=(Fraction(1), Fraction(-1), Fraction(-1)),
            convention="t", definition="d",
        )
        dg2 = DimensionlessGroup(
            name="b", variables=("x", "t", "v"),
            exponents=(Fraction(-1), Fraction(1), Fraction(1)),
            convention="t", definition="d",
        )
        assert dg1.structural_key() == dg2.structural_key()

    def test_discovered_basis_is_canonical(self):
        # First nonzero exponent of every emitted group must be positive,
        # and all exponents must be exact Fractions (never floats).
        from apedp_dyn.conf.dimensionless import build_exponent_matrix

        D, var_order = build_exponent_matrix(CASES["xtv"])
        for g in discover_pi_groups(tuple(var_order), D, convention="t"):
            assert all(isinstance(e, Fraction) for e in g.exponents)
            first = next(e for e in g.exponents if e != 0)
            assert first > 0


class TestDimensionCount:
    def test_dim_ker_equals_n_minus_rank(self):
        expected = {"xtv": 1, "bohm_like": 2, "rank_deficient": 1, "all_dimensionless": 2}
        for name, spec in CASES.items():
            D_int, var_order, _ = build_integer_dimension_matrix(spec)
            r = exact_integer_rank(D_int)
            basis = exact_nullspace_basis(D_int)
            assert len(basis) == len(var_order) - r == expected[name], (
                f"{name}: k={len(basis)}, n-r={len(var_order)}-{r}"
            )

    def test_frozen_version_pinned(self):
        assert D5_CONF_VERSION == "1.0.0-exact"
