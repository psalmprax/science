"""Buckingham-Π engine for dimensionless group discovery.

Given an integer dimensional exponent matrix D (variables × fundamental
dimensions), find the exact rational null space D^T γ = 0. Each independent
null-vector γ generates a dimensionless group Π = ∏ x_i^{γ_i}.

Pipeline (kept separate by design):

    units
      ↓
    integer/rational dimension matrix D
      ↓
    exact nullspace (SymPy, no SVD tolerance, no limit_denominator)
      ↓
    raw Π basis
      ↓
    canonicalization
      ↓
    DimensionlessGroup

Key design decisions:

1. DimensionlessGroup is an explicit, metadata-backed object (not opaque strings).
2. Canonicalization is deterministic: first nonzero exponent positive,
   rational exponents reduced to common denominator.
3. The null-space basis is independence-checked: Π_3 = Π_1 Π_2 is not
   permitted as a new group.
4. Regime-specific scaling generators are stored separately (ScalingInvarianceGate).
5. D5-CONF freeze: exact basis → unit/invariance tests → freeze → noise sweep.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Any

import numpy as np
import sympy as sp

D5_CONF_VERSION = "1.0.0-exact"


@dataclass(frozen=True)
class DimensionlessGroup:
    """An explicit dimensionless group discovered via Buckingham-Π.

    Attributes
    ----------
    name : str
        User-facing group name (e.g. "rho_star", "beta").
    variables : tuple of str
        Variable names appearing in the group.
    exponents : tuple of Fraction
        Rational exponents for each variable, in the same order as `variables`.
        The vector is canonical: first nonzero exponent > 0, rational exponents
        reduced to a common denominator.
    convention : str
        Naming convention convention (e.g. "connor-taylor", "nstx", "psfc").
        Mandatory: conveys model/regime specificity.
    definition : str
        Human-readable definition string.

    Examples
    --------
    rho_star = rho / a   (normalized gyroradius)
    beta = p / (B^2 / 2mu0) (plasma beta)
    q95 = q95 (normalized safety factor)
    """
    name: str
    variables: tuple[str, ...]
    exponents: tuple[Fraction, ...]
    convention: str
    definition: str

    def exponent_map(self) -> Mapping[str, Fraction]:
        """Return variable -> exponent mapping."""
        return dict(zip(self.variables, self.exponents, strict=True))

    def structural_key(self) -> str:
        """Deterministic canonical key for G2 deduplication.

        Two groups that are mathematically the same (e.g. x^1 y^{-2} z
        and z x y^{-2}) map to the same key.
        """
        # Build a canonical representation that is invariant under both
        # variable input order and global sign (Π vs Π^-1): render the key
        # for both orientations and take the lexicographic minimum.
        def _render(pairs: list[tuple[str, Fraction]]) -> str:
            ordered = sorted(pairs, key=lambda iv: (iv[1] == 0, float(iv[1]), iv[0]))
            parts: list[str] = []
            for var, exp in ordered:
                if exp == 0:
                    continue
                sign = "" if exp >= 0 else "-"
                abs_exp = abs(exp)
                if abs_exp.denominator == 1:
                    exp_str = str(abs_exp.numerator)
                else:
                    exp_str = f"{abs_exp.numerator}/{abs_exp.denominator}"
                parts.append(f"{sign}{var}^{exp_str}")
            return "Π=" + "*".join(parts)

        pairs = [(v, e) for v, e in zip(self.variables, self.exponents, strict=True)]
        negated = [(v, -e) for v, e in pairs]
        return min(_render(pairs), _render(negated))


def build_integer_dimension_matrix(
    variables: dict[str, dict[str, int]],
) -> tuple[tuple[tuple[int, ...], ...], list[str], list[str]]:
    """Build the exact integer dimensional exponent matrix.

    This is the physical-unit layer, kept separate from the nullspace solver
    so the mathematical core stays independently testable.

    Parameters
    ----------
    variables : dict mapping var_name -> {base_dimension -> exponent}
        All exponents must be integers (exact rational arithmetic downstream).

    Returns
    -------
    D_int : tuple of tuple of int, shape (n_vars, n_dims)
        D_int[i][j] = exponent of dimension j in variable i.
    var_order : list of str
    dim_order : list of str
    """
    all_vars = list(variables.keys())
    all_dims = sorted({d for v in variables.values() for d in v.keys()})
    D_int: list[tuple[int, ...]] = []
    for var in all_vars:
        row = []
        for dim in all_dims:
            e = variables[var].get(dim, 0)
            if isinstance(e, float):
                if not e.is_integer():
                    raise ValueError(
                        f"non-integer exponent {e!r} for {var!r}.{dim!r}; "
                        "exact path requires integer dimension matrix"
                    )
                e = int(e)
            if not isinstance(e, int):
                raise ValueError(
                    f"non-integer exponent {e!r} for {var!r}.{dim!r}"
                )
            row.append(e)
        D_int.append(tuple(row))
    return tuple(D_int), all_vars, all_dims


def _sympy_rational_vector(vec: Any) -> tuple[Fraction, ...]:
    """Convert a SymPy rational vector to a tuple of Fractions (exact)."""
    out: list[Fraction] = []
    for v in vec:
        r = sp.Rational(v)
        out.append(Fraction(int(r.p), int(r.q)))
    return tuple(out)


def exact_nullspace_basis(
    D_int: tuple[tuple[int, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    """Exact rational nullspace of D^T, where D is (m vars × n dims) integer.

    Solves D^T γ = 0 over QQ via SymPy. Each returned vector has length m
    (one entry per variable). No SVD tolerance, no limit_denominator.

    Returns
    -------
    tuple of gamma vectors, each tuple[Fraction, ...] of length m.
    Number of vectors = dim ker(D^T) = m - rank(D).
    """
    if not D_int:
        return ()
    m = len(D_int)
    n = len(D_int[0])
    M = sp.Matrix([[sp.Rational(D_int[i][j]) for j in range(n)] for i in range(m)]).T
    basis = M.nullspace()
    result: list[tuple[Fraction, ...]] = []
    for b in basis:
        result.append(_sympy_rational_vector(b))
    return tuple(result)


def exact_integer_rank(D_int: tuple[tuple[int, ...], ...]) -> int:
    """Exact rank of the integer dimension matrix over QQ."""
    if not D_int or not D_int[0]:
        return 0
    M = sp.Matrix([[sp.Rational(v) for v in row] for row in D_int])
    return int(M.rank())


def _lcm(a: int, b: int) -> int:
    """Least common multiple via GCD."""
    return a * b // gcd(a, b)


def _reduce_exponents(
    exponents_frac: list[Fraction],
) -> tuple[Fraction, ...]:
    """Reduce rational exponents to common denominator, canonical form.

    1. Find first nonzero exponent.
    2. If it's negative, multiply all by -1 (ensures first nonzero > 0).
    3. Convert all to common denominator.
    4. Return reduced fraction tuple.
    """
    if not exponents_frac:
        return (Fraction(0),)

    # Convert to Fraction objects if not already
    fracs = [Fraction(e) if not isinstance(e, Fraction) else e for e in exponents_frac]

    # Find first nonzero
    first_nz = None
    for k, e in enumerate(fracs):
        if e != 0:
            first_nz = k
            break

    # If first nonzero is negative, flip signs of all
    if first_nz is not None and fracs[first_nz] < 0:
        fracs = [-e for e in fracs]

    # Find common denominator
    denominators = [e.denominator for e in fracs if e.denominator > 0]
    if not denominators:
        common_denom = 1
    else:
        common_denom = 1
        for d in denominators:
            common_denom = _lcm(common_denom, d)

    # Convert to common-denominator representation
    reduced = tuple(
        Fraction(e.numerator * (common_denom // e.denominator), common_denom)
        if e.denominator > 0 else e
        for e in fracs
    )
    return reduced


def _systematic_group_name(
    variable_names: tuple[str, ...],
    exponents: tuple[Fraction, ...],
    convention: str,
) -> str:
    """Generate a systematic group name from variables and exponents.

    Pattern: convention_{var1}^{e1}_var2^{e2}_...
    """
    parts = [convention]
    for var, exp in zip(variable_names, exponents, strict=True):
        if exp == 0:
            continue
        exp_abs = abs(exp)
        exp_str = str(exp_abs.numerator) if exp_abs.denominator == 1 else f"{exp_abs.numerator}/{exp_abs.denominator}"
        sign = "-" if exp < 0 else ""
        parts.append(f"{sign}{var}^{exp_str}")
    return "_".join(parts)


def _normalize_name(
    variables: tuple[str, ...],
    exponents: tuple[Fraction, ...],
) -> str:
    """Normalize variable names for group naming."""
    parts = []
    for v, e in zip(variables, exponents, strict=True):
        if e == 0:
            continue
        sign = "-" if e < 0 else ""
        abs_e = abs(e)
        exp_str = str(abs_e.numerator) if abs_e.denominator == 1 else f"{abs_e.numerator}/{abs_e.denominator}"
        parts.append(f"{sign}{v}^{exp_str}")
    return "_".join(parts)


def _build_definition_string(
    variable_names_selected: tuple[str, ...],
    reduced_exponents: tuple[Fraction, ...],
) -> str:
    """Build a human-readable definition string for the group.

    Example: "rho^2 * beta^{-1} * nu^0.5"
    """
    var_exp_pairs = []
    for v, e in zip(variable_names_selected, reduced_exponents, strict=True):
        if e == 0:
            continue
        sign = "" if e >= 0 else "-"
        abs_e = abs(e)
        exp_str = str(abs_e.numerator) if abs_e.denominator == 1 else f"{abs_e.numerator}/{abs_e.denominator}"
        var_exp_pairs.append(f"{sign}{v}^{exp_str}")
    return " * ".join(var_exp_pairs)


def _svd_nullspace_legacy(D: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Legacy SVD nullspace (pre-freeze, deprecated).

    Kept only for forensic comparison; the frozen D5-CONF path uses
    exact_nullspace_basis(). Not used by discover_pi_groups().
    """
    _u, s, vh = np.linalg.svd(D, full_matrices=True)
    if s.size == 0:
        return np.empty((D.shape[1], 0))
    rank = int(np.sum(s > tol * max(float(s[0]), tol)))
    nullity = D.shape[1] - rank
    if nullity <= 0:
        return np.empty((D.shape[1], 0))
    return vh[rank:].T


def discover_pi_groups(
    variable_names: tuple[str, ...],
    exponent_matrix: np.ndarray,
    variable_units: dict[str, str] | None = None,
    convention: str = "connor-taylor",
) -> tuple[DimensionlessGroup, ...]:
    """Discover dimensionless groups via Buckingham-Π.

    Given m variables with n fundamental dimensions, find the null space
    of the m×n dimensional exponent matrix. Each independent null-vector
    generates a dimensionless group Π_j = ∏ x_i^{γ_{ij}}.

    Parameters
    ----------
    variable_names : tuple of str
        Names of the m variables (must match the rows of exponent_matrix).
    exponent_matrix : np.ndarray of shape (m, n)
        D[i, j] = exponent of variable i in fundamental dimension j.
        Typically built from: each variable's dimensional formula in terms
        of the base dimensions (mass, length, time, temperature, etc.).
    variable_units : dict of str -> str, optional
        Pint-unit-style mapping var_name -> unit string, for validation/debugging.
    convention : str
        Naming convention tag (mandatory, per DimensionlessGroup). Used to
        route the group to the appropriate scaling regime.

    Returns
    -------
    tuple of DimensionlessGroup
        Independent dimensionless groups spanning the null space.
        The number of groups = nullity(D) = n - rank(D).
        Groups are canonicalized (first nonzero exponent > 0, rational exponents
        reduced to common denominator).
    """
    m, n = exponent_matrix.shape
    assert m == len(variable_names), (
        f"variable_names length {len(variable_names)} != matrix rows {m}"
    )

    # Exact path (frozen): float matrix → exact integer matrix → SymPy nullspace.
    # No SVD tolerance, no limit_denominator. Non-integral entries are rejected.
    D_int_rows: list[tuple[int, ...]] = []
    for i in range(m):
        row: list[int] = []
        for j in range(n):
            v = float(exponent_matrix[i, j])
            if not v.is_integer():
                raise ValueError(
                    f"non-integer dimension entry D[{i},{j}]={v!r}; "
                    "frozen exact path requires an integer dimension matrix"
                )
            row.append(int(v))
        D_int_rows.append(tuple(row))
    D_int = tuple(D_int_rows)

    basis = exact_nullspace_basis(D_int)
    if not basis:
        return ()

    groups: list[DimensionlessGroup] = []

    for gamma in basis:
        # gamma is an exact tuple[Fraction, ...] of length m.
        var_names_selected: list[str] = []
        exponents_frac: list[Fraction] = []

        for i in range(m):
            gi = gamma[i]
            if gi != 0:
                var_names_selected.append(variable_names[i])
                exponents_frac.append(gi)

        if not var_names_selected:
            continue

        # Reduce exponents to canonical form
        reduced_exponents = _reduce_exponents(exponents_frac)

        # Build definition string
        definition = _build_definition_string(
            tuple(var_names_selected), reduced_exponents
        )

        # Build systematic group name
        group_name = _systematic_group_name(
            tuple(var_names_selected), reduced_exponents, convention
        )

        groups.append(DimensionlessGroup(
            name=group_name,
            variables=tuple(var_names_selected),
            exponents=reduced_exponents,
            convention=convention,
            definition=definition,
        ))

    return tuple(groups)


def build_exponent_matrix(
    variables: dict[str, dict[str, int]],
) -> tuple[np.ndarray, list[str]]:
    """Build the dimensional exponent matrix from variable dimensional formulas.

    Parameters
    ----------
    variables : dict mapping var_name -> {base_dimension -> exponent}
        e.g. {"tau_E": {"time": 1, "mass": -1}, "rho_star": {"length": -1}}
        Base dimensions typically: mass, length, time, temperature, current

    Returns
    -------
    D : np.ndarray of shape (n_vars, n_dims)
    var_order : list of str
        Variable names in the same row order as D.
    """
    all_vars = list(variables.keys())
    all_dims = sorted(
        {d for v in variables.values() for d in v.keys()}
    )
    m, n = len(all_vars), len(all_dims)
    D = np.zeros((m, n), dtype=float)
    for i, var in enumerate(all_vars):
        for j, dim in enumerate(all_dims):
            D[i, j] = float(variables[var].get(dim, 0))
    return D, all_vars


# Export public API
__all__ = [
    "D5_CONF_VERSION",
    "DimensionlessGroup",
    "_build_definition_string",
    "_reduce_exponents",
    "_systematic_group_name",
    "build_exponent_matrix",
    "build_integer_dimension_matrix",
    "discover_pi_groups",
    "exact_integer_rank",
    "exact_nullspace_basis",
]