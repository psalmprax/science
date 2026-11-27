#!/usr/bin/env python3
"""Development probes of four search ideas an outside assistant proposed on 27 November 2026.

Status: audit-only

**Development, not a result.** Each idea came with a "kill test" in the proposer's own words. These are those tests,
written in a few dozen lines each, on generated rows and on the three real tables that are open for development. The
held-back real tables are not read. Nothing here is a proposer of the route, and nothing is in force.

* ``implicit``: a law written as P(x, y) = 0. Library: monomials in the inputs and y up to a total degree. Every support
  of two to four monomials is fitted by the null vector of its design, reweighted by 1/|dP/dy| (the first-order
  distance in y), and supports are ranked by a BIC on that distance. Scored: is the first support the law's own; is its
  explicit branch accurate inside the measured range and outside it. Three controls are not algebraic at this degree.
* ``real``: the same search on the three development tables (Kepler, Langmuir, Leavitt): where the known law's own
  form stands in the ranking.
* ``gradient``: if y = Phi(u(x_A), x_B), the gradient of y in the x_A directions keeps its direction when x_B changes.
  Ten laws of four and five inputs; a Gaussian process fitted to 1,000 noisy rows supplies the gradient; the exact
  gradient labels every subset of two and three inputs. Does the statistic tell compositions from the rest?
* ``weak``: a function of one input satisfies sum_k p_k(z) f^(k) = 0 with polynomial p_k. Integrated against bumps, no
  derivative of the data is taken. Fifteen functions, rows evenly spaced and at random positions. Any equation the
  function truly satisfies counts.
* ``lagrangian``: the weak Euler-Lagrange form on a noisy trajectory of three conservative systems. The integral is
  of dL/dq * psi + dL/dqdot * dpsi/dt (the proposal printed a minus there; with it the potential comes out with the
  wrong sign). It needs one derivative of the data, from a smoothing spline.

    python scripts/probe_outside_proposals.py all --out benchmarks/outside_proposals/dev_probes.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from numpy.polynomial import Polynomial

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

TAG = "DEV-OUTSIDE-PROPOSALS-2026-11"
R2_ACCURACY = 0.999
_trapezoid: Callable[..., Any] = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]  # numpy 1 and 2


# --------------------------------------------------------------------------------------------- the implicit form
def exponents(n_vars: int, degree: int) -> list[tuple[int, ...]]:
    return [e for e in itertools.product(range(degree + 1), repeat=n_vars) if 0 < sum(e) <= degree] + [tuple([0] * n_vars)]


def _columns(V: np.ndarray, exps: Sequence[tuple[int, ...]]) -> np.ndarray:
    return np.column_stack([np.prod(V ** np.asarray(e), axis=1) for e in exps])


def _d_dy(V: np.ndarray, exps: Sequence[tuple[int, ...]], coef: np.ndarray) -> np.ndarray:
    out = np.zeros(len(V))
    for e, c in zip(exps, coef, strict=True):
        if e[-1] > 0:
            lowered = list(e)
            lowered[-1] -= 1
            out += c * e[-1] * np.prod(V ** np.asarray(lowered), axis=1)
    return out


def fit_support(V: np.ndarray, exps: Sequence[tuple[int, ...]], iters: int = 4) -> tuple[np.ndarray, np.ndarray] | None:
    """The coefficients of one support and each row's first-order distance in y, or None if it cannot be fitted."""
    A = _columns(V, exps)
    scale = np.linalg.norm(A, axis=0)
    if np.any(scale == 0) or not np.all(np.isfinite(A)):
        return None
    w = np.ones(len(V))
    coef = np.zeros(len(exps))
    for _ in range(iters):
        try:
            coef = np.linalg.svd((A / scale) * w[:, None], full_matrices=False)[2][-1] / scale
        except np.linalg.LinAlgError:
            return None
        slope = np.abs(_d_dy(V, exps, coef))
        if not np.all(np.isfinite(slope)) or np.median(slope) == 0:
            return None
        w = 1.0 / np.maximum(slope, 1e-3 * np.median(slope))
    slope = _d_dy(V, exps, coef)
    floor = 1e-3 * np.median(np.abs(slope))
    return coef, (A @ coef) / np.where(np.abs(slope) > floor, slope, np.sign(slope + 1e-300) * floor)


def implicit_search(X: np.ndarray, y: np.ndarray, degree: int, max_terms: int) -> list[tuple[float, tuple[tuple[int, ...], ...], np.ndarray, float]]:
    """Every support of 2..max_terms monomials holding y and an input, best first: (bic, support, coefficients, distance)."""
    V = np.column_stack([X, y])
    n = len(V)
    exps = exponents(V.shape[1], degree)
    spread = float(np.std(y)) or 1.0
    ranked = []
    for k in range(2, max_terms + 1):
        if k >= n - 1:
            break
        for support in itertools.combinations(exps, k):
            if not any(e[-1] > 0 for e in support) or not any(sum(e[:-1]) > 0 for e in support):
                continue
            got = fit_support(V, support)
            if got is None:
                continue
            rss = float(np.sum(got[1] ** 2))
            if not np.isfinite(rss):
                continue
            rss = max(rss, n * (1e-7 * spread) ** 2)        # exact rows: every superset fits too; rounding is not rewarded
            ranked.append((n * float(np.log(rss / n)) + (k - 1) * float(np.log(n)), support, got[0], float(np.sqrt(rss / n))))
    ranked.sort(key=lambda row: row[0])
    return ranked


def solve_y(support: Sequence[tuple[int, ...]], coef: np.ndarray, X: np.ndarray, y_near: np.ndarray) -> np.ndarray:
    """The real root in y of P(x, y) = 0 nearest to ``y_near`` (the kindest choice of branch), row by row."""
    degree = max(e[-1] for e in support)
    out = np.full(len(X), np.nan)
    for i, x in enumerate(X):
        poly = np.zeros(degree + 1)
        for e, c in zip(support, coef, strict=True):
            poly[degree - e[-1]] += c * np.prod(x ** np.asarray(e[:-1]))
        roots = np.roots(poly) if np.any(poly[:-1] != 0) else np.array([])
        real = roots[np.abs(roots.imag) < 1e-7 * (1 + np.abs(roots.real))].real
        if len(real):
            out[i] = real[np.argmin(np.abs(real - y_near[i]))]
    return out


def _r2(truth: np.ndarray, predicted: np.ndarray) -> float:
    if not np.all(np.isfinite(predicted)):
        return float("-inf")
    return float(1.0 - np.sum((truth - predicted) ** 2) / np.sum((truth - truth.mean()) ** 2))


#: name: (f, the inputs' ranges, the law's own support in (inputs..., y) exponents or None for a control, degree, rows of
#: the real table of that form or 20)
IMPLICIT_LAWS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], list[tuple[float, float]], set[tuple[int, ...]] | None, int, int]] = {
    "time dilation a*(sqrt(1 - v^2/b^2) - 1)": (lambda X: np.sqrt(1 - X[:, 0] ** 2) - 1, [(0.1, 0.9)], {(0, 2), (0, 1), (2, 0)}, 2, 7),
    "Langmuir a*p/(1 + b*p)": (lambda X: 2.0 * X[:, 0] / (1 + 0.8 * X[:, 0]), [(0.1, 10.0)], {(1, 1), (0, 1), (1, 0)}, 2, 17),
    "Kepler with two masses c*d^1.5/sqrt(m1 + m2)": (lambda X: X[:, 2] ** 1.5 / np.sqrt(X[:, 0] + X[:, 1]), [(0.5, 2.0), (0.05, 1.0), (0.1, 5.0)],
                                                     {(1, 0, 0, 2), (0, 1, 0, 2), (0, 0, 3, 0)}, 3, 12),
    "Hubble a*D": (lambda X: 0.5 * X[:, 0], [(0.03, 2.0)], {(0, 1), (1, 0)}, 2, 32),
    "lens u*v/(u + v)": (lambda X: X[:, 0] * X[:, 1] / (X[:, 0] + X[:, 1]), [(0.5, 5.0), (0.5, 5.0)], {(1, 0, 1), (0, 1, 1), (1, 1, 0)}, 2, 20),
    "energy sqrt(p^2 + m^2)": (lambda X: np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2), [(0.2, 3.0), (0.2, 3.0)], {(0, 0, 2), (2, 0, 0), (0, 2, 0)}, 2, 20),
    "control: exponential 2*exp(-0.7*x)": (lambda X: 2.0 * np.exp(-0.7 * X[:, 0]), [(0.2, 4.0)], None, 2, 20),
    "control: logarithm a*log(x) + b (Tully and Fisher's form)": (lambda X: -7.0 * np.log10(X[:, 0]) - 2.0, [(100.0, 600.0)], None, 2, 18),
    "control: sine sin(1.3*x) + 2": (lambda X: np.sin(1.3 * X[:, 0]) + 2.0, [(0.2, 4.0)], None, 2, 20),
}


def _draw(law: str, n: int, noise: float, rng: np.random.Generator, widen: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f, ranges = IMPLICIT_LAWS[law][0], IMPLICIT_LAWS[law][1]
    X = np.column_stack([rng.uniform(lo, lo + (hi - lo) * widen, n) for lo, hi in ranges])
    with np.errstate(all="ignore"):
        clean = f(X)
    return X, clean + noise * float(np.nanstd(clean)) * rng.standard_normal(n), clean


def implicit_trial(law: str, n: int, noise: float, seed: int) -> dict[str, Any] | None:
    truth, degree = IMPLICIT_LAWS[law][2], IMPLICIT_LAWS[law][3]
    X, y, _ = _draw(law, n, noise, np.random.default_rng([seed, n, round(noise * 1e4)]))
    ranked = implicit_search(X, y, degree, 3 if X.shape[1] >= 3 else 4)
    if not ranked:
        return None
    _, support, coef, _ = ranked[0]
    X_in, _, clean_in = _draw(law, 400, 0.0, np.random.default_rng([seed, 1]))
    X_out, _, clean_out = _draw(law, 400, 0.0, np.random.default_rng([seed, 2]), widen=1.5)
    ok = np.isfinite(clean_out)
    with np.errstate(all="ignore"):
        inside = _r2(clean_in, solve_y(support, coef, X_in, clean_in)) >= R2_ACCURACY
        outside = _r2(clean_out[ok], solve_y(support, coef, X_out[ok], clean_out[ok])) >= R2_ACCURACY
    place = next((i for i, row in enumerate(ranked[:20]) if set(row[1]) == truth), None) if truth else None
    return {"own_form_first": place == 0, "own_form_in_first_three": place is not None and place < 3, "inside": bool(inside),
            "outside": bool(outside), "terms": len(support)}


def probe_implicit(seeds: int = 30, laws: Sequence[str] | None = None, sizes: Sequence[int] | None = None) -> list[dict[str, Any]]:
    rows = []
    for law in laws or list(IMPLICIT_LAWS):
        for n in sizes or sorted({IMPLICIT_LAWS[law][4], 200}):
            for noise in (0.0, 0.01, 0.05):
                got = [t for t in (implicit_trial(law, n, noise, seed) for seed in range(seeds if n <= 40 else max(2, seeds // 4))) if t]
                control = IMPLICIT_LAWS[law][2] is None
                rows.append({"law": law, "rows": n, "noise": noise, "draws": len(got),
                             "own_form_first": None if control else sum(t["own_form_first"] for t in got),
                             "own_form_in_first_three": None if control else sum(t["own_form_in_first_three"] for t in got),
                             "accurate_inside": sum(t["inside"] for t in got), "accurate_outside": sum(t["outside"] for t in got),
                             "terms_chosen": sorted({t["terms"] for t in got})})
    return rows


REAL_FORMS = {"kepler_solar_system": {(0, 2), (3, 0)}, "langmuir_table_ix": {(0, 1), (1, 0), (1, 1)}, "leavitt": {(0, 1), (1, 0), (0, 0)}}


def probe_real() -> list[dict[str, Any]]:
    """The implicit search on the three real development tables. Columns are divided by their medians (a change of units)."""
    import real_data

    rows = []
    for key, own in REAL_FORMS.items():
        got = real_data.load(key)
        X = np.column_stack([np.asarray(got["X"][name], dtype=float) for name in got["X"]])
        y = np.asarray(got["y"], dtype=float)
        ranked = implicit_search(X / np.median(np.abs(X), axis=0), y / np.median(np.abs(y)), 3, 4 if len(y) > 8 else 3)
        overall = next(i for i, row in enumerate(ranked) if set(row[1]) == own)
        same = [row for row in ranked if len(row[1]) == len(own)]
        rows.append({"table": key, "rows": len(y), "known_law": got["table"].get("law"), "supports_ranked": len(ranked),
                     "own_form_rank": overall + 1, "own_form_rank_among_its_size": next(i for i, row in enumerate(same) if set(row[1]) == own) + 1,
                     "supports_of_its_size": len(same), "own_form_distance": round(ranked[overall][3], 4), "best_distance": round(ranked[0][3], 4),
                     "terms_of_the_best": len(ranked[0][1])})
    return rows


# --------------------------------------------------------------------------------------------- gradient rank
_x = sp.symbols("x1:6")
GRADIENT_LAWS = {
    "sin(x1*x2)*x3 + x4": sp.sin(_x[0] * _x[1]) * _x[2] + _x[3],
    "x4/sqrt(x1^2 + x2^2 + x3^2)": _x[3] / sp.sqrt(_x[0] ** 2 + _x[1] ** 2 + _x[2] ** 2),
    "exp(-(x1 + 2*x2))*x3^2 + x4": sp.exp(-(_x[0] + 2 * _x[1])) * _x[2] ** 2 + _x[3],
    "log(1 + x1*x2)*(x3 + x4)": sp.log(1 + _x[0] * _x[1]) * (_x[2] + _x[3]),
    "sin(x1*x2 + x3) + x4": sp.sin(_x[0] * _x[1] + _x[2]) + _x[3],
    "x3*cos(x1^2 + x2^2) + x4*x5": _x[2] * sp.cos(_x[0] ** 2 + _x[1] ** 2) + _x[3] * _x[4],
    "sqrt(x1*x2 + x3)/(1 + x4)": sp.sqrt(_x[0] * _x[1] + _x[2]) / (1 + _x[3]),
    "tanh(x1 - x2) + x3*x4": sp.tanh(_x[0] - _x[1]) + _x[2] * _x[3],
    "(x1^2 + x2^2)*exp(-x3) + x4": (_x[0] ** 2 + _x[1] ** 2) * sp.exp(-_x[2]) + _x[3],
    "x4*exp(-x1*x2/x3) + x5": _x[3] * sp.exp(-_x[0] * _x[1] / _x[2]) + _x[4],
}
LOW, HIGH, ANCHORS, VARIED = 0.5, 2.0, 16, 12


def rank_statistic(grad: Callable[[np.ndarray], np.ndarray], X: np.ndarray, subset: Sequence[int], rng: np.random.Generator) -> float:
    """Median over anchors of sigma2/sigma1 of the gradients in the subset's directions, x_A fixed and x_B varied."""
    rest = [j for j in range(X.shape[1]) if j not in subset]
    ratios = []
    for _ in range(ANCHORS):
        points = np.tile(X[rng.integers(len(X))], (VARIED, 1))
        points[:, rest] = X[rng.choice(len(X), VARIED, replace=False)][:, rest]
        s = np.linalg.svd(grad(points)[:, list(subset)], compute_uv=False)
        if s[0] > 0:
            ratios.append(s[1] / s[0])
    return float(np.median(ratios)) if ratios else float("nan")


def _surrogate_gradient(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> Callable[[np.ndarray], np.ndarray]:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(length_scale=2.0 * X.std(axis=0), length_scale_bounds=(0.05, 50.0)) + WhiteKernel(1e-3, (1e-8, 1.0))
    part = rng.choice(len(X), min(300, len(X)), replace=False)       # the kernel's settings from a part, the fit from all rows
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        small = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0).fit(X[part], y[part])
        full = GaussianProcessRegressor(kernel=small.kernel_, normalize_y=True, optimizer=None).fit(X, y)
    h = 1e-3 * (HIGH - LOW)

    def grad(points: np.ndarray) -> np.ndarray:
        d = points.shape[1]
        up = np.repeat(points, d, axis=0)
        down = up.copy()
        which = np.tile(np.arange(d), len(points))
        up[np.arange(len(up)), which] += h
        down[np.arange(len(down)), which] -= h
        out: np.ndarray = ((full.predict(up) - full.predict(down)) / (2 * h)).reshape(len(points), d)
        return out

    return grad


def probe_gradient(rows: int = 1000, laws: Sequence[str] | None = None, noises: Sequence[float] = (0.0, 0.01, 0.05)) -> list[dict[str, Any]]:
    out = []
    for index, name in enumerate(laws or list(GRADIENT_LAWS)):
        expr = GRADIENT_LAWS[name]
        used = sorted(expr.free_symbols, key=lambda s: s.name)
        f = sp.lambdify(used, expr, "numpy")
        partials = sp.lambdify(used, [sp.diff(expr, v) for v in used], "numpy")

        def exact(points: np.ndarray, partials: Any = partials) -> np.ndarray:
            return np.column_stack([np.asarray(g, dtype=float) * np.ones(len(points)) for g in partials(*points.T)])

        X = np.random.default_rng([2026112801, index]).uniform(LOW, HIGH, (rows, len(used)))
        clean = np.asarray(f(*X.T), dtype=float)
        subsets = [s for k in (2, 3) for s in itertools.combinations(range(len(used)), k)]
        truth = {s: rank_statistic(exact, X, s, np.random.default_rng(1)) for s in subsets}
        for noise in noises:
            y = clean + noise * clean.std() * np.random.default_rng([7, index]).standard_normal(rows)
            grad = _surrogate_gradient(X, y, np.random.default_rng(3))
            got = {s: rank_statistic(grad, X, s, np.random.default_rng(1)) for s in subsets}
            compositions = [got[s] for s in subsets if truth[s] < 1e-6]
            others = [got[s] for s in subsets if truth[s] >= 1e-6]
            error = np.median(np.linalg.norm(grad(X[:200]) - exact(X[:200]), axis=1) / np.linalg.norm(exact(X[:200]), axis=1))
            out.append({"law": name, "inputs": len(used), "noise": noise, "compositions": len(compositions), "others": len(others),
                        "told_apart": bool(max(compositions) < min(others)),
                        "share_of_pairs_in_the_right_order": float(np.mean([(c < o) + 0.5 * (c == o) for c in compositions for o in others])),
                        "largest_for_a_composition": round(max(compositions), 4), "smallest_for_another": round(min(others), 4),
                        "smallest_for_another_with_the_exact_gradient": round(min(v for v in truth.values() if v >= 1e-6), 4),
                        "gradient_relative_error": round(float(error), 4)})
    return out


# --------------------------------------------------------------------------------------------- the weak form, one input
ORDER, DEGREE, BUMPS, POWER = 2, 2, 24, 6
TERMS = [(k, i) for k in range(ORDER + 1) for i in range(DEGREE + 1)]          #: (order of the derivative, power of z)
WEAK_FUNCTIONS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], tuple[float, float], str]] = {
    "sin(1.0*z + 0.3)": (lambda z: np.sin(1.0 * z + 0.3), (0.0, 6.0), "sine"), "sin(2.0*z + 1.0)": (lambda z: np.sin(2.0 * z + 1.0), (0.0, 6.0), "sine"),
    "sin(3.5*z)": (lambda z: np.sin(3.5 * z), (0.0, 6.0), "sine"),
    "exp(0.5*z)": (lambda z: np.exp(0.5 * z), (0.0, 4.0), "exponential"), "exp(-1.2*z)": (lambda z: np.exp(-1.2 * z), (0.0, 4.0), "exponential"),
    "exp(-0.3*z)": (lambda z: np.exp(-0.3 * z), (0.0, 4.0), "exponential"),
    "(z + 1)^-2": (lambda z: (z + 1.0) ** -2, (0.0, 4.0), "pole"), "(z + 0.5)^-2": (lambda z: (z + 0.5) ** -2, (0.0, 4.0), "pole"),
    "(z - 6)^-2": (lambda z: (z - 6.0) ** -2, (0.0, 4.0), "pole"),
    "log(z + 1)": (lambda z: np.log(z + 1.0), (0.0, 4.0), "logarithm"), "log(2*z + 0.5)": (lambda z: np.log(2.0 * z + 0.5), (0.0, 4.0), "logarithm"),
    "log(0.5*z + 3)": (lambda z: np.log(0.5 * z + 3.0), (0.0, 4.0), "logarithm"),
    "sqrt(1 - z^2)": (lambda z: np.sqrt(1.0 - z ** 2), (-0.9, 0.9), "root"), "sqrt(4 - z^2)": (lambda z: np.sqrt(4.0 - z ** 2), (-1.8, 1.8), "root"),
    "sqrt(1 - z^2/9)": (lambda z: np.sqrt(1.0 - z ** 2 / 9.0), (-2.7, 2.7), "root"),
}


def weak_matrix(z: np.ndarray, f: np.ndarray, lo: float, hi: float) -> np.ndarray:
    order = np.argsort(z)
    z, f = z[order], f[order]
    half = (hi - lo) / 3.0
    M = np.zeros((BUMPS, len(TERMS)))
    for j, centre in enumerate(np.linspace(lo + half, hi - half, BUMPS)):
        inside = (z > centre - half) & (z < centre + half)
        if inside.sum() < 6:
            continue
        u = Polynomial([-centre / half, 1.0 / half])
        bump = (Polynomial([1.0]) - u * u) ** POWER
        for col, (k, i) in enumerate(TERMS):
            g = Polynomial([0.0] * i + [1.0]) * bump
            M[j, col] = (-1) ** k * _trapezoid(f[inside] * (g.deriv(k) if k else g)(z[inside]), z[inside])
    return M


def rank_equations(M: np.ndarray) -> list[tuple[float, float, frozenset[tuple[int, int]]]]:
    scale = np.linalg.norm(M, axis=0)
    out = []
    for size in (2, 3):
        for support in itertools.combinations(range(len(TERMS)), size):
            if all(TERMS[s][0] == 0 for s in support) or np.any(scale[list(support)] == 0):
                continue                                         # no derivative in it: not an equation for f
            s = np.linalg.svd(M[:, support] / scale[list(support)], compute_uv=False)
            out.append((len(M) * float(np.log(max(s[-1] / s[0], 1e-12) ** 2)) + (size - 1) * float(np.log(len(M))), float(s[-1] / s[0]),
                        frozenset(TERMS[t] for t in support)))
    out.sort(key=lambda row: row[0])
    return out


def true_equations(name: str) -> list[frozenset[tuple[int, int]]]:
    """Every support the function satisfies exactly (multiples and derivatives of its equation are true statements about
    it), read from a dense table without noise."""
    f, (lo, hi), _ = WEAK_FUNCTIONS[name]
    z = np.linspace(lo, hi, 6001)
    return [row[2] for row in rank_equations(weak_matrix(z, f(z), lo, hi)) if row[1] < 1e-5]


def probe_weak(seeds: int = 20, functions: Sequence[str] | None = None, sizes: Sequence[int] = (50, 500)) -> list[dict[str, Any]]:
    names = list(functions or WEAK_FUNCTIONS)
    right = {name: true_equations(name) for name in names}
    out = []
    for positions in ("evenly spaced", "random"):
        for n in sizes:
            for noise in (0.0, 0.01, 0.05):
                first: dict[str, list[int]] = {}
                for name in names:
                    f, (lo, hi), family = WEAK_FUNCTIONS[name]
                    for seed in range(seeds):
                        rng = np.random.default_rng([seed, n, round(noise * 1e4)])
                        z = np.linspace(lo, hi, n) if positions == "evenly spaced" else rng.uniform(lo, hi, n)
                        clean = f(z)
                        ranked = rank_equations(weak_matrix(z, clean + noise * clean.std() * rng.standard_normal(n), lo, hi))
                        first.setdefault(family, []).append(int(bool(ranked) and ranked[0][2] in right[name]))
                out.append({"positions": positions, "rows": n, "noise": noise, "draws": sum(len(v) for v in first.values()),
                            "a_true_equation_first": sum(sum(v) for v in first.values()),
                            "by_family": {family: f"{sum(v)} of {len(v)}" for family, v in first.items()}})
    out.append({"true_equations_of_120_supports": {name: len(right[name]) for name in names}})
    return out


# --------------------------------------------------------------------------------------------- the weak Lagrangian
LIBRARY: dict[str, tuple[Callable[..., np.ndarray], Callable[..., np.ndarray]]] = {   # name: (dphi/dq, dphi/dqdot)
    "qd^2": (lambda q, v: 0 * q, lambda q, v: 2 * v), "cos q": (lambda q, v: -np.sin(q), lambda q, v: 0 * q),
    "sin q": (lambda q, v: np.cos(q), lambda q, v: 0 * q), "q^2": (lambda q, v: 2 * q, lambda q, v: 0 * q),
    "q^4": (lambda q, v: 4 * q ** 3, lambda q, v: 0 * q), "qd^4": (lambda q, v: 0 * q, lambda q, v: 4 * v ** 3),
    "q^2 qd^2": (lambda q, v: 2 * q * v ** 2, lambda q, v: 2 * q ** 2 * v),
    "qd^2 cos q": (lambda q, v: -v ** 2 * np.sin(q), lambda q, v: 2 * v * np.cos(q)),
}
#: name: (right-hand side, start, the Lagrangian's terms, (term, over, ratio of their coefficients), the force terms of
#: the strong form and how the same ratio is read from their coefficients)
SYSTEMS: dict[str, tuple[Callable[[float, Sequence[float]], list[float]], list[float], frozenset[str], tuple[str, str, float], str]] = {
    "pendulum: qdd = -2.25*sin(q)": (lambda t, s: [s[1], -2.25 * np.sin(s[0])], [2.0, 0.0], frozenset({"qd^2", "cos q"}), ("cos q", "qd^2", 4.5), "sine"),
    "double well: qdd = q - q^3": (lambda t, s: [s[1], s[0] - s[0] ** 3], [1.6, 0.0], frozenset({"qd^2", "q^2", "q^4"}), ("q^2", "qd^2", 1.0), "cubic"),
    "soft spring: qdd = -q - 0.5*q^3": (lambda t, s: [s[1], -s[0] - 0.5 * s[0] ** 3], [1.5, 0.0], frozenset({"qd^2", "q^2", "q^4"}), ("q^2", "qd^2", -1.0),
                                        "cubic"),
}
SPAN, LAGRANGIAN_BUMPS, LAGRANGIAN_POWER = 20.0, 30, 4


def lagrangian_trial(system: str, n: int, noise: float, seed: int) -> dict[str, Any]:
    from scipy.integrate import solve_ivp
    from scipy.interpolate import UnivariateSpline

    rhs, start, terms, (top, bottom, ratio), force = SYSTEMS[system]
    t = np.linspace(0.0, SPAN, n)
    q_true = solve_ivp(rhs, (0.0, SPAN), start, t_eval=t, rtol=1e-10, atol=1e-10).y[0]
    sigma = noise * q_true.std()
    spline = UnivariateSpline(t, q_true + sigma * np.random.default_rng([seed, n, round(noise * 1e4)]).standard_normal(n), k=5,
                              s=n * sigma ** 2 if noise > 0 else 0)
    q, v, a = spline(t), spline.derivative(1)(t), spline.derivative(2)(t)
    names = list(LIBRARY)
    half = SPAN / 8.0
    M = np.zeros((LAGRANGIAN_BUMPS, len(names)))
    for j, centre in enumerate(np.linspace(half, SPAN - half, LAGRANGIAN_BUMPS)):
        inside = (t > centre - half) & (t < centre + half)
        u = Polynomial([-centre / half, 1.0 / half])
        psi = (Polynomial([1.0]) - u * u) ** LAGRANGIAN_POWER
        for col, name in enumerate(names):
            dq, dv = LIBRARY[name]
            M[j, col] = _trapezoid(dq(q[inside], v[inside]) * psi(t[inside]) + dv(q[inside], v[inside]) * psi.deriv()(t[inside]), t[inside])
    scale = np.linalg.norm(M, axis=0)
    ranked = []
    for size in (2, 3):
        for support in itertools.combinations(range(len(names)), size):
            _, s, vt = np.linalg.svd(M[:, support] / scale[list(support)], full_matrices=False)
            ranked.append((len(M) * float(np.log(max(s[-1] / s[0], 1e-5) ** 2)) + (size - 1) * float(np.log(len(M))),
                           frozenset(names[i] for i in support), dict(zip([names[i] for i in support], vt[-1] / scale[list(support)], strict=True))))
    ranked.sort(key=lambda row: row[0])
    forced = next(row[2] for row in ranked if row[1] == terms)
    columns = [np.sin(q)] if force == "sine" else [q, q ** 3]
    coef = np.linalg.lstsq(np.column_stack(columns)[20:-20], a[20:-20], rcond=None)[0]
    strong = -2 * coef[0] if force == "sine" else coef[0]
    return {"first": ranked[0][1] == terms, "weak_error": abs(forced[top] / forced[bottom] / ratio - 1), "strong_error": abs(strong / ratio - 1)}


def probe_lagrangian(seeds: int = 20, sizes: Sequence[int] = (200, 2000)) -> list[dict[str, Any]]:
    out = []
    for system in SYSTEMS:
        for n in sizes:
            for noise in (0.0, 0.01, 0.05):
                got = [lagrangian_trial(system, n, noise, seed) for seed in range(seeds)]
                out.append({"system": system, "rows": n, "noise": noise, "draws": seeds, "own_terms_first": sum(g["first"] for g in got),
                            "constant_error_weak_form": round(float(np.median([g["weak_error"] for g in got])), 4),
                            "constant_error_second_derivative": round(float(np.median([g["strong_error"] for g in got])), 4)})
    return out


PROBES: dict[str, Callable[[], list[dict[str, Any]]]] = {"implicit": probe_implicit, "real": probe_real, "gradient": probe_gradient, "weak": probe_weak,
                                                         "lagrangian": probe_lagrangian}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("probe", choices=[*PROBES, "all"])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    record: dict[str, Any] = {"tag": TAG, "development_not_a_result": True,
                              "what": "kill tests of four outside proposals, on generated rows and the three real development tables"}
    for name in PROBES if args.probe == "all" else [args.probe]:
        record[name] = PROBES[name]()
        print(name, json.dumps(record[name], indent=1)[:1500], flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
