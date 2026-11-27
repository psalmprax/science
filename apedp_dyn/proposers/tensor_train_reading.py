"""A tensor train fitted to the rows of a table, and read as a formula where it is a product.

Status: experimental

**Not in force.** Used by ``scripts/study_symbolic_tensor_train.py`` (the feasibility check of 27 November
2026), by the fallback chain of ``discovery_l2/fallbacks.py`` and by their tests. It is a proposer's
tool: what it reads goes to the referee like any proposal (AGENTS.md 1.1).

**The idea.** A law of many inputs made of sums and products of one-input functions is a tensor train

    y = G1(x1) G2(x2) ... Gd(xd)

whose cores ``Gi(xi)`` are small matrices of one-input functions (a row at the left end, a column at the
right). One product of one-input functions is rank one; a sum of ``r`` products has rank at most ``r`` at
every cut. The cost of a fit grows in proportion to the number of inputs.

**What is here.**

* ``Train``: every entry of every core a cubic spline of its input. With the other cores held the model
  is linear in one core, so a fit is alternating least squares, with a smoothness penalty.
* ``grown`` and ``choose``: ranks 1, 2, 3 at every cut grown one product at a time (a product fitted to
  what is left, then every core refined), and the smallest rank that no higher rank beats on rows left
  out of the fit (five folds, squared errors compared row by row).
* ``read_product``: where the chosen rank is one, each core is one function of one input up to a factor.
  ``block_search`` proposes laws for it, the simplest is taken whose whole (that law in place of the
  core, the other cores as fitted) misses the real rows by no more than ``MARGIN`` times what the spline
  model misses rows left out of its fit, and the product is refitted, every constant free, on the real
  rows.

**What it does not do yet.** Read a formula at rank two or three: a core of rank two holds its functions
only up to a mixing of its entries, and unmixing them is not written. At 5 % noise the rank choice
under-reads (it took six laws that are not products for products on the development tables,
``benchmarks/symbolic_tensor_train/dev_probe_1.json``), so what ``read_product`` returns there is often
not accurate; the referee is what stands between such a reading and a report.
"""
from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np
import sympy as sp
from scipy.interpolate import BSpline
from scipy.optimize import least_squares

KNOTS_INSIDE, DEGREE = 10, 3
LAMBDA = 1.0e-3            #: weight of the smoothness penalty (second differences of the spline's coefficients)
RANK_MAX = 3
FOLDS, STARTS, SWEEPS = 5, 3, 60
Z_BETTER = 2.0             #: a higher rank is taken only if it is better on left-out rows by this many standard errors
GRID = 200
MARGIN = 1.02
MAX_INPUTS = 9             #: a wider table than this is not read


def basis(x: np.ndarray, low: float, high: float) -> np.ndarray:
    """Cubic B-splines on ``[low, high]``: one column a spline, one row a value of the input."""
    knots = np.r_[[low] * DEGREE, np.linspace(low, high, KNOTS_INSIDE + 2), [high] * DEGREE]
    columns: np.ndarray = np.asarray(BSpline.design_matrix(np.clip(x, low, high), knots, DEGREE).toarray())
    return columns


N_BASIS = KNOTS_INSIDE + DEGREE + 1
_D2 = np.diff(np.eye(N_BASIS), 2, axis=0)
_PENALTY = _D2.T @ _D2


class Train:
    """A tensor train whose cores hold splines of one input: ``cores[i][a, j, b]`` is the weight of spline ``j`` in entry ``(a, b)``."""

    def __init__(self, inputs: int, ranks: Sequence[int], seed: int = 0) -> None:
        self.inputs, self.ranks = inputs, [1, *ranks, 1]
        rng = np.random.default_rng(seed)
        self.cores = []
        for i in range(inputs):
            core = 0.3 * rng.standard_normal((self.ranks[i], N_BASIS, self.ranks[i + 1]))
            core[0, :, 0] += 1.0                      # B-splines sum to one: the constant function
            self.cores.append(core)

    def _matrices(self, bases: Sequence[np.ndarray], i: int) -> np.ndarray:
        matrices: np.ndarray = np.einsum("ajb,nj->nab", self.cores[i], bases[i])
        return matrices

    def predict(self, bases: Sequence[np.ndarray]) -> np.ndarray:
        left = np.ones((len(bases[0]), 1))
        for i in range(self.inputs):
            left = np.einsum("na,nab->nb", left, self._matrices(bases, i))
        predicted: np.ndarray = np.asarray(left[:, 0])
        return predicted

    def fit(self, bases: Sequence[np.ndarray], y: np.ndarray, sweeps: int = 40, tol: float = 1.0e-9) -> Train:
        """Alternating least squares from the cores as they are: one core at a time, the others held."""
        n, last = len(y), np.inf
        order = [*range(self.inputs), *range(self.inputs - 2, 0, -1)]
        for _ in range(sweeps):
            for i in order:
                left = np.ones((n, 1))
                for k in range(i):
                    left = np.einsum("na,nab->nb", left, self._matrices(bases, k))
                right = np.ones((n, 1))
                for k in range(self.inputs - 1, i, -1):
                    right = np.einsum("nab,nb->na", self._matrices(bases, k), right)
                ra, rb = self.ranks[i], self.ranks[i + 1]
                design = np.einsum("na,nj,nb->najb", left, bases[i], right).reshape(n, ra * N_BASIS * rb)
                scale = float(np.sqrt(np.mean(design ** 2))) + 1.0e-300
                design = design / scale
                penalty = np.kron(np.kron(np.eye(ra), _PENALTY), np.eye(rb))
                gram = design.T @ design / n + LAMBDA * penalty + 1.0e-10 * np.eye(design.shape[1])
                self.cores[i] = (np.linalg.solve(gram, design.T @ y / n) / scale).reshape(ra, N_BASIS, rb)
                # the whole is unchanged when a factor moves from one core to its neighbour: keep the cores of one size
                size = float(np.sqrt(np.mean(self.cores[i] ** 2))) + 1.0e-300
                neighbour = i + 1 if i + 1 < self.inputs else i - 1
                self.cores[i] /= size
                self.cores[neighbour] *= size
            error = float(np.mean((self.predict(bases) - y) ** 2))
            if abs(last - error) < tol * max(error, 1.0e-300):
                break
            last = error
        return self


def add(first: Train, second: Train) -> Train:
    """The sum of two tensor trains is a tensor train: the cores side by side, joined at the two ends."""
    out = Train(first.inputs, [a + b for a, b in zip(first.ranks[1:-1], second.ranks[1:-1], strict=True)])
    for i in range(first.inputs):
        one, two = first.cores[i], second.cores[i]
        if i == 0:
            core = np.concatenate([one, two], axis=2)
        elif i == first.inputs - 1:
            core = np.concatenate([one, two], axis=0)
        else:
            core = np.zeros((one.shape[0] + two.shape[0], N_BASIS, one.shape[2] + two.shape[2]))
            core[: one.shape[0], :, : one.shape[2]] = one
            core[one.shape[0]:, :, one.shape[2]:] = two
        out.cores[i] = core
    return out


def _mse(model: Train, bases: Sequence[np.ndarray], y: np.ndarray) -> float:
    return float(np.mean((model.predict(bases) - y) ** 2))


def grown(bases: Sequence[np.ndarray], y: np.ndarray, rank_max: int = RANK_MAX, starts: int = STARTS) -> list[Train]:
    """Models of rank 1 to ``rank_max`` at every cut: one more product fitted to what is left, then every core refined."""
    inputs, out, current, left = len(bases), [], None, y.copy()
    for _ in range(rank_max):
        one = min((Train(inputs, [1] * (inputs - 1), seed=s).fit(bases, left) for s in range(starts)),
                  key=lambda m: _mse(m, bases, left))
        current = one if current is None else add(current, one)
        current.fit(bases, y, sweeps=SWEEPS)
        out.append(copy.deepcopy(current))
        left = y - current.predict(bases)
    return out


def choose(bases: Sequence[np.ndarray], y: np.ndarray, rank_max: int = RANK_MAX, folds: int = FOLDS,
           seed: int = 1) -> dict[str, Any]:
    """The smallest rank that no higher rank beats on rows left out of the fit; the model of that rank fitted on all rows."""
    n = len(y)
    size = float(np.sqrt(np.mean(y ** 2)))
    order = np.random.default_rng(seed).permutation(n)
    squared = np.zeros((rank_max, n))
    for fold in range(folds):
        test = order[fold::folds]
        train = np.setdiff1d(order, test)
        for r, model in enumerate(grown([b[train] for b in bases], y[train] / size, rank_max)):
            squared[r, test] = (size * model.predict([b[test] for b in bases]) - y[test]) ** 2
    rank = 0
    for higher in range(1, rank_max):
        gain = squared[rank] - squared[higher]
        if gain.mean() / (gain.std() / np.sqrt(n) + 1.0e-300) > Z_BETTER:
            rank = higher
    model = grown(bases, y / size, rank + 1)[rank]
    model.cores[0] = model.cores[0] * size
    return {"rank": rank + 1, "model": model, "left_out_rmse": float(np.sqrt(squared[rank].mean())),
            "left_out_rmse_by_rank": [float(np.sqrt(row.mean())) for row in squared]}



def stand_in(scored: Sequence[tuple[float, int, bool, str]], yard: float, margin: float = MARGIN) -> tuple[str | None, bool]:
    """Of ``(error on the real rows, blocks, has an additive constant, law)``: the simplest within the yardstick, else the best fit.

    Simplest: fewest blocks, then no additive constant, then the smaller error.
    """
    passing = [item for item in scored if item[0] <= margin * yard]
    if passing:
        return min(passing, key=lambda item: (item[1], item[2], item[0]))[3], True
    return (min(scored, key=lambda item: item[0])[3], False) if scored else (None, False)


def read_core(i: int, X: np.ndarray, y: np.ndarray, model: Train, bases: Sequence[np.ndarray], grid: np.ndarray,
              low: float, high: float, yard: float, *, name: str | None = None) -> tuple[str | None, bool]:
    """A law of one input in place of core ``i`` of a rank-one model, the other cores as fitted, judged on the real rows."""
    from apedp_dyn.proposers import block_search

    others = np.ones(len(y))
    for k in range(model.inputs):
        if k != i:
            others = others * (bases[k] @ model.cores[k][0, :, 0])
    on_grid = basis(grid, low, high) @ model.cores[i][0, :, 0]
    name = sp.Symbol(name or f"x{i}")
    scored = []
    for _, law, blocks in block_search.search(grid, on_grid / float(np.sqrt(np.mean(on_grid ** 2))), name=str(name)):
        with np.errstate(all="ignore"):
            values = np.asarray(sp.lambdify(name, sp.sympify(law, locals={str(name): name}), "numpy")(X[:, i]),
                                dtype=float) * np.ones(len(y))
        if not np.all(np.isfinite(values)):
            continue
        whole = others * values
        factor = float(np.dot(whole, y) / (np.dot(whole, whole) + 1.0e-300))
        scored.append((float(np.sqrt(np.mean((factor * whole - y) ** 2))), int(blocks), not law.startswith("("), law))
    return stand_in(scored, yard)


def polish(expr: sp.Expr, names: Sequence[str], X: np.ndarray, y: np.ndarray) -> tuple[sp.Expr, float]:
    """Every constant of ``expr`` but the exponents refitted on the rows given."""
    exponents = {e for power in expr.atoms(sp.Pow) for e in power.exp.atoms(sp.Float)}
    floats = sorted(set(expr.atoms(sp.Float)) - exponents, key=str)
    params = [sp.Symbol(f"_p{k}") for k in range(len(floats))]
    template = expr.xreplace(dict(zip(floats, params, strict=True)))
    fn = sp.lambdify([*params, *[sp.Symbol(n) for n in names]], template, "numpy")
    size = float(np.sqrt(np.mean(y ** 2)))

    def residual(theta: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            values = np.asarray(fn(*theta, *X.T), dtype=float) * np.ones(len(y))
        missed: np.ndarray = np.asarray(np.where(np.isfinite(values), values, 1.0e6 * size) - y)
        return missed

    fit = least_squares(residual, np.array([float(f) for f in floats]), max_nfev=400)
    return (template.xreplace({p: sp.Float(float(v)) for p, v in zip(params, fit.x, strict=True)}),
            float(np.sqrt(np.mean(fit.fun ** 2))))



def read_product(X: np.ndarray, y: np.ndarray, names: Sequence[str], *, seed: int = 1) -> dict[str, Any]:
    """A table read as a product of one-input functions, if it is one: ``law`` (or ``None`` with ``why_not``), ``rank``,
    and the spline model's error on rows left out of its fit.

    Each input's splines span the range that input has in the rows given.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    inputs = X.shape[1]
    out: dict[str, Any] = {"law": None, "rank": None, "left_out_rmse": None, "why_not": None}
    if not 2 <= inputs <= MAX_INPUTS or len(y) < 10 * N_BASIS:
        out["why_not"] = "the table has one input, too many inputs or too few rows"
        return out
    ranges = [(float(np.min(X[:, i])), float(np.max(X[:, i]))) for i in range(inputs)]
    if any(high <= low for low, high in ranges) or not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        out["why_not"] = "an input does not vary, or a value is not finite"
        return out
    bases = [basis(X[:, i], *ranges[i]) for i in range(inputs)]
    chosen = choose(bases, y, seed=seed)
    out.update({"rank": chosen["rank"], "left_out_rmse": chosen["left_out_rmse"]})
    if chosen["rank"] != 1:
        out["why_not"] = f"rank {chosen['rank']}: only a product is read as a formula yet"
        return out
    parts = []
    for i in range(inputs):
        grid = np.linspace(*ranges[i], GRID)
        parts.append(read_core(i, X, y, chosen["model"], bases, grid, *ranges[i], chosen["left_out_rmse"], name=str(names[i])))
    if not all(law for law, _ in parts):
        out["why_not"] = "a core has no law of one input"
        return out
    symbols = {str(n): sp.Symbol(str(n)) for n in names}
    whole: sp.Expr = sp.Integer(1)
    for law, _ in parts:
        whole = whole * sp.sympify(law, locals=symbols)
    read, error = polish(whole, [str(n) for n in names], X, y)
    out.update({"law": str(read), "refit_rmse": error, "stand_ins_within_the_yardstick": [bool(ok) for _, ok in parts]})
    return out
